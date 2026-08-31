// Copyright 2025 Neumove
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Two compliant-tendon Hill muscles that share one solver:
//
//   gaintype="millard_mtu"  OpenSim Millard2012EquilibriumMuscle (the damped variant, i.e.
//                           fiber_damping > 0), with OpenSim's own quintic-Bezier curves,
//                           baked at runtime by engine_muscle_bake.cc.
//   gaintype="hyfydy_mtu"   Hyfydy's `muscle_force_m2012fast`, which the Hyfydy User Manual
//                           v1.0.2 states implements Millard et al. (2013) but replaces the
//                           Bezier splines with polynomials, so its curves "differ slightly
//                           from the curves used in the OpenSim implementation". Closed
//                           form: two cubics, two rationals and a quadratic, no table and
//                           no transcendental.
//
// Both solve the SAME equilibrium, in the same state variable (fiber length l_ce), with the
// same fixed-width pennation model; only the four normalized curves differ. Writing them as
// one solver with two curve back-ends is deliberate -- it is what makes a muscle-model A/B
// a one-attribute change rather than a different code path with its own bugs.
//
//   R(l_ce) = f_T(l_T/l_slack) - cos(phi) * [ A f_L(l0) f_V(v0) + f_P(l0) + beta v0 ]
//
//   l0    = l_ce / l_opt
//   v0    = (l_ce - l_ce_prev) / (dt * v_max * l_opt)        backward Euler
//   phi   = asin(h / l_ce), h = l_opt sin(phi_opt)           fixed-width pennation
//   l_T   = l_mtu - l_ce cos(phi)
//   F_mtu = F_max f_T(l_T/l_slack)                           tendon force = what the path pulls with
//
// WHY BACKWARD EULER IN l_ce, AND NOT OpenSim's FIBER-VELOCITY SOLVE. OpenSim integrates l_ce
// as a continuous state and solves the equilibrium for dl_ce/dt at the current l_ce. Here l_ce
// is a per-actuator state advanced once per mj_step, so the natural discretisation is implicit
// in l_ce: it is unconditionally stable in the stiff tendon limit, needs one scalar Newton, and
// its fixed point is the same equilibrium. This is the same discretisation the existing
// gaintype="compliant_mtu" (Song) actuator uses, so all three muscle models step alike.
//
// PARAMETERS COME STRAIGHT FROM THE SOURCE MODEL. The 32 gainprm slots are a 1:1 map of an
// OpenSim Millard2012EquilibriumMuscle's properties -- including all four curve shapes, which
// real models do override (Rajagopal2016 retunes the active force-length curve model-wide, and
// RajagopalLaiUhlrich2023 retunes the passive curve per muscle) -- with zero meaning "OpenSim's
// default". Nothing has to be refitted or reinterpreted on the way in.
//
// THROUGHPUT. The three things that cost are curve evaluation, Jacobian evaluation and Newton
// trip count, and each is addressed directly:
//   * the Millard curves are read from a uniform quintic-Hermite table -- an index, a clamp and
//     a polynomial. No section scan and no nested Newton on the inner loop, which is what a
//     direct Bezier evaluation would cost. Value AND slope come from the same interpolant, so
//     the Jacobian differentiates exactly what the residual evaluates.
//   * the Jacobian is ANALYTIC. The existing Song path finite-differences it, which doubles the
//     curve evaluations per iteration and caps the attainable accuracy at sqrt(eps).
//   * the Newton is warm-started from the previous step's l_ce and stops on the residual, so a
//     muscle in a smooth regime converges in one to three iterations rather than a fixed count.
// The Hyfydy back-end additionally has no table at all and no transcendental, which is the
// point of it: it is the cheapest of the three muscle models this engine ships.

#include "engine/engine_muscle_mtu.h"

#include <math.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include "engine/engine_array_safety.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"


//------------------------------ Millard2012 curve table -------------------------------------------

// defined with the rest of the parameter handling below; needed here by mju_millardCurve
static void mtuResolveDefaults(const mjtNum* in, mjtNum* out);


// Value and slope of a baked curve at x. Either output may be NULL.
//
// Outside [x0, x1] this reproduces OpenSim's `y1 + dydx1*(x - x1)` EXACTLY rather than
// approximately: the anchors are the declared biomechanical values, not interpolated ones.
// The boundary knots carry those same anchor value and slope, so the interpolated and
// extrapolated branches agree at the seam. The seam is C1 and not C2, which is faithful --
// OpenSim's own curve is C1 there too, its extrapolation having zero curvature while the
// Bezier end does not.
void mju_curveTableEval(const mjCurveTable* t, mjtNum x, mjtNum* y, mjtNum* dydx) {
  if (x <= t->x0) {
    if (y) *y = t->a_y0 + t->a_d0*(x - t->x0);
    if (dydx) *dydx = t->a_d0;
    return;
  }
  if (x >= t->x1) {
    if (y) *y = t->a_y1 + t->a_d1*(x - t->x1);
    if (dydx) *dydx = t->a_d1;
    return;
  }

  mjtNum s = (x - t->x0)*t->inv_dx;
  int i = (int)s;                       // floor, since s >= 0 here
  if (i < 0) i = 0;
  if (i > t->n - 2) i = t->n - 2;

  mjtNum u = s - (mjtNum)i;
  mjtNum u2 = u*u, u3 = u2*u, u4 = u3*u, u5 = u4*u;
  mjtNum h = t->dx;
  const mjtNum* k = t->knot + 3*i;
  mjtNum y_i = k[0], d_i = k[1], e_i = k[2];
  mjtNum y_j = k[3], d_j = k[4], e_j = k[5];

  if (y) {
    // quintic Hermite basis on the unit cell: matches value, slope and curvature at both ends
    mjtNum H0 = 1 - 10*u3 + 15*u4 - 6*u5;
    mjtNum H1 = u - 6*u3 + 8*u4 - 3*u5;
    mjtNum H2 = 0.5*u2 - 1.5*u3 + 1.5*u4 - 0.5*u5;
    mjtNum G0 = 10*u3 - 15*u4 + 6*u5;
    mjtNum G1 = -4*u3 + 7*u4 - 3*u5;
    mjtNum G2 = 0.5*u3 - u4 + 0.5*u5;
    *y = H0*y_i + H1*h*d_i + H2*h*h*e_i + G0*y_j + G1*h*d_j + G2*h*h*e_j;
  }
  if (dydx) {
    // d/dx of the basis above; the 1/h cancels against the h on the slope terms
    mjtNum P0 = -30*u2 + 60*u3 - 30*u4;
    mjtNum P1 = 1 - 18*u2 + 32*u3 - 15*u4;
    mjtNum P2 = u - 4.5*u2 + 6*u3 - 2.5*u4;
    mjtNum Q0 = 30*u2 - 60*u3 + 30*u4;
    mjtNum Q1 = -12*u2 + 28*u3 - 15*u4;
    mjtNum Q2 = 1.5*u2 - 4*u3 + 2.5*u4;
    *dydx = (P0*y_i + Q0*y_j)*t->inv_dx + P1*d_i + Q1*d_j + (P2*e_i + Q2*e_j)*h;
  }
}


// public accessor, for checking a model's curves against OpenSim directly. `prm` is a gainprm
// block (zero means default); NULL asks for OpenSim's defaults throughout.
mjtNum mju_millardCurve(int curve, mjtNum x, const mjtNum* prm, mjtNum* deriv) {
  if (curve < 0 || curve >= mjNMUSCLECURVE) {
    mju_error("mju_millardCurve: unknown curve %d", curve);
  }

  mjtNum zero[mjNGAIN] = {0};
  mjtNum resolved[mjNGAIN];
  mtuResolveDefaults(prm ? prm : zero, resolved);

  const mjCurveTable* table[mjNMUSCLECURVE];
  mju_millardBakeCurves(resolved, " (mju_millardCurve)", table);

  mjtNum y;
  mju_curveTableEval(table[curve], x, &y, deriv);
  return y;
}


//------------------------------ Hyfydy muscle_force_m2012fast curves ------------------------------

// Every coefficient below is published in the Hyfydy User Manual v1.0.2 (7 April 2026),
// section "Actuator Forces". They are the polynomial replacements for the Millard Bezier
// splines, which is why a Hyfydy muscle and a Millard muscle with identical mechanical
// parameters do not produce identical forces -- the manual says as much.
//
// tendon:      f_T(e) = c_T1 e^2 + c_T2 e,   e = l_T/l_slack - 1, zero when slack
// active F-L:  f_L(l) = c_L1 (l-1)^3 + c_L2 (l-1)^2 + 1 on (r1, r2), zero outside
// passive F-L: f_P(l) = c_P1 (l-1)^3 + c_P2 (l-1)^2 for l > 1, zero below
// F-V:         0 for v <= -1; c_V1(v+1)/(c_V1-v) on (-1,0); (Fvmax v + c_V2)/(c_V2+v) for v >= 0
#define mjHYF_cT1    260.972
#define mjHYF_cT2    7.9706
#define mjHYF_cL1    1.5
#define mjHYF_cL2    (-2.75)
#define mjHYF_r1     0.46899
#define mjHYF_r2     1.80528
#define mjHYF_cV1    0.227
#define mjHYF_cV2    0.110
#define mjHYF_FVMAX  1.6
#define mjHYF_cP1    1.08027
// The manual prints the passive curve as `c_P1 (l-1)^3 + c_P1 (l-1)^2` -- c_P1 twice -- while
// defining both c_P1 and c_P2. Reading the second coefficient as c_P2 gives f_P(1.4) = 0.2729
// against OpenSim's 0.2630; reading it as c_P1 gives 0.2420. c_P2 is much closer to the curve
// Hyfydy says it approximates and is almost certainly what is meant, so that is what is used
// here -- but that is inference from the numbers, not something the manual states.
#define mjHYF_cP2    1.27368


// active force-length, zero (with zero slope) outside (r1, r2) -- a hard cut, as published
static mjtNum hyfydyFL(mjtNum l, mjtNum* dval) {
  if (l <= mjHYF_r1 || l >= mjHYF_r2) {
    if (dval) *dval = 0;
    return 0;
  }
  mjtNum u = l - 1;
  if (dval) *dval = 3*mjHYF_cL1*u*u + 2*mjHYF_cL2*u;
  return mjHYF_cL1*u*u*u + mjHYF_cL2*u*u + 1;
}


// passive force-length, zero below l = 1 (slack parallel element)
static mjtNum hyfydyFP(mjtNum l, mjtNum* dval) {
  if (l <= 1) {
    if (dval) *dval = 0;
    return 0;
  }
  mjtNum u = l - 1;
  if (dval) *dval = 3*mjHYF_cP1*u*u + 2*mjHYF_cP2*u;
  return mjHYF_cP1*u*u*u + mjHYF_cP2*u*u;
}


// force-velocity. Saturates at 0 below v = -1 rather than going negative.
//
// At v = 0 the two branches meet exactly in value and to 0.91% in slope (5.4053 from below
// against 5.4545 from above), so the curve is effectively C1 at the origin -- which the Song
// f_vce0 in engine_util_misc.c is not: its slopes there differ by 2.4x.
static mjtNum hyfydyFV(mjtNum v, mjtNum* dval) {
  if (v <= -1) {
    if (dval) *dval = 0;
    return 0;
  }
  if (v < 0) {
    mjtNum den = mjHYF_cV1 - v;                   // > 0 on (-1, 0)
    if (dval) *dval = mjHYF_cV1*(mjHYF_cV1 + 1)/(den*den);
    return mjHYF_cV1*(v + 1)/den;
  }
  mjtNum den = mjHYF_cV2 + v;                     // > 0 for v >= 0
  if (dval) *dval = mjHYF_cV2*(mjHYF_FVMAX - 1)/(den*den);
  return (mjHYF_FVMAX*v + mjHYF_cV2)/den;
}


// tendon force-length as a function of the length RATIO x = l_T/l_slack, so strain e = x - 1
static mjtNum hyfydyFT(mjtNum x, mjtNum* dval) {
  mjtNum e = x - 1;
  if (e <= 0) {
    if (dval) *dval = 0;
    return 0;
  }
  if (dval) *dval = 2*mjHYF_cT1*e + mjHYF_cT2;
  return mjHYF_cT1*e*e + mjHYF_cT2*e;
}


// public accessor, mirroring mju_millardCurve
mjtNum mju_hyfydyCurve(int curve, mjtNum x, mjtNum* deriv) {
  switch (curve) {
    case mjMUSCLECURVE_ACTIVE_FL:  return hyfydyFL(x, deriv);
    case mjMUSCLECURVE_PASSIVE_FL: return hyfydyFP(x, deriv);
    case mjMUSCLECURVE_TENDON_FL:  return hyfydyFT(x, deriv);
    case mjMUSCLECURVE_FV:         return hyfydyFV(x, deriv);
    default:
      mju_error("mju_hyfydyCurve: unknown curve %d", curve);
      return 0;
  }
}


//------------------------------ shared parameters and equilibrium ---------------------------------

// The gainprm layout is documented slot by slot in engine_muscle_mtu.h. In short it is a 1:1 map
// of what an OpenSim Millard2012EquilibriumMuscle (or a Hyfydy .hfd muscle) declares, in the
// same units, and a zero in any slot means "OpenSim's default for that property".
//
// v_max is in optimal fiber lengths per second and is NOT a speed: the normalized fiber velocity
// is v_ce / (v_max * l_opt). Four independent sources agree on this convention -- OpenSim
// (Millard2012EquilibriumMuscle.cpp, `dlce = dlceN * max_contraction_velocity *
// optimal_fiber_length`), the Hyfydy manual's own unit annotation, MuJoCo's own mju_muscleGain,
// and the Song muscle in engine_util_misc.c.
//
// beta defaults to 0.1, OpenSim's fiber_damping default, rather than 0. With the damped Millard
// curves the at-vmax force-velocity slopes are ZERO, so outside |v0| <= 1 the F-V term
// contributes nothing to dR/dl_ce and beta is the only thing keeping the Newton Jacobian bounded
// away from zero. It is a well-posedness condition, not a decoration. It is also what lets an
// inactive muscle be a damper: the term is NOT scaled by activation, which is OpenSim's stated
// reason for it ("so that the muscle can have an activation of zero"). Pass a negative value to
// ask for exactly zero damping, since 0 itself means "take the default".
typedef struct mjMtuParams_ {
  mjtNum F_max;      // maximum isometric force
  mjtNum l_opt;      // optimal fiber length
  mjtNum l_slack;    // tendon slack length
  mjtNum v_max_ms;   // v_max*l_opt, i.e. the actual m/s the normalization divides by
  mjtNum pen_h;      // l_opt*sin(phi_opt): the fixed pennation width
  mjtNum beta;       // fiber damping
  mjtNum min_act;    // minimum_activation
  mjtNum tol;        // Newton residual tolerance
  mjtNum lce_min;    // lower clamp on l_ce
  int hyfydy;        // 1 = Hyfydy polynomial curves, 0 = Millard table
  const mjCurveTable* curve[mjNMUSCLECURVE];   // NULL for a Hyfydy muscle
} mjMtuParams;


// sin(acos(0.1)): OpenSim's MuscleFixedWidthPennationModel maximum pennation angle
#define mjMTU_SINPHIMAX 0.9949874371066201

// Left end of Hyfydy's active force-length support; the Millard equivalent is per muscle, being
// OpenSim's min_norm_active_fiber_length property.
#define mjMTU_HYFYDY_LCE_MIN mjHYF_r1


// Apply the zero-means-OpenSim-default rule to a whole gainprm block. Order matters where a
// default is derived from another slot (both stiffness_at_one_norm_force defaults are).
static void mtuResolveDefaults(const mjtNum* in, mjtNum* out) {
  for (int k = 0; k < mjNGAIN; k++) {
    out[k] = in[k];
  }

  if (out[mjMTU_VMAX] == 0)  out[mjMTU_VMAX] = 10;              // max_contraction_velocity
  if (out[mjMTU_BETA] == 0)  out[mjMTU_BETA] = 0.1;             // fiber_damping
  if (out[mjMTU_BETA] < 0)   out[mjMTU_BETA] = 0;               // explicit request for none
  if (out[mjMTU_TOL] == 0)   out[mjMTU_TOL] = 1e-9;

  // minimum_activation. OpenSim clamps the activation to [minimum_activation, 1] wherever it
  // builds a force (Millard2012EquilibriumMuscle.cpp: `a = clampActivation(...)`), and sets
  // min_control to match, so the muscle never fully switches off. Same negative-means-zero
  // convention as the damping.
  if (out[mjMTU_MINACT] == 0) out[mjMTU_MINACT] = 0.01;
  if (out[mjMTU_MINACT] < 0)  out[mjMTU_MINACT] = 0;

  // ActiveForceLengthCurve. x2 (1.0), curviness (1.0) and minimum_value (0, forced by the damped
  // Millard model) are not parameters here: OpenSim passes them as literals.
  if (out[mjMTU_AFL_MIN] == 0)   out[mjMTU_AFL_MIN] = 0.47 - 0.0259;   // 0.4441, as OpenSim spells it
  if (out[mjMTU_AFL_TRANS] == 0) out[mjMTU_AFL_TRANS] = 0.73;
  if (out[mjMTU_AFL_MAX] == 0)   out[mjMTU_AFL_MAX] = 1.8123;
  if (out[mjMTU_AFL_SLOPE] == 0) out[mjMTU_AFL_SLOPE] = 0.8616;

  // FiberForceLengthCurve. strain_at_zero_force's default IS 0, and real models that set it use
  // a negative value, so it needs no rule.
  if (out[mjMTU_PFL_E1] == 0)   out[mjMTU_PFL_E1] = 0.7;
  if (out[mjMTU_PFL_KLOW] == 0) out[mjMTU_PFL_KLOW] = 0.2;      // absolute in OpenSim, not scaled
  if (out[mjMTU_PFL_KISO] == 0) out[mjMTU_PFL_KISO] = 2.0/(out[mjMTU_PFL_E1] - out[mjMTU_PFL_E0]);
  if (out[mjMTU_PFL_CURV] == 0) out[mjMTU_PFL_CURV] = 0.75;

  // TendonForceLengthCurve
  if (out[mjMTU_TFL_E1] == 0)   out[mjMTU_TFL_E1] = 0.049;
  if (out[mjMTU_TFL_KISO] == 0) out[mjMTU_TFL_KISO] = 1.375/out[mjMTU_TFL_E1];
  if (out[mjMTU_TFL_FTOE] == 0) out[mjMTU_TFL_FTOE] = 2.0/3.0;
  if (out[mjMTU_TFL_CURV] == 0) out[mjMTU_TFL_CURV] = 0.5;

  // ForceVelocityCurve. Both slopes at vmax default to 0, which is the damped model.
  if (out[mjMTU_FV_FMAXE] == 0)     out[mjMTU_FV_FMAXE] = 1.4;
  if (out[mjMTU_FV_DYDXNEARC] == 0) out[mjMTU_FV_DYDXNEARC] = 0.25;
  if (out[mjMTU_FV_DYDXISO] == 0)   out[mjMTU_FV_DYDXISO] = 5.0;
  if (out[mjMTU_FV_DYDXNEARE] == 0) out[mjMTU_FV_DYDXNEARE] = 0.15;
  if (out[mjMTU_FV_CONCCURV] == 0)  out[mjMTU_FV_CONCCURV] = 0.6;
  if (out[mjMTU_FV_ECCCURV] == 0)   out[mjMTU_FV_ECCCURV] = 0.9;
}


// Read one actuator's parameters. Cheap: no transcendental unless the muscle is pennated, and
// the curve tables were resolved once at mj_resetData.
static void mtuGetParams(const mjModel* m, const mjData* d, int id, mjMtuParams* p) {
  mjtNum prm[mjNGAIN];
  mtuResolveDefaults(m->actuator_gainprm + mjNGAIN*id, prm);

  p->hyfydy = (m->actuator_gaintype[id] == mjGAIN_HYFYDY_MTU);
  p->F_max = prm[mjMTU_FMAX];
  p->l_opt = prm[mjMTU_LOPT];
  p->l_slack = prm[mjMTU_LSLACK];
  p->v_max_ms = prm[mjMTU_VMAX]*p->l_opt;
  p->beta = prm[mjMTU_BETA];
  p->tol = prm[mjMTU_TOL];
  p->min_act = prm[mjMTU_MINACT];

  mjtNum phi_opt = prm[mjMTU_PENNATION];
  p->pen_h = phi_opt != 0 ? p->l_opt*mju_sin(phi_opt) : 0;   // the common case is unpennated

  // Lower bound on the fiber length. Below the active curve's left foot the active curve is
  // identically zero, so nothing is lost by clamping there, and the pennation term
  // 1/sqrt(1-(h/l_ce)^2) diverges at l_ce = h. OpenSim clamps in the same place for the same
  // reason (Millard2012EquilibriumMuscle's m_minimumFiberLength).
  mjtNum lce_min_norm = p->hyfydy ? mjMTU_HYFYDY_LCE_MIN : prm[mjMTU_AFL_MIN];
  p->lce_min = mju_max(lce_min_norm*p->l_opt, p->pen_h/mjMTU_SINPHIMAX);

  for (int c = 0; c < mjNMUSCLECURVE; c++) {
    p->curve[c] = p->hyfydy ? NULL
                            : (const mjCurveTable*)d->muscle_curve[mjNMUSCLECURVE*id + c];
  }
}


// Fail loudly on parameters the equilibrium is not defined for. Called from mju_mtuMuscleInit;
// the curve shape parameters are validated separately, by the factories in
// engine_muscle_millard_bezier.h, which enforce OpenSim's own admissibility conditions.
static void mtuCheckParams(const mjModel* m, int id, const mjMtuParams* p) {
  const char* kind = p->hyfydy ? "hyfydy_mtu" : "millard_mtu";
  const mjtNum* prm = m->actuator_gainprm + mjNGAIN*id;

  if (!(p->F_max > 0)) {
    mju_error("%s actuator %d: gainprm[0] (max_isometric_force) must be positive", kind, id);
  }
  if (!(p->l_opt > 0)) {
    mju_error("%s actuator %d: gainprm[1] (optimal_fiber_length) must be positive", kind, id);
  }
  if (!(p->l_slack > 0)) {
    mju_error("%s actuator %d: gainprm[2] (tendon_slack_length) must be positive", kind, id);
  }
  if (prm[mjMTU_PENNATION] < 0 || prm[mjMTU_PENNATION] >= mjPI/2) {
    mju_error("%s actuator %d: gainprm[4] (pennation_angle_at_optimal) must be in [0, pi/2)",
              kind, id);
  }
  if (prm[mjMTU_MINACT] > 1) {
    mju_error("%s actuator %d: gainprm[7] (minimum_activation) must be <= 1", kind, id);
  }

  // Hyfydy's curves are published polynomials with no per-muscle shape. Silently ignoring a
  // shape parameter someone set would hide a modelling mistake, so reject it.
  if (p->hyfydy) {
    for (int k = 8; k < mjNGAIN; k++) {
      if (prm[k] != 0) {
        mju_error("hyfydy_mtu actuator %d: gainprm[%d] is a Millard curve shape parameter; "
                  "Hyfydy's curves are fixed polynomials with no per-muscle shape", id, k);
      }
    }
  } else if (prm[mjMTU_AFL_RESERVED12] != 0 || prm[mjMTU_AFL_RESERVED13] != 0 ||
             prm[mjMTU_PFL_RESERVED19] != 0) {
    // slot 12 is OpenSim's ActiveForceLengthCurve minimum_value, which
    // Millard2012EquilibriumMuscle forces to 0 for the damped model this implements
    mju_error("millard_mtu actuator %d: gainprm[12], [13] and [19] are reserved and must be 0",
              id);
  }

  if (m->actuator_trntype[id] != mjTRN_TENDON && m->actuator_trntype[id] != mjTRN_JOINT &&
      m->actuator_trntype[id] != mjTRN_SITE) {
    mju_error("%s actuator %d: transmission must be tendon, joint or site", kind, id);
  }
}


// Everything one residual evaluation produces. Kept as a struct because the residual, its
// Jacobian and the reported force all read the same pennation algebra, and four copies of the
// same signs is how a sign error gets in.
typedef struct mjMtuEval_ {
  mjtNum cos_phi, sin_phi, dphi;   // pennation, and dphi/dl_ce
  mjtNum l_T, dlT;                 // tendon length and dl_T/dl_ce
  mjtNum dcos;                     // dcos(phi)/dl_ce
  mjtNum l0, v0;                   // normalized fiber length and velocity
  mjtNum f_T, d_T;                 // tendon curve: value and d/d(l_T/l_slack)
  mjtNum f_l, d_l;                 // active F-L: value and d/dl0
  mjtNum f_p, d_p;                 // passive F-L: value and d/dl0
  mjtNum f_v, d_v;                 // F-V: value and d/dv0
  mjtNum fib;                      // A f_L f_V + f_P + beta v0
  mjtNum R, dR;                    // residual and dR/dl_ce
} mjMtuEval;


// Residual and analytic Jacobian at l_ce.
//
// `dtv` is dt*v_max*l_opt, the length change that corresponds to one unit of normalized
// velocity. Pass dtv <= 0 for the ISOMETRIC residual (v0 forced to 0 and its Jacobian
// contribution dropped), which is what the steady-state solve after a reset or a teleport
// needs.
//
// Signs, from OpenSim's MuscleFixedWidthPennationModel:
//   phi        = asin(h/l_ce), clamped to phi_max        (identically 0 when h == 0)
//   dphi/dl_ce = (-h/l_ce^2) / sqrt(1 - (h/l_ce)^2)
//   dl_T/dl_ce = l_ce sin(phi) dphi/dl_ce - cos(phi)
static void mtuEval(mjMtuEval* e, mjtNum l_ce, mjtNum l_mtu, mjtNum l_ce_prev,
                    mjtNum A, mjtNum dtv, const mjMtuParams* p) {
  e->l0 = l_ce/p->l_opt;
  e->v0 = dtv > 0 ? (l_ce - l_ce_prev)/dtv : 0;

  // fixed-width pennation; h == 0 is the unpennated case, cos(phi) == 1 and dphi == 0 exactly
  e->cos_phi = 1;
  e->sin_phi = 0;
  e->dphi = 0;
  if (p->pen_h > 0) {
    mjtNum sp = p->pen_h/l_ce;
    if (sp > mjMTU_SINPHIMAX) {
      // GUARD, not live logic: lce_min >= pen_h/sin(phi_max) and every evaluation is clamped to
      // lce_min, so sin(phi) = pen_h/l_ce cannot exceed sin(phi_max). It stays because a caller
      // can write any fiber length into act, and 1/sqrt(1-sp^2) is imaginary past this point.
      sp = mjMTU_SINPHIMAX;                 // phi frozen at phi_max, so dphi stays 0
    } else {
      mjtNum r = 1 - sp*sp;
      e->dphi = -(sp/l_ce)/(r > 0 ? mju_sqrt(r) : 1);
    }
    e->sin_phi = sp;
    e->cos_phi = mju_sqrt(1 - sp*sp);
  }
  e->l_T = l_mtu - l_ce*e->cos_phi;
  e->dlT = l_ce*e->sin_phi*e->dphi - e->cos_phi;
  e->dcos = -e->sin_phi*e->dphi;

  if (p->hyfydy) {
    e->f_T = hyfydyFT(e->l_T/p->l_slack, &e->d_T);
    e->f_l = hyfydyFL(e->l0, &e->d_l);
    e->f_p = hyfydyFP(e->l0, &e->d_p);
    e->f_v = hyfydyFV(e->v0, &e->d_v);
  } else {
    mju_curveTableEval(p->curve[mjMUSCLECURVE_TENDON_FL], e->l_T/p->l_slack, &e->f_T, &e->d_T);
    mju_curveTableEval(p->curve[mjMUSCLECURVE_ACTIVE_FL], e->l0, &e->f_l, &e->d_l);
    mju_curveTableEval(p->curve[mjMUSCLECURVE_PASSIVE_FL], e->l0, &e->f_p, &e->d_p);
    mju_curveTableEval(p->curve[mjMUSCLECURVE_FV], e->v0, &e->f_v, &e->d_v);
  }

  mjtNum inv_dtv = dtv > 0 ? 1/dtv : 0;
  e->fib = A*e->f_l*e->f_v + e->f_p + p->beta*e->v0;
  mjtNum d_fib = A*(e->d_l*e->f_v/p->l_opt + e->f_l*e->d_v*inv_dtv)
               + e->d_p/p->l_opt + p->beta*inv_dtv;

  e->R = e->f_T - e->cos_phi*e->fib;
  e->dR = e->d_T*(e->dlT/p->l_slack) - (e->dcos*e->fib + e->cos_phi*d_fib);
}


// Newton on R(l_ce) = 0, warm-started at `l_ce`, returning the converged fiber length.
//
// Two safeguards, both cheap and both load-bearing:
//   * l_ce is clamped from below at p->lce_min. The pennation term 1/sqrt(1-(h/l_ce)^2)
//     diverges at l_ce = h, and below the active curve's left foot the residual carries no
//     active information anyway. OpenSim clamps in the same place for the same reason.
//   * the step is capped at half an optimal fiber length. The tendon curve is stiff enough
//     that an unguarded Newton step from a bad initial state can leave the physical range
//     entirely; near the root the cap never binds.
static mjtNum mtuSolve(mjtNum l_ce, mjtNum l_mtu, mjtNum l_ce_prev, mjtNum A,
                       mjtNum dtv, const mjMtuParams* p, int niter, mjMtuEval* e_out) {
  mjtNum max_step = 0.5*p->l_opt;
  mjMtuEval e;
  int stale = 1;                  // is `e` evaluated somewhere other than the current l_ce?

  if (l_ce < p->lce_min) {
    l_ce = p->lce_min;
  }

  for (int it = 0; it < niter; it++) {
    mtuEval(&e, l_ce, l_mtu, l_ce_prev, A, dtv, p);
    stale = 0;

    if (mju_abs(e.R) < p->tol) {
      break;
    }

    // dR is negative wherever the equilibrium is well posed (a longer fiber means a shorter,
    // slacker tendon and a stiffer fiber). An exact zero means a flat residual -- a fully
    // slack tendon and a slack fiber at zero activation -- where every l_ce is a root and the
    // warm start is as good an answer as any.
    if (e.dR == 0) {
      break;
    }

    mjtNum step = -e.R/e.dR;
    if (step > max_step) {
      step = max_step;
    } else if (step < -max_step) {
      step = -max_step;
    }
    l_ce += step;

    if (l_ce < p->lce_min) {
      l_ce = p->lce_min;
    }
    stale = 1;
  }

  // report the state at the returned l_ce, not at the last iterate. Converging on the
  // residual leaves `e` already there, so the common path pays nothing for this.
  if (stale) {
    mtuEval(&e, l_ce, l_mtu, l_ce_prev, A, dtv, p);
  }
  if (e_out) {
    *e_out = e;
  }
  return l_ce;
}


//------------------------------ rigid-tendon fallback ----------------------------------------------

// A tendon much shorter than the fiber makes the series-elastic normalization l_T/l_slack
// singular and the equilibrium ill-conditioned, for no physical gain: the tendon's stretch is
// negligible against the fiber's operating range. Below this ratio the tendon is treated as
// inextensible.
#define mjMTU_RIGID_RATIO 0.05


// Rigid-tendon force. This is OpenSim's own ignore_tendon_compliance path, rule for rule
// (Millard2012EquilibriumMuscle::calcMuscleLengthInfo / calcFiberVelocityInfo /
// calcMuscleDynamicsInfo, and isFiberStateClamped):
//
//   l_ce   = clamp(sqrt((l_MTU - l_slack)^2 + h^2), lce_min)     calcFiberLength + clampFiberLength
//   cos(phi) = cos(asin(h/l_ce))                                 ALWAYS >= 0, never signed by the path
//   l_T    = l_MTU - l_ce cos(phi)                               "necessary even for the rigid
//                                                                 tendon, as it might have gone slack"
//   if l_T < l_slack     -> tendon buckling: v_ce = 0, f_V = 1
//   else                 -> v_ce = v_MTU cos(phi)
//   if fiber clamped     -> ALL forces are zero. isFiberStateClamped is
//                           (l_ce <= lce_min and v_ce <= 0) or l_ce < lce_min
//   else   F_fiber = F_max (a f_L f_V + f_P + beta v0), saturated at 0 so a rigid tendon's
//                           parallel damping cannot make the fiber push,
//          F       = F_fiber cos(phi)
//
// NOTE for hyfydy_mtu: the Hyfydy manual documents no rigid-tendon variant -- its tendon is
// always the compliant quadratic. This path therefore evaluates Hyfydy's curves inside OpenSim's
// rigid-tendon formulation, which is our extension and not something Hyfydy defines. It exists so
// a short-tendon muscle degrades gracefully instead of stalling the solver; a model that cares
// about Hyfydy parity should not be in this regime.
static mjtNum mtuRigidForce(mjtNum A, mjtNum l_mtu, mjtNum v_mtu, const mjMtuParams* p,
                            mjtNum* l_ce_out, mjtNum* l_T_out, mjtNum* v_ce_out,
                            mjtNum* dF_dvmtu) {
  mjtNum along = l_mtu - p->l_slack;
  mjtNum l_ce = mju_sqrt(along*along + p->pen_h*p->pen_h);
  if (l_ce < p->lce_min) {
    l_ce = p->lce_min;
  }

  // the pennation angle comes from asin(h/l_ce), so its cosine is non-negative whatever the path
  // did; it is NOT (l_mtu - l_slack)/l_ce, which would be signed
  mjtNum sp = p->pen_h > 0 ? p->pen_h/l_ce : 0;
  if (sp > mjMTU_SINPHIMAX) {
    sp = mjMTU_SINPHIMAX;
  }
  mjtNum cos_phi = mju_sqrt(1 - sp*sp);
  mjtNum l_T = l_mtu - l_ce*cos_phi;

  // a buckled tendon cannot impose a velocity on the fiber
  int buckled = l_T < p->l_slack;
  mjtNum v_ce = buckled ? 0 : v_mtu*cos_phi;

  if (l_ce_out) *l_ce_out = l_ce;
  if (l_T_out) *l_T_out = l_T;
  if (v_ce_out) *v_ce_out = v_ce;
  if (dF_dvmtu) *dF_dvmtu = 0;

  // a fiber sitting on its lower clamp, or being pushed below it, carries nothing at all
  if (l_ce < p->lce_min || (l_ce <= p->lce_min && v_ce <= 0)) {
    return 0;
  }

  mjtNum l0 = l_ce/p->l_opt;
  mjtNum v0 = v_ce/p->v_max_ms;

  mjtNum f_l, f_p, f_v, d_v;
  if (buckled) {
    f_v = 1;                                  // consistent with a fiber velocity of zero
    d_v = 0;
  }
  if (p->hyfydy) {
    f_l = hyfydyFL(l0, NULL);
    f_p = hyfydyFP(l0, NULL);
    if (!buckled) f_v = hyfydyFV(v0, &d_v);
  } else {
    mju_curveTableEval(p->curve[mjMUSCLECURVE_ACTIVE_FL], l0, &f_l, NULL);
    mju_curveTableEval(p->curve[mjMUSCLECURVE_PASSIVE_FL], l0, &f_p, NULL);
    if (!buckled) mju_curveTableEval(p->curve[mjMUSCLECURVE_FV], v0, &f_v, &d_v);
  }

  mjtNum f_fiber = A*f_l*f_v + f_p + p->beta*v0;
  if (f_fiber <= 0) {                         // saturate the damping: the fiber only pulls
    return 0;
  }
  if (dF_dvmtu && !buckled) {
    // v0 = v_mtu*cos(phi)/(v_max*l_opt) at fixed length, so the chain rule brings a second cos(phi)
    *dF_dvmtu = p->F_max*cos_phi*cos_phi*(A*f_l*d_v + p->beta)/p->v_max_ms;
  }
  return p->F_max*f_fiber*cos_phi;
}


static int mtuIsRigid(const mjMtuParams* p) {
  return p->l_slack < mjMTU_RIGID_RATIO*p->l_opt;
}


//------------------------------ public entry points ------------------------------------------------

// Where this actuator's two activation variables live. MuJoCo's convention is that the LAST
// activation variable is the one that multiplies the gain; earlier ones are internal state, which
// is what actuator plugins already rely on. So act = [l_ce, activation].
static void mtuActAdr(const mjModel* m, int id, int* fiber, int* activation) {
  int first = m->actuator_actadr[id];
  *fiber = first;
  *activation = first + m->actuator_actnum[id] - 1;
}


// The activation the force should be built from, clamped to [0, 1]. With actuator_actearly the
// end-of-step activation is used, exactly as the gain*act path does; dyntype is required to be
// mjDYN_MUSCLE for these gains, so the integration is plain Euler and there is no actrange to
// clamp against (the compiler rejects one, since a range is per-actuator and would clamp the
// fiber length too).
static mjtNum mtuActivation(const mjModel* m, const mjData* d, int id, int act_adr,
                            const mjMtuParams* p) {
  mjtNum A = d->act[act_adr];
  if (m->actuator_actearly[id]) {
    A += m->opt.timestep*d->act_dot[act_adr];
  }
  return mju_clip(A, p->min_act, 1);
}


// Solve the fiber equilibrium at the CURRENT state and write the muscle outputs. Returns the
// end-of-step fiber length. Pure: it depends only on (A, l_ce, l_mtu) and the model, and it
// writes nothing that is state.
//
// `dtv <= 0` asks for the ISOMETRIC solve (zero fiber velocity), which is what equilibration
// after a reset wants.
static mjtNum mtuSolveAndReport(const mjModel* m, mjData* d, int id, const mjMtuParams* p,
                                mjtNum A, mjtNum l_ce, mjtNum l_mtu, mjtNum v_mtu, mjtNum dtv) {
  // rigid tendon: the fiber length is algebraic, so there is nothing to solve
  if (mtuIsRigid(p)) {
    mjtNum l_ce_next, l_T, v_ce;
    mjtNum F = mtuRigidForce(A, l_mtu, dtv > 0 ? v_mtu : 0, p, &l_ce_next, &l_T, &v_ce, NULL);
    d->muscle_l_ce[id] = l_ce_next;
    d->muscle_v_ce[id] = v_ce;
    d->muscle_l_se[id] = l_T;
    d->muscle_F_mtu[id] = F;
    return l_ce_next;
  }

  int niter = dtv > 0 ? (m->opt.cmtu_iter > 0 ? m->opt.cmtu_iter : 12) : 100;
  mjMtuEval e;
  mjtNum l_ce_next = mtuSolve(l_ce, l_mtu, l_ce, A, dtv, p, niter, &e);

  d->muscle_l_ce[id] = l_ce_next;
  d->muscle_l_se[id] = e.l_T;
  d->muscle_v_ce[id] = dtv > 0 ? (l_ce_next - l_ce)/m->opt.timestep : 0;
  d->muscle_F_mtu[id] = p->F_max*e.f_T;
  return l_ce_next;
}


// d(actuator_force)/d(actuator_velocity), for the implicit integrators (mjd_actuator_vel).
//
// ZERO for a compliant tendon, and that is exact rather than an omission: the actuator force is
// the TENDON force, which depends only on the tendon's length -- i.e. on the path length and the
// fiber length, both of which are held fixed when differentiating with respect to velocity. The
// muscle's velocity dependence is real but it is mediated entirely through the fiber state, so it
// shows up in the next step, not in this partial derivative.
//
// A RIGID tendon is different: there the fiber velocity is the path velocity projected onto the
// fiber, so the force-velocity curve and the fiber damping enter directly. The actuator force is
// the negated tendon force, hence the sign.
mjtNum mju_mtuMuscleForceVel(const mjModel* m, const mjData* d, int id) {
  mjMtuParams p;
  mtuGetParams(m, d, id, &p);
  if (!mtuIsRigid(&p)) {
    return 0;
  }

  int fiber, activation;
  mtuActAdr(m, id, &fiber, &activation);
  mjtNum A = mju_clip(d->act[activation], p.min_act, 1);

  mjtNum dF_dvmtu = 0;
  mtuRigidForce(A, d->actuator_length[id], d->actuator_velocity[id], &p,
                NULL, NULL, NULL, &dF_dvmtu);
  return -dF_dvmtu;
}


void mju_mtuMuscleActDot(const mjModel* m, mjData* d, int id) {
  mjMtuParams p;
  mtuGetParams(m, d, id, &p);

  int fiber, activation;
  mtuActAdr(m, id, &fiber, &activation);

  mjtNum A = mtuActivation(m, d, id, activation, &p);
  mjtNum l_ce = d->act[fiber];
  mjtNum dt = m->opt.timestep;

  mjtNum l_ce_next = mtuSolveAndReport(m, d, id, &p, A, l_ce, d->actuator_length[id],
                                       d->actuator_velocity[id], dt*p.v_max_ms);

  // THE FIBER VELOCITY IS THE IMPLICIT ONE, and that is deliberate.
  //
  // The obvious alternative -- solve the equilibrium for the instantaneous fiber velocity, which
  // is what OpenSim integrates -- is stiff. Linearising it, d(v_ce)/d(l_ce) is
  // (tendon stiffness)/(a f_L f_V' + beta) in fiber units, and at zero activation the denominator
  // is beta alone: for a typical muscle that is ~2800 1/s, so explicit Euler would need
  // dt < 0.7 ms, and less for a short tendon. An INACTIVE muscle is the stiff one.
  //
  // Reporting the backward-Euler step as a velocity instead keeps the unconditional stability of
  // an implicit solve while leaving act_dot a genuine function of the current state, so
  // mj_forward stays pure. It is consistent: (l_ce* - l_ce)/dt is the backward-Euler estimate of
  // v_ce and converges to it as dt -> 0. Under MuJoCo's Euler integrator act[fiber] lands exactly
  // on l_ce*, i.e. on the implicit solution.
  d->act_dot[fiber] = (l_ce_next - l_ce)/dt;
}


// Put every muscle-tendon unit at its isometric equilibrium for the current pose, the analogue of
// OpenSim's Model::equilibrateMuscles. Call it after mj_forward (which is what fills
// actuator_length) whenever the pose was set rather than integrated -- a reset, a keyframe, a
// qpos edit. Without it the first step sees whatever fiber length the state carried and takes one
// large, bounded fiber excursion to recover.
void mju_mtuMuscleEquilibrate(const mjModel* m, mjData* d) {
  for (int i = 0; i < m->nu; i++) {
    if (m->actuator_gaintype[i] != mjGAIN_MILLARD_MTU &&
        m->actuator_gaintype[i] != mjGAIN_HYFYDY_MTU) {
      continue;
    }
    mjMtuParams p;
    mtuGetParams(m, d, i, &p);

    int fiber, activation;
    mtuActAdr(m, i, &fiber, &activation);
    mjtNum A = mju_clip(d->act[activation], p.min_act, 1);

    // dtv = 0 selects the isometric residual, so this is the fiber length at which the tendon
    // balances the fiber's static force
    d->act[fiber] = mtuSolveAndReport(m, d, i, &p, A, d->act[fiber], d->actuator_length[i],
                                      0, 0);
    d->act_dot[fiber] = 0;
  }
}


void mju_mtuMuscleInit(const mjModel* m, mjData* d) {
  for (int i = 0; i < m->nu; i++) {
    int millard = m->actuator_gaintype[i] == mjGAIN_MILLARD_MTU;
    if (!millard && m->actuator_gaintype[i] != mjGAIN_HYFYDY_MTU) {
      continue;
    }

    // Resolve (and, the first time a shape is seen, bake) this actuator's curve tables. This is
    // the ONLY place the bake can happen: the stepping path reads mjData.muscle_curve.
    if (millard) {
      mjtNum prm[mjNGAIN];
      mtuResolveDefaults(m->actuator_gainprm + mjNGAIN*i, prm);
      char context[64];
      mjSNPRINTF(context, " for millard_mtu actuator %d", i);
      const mjCurveTable* table[mjNMUSCLECURVE];
      mju_millardBakeCurves(prm, context, table);
      for (int c = 0; c < mjNMUSCLECURVE; c++) {
        d->muscle_curve[mjNMUSCLECURVE*i + c] = (uintptr_t)table[c];
      }
    }

    mjMtuParams p;
    mtuGetParams(m, d, i, &p);
    mtuCheckParams(m, i, &p);

    // Seed the fiber at its optimal length. mj_resetData has just zeroed act, and a zero fiber
    // length is not a physical state (l0 = 0, and the pennation term divides by l_ce), so this is
    // the model's own default rather than a hint: it is the analogue of an actuator plugin's
    // reset callback. A keyframe applied afterwards overrides it, which is the right precedence.
    // Note this does NOT equilibrate -- that needs a position pass, so it is
    // mju_mtuMuscleEquilibrate's job and the caller's choice.
    int fiber, activation;
    mtuActAdr(m, i, &fiber, &activation);
    d->act[fiber] = p.l_opt;
    d->act[activation] = 0;

    d->muscle_l_ce[i] = p.l_opt;
    d->muscle_l_se[i] = p.l_slack;
    d->muscle_v_ce[i] = 0;
    d->muscle_F_mtu[i] = 0;
  }
}
