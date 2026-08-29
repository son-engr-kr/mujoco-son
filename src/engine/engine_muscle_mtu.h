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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_MUSCLE_MTU_H_
#define MUJOCO_SRC_ENGINE_ENGINE_MUSCLE_MTU_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------ gainprm layout ----------------------------------------------------

// gaintype="millard_mtu" and gaintype="hyfydy_mtu" read all 32 gainprm slots. The layout is a
// 1:1 map of the properties an OpenSim Millard2012EquilibriumMuscle (or a Hyfydy .hfd muscle)
// declares, in the same units, so a model's parameters transfer without reinterpretation.
//
// ZERO MEANS "OpenSim DEFAULT" for every slot. That rule is complete, not a convenience: the
// only parameters whose OpenSim default is itself zero (afl minimum_value, pfl
// strain_at_zero_force, the two force-velocity slopes at vmax) are exactly the ones for which
// zero is also the value you would want, and pfl strain_at_zero_force -- the one parameter real
// models set to something that is neither zero nor positive -- is negative and so still
// expressible. A hand-written model therefore fills in the first six slots and leaves the rest
// blank; a converted .osim fills in whatever it overrode.
//
// Slots 8..31 are curve SHAPE parameters and apply to millard_mtu only. Hyfydy's curves are
// published polynomials with no per-muscle shape, so hyfydy_mtu requires them to be zero rather
// than silently ignoring a value someone meant to have an effect.
enum {
  // --- mechanical, both models -------------------------------------------------------------
  mjMTU_FMAX = 0,     // max_isometric_force                              [N]
  mjMTU_LOPT,         // optimal_fiber_length                             [m]
  mjMTU_LSLACK,       // tendon_slack_length                              [m]
  mjMTU_VMAX,         // max_contraction_velocity  [l_opt/s]              default 10
  mjMTU_PENNATION,    // pennation_angle_at_optimal                       [rad]
  mjMTU_BETA,         // fiber_damping                                    default 0.1
  mjMTU_TOL,          // fiber Newton residual tolerance                  default 1e-9
  mjMTU_RESERVED7,

  // --- OpenSim ActiveForceLengthCurve, millard_mtu only --------------------------------------
  mjMTU_AFL_MIN = 8,  // min_norm_active_fiber_length                     default 0.4441
  mjMTU_AFL_TRANS,    // transition_norm_fiber_length                     default 0.73
  mjMTU_AFL_MAX,      // max_norm_active_fiber_length                     default 1.8123
  mjMTU_AFL_SLOPE,    // shallow_ascending_slope                          default 0.8616
  mjMTU_AFL_RESERVED12,   // minimum_value: forced to 0 by the damped model, must be 0
  mjMTU_AFL_RESERVED13,

  // --- OpenSim FiberForceLengthCurve (passive), millard_mtu only -----------------------------
  mjMTU_PFL_E0 = 14,  // strain_at_zero_force                             default 0 (may be < 0)
  mjMTU_PFL_E1,       // strain_at_one_norm_force                         default 0.7
  mjMTU_PFL_KLOW,     // stiffness_at_low_force                           default 0.2
  mjMTU_PFL_KISO,     // stiffness_at_one_norm_force                      default 2/(e1-e0)
  mjMTU_PFL_CURV,     // curviness                                        default 0.75
  mjMTU_PFL_RESERVED19,

  // --- OpenSim TendonForceLengthCurve, millard_mtu only --------------------------------------
  mjMTU_TFL_E1 = 20,  // strain_at_one_norm_force                         default 0.049
  mjMTU_TFL_KISO,     // stiffness_at_one_norm_force                      default 1.375/e1
  mjMTU_TFL_FTOE,     // norm_force_at_toe_end                            default 2/3
  mjMTU_TFL_CURV,     // curviness                                        default 0.5

  // --- OpenSim ForceVelocityCurve, millard_mtu only ------------------------------------------
  mjMTU_FV_FMAXE = 24,  // max_eccentric_velocity_force_multiplier        default 1.4
  mjMTU_FV_DYDXC,       // concentric_slope_at_vmax                       default 0
  mjMTU_FV_DYDXNEARC,   // concentric_slope_near_vmax                     default 0.25
  mjMTU_FV_DYDXISO,     // isometric_slope                                default 5.0
  mjMTU_FV_DYDXE,       // eccentric_slope_at_vmax                        default 0
  mjMTU_FV_DYDXNEARE,   // eccentric_slope_near_vmax                      default 0.15
  mjMTU_FV_CONCCURV,    // concentric_curviness                           default 0.6
  mjMTU_FV_ECCCURV      // eccentric_curviness                            default 0.9
};


// Which of the four muscle curves. Also the index into mjData.muscle_curve, whose four entries
// per actuator are the baked tables a millard_mtu actuator evaluates.
typedef enum mjtMuscleCurve_ {
  mjMUSCLECURVE_ACTIVE_FL = 0,  // active force-length,  x = l_ce / l_opt
  mjMUSCLECURVE_PASSIVE_FL,     // passive force-length, x = l_ce / l_opt
  mjMUSCLECURVE_TENDON_FL,      // tendon force-length,  x = l_T / l_slack
  mjMUSCLECURVE_FV,             // force-velocity,       x = v_ce / (v_max*l_opt)
  mjNMUSCLECURVE                // number of curves
} mjtMuscleCurve;


//------------------------------ baked curve table -------------------------------------------------

// One Millard curve sampled onto a uniform grid over [x0, x1], extrapolated linearly from the
// exact anchors outside it. Immutable and shared: many actuators point at the same table.
//
// Quintic (not cubic) Hermite, because storing the second derivative -- which the Bezier
// construction gives exactly, for free -- buys value error O(h^6) against cubic's O(h^4).
// Measured on these curves that is the difference between 513 knots and several thousand.
//
// The per-knot triple is interleaved, so the six numbers one cell needs are 48 contiguous bytes.
typedef struct mjCurveTable_ {
  const mjtNum* knot;     // [3n] (y, dy/dx, d2y/dx2) at knot i
  int n;                  // knot count (>= 2); knot 0 sits at x0, knot n-1 at x1
  mjtNum x0, x1;          // tabulated domain
  mjtNum dx, inv_dx;      // uniform spacing and its reciprocal
  mjtNum a_y0, a_d0;      // exact extrapolation anchors below x0
  mjtNum a_y1, a_d1;      // ... and above x1
} mjCurveTable;


// Value and slope of a baked curve at x. Either output may be NULL.
MJAPI void mju_curveTableEval(const mjCurveTable* t, mjtNum x, mjtNum* y, mjtNum* dydx);


//------------------------------ curve baking (engine_muscle_bake.cc) ------------------------------

// Bake, or find already baked, the four Millard curves for one resolved gainprm block, and
// write the four table pointers into `out`. `prm` must already have had the zero-means-default
// rule applied. `context` names the actuator in any error message.
//
// The tables live in a process-wide cache keyed by the shape parameters, built lazily on first
// use, so a whole model of muscles that share a curve shape bakes it once and every actuator
// points at the same table. Nothing is baked into mjModel: a stored table could go stale against
// the parameters that produced it.
MJAPI void mju_millardBakeCurves(const mjtNum* prm, const char* context,
                                 const mjCurveTable* out[mjNMUSCLECURVE]);

// Number of distinct curves the process-wide cache currently holds. For tests and diagnostics.
MJAPI int mju_millardCurveCacheSize(void);


//------------------------------ actuator entry points ---------------------------------------------

// Initialize the muscle state (mjData muscle_l_ce / muscle_l_se / muscle_v_ce / muscle_F_mtu)
// of every millard_mtu and hyfydy_mtu actuator, and resolve their baked curve tables into
// mjData.muscle_curve. Called from mj_resetData, which is the "once, at startup" point: the
// bake happens here and never on the stepping path.
//
// The consequence, and it is the contract: the MECHANICAL parameters (slots 0..7) are re-read
// every step, so writing to actuator_gainprm takes effect immediately, which is what randomizing
// muscle strength in a training loop needs. The curve SHAPE parameters (slots 8..31) are resolved
// here, so changing one takes effect at the next reset. Making them per-step would mean a cache
// lookup inside the stepping path, which costs more than the muscle solve itself.
MJAPI void mju_mtuMuscleInit(const mjModel* m, mjData* d);

// Compute one millard_mtu or hyfydy_mtu actuator's fiber velocity into act_dot, and its force
// and diagnostics into mjData. Called from the act_dot loop in mj_fwdActuation.
//
// The actuator carries TWO activation variables, act = [l_ce, activation]: MuJoCo's convention is
// that the last one multiplies the gain while earlier ones are internal state. Keeping the fiber
// length there rather than in a side array is what makes mj_forward a pure function of the state,
// so RK4 integrates the fiber correctly, mjd_transitionFD differences it correctly, and
// mj_getState/mj_setState capture it under mjSTATE_ACT.
MJAPI void mju_mtuMuscleActDot(const mjModel* m, mjData* d, int id);

// Put every millard_mtu / hyfydy_mtu fiber at its isometric equilibrium for the current pose --
// the analogue of OpenSim's Model::equilibrateMuscles. Call after mj_forward whenever the pose
// was set rather than integrated (a reset, a keyframe, a qpos edit).
MJAPI void mju_mtuMuscleEquilibrate(const mjModel* m, mjData* d);

// d(actuator_force)/d(actuator_velocity), for the implicit integrators. Exactly zero for a
// compliant tendon -- the tendon force is a function of tendon LENGTH -- and nonzero only on
// the rigid-tendon path, where the fiber velocity is the path velocity.
MJAPI mjtNum mju_mtuMuscleForceVel(const mjModel* m, const mjData* d, int id);

// One normalized Millard2012 curve at the shape parameters in `prm` (zero means default, as in
// gainprm), and optionally its slope. Exposed so a model's curves can be checked against
// OpenSim directly; the solver reads the same tables.
MJAPI mjtNum mju_millardCurve(int curve, mjtNum x, const mjtNum* prm, mjtNum* deriv);

// One normalized Hyfydy `muscle_force_m2012fast` curve, and optionally its slope. Hyfydy's
// curves are fixed polynomials, so this takes no shape parameters.
MJAPI mjtNum mju_hyfydyCurve(int curve, mjtNum x, mjtNum* deriv);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_MUSCLE_MTU_H_
