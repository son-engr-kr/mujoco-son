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

// Port of OpenSim's quintic-Bezier muscle-curve toolkit (SegmentedQuinticBezierToolkit and
// SmoothSegmentedFunctionFactory), used to BAKE the Millard2012 curves that
// gaintype="millard_mtu" evaluates.
//
// WHY A BAKER AND NOT A DIRECT EVALUATOR. These curves are parametric: x(u) and y(u) are
// separate quintic Beziers, so getting y at a given x means first solving x(u) = x for u,
// which OpenSim does with a Newton iteration (calcU, MAXITER 20) after a linear scan for the
// section. The muscle solver already runs a Newton per muscle per step and evaluates four
// curves inside every iteration of it, so evaluating the Bezier directly would put Newton
// inside Newton with data-dependent trip counts at both levels. Sampling each curve once onto
// a uniform grid of (y, dy/dx, d2y/dx2) moves that inversion off the inner loop for good: a
// runtime curve evaluation becomes an index, a clamp and a quintic Hermite polynomial.
//
// WHEN THIS RUNS. Once per distinct curve shape per process, lazily, from
// engine_muscle_bake.cc. Nothing baked is ever written into an mjModel: a table stored in a
// model file could go stale against the shape parameters that produced it, and the bake is
// microseconds against a simulation's lifetime.
//
// This header is C++ (std::vector, std::string) and is compiled only into
// engine_muscle_bake.cc. The solver itself is C and never includes it.
//
// PORTING NOTE -- Bernstein/de Casteljau instead of the upstream expanded polynomials. OpenSim
// carries machine-generated expanded forms (Maple output: `t2 = u1 * 0.5e1` and so on).
// Transcribing those is error-prone and unauditable, so the same quantities are computed here
// from the standard Bernstein derivative identity
//     B^(k)(u) = n!/(n-k)! * [degree-(n-k) Bezier over the k-th forward differences]
// evaluated by de Casteljau, which is mathematically identical, has no binomial coefficients to
// get wrong, and is numerically stable.

#ifndef MUJOCO_SRC_ENGINE_ENGINE_MUSCLE_MILLARD_BEZIER_H_
#define MUJOCO_SRC_ENGINE_ENGINE_MUSCLE_MILLARD_BEZIER_H_

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mjtnum.h>
#include "engine/engine_muscle_mtu.h"
#include "engine/engine_util_errmem.h"

namespace mujoco {
namespace millard {

// Shape parameters reach here straight from a model's gainprm, so a bad one is a user error,
// not an internal invariant: fail with mju_error rather than an assert that a release build
// would drop. The caller sets `context` to the actuator being built so the message names it.
inline thread_local const char* check_context = "";

inline void Check(bool ok, const char* msg) {
  if (!ok) {
    mju_error("millard curve%s: %s", check_context, msg);
  }
}

// Six control points per axis, matching SimTK::Vec6.
struct Vec6 {
  double v[6];
  double operator[](int i) const { return v[i]; }
};
struct CtrlPts {
  Vec6 x, y;
};

// Port of SegmentedQuinticBezierToolkit::calcQuinticBezierCornerControlPoints.
//
// The two endpoints plus their slopes define two lines; C is where they intersect.
// `curviness` slides the interior control points from the endpoints (0) toward C (1).
// The doubled interior points are what make the endpoint slopes exactly dydx0 and dydx1.
inline CtrlPts CornerControlPoints(double x0, double y0, double dydx0,
                                   double x1, double y1, double dydx1,
                                   double curviness) {
  Check(curviness > 0.0 && curviness <= 1.0,
        "curviness must be in (0, 1]: 0 makes x(u) non-invertible at the endpoints");

  const double root_eps = std::sqrt(2.220446049250313e-16);  // sqrt(SimTK::Eps)
  double xC;
  if (std::fabs(dydx0 - dydx1) > root_eps) {
    xC = (y1 - y0 - x1*dydx1 + x0*dydx0) / (dydx0 - dydx1);
  } else {
    xC = (x1 + x0) / 2.0;
  }
  const double yC = (xC - x1)*dydx1 + y1;

  // 'C' vs 'S' shape check (upstream SimTK_ERRCHK): the endpoint separation must exceed
  // the distance from either endpoint to the intersection.
  const double a = (xC - x0)*(xC - x0) + (yC - y0)*(yC - y0);
  const double b = (xC - x1)*(xC - x1) + (yC - y1)*(yC - y1);
  const double c = (x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0);
  Check(c > a && c > b, "corner control points require a C-shaped corner");

  const double x0m = x0 + curviness*(xC - x0), y0m = y0 + curviness*(yC - y0);
  const double x1m = x1 + curviness*(xC - x1), y1m = y1 + curviness*(yC - y1);
  return CtrlPts{Vec6{{x0, x0m, x0m, x1m, x1m, x1}},
                 Vec6{{y0, y0m, y0m, y1m, y1m, y1}}};
}

// d^order/du^order of the quintic Bezier through `p`. order 0 is the value.
inline double DerivU(const Vec6& p, double u, int order) {
  Check(order >= 0, "negative derivative order");
  Check(u >= 0.0 && u <= 1.0, "u out of [0, 1]");
  if (order > 5) {
    return 0.0;
  }
  double q[6];
  for (int i = 0; i < 6; i++) {
    q[i] = p.v[i];
  }
  int n = 5;
  double scale = 1.0;
  for (int k = 0; k < order; k++) {  // k-th forward differences, scaled by n!/(n-k)!
    for (int i = 0; i < n; i++) {
      q[i] = q[i+1] - q[i];
    }
    scale *= static_cast<double>(n);
    n--;
  }
  for (int r = 1; r <= n; r++) {  // de Casteljau: stable, no binomials
    for (int i = 0; i <= n - r; i++) {
      q[i] = q[i]*(1.0 - u) + q[i+1]*u;
    }
  }
  return scale*q[0];
}

// d^order y / dx^order via the chain rule. Port of calcQuinticBezierCurveDerivDYDX,
// orders 0-2 (value, slope, curvature) which is all the baked table stores.
inline double DerivDyDx(const CtrlPts& p, double u, int order) {
  switch (order) {
    case 0:
      return DerivU(p.y, u, 0);
    case 1: {
      const double dxdu = DerivU(p.x, u, 1);
      return DerivU(p.y, u, 1) / dxdu;
    }
    case 2: {
      const double dxdu = DerivU(p.x, u, 1), dydu = DerivU(p.y, u, 1);
      const double d2xdu2 = DerivU(p.x, u, 2), d2ydu2 = DerivU(p.y, u, 2);
      const double t1 = 1.0 / dxdu;
      return (d2ydu2*t1 - dydu/(dxdu*dxdu)*d2xdu2) * t1;
    }
    default:
      Check(false, "DerivDyDx supports order 0-2");
      return 0.0;
  }
}

inline double ClampU(double u) { return u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u); }

// Invert x(u) on one section by Newton. Upstream seeds this from a 100-sample-per-section
// cubic spline of the approximate inverse; the baker walks a monotone x-grid instead, so
// the previous grid point's u is a strictly better seed and SimTK::Spline is not needed.
// Tolerance and iteration cap match upstream (UTOL = eps*1e2, MAXITER = 20).
inline double CalcU(double x, const Vec6& xpts, double u_guess,
                    double tol = 2.220446049250313e-16*1e2, int max_iter = 20) {
  Check(x >= xpts[0] - 1e-12 && x <= xpts[5] + 1e-12, "x outside the section");
  double u = ClampU(u_guess);
  double f = DerivU(xpts, u, 0) - x;
  int it = 0;
  while (std::fabs(f) > tol && it < max_iter) {
    const double df = DerivU(xpts, u, 1);
    if (std::fabs(df) <= 0.0) {
      break;  // upstream calls this 'pathologic' and bails
    }
    u = ClampU(u - f/df);
    f = DerivU(xpts, u, 0) - x;
    it++;
  }
  Check(std::fabs(f) <= 1e-9, "CalcU failed to converge on the section");
  return u;
}

// One or more Bezier sections over [x0, x1], linearly extrapolated outside. The analog of
// OpenSim::SmoothSegmentedFunction. The domain and extrapolation anchors are passed
// explicitly, exactly as the upstream constructor receives them, so the extrapolation
// slope stays exactly the biomechanical parameter that was declared rather than the
// ~5e-13 of roundoff the corner construction carries.
struct SegFn {
  std::string name;
  std::vector<CtrlPts> sec;  // ordered, contiguous in x
  double x0 = 0, x1 = 0, y0 = 0, y1 = 0, dydx0 = 0, dydx1 = 0;

  static SegFn FromSections(std::string nm, std::vector<CtrlPts> s,
                            double x0, double x1, double y0, double y1,
                            double dydx0, double dydx1) {
    Check(!s.empty(), "no sections");
    SegFn f;
    f.name = std::move(nm);
    f.sec = std::move(s);
    for (size_t i = 1; i < f.sec.size(); i++) {  // section i starts where i-1 ended
      Check(std::fabs(f.sec[i].x[0] - f.sec[i-1].x[5]) < 1e-12, "sections not contiguous");
    }
    f.x0 = x0; f.x1 = x1; f.y0 = y0; f.y1 = y1; f.dydx0 = dydx0; f.dydx1 = dydx1;
    // cross-check the explicit anchors against what the sections actually produce; a
    // disagreement means a wrong intermediate control point
    Check(std::fabs(f.sec.front().x[0] - x0) < 1e-12, "x0 anchor mismatch");
    Check(std::fabs(f.sec.back().x[5] - x1) < 1e-12, "x1 anchor mismatch");
    Check(std::fabs(f.sec.front().y[0] - y0) < 1e-12, "y0 anchor mismatch");
    Check(std::fabs(f.sec.back().y[5] - y1) < 1e-12, "y1 anchor mismatch");
    Check(std::fabs(DerivDyDx(f.sec.front(), 0.0, 1) - dydx0) < 1e-8, "dydx0 anchor mismatch");
    Check(std::fabs(DerivDyDx(f.sec.back(), 1.0, 1) - dydx1) < 1e-8, "dydx1 anchor mismatch");
    return f;
  }

  // Which section holds x. Upstream scans linearly (calcIndex).
  int IndexOf(double x) const {
    for (size_t i = 0; i < sec.size(); i++) {
      if (x >= sec[i].x[0] && x < sec[i].x[5]) {
        return static_cast<int>(i);
      }
    }
    if (x == sec.back().x[5]) {
      return static_cast<int>(sec.size()) - 1;
    }
    Check(false, "x outside the Bezier section set");
    return 0;
  }

  // Value and first derivative at x. `u_hint` warm-starts the inversion (pass the previous
  // grid point's u when walking a monotone grid; 0.5 is a safe cold start).
  void Eval(double x, double* y, double* dydx, double* u_hint = nullptr) const {
    if (x < x0) {  // linear extrapolation, exactly as upstream
      if (y) *y = y0 + dydx0*(x - x0);
      if (dydx) *dydx = dydx0;
      return;
    }
    if (x > x1) {
      if (y) *y = y1 + dydx1*(x - x1);
      if (dydx) *dydx = dydx1;
      return;
    }
    const int i = IndexOf(x);
    const double u = CalcU(x, sec[i].x, u_hint ? *u_hint : 0.5);
    if (u_hint) *u_hint = u;
    if (y) *y = DerivDyDx(sec[i], u, 0);
    if (dydx) *dydx = DerivDyDx(sec[i], u, 1);
  }

  double Value(double x) const { double y; Eval(x, &y, nullptr); return y; }
  double Slope(double x) const { double d; Eval(x, nullptr, &d); return d; }
};

// ---------------------------------------------------------------------------------------
// The four curve factories. Port of SmoothSegmentedFunctionFactory's
// create{FiberActiveForceLength,FiberForceLength,TendonForceLength,FiberForceVelocity}Curve.
//
// The user-facing curviness is SCALED: scaleCurviness(c) = 0.1 + 0.8*c, so a declared 0
// becomes 0.1 and the degenerate curviness == 0 case is unreachable through these.

inline double ScaleCurviness(double c) { return 0.1 + 0.8*c; }

// Active force-length. Five elbow sections over [x0, x3]; both extrapolation slopes are
// ZERO, so the curve flattens to `ylow` outside the domain.
// Defaults (ActiveForceLengthCurve.cpp): x0 = 0.4441, x1 = 0.73, x2 = 1.0, x3 = 1.8123,
// ylow = 0 (damped model), dydx = 0.8616, curviness = 1.0.
inline SegFn ActiveForceLength(double x0, double x1, double x2, double x3,
                               double ylow, double dydx, double curviness) {
  Check(x0 < x1 && x1 < x2 && x2 < x3, "active F-L knots must be increasing");
  Check(ylow >= 0.0 && dydx >= 0.0, "active F-L ylow/dydx must be non-negative");
  const double c = ScaleCurviness(curviness);
  const double xDelta = 0.05*x2;  // half-width of the sarcomere shoulder
  const double xs = x2 - xDelta;

  const double y0 = 0.0, dydx0 = 0.0;
  const double y1 = 1.0 - dydx*(xs - x1);
  const double dydx01 = 1.25*(y1 - y0)/(x1 - x0);
  const double x01 = x0 + 0.5*(x1 - x0);
  const double y01 = y0 + 0.5*(y1 - y0);

  const double x1s = x1 + 0.5*(xs - x1);
  const double y1s = y1 + 0.5*(1.0 - y1);
  const double dydx1s = dydx;

  const double y2 = 1.0, dydx2 = 0.0;
  const double y3 = 0.0, dydx3 = 0.0;
  const double x23 = (x2 + xDelta) + 0.5*(x3 - (x2 + xDelta));
  const double y23 = y2 + 0.5*(y3 - y2);
  const double dydx23 = (y3 - y2)/((x3 - xDelta) - (x2 + xDelta));

  std::vector<CtrlPts> s;
  s.push_back(CornerControlPoints(x0,  ylow, dydx0,  x01, y01, dydx01, c));
  s.push_back(CornerControlPoints(x01, y01,  dydx01, x1s, y1s, dydx1s, c));
  s.push_back(CornerControlPoints(x1s, y1s,  dydx1s, x2,  y2,  dydx2,  c));
  s.push_back(CornerControlPoints(x2,  y2,   dydx2,  x23, y23, dydx23, c));
  s.push_back(CornerControlPoints(x23, y23,  dydx23, x3,  ylow, dydx3, c));
  // upstream: (x0, x3, ylow, ylow, 0, 0) -- flat extrapolation both ways
  return SegFn::FromSections("fiberActiveForceLength", std::move(s),
                             x0, x3, ylow, ylow, 0.0, 0.0);
}

// Passive fiber force-length. Two sections over [1+eZero, 1+eIso]; extrapolates flat below
// and with slope kIso above.
// Defaults (FiberForceLengthCurve.cpp): eZero = 0, eIso = 0.7, kLow = 0.2,
// kIso = 2/(eIso-eZero), curviness = 0.75.
inline SegFn FiberForceLength(double eZero, double eIso, double kLow, double kIso,
                              double curviness) {
  Check(eIso > eZero, "passive F-L needs eIso > eZero");
  Check(kIso > 1.0/(eIso - eZero), "passive F-L needs kIso > 1/(eIso-eZero)");
  Check(kLow > 0.0 && kLow < 1.0/(eIso - eZero), "passive F-L kLow out of range");
  const double c = ScaleCurviness(curviness);
  const double xZero = 1.0 + eZero, yZero = 0.0;
  const double xIso = 1.0 + eIso, yIso = 1.0;
  const double deltaX = std::fmin(0.1*(1.0/kIso), 0.1*(xIso - xZero));
  const double xLow = xZero + deltaX;
  const double xfoot = xZero + 0.5*(xLow - xZero), yfoot = 0.0;
  const double yLow = yfoot + kLow*(xLow - xfoot);

  std::vector<CtrlPts> s;
  s.push_back(CornerControlPoints(xZero, yZero, 0.0,  xLow, yLow, kLow, c));
  s.push_back(CornerControlPoints(xLow,  yLow,  kLow, xIso, yIso, kIso, c));
  // upstream: (xZero, xIso, yZero, yIso, 0.0, kIso)
  return SegFn::FromSections("fiberForceLength", std::move(s),
                             xZero, xIso, yZero, yIso, 0.0, kIso);
}

// Tendon force-length. Two sections over [1.0, xToe] ONLY -- the stiff linear leg above
// xToe is the LINEAR EXTRAPOLATION, not a Bezier section, which is why the tabulated
// domain is narrow (~0.037 wide at the defaults) and the stiff part is reproduced exactly.
// Defaults (TendonForceLengthCurve.cpp): eIso = 0.049, kIso = 1.375/eIso, fToe = 2/3,
// curviness = 0.5.
inline SegFn TendonForceLength(double eIso, double kIso, double fToe, double curviness) {
  Check(eIso > 0.0, "tendon eIso must be positive");
  Check(fToe > 0.0 && fToe < 1.0, "tendon fToe must be in (0, 1)");
  Check(kIso > 1.0/eIso, "tendon needs kIso > 1/eIso");
  const double c = ScaleCurviness(curviness);
  const double x0 = 1.0, y0 = 0.0, dydx0 = 0.0;
  const double xIso = 1.0 + eIso, yIso = 1.0, dydxIso = kIso;
  const double yToe = fToe;
  const double xToe = (yToe - 1.0)/kIso + xIso;  // where the curve becomes linear
  // the toe's asymptote must cross the x axis right of the origin, hence xFoot
  const double xFoot = 1.0 + (xToe - 1.0)/10.0, yFoot = 0.0;
  const double yToeMid = yToe*0.5;
  const double xToeMid = (yToeMid - yIso)/kIso + xIso;
  const double dydxToeMid = (yToeMid - yFoot)/(xToeMid - xFoot);
  const double xToeCtrl = xFoot + 0.5*(xToeMid - xFoot);
  const double yToeCtrl = yFoot + dydxToeMid*(xToeCtrl - xFoot);

  std::vector<CtrlPts> s;
  s.push_back(CornerControlPoints(x0, y0, dydx0, xToeCtrl, yToeCtrl, dydxToeMid, c));
  s.push_back(CornerControlPoints(xToeCtrl, yToeCtrl, dydxToeMid, xToe, yToe, dydxIso, c));
  // upstream: (x0, xToe, y0, yToe, dydx0, dydxIso) -- note x1 is xToe, NOT xIso
  return SegFn::FromSections("tendonForceLength", std::move(s),
                             x0, xToe, y0, yToe, dydx0, dydxIso);
}

// Force-velocity. Four sections over [-1, 1] in normalized fiber velocity (positive =
// lengthening). Defaults (ForceVelocityCurve.cpp) with the at-vmax slopes FORCED to 0,
// which is the damped (beta > 0) model: fmaxE = 1.4, dydxC = 0, dydxNearC = 0.25,
// dydxIso = 5.0, dydxE = 0, dydxNearE = 0.15, concCurviness = 0.6, eccCurviness = 0.9.
//
// With dydxC = dydxE = 0 the extrapolation outside [-1, 1] is FLAT, so f_V contributes
// nothing to dR/dv there. That is why beta > 0 is a well-posedness condition and not a
// decoration: beta keeps the Newton Jacobian bounded away from zero when the solve
// transiently leaves the velocity domain.
inline SegFn ForceVelocity(double fmaxE, double dydxC, double dydxNearC, double dydxIso,
                           double dydxE, double dydxNearE, double concCurviness,
                           double eccCurviness) {
  Check(fmaxE > 1.0, "F-V fmaxE must exceed 1");
  Check(dydxC >= 0.0 && dydxC < 1.0, "F-V dydxC out of range");
  Check(dydxNearC > dydxC && dydxNearC <= 1.0, "F-V dydxNearC out of range");
  Check(dydxIso > 1.0, "F-V dydxIso must exceed 1");
  Check(dydxE >= 0.0 && dydxE < (fmaxE - 1.0), "F-V dydxE out of range");
  Check(dydxNearE >= dydxE && dydxNearE < (fmaxE - 1.0), "F-V dydxNearE out of range");
  const double cC = ScaleCurviness(concCurviness), cE = ScaleCurviness(eccCurviness);

  const double xC = -1.0, yC = 0.0;
  const double xNearC = -0.9;
  const double yNearC = yC + 0.5*dydxNearC*(xNearC - xC) + 0.5*dydxC*(xNearC - xC);
  const double xIso = 0.0, yIso = 1.0;
  const double xE = 1.0, yE = fmaxE;
  const double xNearE = 0.9;
  const double yNearE = yE + 0.5*dydxNearE*(xNearE - xE) + 0.5*dydxE*(xNearE - xE);

  std::vector<CtrlPts> s;
  s.push_back(CornerControlPoints(xC, yC, dydxC, xNearC, yNearC, dydxNearC, cC));
  s.push_back(CornerControlPoints(xNearC, yNearC, dydxNearC, xIso, yIso, dydxIso, cC));
  s.push_back(CornerControlPoints(xIso, yIso, dydxIso, xNearE, yNearE, dydxNearE, cE));
  s.push_back(CornerControlPoints(xNearE, yNearE, dydxNearE, xE, yE, dydxE, cE));
  // upstream: (xC, xE, yC, yE, dydxC, dydxE) -- dydxC = dydxE = 0 => FLAT extrapolation
  return SegFn::FromSections("fiberForceVelocity", std::move(s),
                             xC, xE, yC, yE, dydxC, dydxE);
}

// ---------------------------------------------------------------------------------------
// The baker: samples an exact SegFn onto the uniform grid the engine reads. Two details matter.
//   * the grid is walked MONOTONICALLY and each u inversion is warm-started from the previous
//     knot's u, which keeps the whole bake to a couple of Newton steps per knot and removes the
//     seed spline upstream needs entirely.
//   * the BOUNDARY knots take the SegFn's exact anchor VALUE and SLOPE rather than the Bezier
//     evaluation. The two agree to ~1e-13, and using the anchors makes the interpolated branch
//     agree with the extrapolated branch at the seam by construction. Their SECOND derivative is
//     still the curve's true one: the seam is C1 and not C2, which is faithful (OpenSim's
//     extrapolation has zero curvature while its Bezier end does not), and the interior
//     interpolation needs the real value.
//
// The three per-knot quantities are INTERLEAVED as (y, dy/dx, d2y/dx2), so the six numbers one
// cell needs -- knot i and knot i+1 -- are 48 contiguous bytes and normally a single cache line.
// Three parallel arrays would touch three lines per curve, twelve per residual evaluation, for
// exactly the same arithmetic.
struct Baked {
  std::vector<mjtNum> knot;   // [3n] (y, dy/dx, d2y/dx2) per knot
  mjCurveTable table;         // `knot` field points into the vector above
};

inline Baked Bake(const SegFn& f, int n) {
  Check(n >= 2, "need at least 2 knots");
  Baked b;
  b.knot.resize(3*static_cast<size_t>(n));

  double u_hint = 0.0;  // monotone walk => previous u is the best seed
  const double dx = (f.x1 - f.x0)/static_cast<double>(n - 1);
  for (int i = 0; i < n; i++) {
    const double x = (i == 0) ? f.x0 : (i == n-1 ? f.x1 : f.x0 + dx*static_cast<double>(i));
    const int s = f.IndexOf(i == n-1 ? f.x1 - 1e-15 : x);
    const double xs0 = f.sec[s].x[0], xs5 = f.sec[s].x[5];
    const double xc = x < xs0 ? xs0 : (x > xs5 ? xs5 : x);
    const double u = CalcU(xc, f.sec[s].x, u_hint);
    u_hint = u;
    b.knot[3*i+0] = (i == 0) ? f.y0 : (i == n-1 ? f.y1 : DerivDyDx(f.sec[s], u, 0));
    b.knot[3*i+1] = (i == 0) ? f.dydx0 : (i == n-1 ? f.dydx1 : DerivDyDx(f.sec[s], u, 1));
    b.knot[3*i+2] = DerivDyDx(f.sec[s], u, 2);   // always the curve's true curvature
  }

  b.table.knot = b.knot.data();
  b.table.n = n;
  b.table.x0 = f.x0;
  b.table.x1 = f.x1;
  b.table.dx = dx;
  b.table.inv_dx = 1.0/dx;
  b.table.a_y0 = f.y0;
  b.table.a_d0 = f.dydx0;
  b.table.a_y1 = f.y1;
  b.table.a_d1 = f.dydx1;
  return b;
}

}  // namespace millard
}  // namespace mujoco

#endif  // MUJOCO_SRC_ENGINE_ENGINE_MUSCLE_MILLARD_BEZIER_H_
