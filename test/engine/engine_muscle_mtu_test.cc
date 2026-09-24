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

// Tests for engine/engine_muscle_mtu.c and engine/engine_muscle_bake.cc -- the
// gaintype="millard_mtu" (OpenSim Millard2012EquilibriumMuscle) and gaintype="hyfydy_mtu"
// (Hyfydy muscle_force_m2012fast) actuators.

#include "src/engine/engine_muscle_mtu.h"

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_muscle_millard_bezier.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::DoubleNear;
using MuscleMtuTest = MujocoTest;

// A point mass hung from a single muscle. `gain` selects the muscle model and `prm` its gainprm.
std::string HangingMuscle(const char* gain, const char* prm, double mass = 20.0) {
  return std::string(R"(
  <mujoco>
    <option timestep="0.001" gravity="0 0 -9.81"/>
    <worldbody>
      <site name="anchor" pos="0 0 0.3" size="0.005"/>
      <body name="m">
        <joint name="s" type="slide" axis="0 0 1"/>
        <site name="ins" size="0.005"/>
        <geom type="sphere" size="0.01" mass=")") + std::to_string(mass) + R"("/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="t"><site site="anchor"/><site site="ins"/></spatial>
    </tendon>
    <actuator>
      <general name="a" tendon="t" gaintype=")" + gain + R"(" biastype="none"
               dyntype="muscle" dynprm="0.01 0.04" ctrllimited="true" ctrlrange="0 1"
               gainprm=")" + prm + R"("/>
    </actuator>
  </mujoco>)";
}

// F_max l_opt l_slack v_max pennation beta -- the rest default
constexpr char kDefaultPrm[] = "3000 0.15 0.15 10 0.15 0.1";


// ------------------------------------------------------------------------------------------
// The curves themselves.

// Landmark values every one of OpenSim's four curves is defined by. If the bake or the table
// interpolation is wrong, these are what move.
TEST_F(MuscleMtuTest, MillardCurveLandmarks) {
  mjtNum d;

  // active force-length: peaks at 1 at the optimal length, vanishes at both ends of its domain
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_ACTIVE_FL, 1.0, nullptr, nullptr),
              DoubleNear(1.0, 1e-9));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_ACTIVE_FL, 0.4441, nullptr, nullptr),
              DoubleNear(0.0, 1e-12));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_ACTIVE_FL, 1.8123, nullptr, nullptr),
              DoubleNear(0.0, 1e-12));
  // flat extrapolation outside the domain, both ways
  EXPECT_EQ(mju_millardCurve(mjMUSCLECURVE_ACTIVE_FL, 0.2, nullptr, nullptr), 0.0);
  EXPECT_EQ(mju_millardCurve(mjMUSCLECURVE_ACTIVE_FL, 3.0, nullptr, nullptr), 0.0);

  // passive: one normalized force at 70% strain, with the derived stiffness 2/eIso there
  EXPECT_EQ(mju_millardCurve(mjMUSCLECURVE_PASSIVE_FL, 1.0, nullptr, nullptr), 0.0);
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_PASSIVE_FL, 1.7, nullptr, &d),
              DoubleNear(1.0, 1e-9));
  EXPECT_THAT(d, DoubleNear(2.0/0.7, 1e-8));

  // tendon: one normalized force at 4.9% strain, stiffness 1.375/eIso, linear above the toe
  EXPECT_EQ(mju_millardCurve(mjMUSCLECURVE_TENDON_FL, 1.0, nullptr, nullptr), 0.0);
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_TENDON_FL, 1.049, nullptr, &d),
              DoubleNear(1.0, 1e-8));
  EXPECT_THAT(d, DoubleNear(1.375/0.049, 1e-8));

  // force-velocity: 1 isometric with slope 5, 0 at max shortening, fmaxE at max lengthening
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_FV, 0.0, nullptr, &d), DoubleNear(1.0, 1e-9));
  EXPECT_THAT(d, DoubleNear(5.0, 1e-7));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_FV, -1.0, nullptr, nullptr), DoubleNear(0.0, 1e-12));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_FV, 1.0, nullptr, nullptr), DoubleNear(1.4, 1e-9));
}


// ------------------------------------------------------------------------------------------
// Independent cross-check of the Bezier port.
//
// engine_muscle_millard_bezier.h deliberately does NOT transcribe OpenSim's evaluator: upstream
// carries machine-generated expanded polynomials (Maple output, `t2 = u1 * 0.5e1` and so on),
// which are error-prone to copy and unauditable, so the port computes the same quantities from
// the Bernstein derivative identity by de Casteljau instead.
//
// That choice is only safe if the two agree, which is what this checks. `UpstreamDerivU` below is
// SegmentedQuinticBezierToolkit::calcQuinticBezierCurveDerivU transcribed VERBATIM from
// OpenSim/Common/SegmentedQuinticBezierToolkit.cpp, error checks aside. It exists solely as an
// oracle for this test and is not compiled into the engine.
//
// (The chain rule on top of it, calcQuinticBezierCurveDerivDYDX orders 0-2, is short enough that
// the port copies it directly and it needs no separate oracle.)
double UpstreamDerivU(const millard::Vec6& pts, double u, int order) {
  double val = -1;
  double p0 = pts[0], p1 = pts[1], p2 = pts[2], p3 = pts[3], p4 = pts[4], p5 = pts[5];

  switch (order) {
    case 0: {
      double u5 = 1;
      double u4 = u;
      double u3 = u4*u;
      double u2 = u3*u;
      double u1 = u2*u;
      double u0 = u1*u;

      double t2 = u1 * 0.5e1;
      double t3 = u2 * 0.10e2;
      double t4 = u3 * 0.10e2;
      double t5 = u4 * 0.5e1;
      double t9 = u0 * 0.5e1;
      double t10 = u1 * 0.20e2;
      double t11 = u2 * 0.30e2;
      double t15 = u0 * 0.10e2;
      val = p0 * (u0 * (-0.1e1) + t2 - t3 + t4 - t5 + u5 * 0.1e1)
          + p1 * (t9 - t10 + t11 + u3 * (-0.20e2) + t5)
          + p2 * (-t15 + u1 * 0.30e2 - t11 + t4)
          + p3 * (t15 - t10 + t3)
          + p4 * (-t9 + t2) + p5 * u0 * 0.1e1;
    } break;
    case 1: {
      double t1 = u*u;
      double t2 = t1*t1;
      double t4 = t1 * u;
      double t5 = t4 * 0.20e2;
      double t6 = t1 * 0.30e2;
      double t7 = u * 0.20e2;
      double t10 = t2 * 0.25e2;
      double t11 = t4 * 0.80e2;
      double t12 = t1 * 0.90e2;
      double t16 = t2 * 0.50e2;
      val = p0 * (t2 * (-0.5e1) + t5 - t6 + t7 - 0.5e1)
          + p1 * (t10 - t11 + t12 + u * (-0.40e2) + 0.5e1)
          + p2 * (-t16 + t4 * 0.120e3 - t12 + t7)
          + p3 * (t16 - t11 + t6)
          + p4 * (-t10 + t5)
          + p5 * t2 * 0.5e1;
    } break;
    case 2: {
      double t1 = u*u;
      double t2 = t1 * u;
      double t4 = t1 * 0.60e2;
      double t5 = u * 0.60e2;
      double t8 = t2 * 0.100e3;
      double t9 = t1 * 0.240e3;
      double t10 = u * 0.180e3;
      double t13 = t2 * 0.200e3;
      val = p0 * (t2 * (-0.20e2) + t4 - t5 + 0.20e2)
          + p1 * (t8 - t9 + t10 - 0.40e2)
          + p2 * (-t13 + t1 * 0.360e3 - t10 + 0.20e2)
          + p3 * (t13 - t9 + t5)
          + p4 * (-t8 + t4)
          + p5 * t2 * 0.20e2;
    } break;
    case 3: {
      double t1 = u*u;
      double t3 = u * 0.120e3;
      double t6 = t1 * 0.300e3;
      double t7 = u * 0.480e3;
      double t10 = t1 * 0.600e3;
      val = p0 * (t1 * (-0.60e2) + t3 - 0.60e2)
          + p1 * (t6 - t7 + 0.180e3)
          + p2 * (-t10 + u * 0.720e3 - 0.180e3)
          + p3 * (t10 - t7 + 0.60e2)
          + p4 * (-t6 + t3)
          + p5 * t1 * 0.60e2;
    } break;
    case 4: {
      double t4 = u * 0.600e3;
      double t7 = u * 0.1200e4;
      val = p0 * (u * (-0.120e3) + 0.120e3)
          + p1 * (t4 - 0.480e3)
          + p2 * (-t7 + 0.720e3)
          + p3 * (t7 - 0.480e3)
          + p4 * (-t4 + 0.120e3)
          + p5 * u * 0.120e3;
    } break;
    case 5: {
      val = p0 * (-0.120e3)
          + p1 * 0.600e3
          + p2 * (-0.1200e4)
          + p3 * 0.1200e4
          + p4 * (-0.600e3)
          + p5 * 0.120e3;
    } break;
    default:
      val = 0;
  }
  return val;
}


// The port and OpenSim's expanded polynomials must agree to machine precision, on arbitrary
// control points and at every derivative order the curves use.
TEST_F(MuscleMtuTest, BezierEvaluatorMatchesUpstreamExpandedPolynomials) {
  // a deterministic spread of control points, including negative and widely separated ones
  unsigned seed = 12345;
  auto next = [&seed]() {
    seed = seed*1103515245u + 12345u;
    return static_cast<double>((seed >> 8) % 20000)/1000.0 - 10.0;   // [-10, 10]
  };

  double worst[6] = {0, 0, 0, 0, 0, 0};
  for (int trial = 0; trial < 2000; trial++) {
    millard::Vec6 pts;
    double scale = 0;
    for (int i = 0; i < 6; i++) {
      pts.v[i] = next();
      scale = std::fmax(scale, std::fabs(pts.v[i]));
    }
    for (int k = 0; k <= 20; k++) {
      double u = k/20.0;
      for (int order = 0; order <= 5; order++) {
        double got = millard::DerivU(pts, u, order);
        double want = UpstreamDerivU(pts, u, order);
        // relative to the coefficient magnitude, which the high orders multiply by up to 1200
        double tol_scale = scale*(order == 0 ? 1 : (order == 1 ? 5 : 1200));
        worst[order] = std::fmax(worst[order], std::fabs(got - want)/tol_scale);
      }
    }
  }
  for (int order = 0; order <= 5; order++) {
    EXPECT_LT(worst[order], 1e-14) << "derivative order " << order;
  }
}


// d2y/dx2 of an exact curve at x, which SegFn::Eval does not expose (the engine never needs it
// at runtime -- only the baker does, per knot).
double SecondDeriv(const millard::SegFn& f, double x) {
  if (x < f.x0 || x > f.x1) {
    return 0.0;                                    // linear extrapolation has zero curvature
  }
  int i = f.IndexOf(x == f.x1 ? f.x1 - 1e-15 : x);
  double u = millard::CalcU(x, f.sec[i].x, 0.5);
  return millard::DerivDyDx(f.sec[i], u, 2);
}


// The four curves, at OpenSim's damped-model defaults.
millard::SegFn ExactCurve(int which) {
  switch (which) {
    case mjMUSCLECURVE_ACTIVE_FL:
      return millard::ActiveForceLength(0.4441, 0.73, 1.0, 1.8123, 0.0, 0.8616, 1.0);
    case mjMUSCLECURVE_PASSIVE_FL:
      return millard::FiberForceLength(0.0, 0.7, 0.2, 2.0/0.7, 0.75);
    case mjMUSCLECURVE_TENDON_FL:
      return millard::TendonForceLength(0.049, 1.375/0.049, 2.0/3.0, 0.5);
    default:
      return millard::ForceVelocity(1.4, 0.0, 0.25, 5.0, 0.0, 0.15, 0.6, 0.9);
  }
}


// OpenSim's own acceptance criteria for these factories, from
// OpenSim/Common/Test/testSmoothSegmentedFunctionFactory.cpp ("Keypoint Testing"): each curve
// must not only pass through its defining points with the declared slope, but do so with ZERO
// CURVATURE. That is the condition an intermediate control point has to be right for -- the
// endpoint value and slope are pinned by the doubled corner points regardless.
TEST_F(MuscleMtuTest, CurvesMeetOpenSimKeypointConditions) {
  const double e0 = 0.049, kiso = 1.375/0.049;      // tendon defaults
  millard::SegFn tendon = ExactCurve(mjMUSCLECURVE_TENDON_FL);
  EXPECT_THAT(tendon.Value(1.0), DoubleNear(0.0, 1e-12));
  EXPECT_THAT(tendon.Slope(1.0), DoubleNear(0.0, 1e-8));
  EXPECT_THAT(SecondDeriv(tendon, 1.0), DoubleNear(0.0, 1e-6));
  EXPECT_THAT(tendon.Value(1 + e0), DoubleNear(1.0, 1e-9));
  EXPECT_THAT(tendon.Slope(1 + e0), DoubleNear(kiso, 1e-6));
  EXPECT_THAT(SecondDeriv(tendon, 1 + e0), DoubleNear(0.0, 1e-4));

  const double e0f = 0.7, kisof = 2.0/0.7;          // passive fiber defaults
  millard::SegFn fiber = ExactCurve(mjMUSCLECURVE_PASSIVE_FL);
  EXPECT_THAT(fiber.x0, DoubleNear(1.0, 1e-12));
  EXPECT_THAT(fiber.x1, DoubleNear(1.0 + e0f, 1e-12));
  EXPECT_THAT(fiber.Value(1.0), DoubleNear(0.0, 1e-12));
  EXPECT_THAT(fiber.Slope(1.0), DoubleNear(0.0, 1e-8));
  EXPECT_THAT(SecondDeriv(fiber, 1.0), DoubleNear(0.0, 1e-6));
  EXPECT_THAT(fiber.Value(1 + e0f), DoubleNear(1.0, 1e-9));
  EXPECT_THAT(fiber.Slope(1 + e0f), DoubleNear(kisof, 1e-6));
  EXPECT_THAT(SecondDeriv(fiber, 1 + e0f), DoubleNear(0.0, 1e-4));
}


// C2 continuity, the other property OpenSim's factory test asserts. A wrong intermediate control
// point leaves a jump in d2y/dx2 at a section boundary, and a jump is exactly what refinement
// cannot shrink: halving the sample spacing must halve the largest step in the curvature.
TEST_F(MuscleMtuTest, CurvesAreC2Continuous) {
  for (int c = 0; c < mjNMUSCLECURVE; c++) {
    millard::SegFn f = ExactCurve(c);
    double jump[2];
    for (int pass = 0; pass < 2; pass++) {
      int n = 2000 << pass;
      double worst = 0, prev = 0;
      for (int i = 0; i <= n; i++) {
        // stay strictly inside the domain: the seam to the linear extrapolation is C1, not C2,
        // and faithfully so -- OpenSim's extrapolation has zero curvature while its Bezier end
        // does not
        double x = f.x0 + (f.x1 - f.x0)*(0.001 + 0.998*i/n);
        double d2 = SecondDeriv(f, x);
        if (i > 0) worst = std::fmax(worst, std::fabs(d2 - prev));
        prev = d2;
      }
      jump[pass] = worst;
    }
    EXPECT_GT(jump[0]/jump[1], 1.8) << "curve " << c;   // 2.0 if C2, 1.0 at a curvature jump
  }
}


// Monotonicity, the last of OpenSim's factory checks: the tendon and passive fiber curves must
// rise without ever turning back, and so must the force-velocity curve.
TEST_F(MuscleMtuTest, CurvesAreMonotonicWhereRequired) {
  for (int c : {mjMUSCLECURVE_PASSIVE_FL, mjMUSCLECURVE_TENDON_FL, mjMUSCLECURVE_FV}) {
    millard::SegFn f = ExactCurve(c);
    double prev = -std::numeric_limits<double>::infinity();
    for (int i = 0; i <= 4000; i++) {
      double x = f.x0 + (f.x1 - f.x0)*(static_cast<double>(i)/4000);
      double y = f.Value(x);
      EXPECT_GE(y, prev - 1e-12) << "curve " << c << " decreased at x=" << x;
      prev = y;
      EXPECT_GE(f.Slope(x), -1e-9) << "curve " << c << " has negative slope at x=" << x;
    }
  }
}


// THE reference test: OpenSim's own factory and evaluator, compiled from the opensim-core
// sources and sampled into test/engine/testdata/millard_opensim_reference.csv (see
// tools/opensim_curve_dump.cpp). Everything else in this file checks the port against pieces of
// upstream or against upstream's stated criteria; this checks the whole curve against the number
// OpenSim actually produces, including outside the domain where the curves extrapolate.
//
// The tolerances are the baked table's own interpolation error, which is what should be left once
// the port is right: below 1e-8 in normalized force and 1e-5 in slope.
TEST_F(MuscleMtuTest, CurvesMatchOpenSimReference) {
  std::ifstream in("engine/testdata/millard_opensim_reference.csv");
  ASSERT_TRUE(in.good()) << "reference data missing";

  std::string line;
  ASSERT_TRUE(std::getline(in, line));                 // header
  EXPECT_EQ(line, "curve,x,y,dydx");

  const std::map<std::string, int> curve_of = {{"afl", mjMUSCLECURVE_ACTIVE_FL},
                                               {"pfl", mjMUSCLECURVE_PASSIVE_FL},
                                               {"tfl", mjMUSCLECURVE_TENDON_FL},
                                               {"fv", mjMUSCLECURVE_FV}};
  std::map<int, int> counted;
  double worst_y = 0, worst_d = 0;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string name, sx, sy, sd;
    ASSERT_TRUE(std::getline(ss, name, ','));
    ASSERT_TRUE(std::getline(ss, sx, ','));
    ASSERT_TRUE(std::getline(ss, sy, ','));
    ASSERT_TRUE(std::getline(ss, sd, ','));
    auto it = curve_of.find(name);
    ASSERT_NE(it, curve_of.end()) << "unknown curve " << name;

    double x = std::stod(sx), y = std::stod(sy), d = std::stod(sd);
    mjtNum got_d;
    double got_y = mju_millardCurve(it->second, x, nullptr, &got_d);
    worst_y = std::fmax(worst_y, std::fabs(got_y - y));
    worst_d = std::fmax(worst_d, std::fabs(got_d - d));
    counted[it->second]++;
  }

  EXPECT_LT(worst_y, 1e-8);
  EXPECT_LT(worst_d, 1e-5);
  // every curve must actually have been compared
  for (int c = 0; c < mjNMUSCLECURVE; c++) {
    EXPECT_GT(counted[c], 500) << "curve " << c << " under-sampled in the reference";
  }
}


// The baked table must reproduce the exact quintic-Bezier curve it was sampled from. This is
// the accuracy gate on the interpolation: it is checked off-knot, where a quintic Hermite has
// nothing to hide behind.
TEST_F(MuscleMtuTest, MillardTableMatchesExactBezier) {
  const millard::SegFn exact[mjNMUSCLECURVE] = {
      millard::ActiveForceLength(0.4441, 0.73, 1.0, 1.8123, 0.0, 0.8616, 1.0),
      millard::FiberForceLength(0.0, 0.7, 0.2, 2.0/0.7, 0.75),
      millard::TendonForceLength(0.049, 1.375/0.049, 2.0/3.0, 0.5),
      millard::ForceVelocity(1.4, 0.0, 0.25, 5.0, 0.0, 0.15, 0.6, 0.9)};

  for (int c = 0; c < mjNMUSCLECURVE; c++) {
    double worst = 0;
    // sample at 4001 points, none of which is a knot (513 knots, 4000 intervals)
    for (int i = 0; i <= 4000; i++) {
      double x = exact[c].x0 + (exact[c].x1 - exact[c].x0)*((i + 0.5)/4001.0);
      double got = mju_millardCurve(c, x, nullptr, nullptr);
      worst = std::fmax(worst, std::fabs(got - exact[c].Value(x)));
    }
    EXPECT_LT(worst, 1e-7) << "curve " << c;
  }
}


// Hyfydy's curves are published polynomials, so these are exact arithmetic identities.
TEST_F(MuscleMtuTest, HyfydyCurveLandmarks) {
  mjtNum d;
  EXPECT_THAT(mju_hyfydyCurve(mjMUSCLECURVE_ACTIVE_FL, 1.0, nullptr), DoubleNear(1.0, 1e-14));
  EXPECT_EQ(mju_hyfydyCurve(mjMUSCLECURVE_ACTIVE_FL, 0.4, nullptr), 0.0);   // below r1
  EXPECT_EQ(mju_hyfydyCurve(mjMUSCLECURVE_ACTIVE_FL, 1.9, nullptr), 0.0);   // above r2
  EXPECT_EQ(mju_hyfydyCurve(mjMUSCLECURVE_PASSIVE_FL, 1.0, nullptr), 0.0);
  // 1.08027*0.4^3 + 1.27368*0.4^2, the published coefficients read as c_P1 and c_P2
  EXPECT_THAT(mju_hyfydyCurve(mjMUSCLECURVE_PASSIVE_FL, 1.4, nullptr),
              DoubleNear(1.08027*0.064 + 1.27368*0.16, 1e-12));
  EXPECT_EQ(mju_hyfydyCurve(mjMUSCLECURVE_TENDON_FL, 1.0, nullptr), 0.0);
  EXPECT_THAT(mju_hyfydyCurve(mjMUSCLECURVE_FV, 0.0, nullptr), DoubleNear(1.0, 1e-14));
  EXPECT_EQ(mju_hyfydyCurve(mjMUSCLECURVE_FV, -1.0, nullptr), 0.0);
  EXPECT_THAT(mju_hyfydyCurve(mjMUSCLECURVE_FV, 1.0, &d), DoubleNear(1.71/1.11, 1e-12));

  // The two force-velocity branches meet in VALUE exactly at zero and in SLOPE to under 1%.
  // The Song muscle in engine_util_misc.c has a 2.4x kink there; this one effectively does not,
  // which is why it is the better-conditioned model to differentiate through.
  mjtNum d_minus, d_plus;
  mju_hyfydyCurve(mjMUSCLECURVE_FV, -1e-12, &d_minus);
  mju_hyfydyCurve(mjMUSCLECURVE_FV, 0.0, &d_plus);
  EXPECT_LT(std::fabs(d_plus - d_minus)/d_plus, 0.01);
}


// Every slot's zero-means-default rule must be exactly the OpenSim default it claims: writing
// the defaults out explicitly has to change nothing.
TEST_F(MuscleMtuTest, ZeroMeansOpenSimDefault) {
  std::array<mjtNum, mjNGAIN> prm = {};
  prm[mjMTU_AFL_MIN] = 0.4441;    prm[mjMTU_AFL_TRANS] = 0.73;
  prm[mjMTU_AFL_MAX] = 1.8123;    prm[mjMTU_AFL_SLOPE] = 0.8616;
  prm[mjMTU_PFL_E0] = 0.0;        prm[mjMTU_PFL_E1] = 0.7;
  prm[mjMTU_PFL_KLOW] = 0.2;      prm[mjMTU_PFL_KISO] = 2.0/0.7;
  prm[mjMTU_PFL_CURV] = 0.75;
  prm[mjMTU_TFL_E1] = 0.049;      prm[mjMTU_TFL_KISO] = 1.375/0.049;
  prm[mjMTU_TFL_FTOE] = 2.0/3.0;  prm[mjMTU_TFL_CURV] = 0.5;
  prm[mjMTU_FV_FMAXE] = 1.4;      prm[mjMTU_FV_DYDXNEARC] = 0.25;
  prm[mjMTU_FV_DYDXISO] = 5.0;    prm[mjMTU_FV_DYDXNEARE] = 0.15;
  prm[mjMTU_FV_CONCCURV] = 0.6;   prm[mjMTU_FV_ECCCURV] = 0.9;

  for (int c = 0; c < mjNMUSCLECURVE; c++) {
    for (double x = 0.2; x < 2.0; x += 0.013) {
      EXPECT_EQ(mju_millardCurve(c, x, prm.data(), nullptr),
                mju_millardCurve(c, x, nullptr, nullptr)) << "curve " << c << " at x=" << x;
    }
  }
}


// An .osim that overrides a curve's shape must actually get that shape. These are the
// invariants each OpenSim factory is defined by, checked at a non-default parameter set taken
// from RajagopalLaiUhlrich2023 (which tunes the passive curve per muscle).
TEST_F(MuscleMtuTest, CurveShapeOverridesAreHonoured) {
  std::array<mjtNum, mjNGAIN> prm = {};
  prm[mjMTU_AFL_MIN] = 0.25;      // Rajagopal's global active force-length override
  prm[mjMTU_AFL_TRANS] = 0.77;
  prm[mjMTU_AFL_MAX] = 1.9;
  prm[mjMTU_AFL_SLOPE] = 0.75;
  prm[mjMTU_PFL_E0] = -0.1844453874929782;   // addbrev_r's passive curve
  prm[mjMTU_PFL_E1] = 0.5059595363767915;
  prm[mjMTU_PFL_KLOW] = 0.2;
  prm[mjMTU_PFL_KISO] = 2.896858;
  prm[mjMTU_TFL_E1] = 0.06;
  prm[mjMTU_FV_FMAXE] = 1.8;

  mjtNum d;
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_ACTIVE_FL, 0.25, prm.data(), nullptr),
              DoubleNear(0.0, 1e-12));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_ACTIVE_FL, 1.9, prm.data(), nullptr),
              DoubleNear(0.0, 1e-12));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_ACTIVE_FL, 1.0, prm.data(), nullptr),
              DoubleNear(1.0, 1e-8));
  // passive force is zero at strain_at_zero_force and one at strain_at_one_norm_force, with
  // the declared stiffness there -- note eZero is NEGATIVE here, so passive force starts below
  // the optimal fiber length
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_PASSIVE_FL, 1 + prm[mjMTU_PFL_E0], prm.data(),
                               nullptr), DoubleNear(0.0, 1e-12));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_PASSIVE_FL, 1 + prm[mjMTU_PFL_E1], prm.data(), &d),
              DoubleNear(1.0, 1e-9));
  EXPECT_THAT(d, DoubleNear(prm[mjMTU_PFL_KISO], 1e-6));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_TENDON_FL, 1.06, prm.data(), nullptr),
              DoubleNear(1.0, 1e-8));
  EXPECT_THAT(mju_millardCurve(mjMUSCLECURVE_FV, 1.0, prm.data(), nullptr),
              DoubleNear(1.8, 1e-8));

  // and the override really is an override: it differs from the default curve
  EXPECT_GT(std::fabs(mju_millardCurve(mjMUSCLECURVE_PASSIVE_FL, 1.2, prm.data(), nullptr) -
                      mju_millardCurve(mjMUSCLECURVE_PASSIVE_FL, 1.2, nullptr, nullptr)), 0.05);
}


// Muscles that share a curve shape must share the baked table, or a large model would bake the
// same curve once per muscle.
TEST_F(MuscleMtuTest, CurveTablesAreSharedBetweenMuscles) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="anchor" pos="0 0 0.3" size="0.005"/>
      <body><joint type="slide" axis="0 0 1"/><site name="ins" size="0.005"/>
        <geom type="sphere" size="0.01" mass="1"/></body>
    </worldbody>
    <tendon><spatial name="t"><site site="anchor"/><site site="ins"/></spatial></tendon>
    <actuator>
      <general name="a" tendon="t" gaintype="millard_mtu" biastype="none" dyntype="muscle"
               gainprm="1000 0.1 0.1 10 0 0.1"/>
      <general name="b" tendon="t" gaintype="millard_mtu" biastype="none" dyntype="muscle"
               gainprm="2000 0.2 0.2 10 0.2 0.1"/>
      <general name="c" tendon="t" gaintype="millard_mtu" biastype="none" dyntype="muscle"
               gainprm="2000 0.2 0.2 10 0.2 0.1 0 0 0 0 0 0 0 0 0 0.5"/>
    </actuator>
  </mujoco>)";
  mjModel* model = LoadModelFromString(xml);
  ASSERT_THAT(model, ::testing::NotNull());
  mjData* data = mj_makeData(model);

  // a and b differ only in mechanical parameters, so all four curves are shared; c overrides
  // the passive strain, so only its passive curve is distinct
  for (int c = 0; c < mjNMUSCLECURVE; c++) {
    EXPECT_EQ(data->muscle_curve[mjNMUSCLECURVE*0 + c], data->muscle_curve[mjNMUSCLECURVE*1 + c]);
  }
  EXPECT_NE(data->muscle_curve[mjNMUSCLECURVE*2 + mjMUSCLECURVE_PASSIVE_FL],
            data->muscle_curve[mjNMUSCLECURVE*0 + mjMUSCLECURVE_PASSIVE_FL]);
  EXPECT_EQ(data->muscle_curve[mjNMUSCLECURVE*2 + mjMUSCLECURVE_ACTIVE_FL],
            data->muscle_curve[mjNMUSCLECURVE*0 + mjMUSCLECURVE_ACTIVE_FL]);

  // resetting must not bake anything new
  int before = mju_millardCurveCacheSize();
  mj_resetData(model, data);
  EXPECT_EQ(mju_millardCurveCacheSize(), before);

  mj_deleteData(data);
  mj_deleteModel(model);
}


// ------------------------------------------------------------------------------------------
// The equilibrium.

// Recompute the equilibrium residual from the public curves and the reported state. This checks
// the solver from outside: if the Newton stopped early, or the pennation algebra disagrees
// between residual and force, the residual will not be zero.
double Residual(const mjModel* m, const mjData* d, int id, bool hyfydy,
                double act = -1) {
  const mjtNum* p = m->actuator_gainprm + mjNGAIN*id;
  double l_opt = p[mjMTU_LOPT], l_slack = p[mjMTU_LSLACK];
  double v_max = p[mjMTU_VMAX] ? p[mjMTU_VMAX] : 10;
  double beta = p[mjMTU_BETA] == 0 ? 0.1 : (p[mjMTU_BETA] < 0 ? 0 : p[mjMTU_BETA]);
  double h = p[mjMTU_PENNATION] ? l_opt*std::sin(p[mjMTU_PENNATION]) : 0;

  double l_ce = d->muscle_l_ce[id];
  double sin_phi = h > 0 ? h/l_ce : 0;
  double cos_phi = std::sqrt(1 - sin_phi*sin_phi);
  double l0 = l_ce/l_opt;
  double v0 = d->muscle_v_ce[id]/(v_max*l_opt);
  double xT = d->muscle_l_se[id]/l_slack;
  double A = act >= 0 ? act
                      : d->act[m->actuator_actadr[id] + m->actuator_actnum[id] - 1];
  // the solve clamps the activation to [minimum_activation, 1], as OpenSim does wherever it
  // builds a force; the reconstruction has to use the same value
  double min_act = p[mjMTU_MINACT] == 0 ? 0.01 : (p[mjMTU_MINACT] < 0 ? 0 : p[mjMTU_MINACT]);
  A = A < min_act ? min_act : (A > 1 ? 1 : A);

  auto curve = [&](int c, double x) {
    return hyfydy ? mju_hyfydyCurve(c, x, nullptr) : mju_millardCurve(c, x, p, nullptr);
  };
  double fib = A*curve(mjMUSCLECURVE_ACTIVE_FL, l0)*curve(mjMUSCLECURVE_FV, v0) +
               curve(mjMUSCLECURVE_PASSIVE_FL, l0) + beta*v0;
  return curve(mjMUSCLECURVE_TENDON_FL, xT) - cos_phi*fib;
}


// A muscle holding a mass must settle at exactly the weight, and its fiber must sit at the
// equilibrium that produced it. This is the end-to-end statement that the model is a muscle.
TEST_F(MuscleMtuTest, StaticEquilibriumCarriesTheLoad) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    for (double ctrl : {0.0, 1.0}) {
      mjModel* model = LoadModelFromString(HangingMuscle(gain, kDefaultPrm));
      ASSERT_THAT(model, ::testing::NotNull()) << gain;
      mjData* data = mj_makeData(model);
      data->ctrl[0] = ctrl;
      for (int i = 0; i < 6000; i++) mj_step(model, data);

      EXPECT_THAT(data->muscle_F_mtu[0], DoubleNear(20.0*9.81, 1e-2))
          << gain << " at ctrl=" << ctrl;
      EXPECT_THAT(data->qvel[0], DoubleNear(0.0, 1e-4)) << gain << " at ctrl=" << ctrl;
      EXPECT_LT(std::fabs(Residual(model, data, 0, gain[0] == 'h')), 1e-6)
          << gain << " at ctrl=" << ctrl;
      mj_deleteData(data);
      mj_deleteModel(model);
    }
  }
}


// The equilibrium must stay solved while the muscle is moving, not just at rest -- that is what
// the backward-Euler residual and its analytic Jacobian are for.
TEST_F(MuscleMtuTest, EquilibriumHoldsDuringMotion) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* model = LoadModelFromString(HangingMuscle(gain, kDefaultPrm));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    mjData* data = mj_makeData(model);
    double worst = 0;
    for (int i = 0; i < 3000; i++) {
      data->ctrl[0] = 0.5 + 0.5*std::sin(0.02*i);   // keep it moving
      // the solve inside this step sees the PRE-step activation; mj_step integrates act
      // afterwards, so the residual has to be evaluated against the value that went in
      double act_in = data->act[1];
      mj_step(model, data);
      if (i > 10) {
        worst = std::fmax(worst, std::fabs(Residual(model, data, 0, gain[0] == 'h', act_in)));
      }
    }
    EXPECT_LT(worst, 1e-6) << gain;
    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// Worst first and second difference of the quasi-static force over a path-length sweep,
// sampled at spacing `step`. Under refinement these say what kind of function the force is: a
// jump keeps the first difference constant, a kink keeps the second difference constant, and a
// C1 function halves it.
struct SweepDiffs { double d1, d2; };

SweepDiffs SweepForce(const mjModel* m, mjData* d, double lo, double step, int n) {
  std::vector<double> f;
  for (int i = 0; i <= n; i++) {
    d->qpos[0] = lo + step*i;
    d->qvel[0] = 0;
    d->ctrl[0] = 0.5;
    d->act[1] = 0.5;                  // act = [l_ce, activation]
    mj_forward(m, d);
    mju_mtuMuscleEquilibrate(m, d);   // quasi-static: the fiber follows the pose
    mj_forward(m, d);
    f.push_back(d->muscle_F_mtu[0]);
  }
  SweepDiffs out = {0, 0};
  for (size_t i = 1; i + 1 < f.size(); i++) {
    out.d1 = std::fmax(out.d1, std::fabs(f[i+1] - f[i]));
    out.d2 = std::fmax(out.d2, std::fabs(f[i+1] - 2*f[i] + f[i-1]));
  }
  return out;
}


// The force must be a CONTINUOUS function of the path length: halving the sample spacing must
// halve the largest step in the force. A truncated Newton, or a solver that jumps between roots,
// leaves a jump here that refinement cannot shrink.
TEST_F(MuscleMtuTest, ForceIsContinuousInPathLength) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* model = LoadModelFromString(HangingMuscle(gain, kDefaultPrm));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    mjData* data = mj_makeData(model);

    SweepDiffs coarse = SweepForce(model, data, -0.05, 4e-4, 400);
    SweepDiffs fine = SweepForce(model, data, -0.05, 2e-4, 800);
    EXPECT_GT(coarse.d1/fine.d1, 1.8) << gain;   // 2.0 for continuous, 1.0 for a jump

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// Millard's force is C1 as well: refinement must shrink the second difference too. That is what
// the implicit solve and OpenSim's C2 curves buy, and it is what makes the muscle usable under
// finite-difference derivatives (mjd_transitionFD).
//
// hyfydy_mtu is deliberately NOT held to this. Its active force-length curve is published as a
// polynomial with a HARD CUT -- "0 for l <= r1" -- and the polynomial reaches zero at r1 with
// slope 4.19, not 0. So a Hyfydy muscle really does have a slope discontinuity where its fiber
// leaves the curve's support, and reproducing that is the point of shipping the model.
TEST_F(MuscleMtuTest, MillardForceIsC1InPathLength) {
  mjModel* model = LoadModelFromString(HangingMuscle("millard_mtu", kDefaultPrm));
  ASSERT_THAT(model, ::testing::NotNull());
  mjData* data = mj_makeData(model);

  SweepDiffs coarse = SweepForce(model, data, -0.05, 4e-4, 400);
  SweepDiffs fine = SweepForce(model, data, -0.05, 2e-4, 800);
  EXPECT_GT(coarse.d2/fine.d2, 1.8);   // >= 2 for C1, 4 for C2, 1 for a kink

  mj_deleteData(data);
  mj_deleteModel(model);
}


// A tendon far shorter than the fiber makes the series-elastic normalization ill-conditioned;
// the rigid-tendon path takes over and must still carry the load.
TEST_F(MuscleMtuTest, RigidTendonFallback) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* model = LoadModelFromString(HangingMuscle(gain, "3000 0.15 0.005 10 0 0.1"));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    mjData* data = mj_makeData(model);
    data->ctrl[0] = 0.5;
    for (int i = 0; i < 6000; i++) mj_step(model, data);
    EXPECT_THAT(data->muscle_F_mtu[0], DoubleNear(20.0*9.81, 1e-1)) << gain;
    // a rigid tendon reports its real length, l_MTU - l_ce cos(phi), which sits at the slack
    // length whenever the fiber is not on its clamp
    EXPECT_THAT(data->muscle_l_se[0], DoubleNear(0.005, 1e-12)) << gain;
    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// Pennation is fixed-width, so cos(phi) must follow l_ce and the tendon length must close the
// path: l_mtu = l_ce cos(phi) + l_T.
TEST_F(MuscleMtuTest, PennationClosesThePath) {
  mjModel* model = LoadModelFromString(HangingMuscle("millard_mtu", kDefaultPrm));
  ASSERT_THAT(model, ::testing::NotNull());
  mjData* data = mj_makeData(model);
  data->ctrl[0] = 1.0;
  for (int i = 0; i < 2000; i++) mj_step(model, data);

  double h = 0.15*std::sin(0.15);
  double l_ce = data->muscle_l_ce[0];
  double cos_phi = std::sqrt(1 - (h/l_ce)*(h/l_ce));
  EXPECT_THAT(l_ce*cos_phi + data->muscle_l_se[0],
              DoubleNear(data->actuator_length[0], 1e-12));
  mj_deleteData(data);
  mj_deleteModel(model);
}


// ------------------------------------------------------------------------------------------
// The MuJoCo state contract.
//
// The fiber length is an actuator ACTIVATION variable, act = [l_ce, activation], not a side
// array. The three tests below are what that buys, and each of them failed before the fiber
// moved into act.

// mj_forward must be a pure function of the state: calling it twice must not move anything.
TEST_F(MuscleMtuTest, ForwardIsIdempotent) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* model = LoadModelFromString(HangingMuscle(gain, kDefaultPrm));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    mjData* data = mj_makeData(model);
    data->ctrl[0] = 1.0;
    for (int i = 0; i < 300; i++) mj_step(model, data);   // get it moving

    mj_forward(model, data);
    double f1 = data->muscle_F_mtu[0], l1 = data->act[0];
    mj_forward(model, data);
    EXPECT_EQ(data->muscle_F_mtu[0], f1) << gain;
    EXPECT_EQ(data->act[0], l1) << gain;
    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// mjSTATE_PHYSICS must capture the whole muscle: restoring it into a fresh mjData and continuing
// has to reproduce the reference trajectory exactly.
TEST_F(MuscleMtuTest, StateRoundTripsThroughMjStatePhysics) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* model = LoadModelFromString(HangingMuscle(gain, kDefaultPrm));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    EXPECT_EQ(model->na, 2) << gain;   // [l_ce, activation]

    mjData* ref = mj_makeData(model);
    ref->ctrl[0] = 1.0;
    for (int i = 0; i < 300; i++) mj_step(model, ref);

    std::vector<mjtNum> state(mj_stateSize(model, mjSTATE_PHYSICS));
    mj_getState(model, ref, state.data(), mjSTATE_PHYSICS);
    for (int i = 0; i < 200; i++) mj_step(model, ref);

    mjData* restored = mj_makeData(model);
    restored->ctrl[0] = 1.0;
    mj_setState(model, restored, state.data(), mjSTATE_PHYSICS);
    for (int i = 0; i < 200; i++) mj_step(model, restored);

    EXPECT_EQ(restored->qpos[0], ref->qpos[0]) << gain;
    EXPECT_EQ(restored->muscle_F_mtu[0], ref->muscle_F_mtu[0]) << gain;
    mj_deleteData(ref);
    mj_deleteData(restored);
    mj_deleteModel(model);
  }
}


// RK4 evaluates mj_forward at intermediate states; with the fiber in act it integrates the fiber
// alongside everything else instead of advancing it once per evaluation.
TEST_F(MuscleMtuTest, RungeKuttaAgreesWithEuler) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    double q[2];
    for (int j = 0; j < 2; j++) {
      std::string xml = HangingMuscle(gain, kDefaultPrm);
      xml.replace(xml.find("<option "), 8, j ? "<option integrator=\"RK4\" " : "<option ");
      mjModel* model = LoadModelFromString(xml);
      ASSERT_THAT(model, ::testing::NotNull()) << gain;
      mjData* data = mj_makeData(model);
      data->ctrl[0] = 1.0;
      for (int i = 0; i < 2000; i++) mj_step(model, data);
      q[j] = data->qpos[0];
      mj_deleteData(data);
      mj_deleteModel(model);
    }
    EXPECT_NEAR(q[1], q[0], 1e-6) << gain;
  }
}


// Equilibration reproduces the isometric equilibrium: after it, the residual is zero and the
// fiber does not move on the next step.
TEST_F(MuscleMtuTest, EquilibrateSolvesTheIsometricEquilibrium) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* model = LoadModelFromString(HangingMuscle(gain, kDefaultPrm));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    mjData* data = mj_makeData(model);
    data->qpos[0] = 0.02;
    data->act[1] = 0.5;
    mj_forward(model, data);
    mju_mtuMuscleEquilibrate(model, data);
    mj_forward(model, data);

    EXPECT_LT(std::fabs(Residual(model, data, 0, gain[0] == 'h')), 1e-6) << gain;
    EXPECT_NEAR(data->act_dot[0], 0.0, 1e-6) << gain;   // fiber at rest
    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// ------------------------------------------------------------------------------------------
// Regimes the ordinary tests never enter.

// Fixed-width pennation puts a floor under the fiber: below h/sin(phi_max) the angle would
// exceed phi_max and 1/sqrt(1-(h/l_ce)^2) diverges. For a muscle pennated past ~26 degrees that
// floor is what sets lce_min, rather than the active curve's left foot, so this is the case where
// the pennation part of the clamp is load-bearing.
//
// Because lce_min >= h/sin(phi_max) and every evaluation is clamped to it, sin(phi) = h/l_ce can
// never exceed sin(phi_max): the clamp is what makes the guard inside the residual unreachable.
// That is the property under test here, and it is checked where it actually binds -- with a fiber
// length written straight into act, which a keyframe or a user edit can now do.
TEST_F(MuscleMtuTest, HeavilyPennatedMuscleStaysOnTheRightSideOfPhiMax) {
  // phi_opt = 30 deg: sin(30 deg)/sin(acos(0.1)) = 0.5025 > 0.4441, so h/sin(phi_max) wins
  const double phi_opt = 0.5235987755982988;
  const double l_opt = 0.15;
  const double h = l_opt*std::sin(phi_opt);
  const double lce_min = h/0.9949874371066201;
  EXPECT_GT(lce_min, 0.4441*l_opt);   // the pennation floor, not the active curve's, is binding

  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* model = LoadModelFromString(
        HangingMuscle(gain, "3000 0.15 0.15 10 0.5235987755982988 0.1"));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    mjData* data = mj_makeData(model);

    // (1) a fiber length below the floor -- including one below h itself, where the pennation
    // term would be imaginary -- must come back clamped, at exactly cos(phi_max) = 0.1
    for (double bad : {1e-9, 0.5*h, h, 0.99*lce_min}) {
      mj_resetData(model, data);
      data->qpos[0] = 0.0;
      data->act[0] = bad;
      data->act[1] = 0.5;
      data->ctrl[0] = 0.5;
      mj_forward(model, data);

      double l_ce = data->muscle_l_ce[0];
      ASSERT_TRUE(std::isfinite(l_ce)) << gain << " from act[0]=" << bad;
      ASSERT_TRUE(std::isfinite(data->muscle_F_mtu[0])) << gain << " from act[0]=" << bad;
      EXPECT_GE(l_ce, lce_min - 1e-12) << gain << " from act[0]=" << bad;
      double cos_phi = std::sqrt(1 - (h/l_ce)*(h/l_ce));
      EXPECT_GE(cos_phi, 0.1 - 1e-9) << gain << " from act[0]=" << bad;
    }

    // and stepping from there must recover rather than blow up
    mj_resetData(model, data);
    data->act[0] = 1e-6;
    data->act[1] = 0.5;
    data->ctrl[0] = 0.5;
    for (int i = 0; i < 50; i++) {
      mj_step(model, data);
      ASSERT_TRUE(std::isfinite(data->act[0])) << gain << " at step " << i;
      ASSERT_TRUE(std::isfinite(data->muscle_F_mtu[0])) << gain << " at step " << i;
    }
    EXPECT_GE(data->act[0], lce_min - 1e-9) << gain;   // pulled back into the valid range

    // (2) a whole path sweep: the geometry must close and cos(phi) stay above cos(phi_max)
    for (int i = 0; i <= 140; i++) {
      mj_resetData(model, data);
      data->qpos[0] = 0.001*i;
      data->act[1] = 1.0;
      data->ctrl[0] = 1.0;
      mj_forward(model, data);
      mju_mtuMuscleEquilibrate(model, data);
      mj_forward(model, data);

      double l_ce = data->muscle_l_ce[0];
      ASSERT_TRUE(std::isfinite(l_ce)) << gain << " at qpos " << data->qpos[0];
      EXPECT_GE(l_ce, lce_min - 1e-12) << gain << " at qpos " << data->qpos[0];
      double cos_phi = std::sqrt(1 - (h/l_ce)*(h/l_ce));
      EXPECT_GE(cos_phi, 0.1 - 1e-9) << gain;
      EXPECT_THAT(l_ce*cos_phi + data->muscle_l_se[0],
                  DoubleNear(data->actuator_length[0], 1e-12)) << gain;
    }

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// Outside |v0| <= 1 the damped Millard force-velocity curve extrapolates FLAT, so it contributes
// nothing to dR/dl_ce there and the fiber's velocity term is carried by the damping alone. That
// is the regime the fiber damping is a well-posedness condition for, and nothing else in the
// suite reaches it: v0 = v_ce/(v_max*l_opt) needs |v_ce| > 1.5 m/s at these parameters.
TEST_F(MuscleMtuTest, SolveConvergesBeyondTheForceVelocityDomain) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    // beta = 0.1 (default) and beta = 0 (asked for with a negative value)
    for (const char* prm : {"3000 0.15 0.15 10 0.15 0.1", "3000 0.15 0.15 10 0.15 -1"}) {
      mjModel* model = LoadModelFromString(HangingMuscle(gain, prm));
      ASSERT_THAT(model, ::testing::NotNull()) << gain;
      mjData* data = mj_makeData(model);
      data->ctrl[0] = 0.8;
      data->act[1] = 0.8;
      mj_forward(model, data);
      mju_mtuMuscleEquilibrate(model, data);

      // displace the fiber far enough that one implicit step has to move it fast
      double reached = 0;
      for (double push : {0.7, 0.8, 1.2, 1.4}) {
        double eq = data->muscle_l_ce[0];
        data->act[0] = eq*push;
        mj_forward(model, data);

        double v0 = data->muscle_v_ce[0]/(10*0.15);   // v_max*l_opt
        reached = std::fmax(reached, std::fabs(v0));
        EXPECT_TRUE(std::isfinite(data->muscle_F_mtu[0])) << gain << " " << prm;
        EXPECT_LT(std::fabs(Residual(model, data, 0, gain[0] == 'h', 0.8)), 1e-6)
            << gain << " " << prm << " at push " << push;
        data->act[0] = eq;
      }
      EXPECT_GT(reached, 1.0) << gain << " " << prm << ": never left the F-V domain";

      mj_deleteData(data);
      mj_deleteModel(model);
    }
  }
}


// The velocity derivative handed to the implicit integrators has to be the real one. Checked
// against a central difference of the actuator force with respect to the actuator velocity --
// both for the rigid-tendon path, where it is nonzero, and the compliant one, where the claim is
// that it is exactly zero.
TEST_F(MuscleMtuTest, ForceVelocityDerivativeMatchesFiniteDifference) {
  struct Case { const char* gain; const char* prm; bool rigid; };
  const Case cases[] = {
      {"millard_mtu", "3000 0.15 0.005 10 0.15 0.1", true},
      {"hyfydy_mtu",  "3000 0.15 0.005 10 0.15 0.1", true},
      {"millard_mtu", "3000 0.15 0.15 10 0.15 0.1",  false},
      {"hyfydy_mtu",  "3000 0.15 0.15 10 0.15 0.1",  false}};

  for (const Case& c : cases) {
    mjModel* model = LoadModelFromString(HangingMuscle(c.gain, c.prm));
    ASSERT_THAT(model, ::testing::NotNull()) << c.gain;
    mjData* data = mj_makeData(model);

    auto force_at = [&](double qvel) {
      // reset first: the two probes must differ in qvel and nothing else. Carrying the fiber
      // length over from the previous probe would make this measure the equilibration's own
      // tolerance, amplified by 1/(2h), rather than the velocity dependence it is after.
      mj_resetData(model, data);
      data->qpos[0] = 0.02;
      data->qvel[0] = qvel;
      data->act[1] = 0.6;
      data->ctrl[0] = 0.6;
      mj_forward(model, data);
      mju_mtuMuscleEquilibrate(model, data);
      mj_forward(model, data);
      return data->actuator_force[0];
    };

    const double h = 1e-4;
    double f_plus = force_at(h), v_plus = data->actuator_velocity[0];
    double f_minus = force_at(-h), v_minus = data->actuator_velocity[0];
    force_at(0.0);
    double analytic = mju_mtuMuscleForceVel(model, data, 0);
    double fd = (f_plus - f_minus)/(v_plus - v_minus);

    if (c.rigid) {
      EXPECT_GT(std::fabs(analytic), 1.0) << c.gain << ": rigid tendon must depend on velocity";
      EXPECT_NEAR(analytic, fd, 1e-4*std::fabs(analytic)) << c.gain << " rigid";
    } else {
      EXPECT_EQ(analytic, 0.0) << c.gain << ": compliant tendon force is a function of length";
      EXPECT_NEAR(fd, 0.0, 1e-9) << c.gain << " compliant";
    }

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// OpenSim clamps the activation to [minimum_activation, 1] wherever it builds a force, and
// defaults that floor to 0.01, so a Millard muscle never fully switches off. Writing the default
// out explicitly must change nothing, and asking for a true zero must be possible.
TEST_F(MuscleMtuTest, MinimumActivationFloorsTheActiveForce) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    //                                    F_max l_opt l_slack v_max phi  beta  tol  min_act
    mjModel* def = LoadModelFromString(HangingMuscle(gain, "3000 0.15 0.15 10 0.15 0.1"));
    mjModel* exp = LoadModelFromString(HangingMuscle(gain, "3000 0.15 0.15 10 0.15 0.1 0 0.01"));
    mjModel* off = LoadModelFromString(HangingMuscle(gain, "3000 0.15 0.15 10 0.15 0.1 0 -1"));
    ASSERT_THAT(def, ::testing::NotNull()) << gain;

    auto settle = [](mjModel* m) {
      mjData* d = mj_makeData(m);
      d->ctrl[0] = 0.0;                       // no excitation at all
      for (int i = 0; i < 6000; i++) mj_step(m, d);
      double l_ce = d->muscle_l_ce[0];
      mj_deleteData(d);
      return l_ce;
    };

    double l_default = settle(def), l_explicit = settle(exp), l_zero = settle(off);
    EXPECT_EQ(l_default, l_explicit) << gain;             // 0 in the slot means OpenSim's 0.01
    // 1% of activation is a real force: the fiber does not sit in the same place without it
    EXPECT_GT(std::fabs(l_default - l_zero), 1e-4) << gain;

    mj_deleteModel(def);
    mj_deleteModel(exp);
    mj_deleteModel(off);
  }
}


// OpenSim's rigid-tendon path has two rules beyond the algebra, and both zero the force: a fiber
// on (or below) its lower clamp carries nothing, and a tendon whose length has fallen below its
// slack length has buckled and imposes no velocity on the fiber.
TEST_F(MuscleMtuTest, RigidTendonHonoursOpenSimsClampAndBucklingRules) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    // a rigid tendon (l_slack < 0.05 l_opt) on a path that can be driven shorter than it is
    mjModel* model = LoadModelFromString(HangingMuscle(gain, "3000 0.15 0.005 10 0 0.1"));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    mjData* data = mj_makeData(model);

    bool saw_force = false, saw_zero = false;
    for (int i = 0; i <= 300; i++) {
      data->qpos[0] = 0.001*i;                // shortens the path from 0.30 m down to 0.00 m
      data->qvel[0] = 0;
      data->act[1] = 1.0;
      data->ctrl[0] = 1.0;
      mj_forward(model, data);

      double F = data->muscle_F_mtu[0];
      ASSERT_TRUE(std::isfinite(F)) << gain << " at qpos " << data->qpos[0];
      EXPECT_GE(F, 0.0) << gain << ": a rigid tendon cannot push";
      if (F > 1.0) saw_force = true;
      if (F == 0.0) saw_zero = true;

      // the reported tendon length is the real one, l_MTU - l_ce cos(phi), not l_slack
      EXPECT_THAT(data->muscle_l_ce[0] + data->muscle_l_se[0],
                  DoubleNear(data->actuator_length[0], 1e-12)) << gain;   // unpennated
    }
    EXPECT_TRUE(saw_force) << gain << ": the sweep never produced a force";
    EXPECT_TRUE(saw_zero) << gain << ": the sweep never reached the clamped/buckled region";

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// The cache never evicts, because mjData.muscle_curve points into it and nothing can enumerate
// live mjData. A search over curve shapes bakes a fresh set per candidate and never revisits one,
// so it needs a way to drop them; mju_millardCurveCacheClear is it. What has to hold afterwards
// is that a reset re-resolves the pointers and the model still produces the same force.
TEST_F(MuscleMtuTest, CurveCacheClearFreesEntriesAndResetRepopulatesThem) {
  mjModel* model = LoadModelFromString(HangingMuscle("millard_mtu", kDefaultPrm));
  ASSERT_THAT(model, ::testing::NotNull());
  mjData* data = mj_makeData(model);
  data->ctrl[0] = 0.6;
  for (int i = 0; i < 200; i++) mj_step(model, data);
  double force_before = data->muscle_F_mtu[0];
  ASSERT_GT(mju_millardCurveCacheSize(), 0);

  // bake some shapes that will never be used again, as a fit does
  for (int i = 1; i <= 20; i++) {
    model->actuator_gainprm[mjNGAIN*0 + mjMTU_PFL_E1] = 0.5 + 0.01*i;
    mj_resetData(model, data);
  }
  int grown = mju_millardCurveCacheSize();
  EXPECT_GT(grown, 4) << "the sweep should have baked new shapes";

  int freed = mju_millardCurveCacheClear();
  EXPECT_EQ(freed, grown);
  EXPECT_EQ(mju_millardCurveCacheSize(), 0);

  // the precondition: reset before use. That re-resolves muscle_curve.
  model->actuator_gainprm[mjNGAIN*0 + mjMTU_PFL_E1] = 0;   // back to the default shape
  mj_resetData(model, data);
  EXPECT_EQ(mju_millardCurveCacheSize(), mjNMUSCLECURVE) << "reset should re-bake exactly 4";
  for (int c = 0; c < mjNMUSCLECURVE; c++) {
    EXPECT_NE(data->muscle_curve[c], 0u) << "curve " << c << " not re-resolved";
  }

  data->ctrl[0] = 0.6;
  for (int i = 0; i < 200; i++) mj_step(model, data);
  EXPECT_THAT(data->muscle_F_mtu[0], DoubleNear(force_before, 1e-9))
      << "the same model after a clear must produce the same force";

  mj_deleteData(data);
  mj_deleteModel(model);
}


// Clearing an empty cache is a no-op, not an error.
TEST_F(MuscleMtuTest, CurveCacheClearIsIdempotent) {
  mju_millardCurveCacheClear();
  EXPECT_EQ(mju_millardCurveCacheSize(), 0);
  EXPECT_EQ(mju_millardCurveCacheClear(), 0);
  EXPECT_EQ(mju_millardCurveCacheSize(), 0);
}


// Equilibration has to reach the root from the seed mj_resetData leaves, and that seed is the
// worst case rather than a typical one: l_opt is the PEAK of the active force-length curve, where
// the active slope is zero, and if the tendon is slack there and the parallel element is too,
// every term of dR/dl_ce vanishes. Plain Newton has nothing to descend; with a step cap it does
// not diverge, it cycles, and the fiber comes back exactly where it started.
//
// Both muscles below are parameter transfers from real OpenSim muscles that did exactly that.
TEST_F(MuscleMtuTest, EquilibrateReachesTheRootFromTheResetSeed) {
  struct Case {
    const char* name;
    const char* prm;
    double l_mtu;
    double l_ce_expected;     // OpenSim computeEquilibrium
  };
  // BIClong (MoBL-ARMS), pennation 0: the seed sits at the active peak with a slack tendon, so
  // the Jacobian is zero to rounding there.
  // glmax3_r (Rajagopal), pennation 0.3824: the iterate lands on the fiber clamp, where
  // sin(phi) is within an ulp of sin(phi_max).
  const Case cases[] = {
      {"BIClong", "525.1 0.1157 0.2723 10 0 0.1 0 0.01 "
                  "0.4441 0.73 1.8123 0.8616 0 0 "
                  "0 0.7 0.2 2.8571 0.75 0 "
                  "0.049 28.0612 0.6667 0.5 "
                  "1.4 0 0.25 5.0 0 0.15 0.6 0.9", 0.3851, 0.10016},
      {"glmax3_r", "947.754098360656 0.16699999962441678 0.103 10 0.38241613 0.1 0 0.01 "
                   "0.25 0.77 1.9 0.75 0 0 "
                   "0.039777684490424015 0.7004987591743358 0.2 3.026995924046766 0.75 0 "
                   "0.049 28.06122448979592 0.6666666666666666 0.5 "
                   "1.4 0 0.25 5.0 0 0.15 0.6 0.9", 0.1554, 0.08020}};

  for (const Case& c : cases) {
    std::string xml = HangingMuscle("millard_mtu", c.prm);
    mjModel* model = LoadModelFromString(xml);
    ASSERT_THAT(model, ::testing::NotNull()) << c.name;
    mjData* data = mj_makeData(model);

    // put the path at the reported length; the anchor is at 0.3, the site at qpos
    data->qpos[0] = 0.3 - c.l_mtu;
    data->act[1] = 1.0;
    data->ctrl[0] = 1.0;
    mj_forward(model, data);
    ASSERT_THAT(data->actuator_length[0], DoubleNear(c.l_mtu, 1e-9)) << c.name;

    double seed = data->act[0];
    mju_mtuMuscleEquilibrate(model, data);
    mj_forward(model, data);

    EXPECT_NE(data->act[0], seed) << c.name << ": equilibrate was a no-op";
    EXPECT_THAT(data->act[0], DoubleNear(c.l_ce_expected, 1e-4)) << c.name;
    EXPECT_GT(data->muscle_F_mtu[0], 1.0) << c.name << ": a taut tendon should carry force";

    // and the fiber is genuinely at rest there
    EXPECT_NEAR(data->act_dot[0], 0.0, 1e-6) << c.name;
    EXPECT_LT(std::fabs(Residual(model, data, 0, false, 1.0)), 1e-6) << c.name;

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// Equilibration is a function of the pose and the activation, not of where the fiber happened to
// be. Two calls at the same pose from different fiber lengths must agree bit for bit -- a one-ulp
// difference is invisible until something differentiates the result, and then it is not.
TEST_F(MuscleMtuTest, EquilibrateIsIndependentOfTheIncomingFiberLength) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* model = LoadModelFromString(HangingMuscle(gain, kDefaultPrm));
    ASSERT_THAT(model, ::testing::NotNull()) << gain;
    mjData* data = mj_makeData(model);

    double results[3];
    const double starts[3] = {0.05, 0.15, 0.40};   // well below, at, and well above l_opt
    for (int i = 0; i < 3; i++) {
      mj_resetData(model, data);
      data->qpos[0] = 0.02;
      data->act[0] = starts[i];
      data->act[1] = 0.6;
      data->ctrl[0] = 0.6;
      mj_forward(model, data);
      mju_mtuMuscleEquilibrate(model, data);
      results[i] = data->act[0];
    }
    EXPECT_EQ(results[0], results[1]) << gain;
    EXPECT_EQ(results[1], results[2]) << gain;

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// Hyfydy's curves have no per-muscle shape, so a shape parameter set on a hyfydy_mtu actuator
// is a modelling mistake and must be rejected rather than silently ignored.
TEST_F(MuscleMtuTest, HyfydyRejectsCurveShapeParameters) {
  char error[1024];
  mjModel* model = LoadModelFromString(
      HangingMuscle("hyfydy_mtu", "3000 0.15 0.15 10 0 0.1 0 0 0.3"), error, sizeof(error));
  EXPECT_THAT(model, ::testing::IsNull());
  EXPECT_THAT(std::string(error), ::testing::HasSubstr("fixed polynomials"));
  if (model) mj_deleteModel(model);
}


// OpenSim's damped model sets the force-velocity curve's two at-vmax slopes to 0 whenever the fiber
// is damped, and only logs it. A declared non-zero slope with damping is refused rather than
// silently replaced; the undamped model (negative fiber_damping) keeps them.
TEST_F(MuscleMtuTest, DampedModelRefusesNonzeroSlopesAtVmax) {
  struct Case { const char* name; const char* prm; bool loads; };
  const Case cases[] = {
      {"concentric, default damping", "3000 0.15 0.15 10 0 0 0 0 " "0 0 0 0 0 0 0 0 0 0 0 0 "
                                      "0 0 0 0 " "0 0.2", false},
      {"eccentric, explicit damping", "3000 0.15 0.15 10 0 0.05 0 0 " "0 0 0 0 0 0 0 0 0 0 0 0 "
                                      "0 0 0 0 " "0 0 0 0 0.1", false},
      {"both, undamped",              "3000 0.15 0.15 10 0 -1 0 0 " "0 0 0 0 0 0 0 0 0 0 0 0 "
                                      "0 0 0 0 " "0 0.2 0 0 0.1", true},
      {"zero slopes, damped",         "3000 0.15 0.15 10 0 0.1", true}};
  for (const Case& c : cases) {
    char error[1024] = "";
    mjModel* model = LoadModelFromString(HangingMuscle("millard_mtu", c.prm), error,
                                         sizeof(error));
    if (c.loads) {
      EXPECT_THAT(model, ::testing::NotNull()) << c.name << ": " << error;
    } else {
      EXPECT_THAT(model, ::testing::IsNull()) << c.name;
      EXPECT_THAT(std::string(error), ::testing::HasSubstr("must be 0 with fiber damping"))
          << c.name;
    }
    if (model) mj_deleteModel(model);
  }
}


// ------------------------------------------------------------------------------------------
// Active force-length curves fitted to Thelen2003, where min_norm_active_fiber_length goes to 0.

// Slots 8-31 of jinsimul's Millard fit to Lumbar_C_210's Thelen curves
// (data/muscle_fit/millard_thelen_lumbar210.json, block), with minimum_value (0.0074),
// concentric_slope_at_vmax (0.234) and eccentric_slope_at_vmax (0.126) replaced by the 0 that
// OpenSim's damped model forces and slots 12, 25 and 28 therefore require. `afl` is slots 8-11.
std::string ThelenFitShape(const char* afl) {
  return std::string(afl) +
         " 0 0 "
         "0.04179811946585685 0.5964168220217628 0.2734354303931352 7.0483566037528895 "
         "0.6485551560379731 0 "
         "0.0329927724761894 51.63096652474981 0.2084607998115487 0.25021220733287797 "
         "1.7430548947734776 0 0.275657413457181 5.265654631839498 "
         "0 0.1260428439362502 0.5546747462355381 0.8464541431020638";
}
constexpr char kThelenFitAfl[] =
    "3.531853309691377e-10 1.4309121738647031e-09 2.1913352613858628 0.9209090308901695";


// The fit collapses the ascending limb: transition - min = 1.1e-9. OpenSim's
// createFiberActiveForceLengthCurve requires every knot gap to exceed sqrt(Eps) = 1.5e-8, and
// OpenSim 4.6 refuses this shape when the ActiveForceLengthCurve is constructed. So must this.
TEST_F(MuscleMtuTest, ThelenFitWithCollapsedAscendingLimbIsRejectedAsByOpenSim) {
  char error[1024] = "";
  mjModel* model = LoadModelFromString(
      HangingMuscle("millard_mtu", ("97 0.1841 0.0647 10 0 0 0 0 " +
                                    ThelenFitShape(kThelenFitAfl)).c_str()),
      error, sizeof(error));
  EXPECT_THAT(model, ::testing::IsNull());
  EXPECT_THAT(std::string(error), ::testing::HasSubstr("each gap larger than sqrt(eps)"));
  if (model) mj_deleteModel(model);
}


// OpenSim's other two active-curve conditions, which an earlier port did not enforce: x0 >= 0,
// and a shallow ascending slope that stays below the one reaching the plateau before x2 = 1.
TEST_F(MuscleMtuTest, ActiveCurveEnforcesOpenSimsRangeChecks) {
  struct Case { const char* name; const char* afl; const char* why; };
  const Case cases[] = {
      {"negative min", "-0.01 0.73 1.8123 0.8616", "0 <= x0"},
      {"gap below sqrt(eps)", "0.4441 0.44410001 1.8123 0.8616", "each gap larger than sqrt(eps)"},
      {"slope reaching the plateau early", "0.4441 0.73 1.8123 3.8", "dydx must be in"}};
  for (const Case& c : cases) {
    char error[1024] = "";
    std::string prm = std::string("3000 0.15 0.15 10 0 0.1 0 0 ") + c.afl;
    mjModel* model = LoadModelFromString(HangingMuscle("millard_mtu", prm.c_str()),
                                         error, sizeof(error));
    EXPECT_THAT(model, ::testing::IsNull()) << c.name;
    EXPECT_THAT(std::string(error), ::testing::HasSubstr(c.why)) << c.name;
    if (model) mj_deleteModel(model);
  }
}


// min_norm_active_fiber_length follows the same zero-means-default rule as every other slot, so
// a 0 in slot 8 is OpenSim's 0.4441, not zero. A fit that drives it to zero has to write a small
// positive number, and that number -- not 0.4441 -- must then set the fiber's lower clamp.
TEST_F(MuscleMtuTest, ZeroMinActiveFiberLengthMeansTheDefaultNotZero) {
  // the Thelen fit's max and slope; min and transition as below
  const std::string mech = "97 0.1841 0.0647 10 0 0 0 0 ";
  const double l_opt = 0.1841, l_slack = 0.0647;
  struct Case { const char* afl; double lce_min; };
  const Case cases[] = {
      // 0 and 0: OpenSim's 0.4441 and 0.73
      {"0 0 2.1913352613858628 0.9209090308901695", 0.4441*l_opt},
      // the fitted min, with the transition moved just far enough for the shape to be admissible
      {"3.531853309691377e-10 1e-7 2.1913352613858628 0.9209090308901695",
       3.531853309691377e-10*l_opt}};

  for (const Case& c : cases) {
    mjModel* model = LoadModelFromString(
        HangingMuscle("millard_mtu", (mech + ThelenFitShape(c.afl)).c_str()));
    ASSERT_THAT(model, ::testing::NotNull()) << c.afl;
    mjData* data = mj_makeData(model);

    // a path far shorter than the tendon: the fiber is pushed onto its clamp
    data->qpos[0] = 0.3 - 0.5*l_slack;
    data->act[1] = 1.0;
    data->ctrl[0] = 1.0;
    mj_forward(model, data);
    mju_mtuMuscleEquilibrate(model, data);
    EXPECT_NEAR(data->act[0], c.lce_min, 1e-15) << c.afl;

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}


// With min_norm_active_fiber_length near zero the unpennated fiber may shorten almost to nothing
// (lce_min = 6.5e-11 m here), while a pennated one is still held at h/sin(phi_max). Both must
// equilibrate and step without a NaN, from a path so short that the fiber sits on its clamp to
// one long enough to stretch the parallel element.
//
// The shape is the Thelen fit with the transition moved to the nearest round value OpenSim
// admits. Note that the baked table does NOT resolve an ascending limb this narrow -- its first
// cell is 4.3e-3 wide -- so near l0 = 0 it departs from the exact curve by up to 0.09 in
// normalized force (doc/muscle_mtu.rst). This test is about robustness, not accuracy.
TEST_F(MuscleMtuTest, NearZeroMinActiveFiberLengthEquilibratesPennatedAndNot) {
  const char* afl =
      "3.531853309691377e-10 1e-7 2.1913352613858628 0.9209090308901695";
  // Ps_L1_VB_r of Lumbar_C_210 (97 N, l_opt 0.1841, l_slack 0.0647, unpennated). Lumbar_C_210
  // has no pennated muscle, so the pennated case gives the same muscle 0.2 rad; the rigid case
  // shortens its tendon below 0.05 l_opt.
  struct Case { const char* name; const char* mech; double l_slack, pennation; };
  const Case cases[] = {
      {"unpennated", "97 0.1841 0.0647 10 0 0 0 0 ", 0.0647, 0.0},
      {"pennated",   "97 0.1841 0.0647 10 0.2 0 0 0 ", 0.0647, 0.2},
      {"rigid",      "97 0.1841 0.005 10 0 0 0 0 ", 0.005, 0.0}};
  const double l_opt = 0.1841;

  for (const Case& c : cases) {
    std::string prm = c.mech + ThelenFitShape(afl);
    mjModel* model = LoadModelFromString(HangingMuscle("millard_mtu", prm.c_str(), 1.0));
    ASSERT_THAT(model, ::testing::NotNull()) << c.name;
    mjData* data = mj_makeData(model);

    const double h = l_opt*std::sin(c.pennation);
    const double lce_min = std::fmax(3.531853309691377e-10*l_opt, h/0.9949874371066201);
    const bool rigid = c.l_slack < 0.05*l_opt;
    int on_clamp = 0, solved = 0;

    for (int i = 0; i <= 60; i++) {
      double l_mtu = 0.5*c.l_slack + (c.l_slack + 1.9*l_opt)*i/60.0;
      for (double act : {0.0, 0.5, 1.0}) {
        mj_resetData(model, data);
        data->qpos[0] = 0.3 - l_mtu;
        data->act[1] = act;
        data->ctrl[0] = act;
        mj_forward(model, data);
        mju_mtuMuscleEquilibrate(model, data);
        mj_forward(model, data);

        ASSERT_TRUE(std::isfinite(data->act[0])) << c.name << " l_mtu=" << l_mtu;
        ASSERT_TRUE(std::isfinite(data->muscle_F_mtu[0])) << c.name << " l_mtu=" << l_mtu;
        ASSERT_TRUE(std::isfinite(data->act_dot[0])) << c.name << " l_mtu=" << l_mtu;
        EXPECT_GE(data->act[0], lce_min*(1 - 1e-12)) << c.name << " l_mtu=" << l_mtu;
        EXPECT_GE(data->muscle_F_mtu[0], 0) << c.name << " l_mtu=" << l_mtu;
        if (data->act[0] <= lce_min*(1 + 1e-9)) {
          on_clamp++;
        } else if (!rigid) {
          // off the clamp the fiber is at a root of the isometric residual
          EXPECT_LT(std::fabs(Residual(model, data, 0, false, act)), 1e-6)
              << c.name << " l_mtu=" << l_mtu << " a=" << act;
          solved++;
        }
      }
    }
    // the compliant sweeps must have visited both regimes. (The rigid path's fiber is
    // sqrt((l_MTU - l_slack)^2 + h^2), OpenSim's rule, which a 6.5e-11 m clamp does not reach.)
    if (!rigid) {
      EXPECT_GT(on_clamp, 0) << c.name;
      EXPECT_GT(solved, 0) << c.name;
    }

    // and the stepping path, driven, from the fiber on its clamp: the 1 kg mass falls until the
    // muscle catches it
    mj_resetData(model, data);
    data->qpos[0] = 0.3 - 0.5*c.l_slack;
    mj_forward(model, data);
    mju_mtuMuscleEquilibrate(model, data);
    for (int t = 0; t < 4000; t++) {
      data->ctrl[0] = 0.5 + 0.5*std::sin(0.005*t);
      mj_step(model, data);
      ASSERT_TRUE(std::isfinite(data->act[0])) << c.name << " step " << t;
      ASSERT_TRUE(std::isfinite(data->qpos[0])) << c.name << " step " << t;
      EXPECT_GE(data->act[0], lce_min*(1 - 1e-12)) << c.name << " step " << t;
    }

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}

// A taut rigid tendon is not a buckled one. On the rigid path l_T is l_slack only up to the
// roundoff of subtracting l_ce cos(phi) back out of l_MTU, and OpenSim allows SignificantReal for
// it; without that margin the roundoff decided, and a pennated rigid tendon lost its fiber
// velocity -- f_V and the damping with it -- at 383 of these 1000 moving points.
TEST_F(MuscleMtuTest, TautRigidTendonKeepsItsVelocityDependence) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    mjModel* m = LoadModelFromString(HangingMuscle(gain, "3000 0.15 0.005 10 0.15 0.1"));
    ASSERT_THAT(m, ::testing::NotNull()) << gain;
    mjData* d = mj_makeData(m);
    int flat = 0;
    for (int i = 0; i < 1000; i++) {
      double qpos = -0.05 + 0.2*i/1000.0;       // fiber from 1 to 2.3 l_opt, tendon taut
      double F[2];
      for (int k = 0; k < 2; k++) {
        mj_resetData(m, d);
        d->qpos[0] = qpos;
        d->qvel[0] = k ? 0.1 : 0;
        d->act[1] = 0.6;
        mj_forward(m, d);
        F[k] = d->muscle_F_mtu[0];
      }
      if (F[0] == F[1]) flat++;
    }
    EXPECT_EQ(flat, 0) << gain;
    mj_deleteData(d);
    mj_deleteModel(m);
  }
}


// ------------------------------------------------------------------------------------------
// ignore_tendon_compliance (gainprm[19]).

// gainprm with slot 19 set; `mech` is slots 0-7.
std::string WithRigidFlag(const char* mech, const char* flag) {
  std::string prm = mech;
  for (int k = 8; k < 19; k++) prm += " 0";
  return prm + " " + flag;
}


// Where the ratio rule already makes the tendon rigid, declaring it rigid must change nothing, bit
// for bit: the flag selects the same path, it does not add one.
TEST_F(MuscleMtuTest, DeclaredRigidTendonIsTheRatioRulesPath) {
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    const char* mech = "3000 0.15 0.005 10 0.15 0.1 0 0";   // l_slack < 0.05 l_opt
    mjModel* m0 = LoadModelFromString(HangingMuscle(gain, WithRigidFlag(mech, "0").c_str()));
    mjModel* m1 = LoadModelFromString(HangingMuscle(gain, WithRigidFlag(mech, "1").c_str()));
    ASSERT_THAT(m0, ::testing::NotNull()) << gain;
    ASSERT_THAT(m1, ::testing::NotNull()) << gain;
    mjData* d0 = mj_makeData(m0);
    mjData* d1 = mj_makeData(m1);
    for (double qpos : {0.0, 0.05, 0.1}) {
      for (double qvel : {-0.2, 0.0, 0.3}) {
        for (auto [m, d] : {std::pair{m0, d0}, std::pair{m1, d1}}) {
          mj_resetData(m, d);
          d->qpos[0] = qpos;
          d->qvel[0] = qvel;
          d->act[1] = 0.6;
          mj_forward(m, d);
        }
        EXPECT_EQ(d0->muscle_F_mtu[0], d1->muscle_F_mtu[0]) << gain;
        EXPECT_EQ(d0->muscle_l_ce[0], d1->muscle_l_ce[0]) << gain;
        EXPECT_EQ(d0->act_dot[0], d1->act_dot[0]) << gain;
        EXPECT_EQ(mju_mtuMuscleForceVel(m0, d0, 0), mju_mtuMuscleForceVel(m1, d1, 0)) << gain;
      }
    }
    mj_deleteData(d1);
    mj_deleteData(d0);
    mj_deleteModel(m1);
    mj_deleteModel(m0);
  }
}


// Declared rigid, the tendon is inextensible at any slack length: at l_slack = 0.7 l_opt, where the
// ratio rule would solve a compliant tendon, the fiber is OpenSim's calcFiberLength,
// hypot(l_MTU - l_slack, h), and the force its rigid-tendon fiber force.
TEST_F(MuscleMtuTest, DeclaredRigidTendonAtAnySlackLength) {
  struct Case { const char* gain; double pennation; };
  const Case cases[] = {{"millard_mtu", 0}, {"millard_mtu", 0.3}, {"hyfydy_mtu", 0},
                        {"hyfydy_mtu", 0.3}};
  const double l_opt = 0.15, l_slack = 0.105, beta = 0.1, v_max = 10, A = 0.6;

  for (const Case& c : cases) {
    std::string mech = "3000 0.15 0.105 10 " + std::to_string(c.pennation) + " 0.1 0 0";
    mjModel* rigid = LoadModelFromString(
        HangingMuscle(c.gain, WithRigidFlag(mech.c_str(), "1").c_str()));
    mjModel* compliant = LoadModelFromString(
        HangingMuscle(c.gain, WithRigidFlag(mech.c_str(), "0").c_str()));
    ASSERT_THAT(rigid, ::testing::NotNull()) << c.gain;
    ASSERT_THAT(compliant, ::testing::NotNull()) << c.gain;
    mjData* d = mj_makeData(rigid);
    mjData* dc = mj_makeData(compliant);
    const double h = l_opt*std::sin(c.pennation);

    auto curve = [&](int k, double x, double* deriv) {
      return c.gain[0] == 'h' ? mju_hyfydyCurve(k, x, deriv)
                              : mju_millardCurve(k, x, nullptr, deriv);
    };

    for (double l_mtu : {l_slack + 0.9*l_opt, l_slack + 1.1*l_opt, l_slack + 1.3*l_opt}) {
      for (double qvel : {0.0, 0.2}) {                    // qvel > 0 shortens the path
        for (auto [m, dd] : {std::pair{rigid, d}, std::pair{compliant, dc}}) {
          mj_resetData(m, dd);
          dd->qpos[0] = 0.3 - l_mtu;
          dd->qvel[0] = qvel;
          dd->act[1] = A;
          mj_forward(m, dd);
        }
        double l_ce = std::hypot(l_mtu - l_slack, h);
        double cos_phi = std::sqrt(1 - (h/l_ce)*(h/l_ce));
        EXPECT_NEAR(d->muscle_l_ce[0], l_ce, 1e-12) << c.gain << " l_mtu=" << l_mtu;
        EXPECT_NEAR(d->muscle_l_se[0], l_mtu - l_ce*cos_phi, 1e-12) << c.gain;

        double v0 = d->actuator_velocity[0]*cos_phi/(v_max*l_opt);
        double l0 = l_ce/l_opt;
        double fiber = A*curve(mjMUSCLECURVE_ACTIVE_FL, l0, nullptr)*
                       curve(mjMUSCLECURVE_FV, v0, nullptr) +
                       curve(mjMUSCLECURVE_PASSIVE_FL, l0, nullptr) + beta*v0;
        EXPECT_NEAR(d->muscle_F_mtu[0], 3000*std::fmax(fiber, 0)*cos_phi, 1e-9*3000)
            << c.gain << " l_mtu=" << l_mtu << " qvel=" << qvel;

        // and the flag is what did it: the same muscle without it solves a compliant tendon
        EXPECT_GT(std::fabs(dc->muscle_l_ce[0] - l_ce), 1e-4) << c.gain << " l_mtu=" << l_mtu;
      }
    }

    // the rigid path's velocity derivative, against a central difference
    auto force_at = [&](double qvel) {
      mj_resetData(rigid, d);
      d->qpos[0] = 0.3 - (l_slack + 1.1*l_opt);
      d->qvel[0] = qvel;
      d->act[1] = A;
      mj_forward(rigid, d);
      return d->actuator_force[0];
    };
    const double dq = 1e-6;
    double f_plus = force_at(0.1 + dq), v_plus = d->actuator_velocity[0];
    double f_minus = force_at(0.1 - dq), v_minus = d->actuator_velocity[0];
    force_at(0.1);
    double analytic = mju_mtuMuscleForceVel(rigid, d, 0);
    EXPECT_NEAR(analytic, (f_plus - f_minus)/(v_plus - v_minus), 1e-5*std::fabs(analytic))
        << c.gain;

    mj_deleteData(dc);
    mj_deleteData(d);
    mj_deleteModel(compliant);
    mj_deleteModel(rigid);
  }
}


// OpenSim's ignore_tendon_compliance is a boolean. hyfydy_mtu, which refuses every curve shape
// slot, takes this one.
TEST_F(MuscleMtuTest, IgnoreTendonComplianceIsABoolean) {
  const char* mech = "3000 0.15 0.105 10 0 0.1 0 0";
  for (const char* gain : {"millard_mtu", "hyfydy_mtu"}) {
    for (const char* bad : {"0.5", "2", "-1", "inf"}) {
      char error[1024] = "";
      mjModel* m = LoadModelFromString(HangingMuscle(gain, WithRigidFlag(mech, bad).c_str()),
                                       error, sizeof(error));
      EXPECT_THAT(m, ::testing::IsNull()) << gain << " " << bad;
      EXPECT_THAT(std::string(error), ::testing::HasSubstr("must be 0 or 1")) << gain << " " << bad;
      if (m) mj_deleteModel(m);
    }

    // NaN cannot come through MJCF quietly; it can through an edited model
    mjModel* m = LoadModelFromString(HangingMuscle(gain, WithRigidFlag(mech, "1").c_str()));
    ASSERT_THAT(m, ::testing::NotNull()) << gain;
    mjData* d = mj_makeData(m);
    m->actuator_gainprm[mjMTU_IGNORE_TENDON_COMPLIANCE] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THAT(MjuErrorMessageFrom(mj_resetData)(m, d), ::testing::HasSubstr("must be 0 or 1"))
        << gain;
    mj_deleteData(d);
    mj_deleteModel(m);
  }
}


// OpenSim runs its damped-model branch, which zeroes both at-vmax force-velocity slopes, unless
// the tendon is compliant AND the fiber undamped. So a declared rigid tendon refuses them even
// without damping.
TEST_F(MuscleMtuTest, DeclaredRigidTendonRefusesSlopesAtVmaxEvenUndamped) {
  // undamped, slot 19 as given, slot 25 = 0.2
  auto prm = [](const char* flag) {
    std::string p = "3000 0.15 0.105 10 0 -1 0 0";
    for (int k = 8; k < 19; k++) p += " 0";
    p += std::string(" ") + flag + " 0 0 0 0 0 0.2";
    return p;
  };
  char error[1024] = "";
  mjModel* m = LoadModelFromString(HangingMuscle("millard_mtu", prm("1").c_str()), error,
                                   sizeof(error));
  EXPECT_THAT(m, ::testing::IsNull());
  EXPECT_THAT(std::string(error), ::testing::HasSubstr("with fiber damping or a rigid tendon"));
  if (m) mj_deleteModel(m);

  m = LoadModelFromString(HangingMuscle("millard_mtu", prm("0").c_str()));
  EXPECT_THAT(m, ::testing::NotNull()) << "compliant and undamped keeps the slopes";
  if (m) mj_deleteModel(m);
}


// A stiff, short tendon -- strain_at_one_norm_force 3.4e-4 and stiffness 3650, as ARMS's
// lumbricals come out when their tendon is kept at the source's stiffness -- builds in OpenSim for
// every toe force and curviness. The bake used to refuse a third of them on a slope cross-check
// that was absolute, and so tripped on roundoff. It must build them all, and reproduce them.
TEST_F(MuscleMtuTest, StiffShortTendonCurvesBuild) {
  double worst = 0;
  for (int i = 0; i < 19; i++) {
    for (int j = 0; j < 20; j++) {
      double f_toe = 0.05 + 0.05*i, curviness = 0.05 + 0.05*j;
      millard::SegFn exact = millard::TendonForceLength(3.4e-4, 3650, f_toe, curviness);
      std::array<mjtNum, mjNGAIN> prm = {};
      prm[mjMTU_TFL_E1] = 3.4e-4;
      prm[mjMTU_TFL_KISO] = 3650;
      prm[mjMTU_TFL_FTOE] = f_toe;
      prm[mjMTU_TFL_CURV] = curviness;
      for (int k = 0; k <= 400; k++) {
        double x = exact.x0 + (exact.x1 - exact.x0)*((k + 0.5)/401.0);
        double got = mju_millardCurve(mjMUSCLECURVE_TENDON_FL, x, prm.data(), nullptr);
        worst = std::fmax(worst, std::fabs(got - exact.Value(x)));
      }
    }
  }
  EXPECT_LT(worst, 1e-7);
}

}  // namespace
}  // namespace mujoco
