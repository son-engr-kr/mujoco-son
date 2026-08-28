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
      <general name="a" tendon="t" gaintype="millard_mtu" biastype="none"
               gainprm="1000 0.1 0.1 10 0 0.1"/>
      <general name="b" tendon="t" gaintype="millard_mtu" biastype="none"
               gainprm="2000 0.2 0.2 10 0.2 0.1"/>
      <general name="c" tendon="t" gaintype="millard_mtu" biastype="none"
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
      double act_in = data->act[0];
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
    d->act[0] = 0.5;
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
    EXPECT_EQ(data->muscle_l_se[0], 0.005) << gain;
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

}  // namespace
}  // namespace mujoco
