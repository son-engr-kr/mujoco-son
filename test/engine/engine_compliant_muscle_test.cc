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

// Tests for the gaintype="compliant_mtu" (Geyer/Song) actuator in engine/engine_util_misc.c --
// here, its parallel elastic element and its gainprm validation.

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;
using CompliantMuscleTest = MujocoTest;

constexpr double kAnchor = 0.5;   // anchor height: the path length is kAnchor - qpos

// A point mass hung from a single compliant_mtu muscle with gainprm `prm`.
std::string HangingMuscle(const std::string& prm) {
  return R"(
  <mujoco>
    <option timestep="0.001" gravity="0 0 -9.81"/>
    <worldbody>
      <site name="anchor" pos="0 0 )" + std::to_string(kAnchor) + R"(" size="0.005"/>
      <body name="m">
        <joint name="s" type="slide" axis="0 0 1"/>
        <site name="ins" size="0.005"/>
        <geom type="sphere" size="0.01" mass="20"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="t"><site site="anchor"/><site site="ins"/></spatial>
    </tendon>
    <actuator>
      <general name="a" tendon="t" gaintype="compliant_mtu" biastype="none"
               dyntype="muscle" dynprm="0.01 0.04" ctrllimited="true" ctrlrange="0 1"
               gainprm=")" + prm + R"("/>
    </actuator>
  </mujoco>)";
}

// abd_r of jinsimul's myoleg26 Geyer-equivalent fit, with its own parallel element
// (data/muscle_fit/geyer_equivalent_myoleg26_extended_pe.csv):
// F_max l_opt l_slack v_max W C N K E_REF, then L_PE0 E_REF_PE.
constexpr double kFmax = 4460.290481, kLopt = 0.0845, kLslack = 0.053;
constexpr double kW = 0.9219990822, kEref = 0.0492005381;
constexpr double kLpe0 = 1.237603992, kErefPe = 0.3815447921;
const char kSong[] = "4460.290481 0.0845 0.053 15 0.9219990822 -2.995732274 1.5 4 0.0492005381";
const char kOwnPe[] = " 1.237603992 0.3815447921";

// The same muscle with a tendon short enough (l_slack < 0.05 l_opt) for the rigid-tendon path.
constexpr double kLslackRigid = 0.004;
const char kSongRigid[] =
    "4460.290481 0.0845 0.004 15 0.9219990822 -2.995732274 1.5 4 0.0492005381";


// Put the path at `l_mtu`, the activation at `act`, and solve the isometric equilibrium.
void Equilibrate(const mjModel* m, mjData* d, double l_mtu, double act) {
  mj_resetData(m, d);
  d->qpos[0] = kAnchor - l_mtu;
  d->act[1] = act;                      // act = [l_ce, activation]
  d->ctrl[0] = act;
  mj_forward(m, d);
  mju_compliantMuscleEquilibrate(m, d);
}


// ------------------------------------------------------------------------------------------
// The parallel element.

// With Geyer & Herr 2010's slack length and reference strain the new curve is the old one --
// exactly, not approximately, since a model that declares neither must not move by an ulp.
TEST_F(CompliantMuscleTest, Fpe0WithGeyersElementIsFp0) {
  for (double w : {0.05, 0.4, 0.56, 0.9219990822, 1.3}) {
    for (int i = 0; i <= 400; i++) {
      double l0 = 0.5 + 0.005*i;
      EXPECT_EQ(mju_compliantMuscleFpe0(l0, 1.0, w), mju_compliantMuscleFp0(l0, w))
          << "l0=" << l0 << " w=" << w;
    }
  }
  // and its own element is the same quadratic, moved and rescaled
  EXPECT_EQ(mju_compliantMuscleFpe0(1.2, 1.25, 0.4), 0.0);
  EXPECT_EQ(mju_compliantMuscleFpe0(1.25, 1.25, 0.4), 0.0);
  EXPECT_DOUBLE_EQ(mju_compliantMuscleFpe0(1.45, 1.25, 0.4), 0.25);
}


// A muscle declaring (L_PE0, E_REF_PE) = (1, W) must run bit for bit like one declaring
// nothing, on all three solver branches: that is the statement that zero in both slots means
// Geyer's element and nothing else. (The same outputs were also checked bit for bit against
// the build that preceded slots 9 and 10; see CHANGELOG-son.md.)
TEST_F(CompliantMuscleTest, UndeclaredParallelElementIsGeyers) {
  const std::string w = " 1 0.9219990822";   // (1, W), W spelled exactly as in slot 4
  struct Case { const char* name; std::string prm; double l_slack; };
  const Case cases[] = {{"compliant", kSong, kLslack}, {"rigid", kSongRigid, kLslackRigid}};

  for (const Case& c : cases) {
    mjModel* m0 = LoadModelFromString(HangingMuscle(c.prm));
    mjModel* m1 = LoadModelFromString(HangingMuscle(c.prm + w));
    ASSERT_THAT(m0, NotNull()) << c.name;
    ASSERT_THAT(m1, NotNull()) << c.name;
    ASSERT_EQ(m1->actuator_gainprm[9], 1.0);
    ASSERT_EQ(m1->actuator_gainprm[10], m1->actuator_gainprm[4]);
    mjData* d0 = mj_makeData(m0);
    mjData* d1 = mj_makeData(m1);

    int compared = 0;
    for (int il = 0; il <= 20; il++) {
      double l_mtu = c.l_slack + kLopt*(0.3 + 0.08*il);   // fiber 0.3 .. 1.9 l_opt
      for (double act : {0.0, 0.3, 1.0}) {
        // isometric
        Equilibrate(m0, d0, l_mtu, act);
        Equilibrate(m1, d1, l_mtu, act);
        EXPECT_EQ(d0->act[0], d1->act[0]) << c.name << " l_mtu=" << l_mtu << " a=" << act;
        EXPECT_EQ(d0->muscle_F_mtu[0], d1->muscle_F_mtu[0]) << c.name << " l_mtu=" << l_mtu;
        double l_eq = d0->act[0];

        // stepping: backward-Euler Newton (rigid path: algebraic) from off-equilibrium fibers
        for (double scale : {0.9, 1.1}) {
          for (double qvel : {-0.3, 0.0, 0.3}) {
            for (auto [m, d] : {std::pair{m0, d0}, std::pair{m1, d1}}) {
              mj_resetData(m, d);
              d->qpos[0] = kAnchor - l_mtu;
              d->qvel[0] = qvel;
              d->act[0] = scale*l_eq;
              d->act[1] = act;
              d->ctrl[0] = act;
              mj_forward(m, d);
            }
            EXPECT_EQ(d0->act_dot[0], d1->act_dot[0]) << c.name << " l_mtu=" << l_mtu;
            EXPECT_EQ(d0->muscle_l_ce[0], d1->muscle_l_ce[0]) << c.name << " l_mtu=" << l_mtu;
            EXPECT_EQ(d0->muscle_F_mtu[0], d1->muscle_F_mtu[0]) << c.name << " l_mtu=" << l_mtu;
            compared += 3;
          }
        }
        compared += 2;
      }
    }
    EXPECT_EQ(compared, 21*3*(2 + 2*3*3)) << c.name;

    mj_deleteData(d1);
    mj_deleteData(d0);
    mj_deleteModel(m1);
    mj_deleteModel(m0);
  }
}


// At zero activation the contractile element carries nothing, so the parallel element and the
// tendon are two quadratic springs in series carrying one force, and their lengths add up to the
// path. That has a closed form:
//
//   sqrt(F/F_max) = (l_mtu - l_slack - l_opt L_PE0) / (l_slack E_REF + l_opt E_REF_PE)
//
// for l_mtu > l_slack + l_opt L_PE0, and F = 0 below it. The equilibrium solve must land on it.
TEST_F(CompliantMuscleTest, OwnParallelElementMatchesSeriesClosedForm) {
  mjModel* m = LoadModelFromString(HangingMuscle(std::string(kSong) + kOwnPe));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  const double onset = kLslack + kLopt*kLpe0;
  const double compliance = kLslack*kEref + kLopt*kErefPe;

  // taut: up to 1.44 F_max
  for (int i = 1; i <= 40; i++) {
    double l_mtu = onset + 1.2*compliance*i/40.0;
    Equilibrate(m, d, l_mtu, 0.0);
    double root = (l_mtu - onset)/compliance;
    EXPECT_NEAR(d->muscle_F_mtu[0], kFmax*root*root, 2e-6*kFmax) << "l_mtu=" << l_mtu;

    // the stepping solve sees the same element: from this equilibrium the fiber stays put
    mj_forward(m, d);
    EXPECT_NEAR(d->act_dot[0], 0.0, 1e-9) << "l_mtu=" << l_mtu;
  }

  // slack, including the band past l_opt where Geyer's element would already be pulling. Zero to
  // the isometric solve's tolerance, which stops on |residual| < 1e-6 and so may leave the tendon
  // a hair taut.
  for (int i = 0; i <= 20; i++) {
    double l_mtu = kLslack + kLopt*(0.5 + (kLpe0 - 0.5)*i/20.0) - 1e-9;
    Equilibrate(m, d, l_mtu, 0.0);
    EXPECT_NEAR(d->muscle_F_mtu[0], 0.0, 2e-6*kFmax) << "l_mtu=" << l_mtu;
  }

  // and it is not Geyer's element: at the midpoint of that band his carries real force
  mjModel* g = LoadModelFromString(HangingMuscle(kSong));
  ASSERT_THAT(g, NotNull());
  mjData* dg = mj_makeData(g);
  double l_mid = kLslack + kLopt*(1 + kLpe0)/2;
  Equilibrate(g, dg, l_mid, 0.0);
  double root_g = (l_mid - kLslack - kLopt)/(kLslack*kEref + kLopt*kW);
  EXPECT_NEAR(dg->muscle_F_mtu[0], kFmax*root_g*root_g, 2e-6*kFmax);
  EXPECT_GT(dg->muscle_F_mtu[0], 0.01*kFmax);

  mj_deleteData(dg);
  mj_deleteModel(g);
  mj_deleteData(d);
  mj_deleteModel(m);
}


// The rigid-tendon path evaluates the parallel element directly on l_ce = l_mtu - l_slack.
TEST_F(CompliantMuscleTest, OwnParallelElementOnTheRigidTendonPath) {
  mjModel* m = LoadModelFromString(HangingMuscle(std::string(kSongRigid) + kOwnPe));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  for (int i = 0; i <= 30; i++) {
    double l_mtu = kLslackRigid + kLopt*(1.0 + 0.02*i);
    double l0 = (l_mtu - kLslackRigid)/kLopt;
    double x = l0 > kLpe0 ? (l0 - kLpe0)/kErefPe : 0;

    Equilibrate(m, d, l_mtu, 0.0);
    EXPECT_NEAR(d->muscle_F_mtu[0], kFmax*x*x, 1e-12*kFmax) << "l_mtu=" << l_mtu;

    mj_forward(m, d);                   // the stepping path, at rest
    EXPECT_NEAR(d->muscle_F_mtu[0], kFmax*x*x, 1e-12*kFmax) << "l_mtu=" << l_mtu;
  }

  mj_deleteData(d);
  mj_deleteModel(m);
}


// ------------------------------------------------------------------------------------------
// gainprm validation.

// Slots 9 and 10 come as a pair, positive and finite; slots 11-31 are not parameters at all.
// A build that ignored either would run a different muscle from the one declared, silently.
TEST_F(CompliantMuscleTest, RejectsMalformedParallelElementAndUnusedSlots) {
  // slots 9 .. 31 with only `slot` set
  auto only = [](int slot, const char* value) {
    std::string tail;
    for (int k = 9; k <= slot; k++) tail += k == slot ? std::string(" ") + value : " 0";
    return tail;
  };
  struct Case { const char* name; std::string tail; const char* why; };
  const Case cases[] = {
      {"L_PE0 alone",       " 1.2376",          "must both be 0"},
      {"E_REF_PE alone",    " 0 0.3815",        "must both be 0"},
      {"negative L_PE0",    " -1.2376 0.3815",  "must both be 0"},
      {"negative E_REF_PE", " 1.2376 -0.3815",  "must both be 0"},
      {"infinite L_PE0",    " inf 0.3815",      "must both be 0"},
      {"infinite E_REF_PE", " 1.2376 inf",      "must both be 0"},
      {"slot 11",           only(11, "0.5"),    "gainprm[11] is not a compliant_mtu parameter"},
      {"slot 11, own PE",   std::string(kOwnPe) + " 0.5",
                                                "gainprm[11] is not a compliant_mtu parameter"},
      {"slot 31",           only(31, "1"),      "gainprm[31] is not a compliant_mtu parameter"},
  };

  for (const Case& c : cases) {
    char error[1024] = "";
    mjModel* m = LoadModelFromString(HangingMuscle(kSong + c.tail), error, sizeof(error));
    EXPECT_THAT(m, IsNull()) << c.name;
    EXPECT_THAT(std::string(error), HasSubstr(c.why)) << c.name;
    if (m) mj_deleteModel(m);
  }
}


// XML refuses to carry a NaN quietly, so check the other way in: a model edited after loading.
// The validation lives in mj_resetData, not only in the compiler, so it catches this too.
TEST_F(CompliantMuscleTest, ResetRejectsNanInTheParallelElement) {
  mjModel* m = LoadModelFromString(HangingMuscle(std::string(kSong) + kOwnPe));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  for (int slot : {9, 10}) {
    double saved = m->actuator_gainprm[slot];
    m->actuator_gainprm[slot] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THAT(MjuErrorMessageFrom(mj_resetData)(m, d), HasSubstr("must both be 0"))
        << "slot " << slot;
    m->actuator_gainprm[slot] = saved;
  }
  EXPECT_EQ(MjuErrorMessageFrom(mj_resetData)(m, d), "");

  mj_deleteData(d);
  mj_deleteModel(m);
}

}  // namespace
}  // namespace mujoco
