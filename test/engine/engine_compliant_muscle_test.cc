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
std::string HangingMuscle(const std::string& prm, double mass = 20.0) {
  return R"(
  <mujoco>
    <option timestep="0.001" gravity="0 0 -9.81"/>
    <worldbody>
      <site name="anchor" pos="0 0 )" + std::to_string(kAnchor) + R"(" size="0.005"/>
      <body name="m">
        <joint name="s" type="slide" axis="0 0 1"/>
        <site name="ins" size="0.005"/>
        <geom type="sphere" size="0.01" mass=")" + std::to_string(mass) + R"("/>
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
// The force balance: Geyer & Herr 2010, Appendix II.

// Geyer & Herr 2010's soleus (Table II), with the paper's shared parameters.
constexpr double kSolFmax = 4000, kSolLopt = 0.04, kSolLslack = 0.26, kSolVmax = 6;
constexpr double kGeyerW = 0.56, kGeyerC = -2.995732273553991, kGeyerN = 1.5, kGeyerK = 5;
constexpr double kGeyerEref = 0.04;
const char kSol[] = "4000 0.04 0.26 6 0.56 -2.995732273553991 1.5 5 0.04";

// The same muscle with a tendon short enough for the rigid-tendon path.
constexpr double kSolLslackRigid = 0.001;
const char kSolRigid[] = "4000 0.04 0.001 6 0.56 -2.995732273553991 1.5 5 0.04";

// The forward force-velocity curve, transcribed so the tests can rebuild the residual from outside.
// It is the analytic inverse of the public mju_compliantMuscleInvFvce0, region for region, which
// ForwardVelocityIsTheInverseOfInvFvce0 checks.
double Fv(double v0, double K, double N) {
  if (v0 <= 0) return (1 + v0)/(1 - K*v0);
  if (v0 <= 1) {
    double t = (v0 - 1)/(7.56*K*v0 + 1);
    return N - t/(t - 1);
  }
  return N + 100*(v0 - 1);
}

struct Geyer {
  double l_opt, l_slack, W, C, N, K, E_REF, L_PE0, E_REF_PE;
};
constexpr Geyer kSolGeyer = {kSolLopt, kSolLslack, kGeyerW, kGeyerC, kGeyerN, kGeyerK,
                             kGeyerEref, 1.0, kGeyerW};

// Geyer & Herr 2010's force balance, f_se + f_be - f_v (f_pe + A f_l), from the public curves.
double Residual2010(const Geyer& g, double A, double l_ce, double l_mtu, double v0) {
  double l0 = l_ce/g.l_opt;
  double f_se = mju_compliantMuscleFp0((l_mtu - l_ce)/g.l_slack, g.E_REF);
  double f_be = mju_compliantMuscleFp0Ext(l0, 0.5*g.W, 1 - g.W);
  double f_pe = mju_compliantMuscleFpe0(l0, g.L_PE0, g.E_REF_PE);
  double f_l = mju_compliantMuscleFlce0(l0, g.W, g.C);
  return f_se + f_be - Fv(v0, g.K, g.N)*(f_pe + A*f_l);
}

// The fiber length at which the buffer element balances a fiber with a slack tendon,
// A f_l = f_be below l_opt - w, by bisection.
double BufferRoot(const Geyer& g, double A) {
  double lo = 0, hi = 1 - g.W;              // in l_opt
  for (int i = 0; i < 200; i++) {
    double mid = 0.5*(lo + hi);
    double r = mju_compliantMuscleFp0Ext(mid, 0.5*g.W, 1 - g.W) -
               A*mju_compliantMuscleFlce0(mid, g.W, g.C);
    (r > 0 ? lo : hi) = mid;
  }
  return 0.5*(lo + hi)*g.l_opt;
}


TEST_F(CompliantMuscleTest, ForwardVelocityIsTheInverseOfInvFvce0) {
  for (double v0 = -0.99; v0 <= 3.0; v0 += 0.01) {
    double f = Fv(v0, kGeyerK, kGeyerN);
    EXPECT_NEAR(mju_compliantMuscleInvFvce0(f, kGeyerK, kGeyerN), v0, 1e-12) << "v0=" << v0;
  }
}


// Fig. 6 of the paper: "a buffer elasticity (BE) prevents the active CE from collapsing if the SE
// is slack". With the path shorter than the tendon plus the buffered fiber, the tendon is slack and
// carries nothing, and the fiber sits where the buffer element balances its pull. An earlier
// version held the tendon at its slack length instead, which is a tendon pushing.
TEST_F(CompliantMuscleTest, BufferElementHoldsAnActiveFiberWhenTheTendonIsSlack) {
  mjModel* m = LoadModelFromString(HangingMuscle(kSol));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  for (double A : {0.3, 1.0}) {
    double l_ce_star = BufferRoot(kSolGeyer, A);
    ASSERT_LT(l_ce_star, (1 - kGeyerW)*kSolLopt);
    for (double l_mtu : {0.20, 0.25, kSolLslack + 0.9*l_ce_star}) {
      Equilibrate(m, d, l_mtu, A);
      EXPECT_EQ(d->muscle_F_mtu[0], 0.0) << "a slack tendon carries nothing, l_mtu=" << l_mtu;
      EXPECT_NEAR(d->act[0], l_ce_star, 1e-5*kSolLopt) << "A=" << A << " l_mtu=" << l_mtu;
      EXPECT_LT(std::fabs(Residual2010(kSolGeyer, A, d->act[0], l_mtu, 0)), 1e-6);
    }
  }

  mj_deleteData(d);
  mj_deleteModel(m);
}


// The backward-Euler step solves the paper's residual, in which f_v scales the parallel element
// too. Checked from outside at steps that engage the element while the fiber moves: the 2010
// residual is zero there, and the 2003 one -- f_se = f_pe + A f_l f_v, the structure this model
// used before -- is not.
TEST_F(CompliantMuscleTest, SteppingSolvesThe2010ResidualWithTheParallelElementScaledByFv) {
  mjModel* m = LoadModelFromString(HangingMuscle(kSol));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);
  const double dt = m->opt.timestep;

  int engaged = 0;
  for (double A : {0.0, 0.3, 1.0}) {
    for (double l0_eq : {1.15, 1.3}) {
      double l_mtu = kSolLslack*1.01 + l0_eq*kSolLopt;
      for (double scale : {0.97, 1.03}) {
        mj_resetData(m, d);
        d->qpos[0] = kAnchor - l_mtu;
        d->act[0] = scale*l0_eq*kSolLopt;
        d->act[1] = A;
        d->ctrl[0] = A;
        mj_forward(m, d);

        double l_ce = d->muscle_l_ce[0];
        double v0 = d->muscle_v_ce[0]/(kSolLopt*kSolVmax);
        ASSERT_NEAR(d->muscle_v_ce[0], (l_ce - d->act[0])/dt, 1e-9);
        double r2010 = Residual2010(kSolGeyer, A, l_ce, l_mtu, v0);
        EXPECT_LT(std::fabs(r2010), 1e-5) << "A=" << A << " l_mtu=" << l_mtu;

        double l0 = l_ce/kSolLopt;
        double f_pe = mju_compliantMuscleFpe0(l0, 1.0, kGeyerW);
        double f_l = mju_compliantMuscleFlce0(l0, kGeyerW, kGeyerC);
        double f_se = mju_compliantMuscleFp0((l_mtu - l_ce)/kSolLslack, kGeyerEref);
        double r2003 = f_se - (f_pe + A*f_l*Fv(v0, kGeyerK, kGeyerN));
        if (f_pe > 0.01 && std::fabs(v0) > 0.01) {
          engaged++;
          EXPECT_GT(std::fabs(r2003), 1e-3) << "A=" << A << " l_mtu=" << l_mtu;
        }
      }
    }
  }
  EXPECT_GT(engaged, 4);

  mj_deleteData(d);
  mj_deleteModel(m);
}


// Equilibration is a function of the pose and the activation, whatever the fiber was doing. At
// zero activation a band of lengths is in equilibrium -- tendon slack, parallel and buffer elements
// unloaded -- and the solve takes the shortest, the limit as the activation goes to zero.
TEST_F(CompliantMuscleTest, EquilibrateIsIndependentOfTheIncomingFiberLength) {
  mjModel* m = LoadModelFromString(HangingMuscle(kSol));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  auto equilibrate_from = [&](double l_mtu, double A, double start) {
    mj_resetData(m, d);
    d->qpos[0] = kAnchor - l_mtu;
    d->act[0] = start;
    d->act[1] = A;
    mj_forward(m, d);
    mju_compliantMuscleEquilibrate(m, d);
    return d->act[0];
  };

  for (double A : {0.0, 0.02, 0.5, 1.0}) {
    for (double l_mtu : {0.2, kSolLslack + 0.7*kSolLopt, kSolLslack*1.02 + 1.2*kSolLopt}) {
      double first = equilibrate_from(l_mtu, A, 0.5*kSolLopt);
      for (double start : {0.9*kSolLopt, 1.4*kSolLopt}) {
        EXPECT_EQ(equilibrate_from(l_mtu, A, start), first) << "A=" << A << " l_mtu=" << l_mtu;
      }
    }
  }

  // the zero-activation band is [0.7, 1] l_opt here; the tendon goes slack at its lower end
  double band = equilibrate_from(kSolLslack + 0.7*kSolLopt, 0.0, 0.9*kSolLopt);
  EXPECT_NEAR(band, 0.7*kSolLopt, 1e-4*kSolLslack);
  EXPECT_NEAR(d->muscle_F_mtu[0], 0.0, 1e-6*kSolFmax);

  mj_deleteData(d);
  mj_deleteModel(m);
}


// A rigid tendon cannot push: where the buffer element outweighs the fiber's pull the force is 0,
// and so is its velocity derivative. Elsewhere the force is F_max f_v (f_pe + A f_l) - including the
// parallel element under f_v -- and the derivative handed to the implicit integrators matches a
// central difference.
TEST_F(CompliantMuscleTest, RigidTendonForceIsFlooredAndItsVelocityDerivativeIsExact) {
  mjModel* m = LoadModelFromString(HangingMuscle(kSolRigid));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  auto force_at = [&](double l_mtu, double qvel, double A) {
    mj_resetData(m, d);
    d->qpos[0] = kAnchor - l_mtu;
    d->qvel[0] = qvel;
    d->act[1] = A;
    mj_forward(m, d);
    return d->actuator_force[0];
  };

  // buffered: l_ce = 0.3 l_opt, where f_be = 0.25 dwarfs the fiber's pull
  double short_mtu = kSolLslackRigid + 0.3*kSolLopt;
  EXPECT_EQ(force_at(short_mtu, 0.0, 1.0), 0.0);
  EXPECT_EQ(d->muscle_F_mtu[0], 0.0);
  EXPECT_EQ(mju_compliantMuscleForceVel(m, d, 0), 0.0);

  // pulling, parallel element engaged (l0 = 1.3), off the f_v kink at v0 = 0
  const double l0 = 1.3, A = 0.5;
  double long_mtu = kSolLslackRigid + l0*kSolLopt;
  double qvel = 0.2*kSolLopt*kSolVmax;   // shortens the path: v0 = -0.2
  force_at(long_mtu, qvel, A);
  double v0 = d->actuator_velocity[0]/(kSolLopt*kSolVmax);
  double f_pe = mju_compliantMuscleFpe0(l0, 1.0, kGeyerW);
  double f_l = mju_compliantMuscleFlce0(l0, kGeyerW, kGeyerC);
  EXPECT_NEAR(d->muscle_F_mtu[0], kSolFmax*Fv(v0, kGeyerK, kGeyerN)*(f_pe + A*f_l),
              1e-9*kSolFmax);

  const double h = 1e-6;
  double f_plus = force_at(long_mtu, qvel + h, A), v_plus = d->actuator_velocity[0];
  double f_minus = force_at(long_mtu, qvel - h, A), v_minus = d->actuator_velocity[0];
  force_at(long_mtu, qvel, A);
  double analytic = mju_compliantMuscleForceVel(m, d, 0);
  EXPECT_LT(analytic, 0);
  EXPECT_NEAR(analytic, (f_plus - f_minus)/(v_plus - v_minus), 1e-5*std::fabs(analytic));

  mj_deleteData(d);
  mj_deleteModel(m);
}


// The fiber's positivity guard sits far below any length the buffer element holds. With a W fitted
// to a Thelen-sourced curve the element's rest length is a few percent of l_opt, so on a short fiber
// it holds the active fiber under a millimetre -- where the absolute 1 mm clamp this replaced would
// have stopped it first. With W >= 1 the element is off, and only the guard is left.
TEST_F(CompliantMuscleTest, BufferElementIsNotPreemptedByTheFiberGuard) {
  // abd_r's curves (W 0.922) on a 44 mm fiber
  const Geyer g = {0.044, 0.25, kW, kGeyerC, 1.5, 4, kEref, kLpe0, kErefPe};
  mjModel* m = LoadModelFromString(HangingMuscle(
      "3000 0.044 0.25 15 0.9219990822 -2.995732273553991 1.5 4 0.0492005381" +
      std::string(kOwnPe)));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);

  double l_ce_star = BufferRoot(g, 1.0);
  EXPECT_LT(l_ce_star, 0.001) << "the buffered fiber is under 1 mm";
  Equilibrate(m, d, 0.2, 1.0);
  EXPECT_NEAR(d->act[0], l_ce_star, 1e-5*g.l_opt);
  EXPECT_GT(d->act[0], 1e-6*g.l_opt);
  mj_deleteData(d);
  mj_deleteModel(m);

  // W > 1: nothing holds the active fiber, which goes to the guard and stays finite
  m = LoadModelFromString(HangingMuscle("3000 0.044 0.25 15 1.002 -2.995732273553991 1.5 4 0.05"));
  ASSERT_THAT(m, NotNull());
  d = mj_makeData(m);
  Equilibrate(m, d, 0.2, 1.0);
  EXPECT_EQ(d->act[0], 1e-6*0.044);
  EXPECT_EQ(d->muscle_F_mtu[0], 0.0);
  d->ctrl[0] = 1.0;
  for (int t = 0; t < 2000; t++) {
    mj_step(m, d);
    ASSERT_TRUE(std::isfinite(d->act[0]) && std::isfinite(d->qpos[0])) << "step " << t;
    ASSERT_GE(d->act[0], 1e-6*0.044*(1 - 1e-12)) << "step " << t;
  }
  mj_deleteData(d);
  mj_deleteModel(m);
}


// End to end: a muscle holding a mass settles at its weight, and while driven the fiber stays on
// the root of the paper's residual, step after step -- to the stepping solve's 1e-5, except within
// 0.01 of v0 = 1. There f_v's slope jumps from 0.026 (region 2) to 100 (region 3), the
// finite-difference Jacobian straddles the jump, and the solve can stop just short: 1.4e-5 on this
// trajectory, on 39 of 4000 steps near the kink. Region 3 is under review (CHANGELOG-son.md, E3).
TEST_F(CompliantMuscleTest, CarriesTheLoadAndStaysOnTheResidualWhileDriven) {
  mjModel* m = LoadModelFromString(HangingMuscle(kSol, 50.0));
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);
  d->qpos[0] = kAnchor - (kSolLslack + kSolLopt);
  mj_forward(m, d);
  mju_compliantMuscleEquilibrate(m, d);

  d->ctrl[0] = 0.5;
  for (int t = 0; t < 8000; t++) mj_step(m, d);
  EXPECT_NEAR(d->muscle_F_mtu[0], 50.0*9.81, 1e-2);
  EXPECT_NEAR(d->qvel[0], 0.0, 1e-4);

  double worst = 0, worst_at_kink = 0;
  for (int t = 0; t < 4000; t++) {
    d->ctrl[0] = 0.5 + 0.5*std::sin(0.01*t);
    double A_in = d->act[1];
    mj_step(m, d);
    double v0 = d->muscle_v_ce[0]/(kSolLopt*kSolVmax);
    double r = std::fabs(Residual2010(kSolGeyer, A_in, d->muscle_l_ce[0],
                                      d->actuator_length[0], v0));
    double& w = std::fabs(v0 - 1) < 0.01 ? worst_at_kink : worst;
    w = std::fmax(w, r);
  }
  EXPECT_LT(worst, 1.01e-5);          // the tolerance, and the rounding of rebuilding it here
  EXPECT_LT(worst_at_kink, 1e-4);

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
