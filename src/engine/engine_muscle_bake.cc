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

// The process-wide cache of baked Millard2012 curves.
//
// WHY A PROCESS-WIDE CACHE, RATHER THAN mjModel OR mjData.
//   * not mjModel: a table written into a model file can go stale against the shape parameters
//     that produced it, and keeping the two in sync becomes a human-error surface rather than a
//     safeguard. The parameters are the source of truth; the table is derived.
//   * not mjData: the tables are immutable and identical across every mjData made from the same
//     parameters, so putting them there would duplicate them per parallel environment for no
//     benefit -- hundreds of KB times the environment count.
// Keyed by the shape parameters, so a model whose muscles share a curve shape (which is every
// OpenSim model except the ones that tune the passive curve per muscle) bakes it exactly once.
//
// WHEN IT RUNS. From mju_mtuMuscleInit, i.e. mj_resetData -- once at startup, never on the
// stepping path. What the stepping path sees is a pointer, already resolved into
// mjData.muscle_curve. Baking all four curves costs on the order of a hundred microseconds.
//
// THREAD SAFETY. Guarded by one mutex. Contention is irrelevant: the critical section is
// entered once per muscle per mj_resetData, not per step, and after the first reset every
// lookup is a short scan that finds an existing entry. Entries are never evicted and their
// storage never moves, so a mjCurveTable pointer handed out here stays valid for the life of
// the process -- which is what lets mjData hold raw pointers and mj_copyData copy them.

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include "engine/engine_muscle_millard_bezier.h"
#include "engine/engine_muscle_mtu.h"
#include "engine/engine_util_errmem.h"

namespace {

using mujoco::millard::Baked;

// Knot count per curve. At 513 the worst |table - exact| over the tabulated domain is 2.7e-9
// (active F-L), 7.7e-10 (passive), 5.9e-11 (tendon) and 6.7e-9 (force-velocity) -- below single
// precision epsilon relative to 1, so the table contributes nothing measurable to a comparison
// against OpenSim. It costs 12 KB per curve.
constexpr int kKnots = 513;

// Guard against a pathological caller (thousands of scaled subject models, each with its own
// per-muscle passive curve) growing the cache without bound. 4096 curves is ~50 MB.
constexpr int kMaxCurves = 4096;

// The shape parameters that define one curve, at most 8 of them (force-velocity is the widest).
constexpr int kMaxShape = 8;

struct Entry {
  int curve;
  mjtNum shape[kMaxShape];
  Baked baked;
};

std::mutex& cache_mutex() {
  static std::mutex mu;
  return mu;
}

// unique_ptr elements so an entry's storage -- and therefore the mjCurveTable pointer handed to
// mjData -- never moves when the vector grows
std::vector<std::unique_ptr<Entry>>& cache() {
  static std::vector<std::unique_ptr<Entry>> c;
  return c;
}

// How many shape parameters each curve takes, and where they sit in a resolved gainprm block.
struct CurveSpec {
  int nshape;
  int slot[kMaxShape];
};

const CurveSpec& spec_of(int curve) {
  static const CurveSpec specs[mjNMUSCLECURVE] = {
      {4, {mjMTU_AFL_MIN, mjMTU_AFL_TRANS, mjMTU_AFL_MAX, mjMTU_AFL_SLOPE}},
      {5, {mjMTU_PFL_E0, mjMTU_PFL_E1, mjMTU_PFL_KLOW, mjMTU_PFL_KISO, mjMTU_PFL_CURV}},
      {4, {mjMTU_TFL_E1, mjMTU_TFL_KISO, mjMTU_TFL_FTOE, mjMTU_TFL_CURV}},
      {8, {mjMTU_FV_FMAXE, mjMTU_FV_DYDXC, mjMTU_FV_DYDXNEARC, mjMTU_FV_DYDXISO,
           mjMTU_FV_DYDXE, mjMTU_FV_DYDXNEARE, mjMTU_FV_CONCCURV, mjMTU_FV_ECCCURV}}};
  return specs[curve];
}


// Build one curve from its (already defaulted) shape parameters. The factory calls mirror
// OpenSim's createSimTKFunction for each curve class, including the arguments OpenSim passes as
// literals: the active force-length curve's x2 = 1.0 and curviness = 1.0, and its minimum_value
// of 0, which Millard2012EquilibriumMuscle forces for the damped model it uses by default.
mujoco::millard::SegFn build(int curve, const mjtNum* s) {
  switch (curve) {
    case mjMUSCLECURVE_ACTIVE_FL:
      return mujoco::millard::ActiveForceLength(s[0], s[1], 1.0, s[2], 0.0, s[3], 1.0);
    case mjMUSCLECURVE_PASSIVE_FL:
      return mujoco::millard::FiberForceLength(s[0], s[1], s[2], s[3], s[4]);
    case mjMUSCLECURVE_TENDON_FL:
      return mujoco::millard::TendonForceLength(s[0], s[1], s[2], s[3]);
    default:
      return mujoco::millard::ForceVelocity(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
  }
}


const mjCurveTable* resolve_one(int curve, const mjtNum* prm, const char* context) {
  const CurveSpec& sp = spec_of(curve);
  mjtNum shape[kMaxShape] = {0};
  for (int i = 0; i < sp.nshape; i++) {
    shape[i] = prm[sp.slot[i]];
  }

  std::lock_guard<std::mutex> lock(cache_mutex());
  for (const std::unique_ptr<Entry>& e : cache()) {
    if (e->curve == curve && !std::memcmp(e->shape, shape, sizeof(mjtNum)*sp.nshape)) {
      return &e->baked.table;
    }
  }
  if (static_cast<int>(cache().size()) >= kMaxCurves) {
    mju_error("millard curve cache is full (%d distinct curve shapes)", kMaxCurves);
  }

  std::unique_ptr<Entry> e(new Entry);
  e->curve = curve;
  std::memcpy(e->shape, shape, sizeof(shape));
  mujoco::millard::check_context = context;
  e->baked = mujoco::millard::Bake(build(curve, shape), kKnots);
  mujoco::millard::check_context = "";
  // Bake() stores knot.data() in the table, so re-point it after the move into the entry.
  e->baked.table.knot = e->baked.knot.data();
  const mjCurveTable* out = &e->baked.table;
  cache().push_back(std::move(e));
  return out;
}

}  // namespace


void mju_millardBakeCurves(const mjtNum* prm, const char* context,
                           const mjCurveTable* out[mjNMUSCLECURVE]) {
  for (int c = 0; c < mjNMUSCLECURVE; c++) {
    out[c] = resolve_one(c, prm, context ? context : "");
  }
}


int mju_millardCurveCacheSize(void) {
  std::lock_guard<std::mutex> lock(cache_mutex());
  return static_cast<int>(cache().size());
}
