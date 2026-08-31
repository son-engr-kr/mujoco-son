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

// Generator for test/engine/testdata/millard_opensim_reference.csv -- the OpenSim reference the
// engine's Millard curves are checked against.
//
// This is the real thing: OpenSim's own SmoothSegmentedFunctionFactory and its own evaluator,
// compiled straight from the opensim-core sources. Only three files are needed
// (SegmentedQuinticBezierToolkit, SmoothSegmentedFunctionFactory, SmoothSegmentedFunction), and
// they depend on SimTK alone -- Vec6, Array_, Matrix, Eps, Pi -- not on OpenSim's Object or Model
// framework. So the oracle costs SimTKcommon + SimTKmath, not a full OpenSim build.
//
// BUILDING IT. Simbody's headers need C++20.
//
//   git clone --depth 1 https://github.com/simbody/simbody.git
//   cmake -S simbody -B sbbuild -G Ninja -DCMAKE_BUILD_TYPE=Release \
//         -DBUILD_VISUALIZER=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
//   cmake --build sbbuild --target SimTKmath -j
//
//   OSIM=<path to opensim-core>
//   INC=(); for d in $(find simbody/SimTKcommon simbody/SimTKmath -type d -name include); do
//     INC+=(-I$d); done
//   c++ -O2 -std=c++20 -o opensim_curve_dump tools/opensim_curve_dump.cpp \
//       $OSIM/OpenSim/Common/SegmentedQuinticBezierToolkit.cpp \
//       $OSIM/OpenSim/Common/SmoothSegmentedFunction.cpp \
//       $OSIM/OpenSim/Common/SmoothSegmentedFunctionFactory.cpp \
//       -I $OSIM/OpenSim/Common -I $OSIM "${INC[@]}" \
//       -I sbbuild/SimTKcommon -I sbbuild/SimTKmath \
//       -L sbbuild -lSimTKcommon -lSimTKmath -Wl,-rpath,sbbuild
//
//   ./opensim_curve_dump > test/engine/testdata/millard_opensim_reference.csv
//
// Regenerate only when the curve defaults change. The checked-in CSV is what
// MuscleMtuTest.CurvesMatchOpenSimReference reads.

#include <cstdio>
#include <string>
#include "SmoothSegmentedFunctionFactory.h"

using OpenSim::SmoothSegmentedFunction;
using OpenSim::SmoothSegmentedFunctionFactory;

static void dump(const char* name, SmoothSegmentedFunction* f, double lo, double hi, int n) {
  for (int i = 0; i <= n; i++) {
    double x = lo + (hi - lo)*(static_cast<double>(i)/n);
    std::printf("%s,%.17g,%.17g,%.17g\n", name, x, f->calcValue(x), f->calcDerivative(x, 1));
  }
}

int main() {
  // exactly the damped-model defaults the engine bakes at
  SmoothSegmentedFunction* afl = SmoothSegmentedFunctionFactory::
      createFiberActiveForceLengthCurve(0.4441, 0.73, 1.0, 1.8123, 0.0, 0.8616, 1.0, false, "afl");
  SmoothSegmentedFunction* pfl = SmoothSegmentedFunctionFactory::
      createFiberForceLengthCurve(0.0, 0.7, 0.2, 2.0/0.7, 0.75, false, "pfl");
  SmoothSegmentedFunction* tfl = SmoothSegmentedFunctionFactory::
      createTendonForceLengthCurve(0.049, 1.375/0.049, 2.0/3.0, 0.5, false, "tfl");
  SmoothSegmentedFunction* fv = SmoothSegmentedFunctionFactory::
      createFiberForceVelocityCurve(1.4, 0.0, 0.25, 5.0, 0.0, 0.15, 0.6, 0.9, false, "fv");

  std::printf("curve,x,y,dydx\n");
  // sampled past both ends of each domain, so the linear extrapolation is compared too
  dump("afl", afl, 0.2, 2.1, 977);
  dump("pfl", pfl, 0.9, 2.0, 977);
  dump("tfl", tfl, 0.98, 1.20, 977);
  dump("fv", fv, -1.5, 1.5, 977);
  return 0;
}
