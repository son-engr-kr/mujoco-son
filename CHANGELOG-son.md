# mujoco-son changelog

Changes made by this fork, on top of upstream MuJoCo. Upstream's own changelog is
[doc/changelog.rst](doc/changelog.rst) and is left untouched so it stays mergeable.

## v3.3.3+son4.0a3 — alpha

Supersedes `son4.0a2` for anyone driving the muscle models from Python: the helpers
`doc/muscle_mtu.rst` tells you to call were exported from the C library but never bound, so the
documented equilibration sequence could not be run at all. `a2` remains correct for C callers and
for models driven purely through `mj_step`.


### Fixed: the MTU muscle helpers are now reachable from Python

`doc/muscle_mtu.rst` tells users to call `mju_mtuMuscleEquilibrate` after setting a pose rather
than integrating one, and points at `mju_millardCurve` / `mju_hyfydyCurve` for comparing a model
against OpenSim. None of them were bound, so from Python the documented sequence could not be
run at all. Reported against `son4.0a2`.

The whole muscle surface of `mujoco.h` is now bound, not a subset — partial exposure is what
produced the report:

`mju_compliantMuscleInvFvce0`, `mju_compliantMuscleFlce0`, `mju_compliantMuscleFp0`,
`mju_compliantMuscleFp0Ext`, `mju_compliantMuscleInit`, `mju_compliantMuscleActDot`,
`mju_compliantMuscleEquilibrate`, `mju_compliantMuscleForceVel`, `mju_compliantMuscleECC`,
`mju_mtuMuscleInit`, `mju_mtuMuscleActDot`, `mju_mtuMuscleEquilibrate`,
`mju_mtuMuscleForceVel`, `mju_millardCurve`, `mju_hyfydyCurve`, `mju_millardCurveCacheSize`.

Binding them surfaced a second, older defect: **six of those were declared `MJAPI` in `mujoco.h`
but never exported from the library.** The build uses `-fvisibility=hidden`, and a definition
only gets default visibility if an `MJAPI` declaration is in scope where it is defined;
`engine_util_misc.c` does not include `mujoco.h`, so the public header was promising symbols a C
user could not link against. They are now declared in `engine_util_misc.h` as well.

The optional `deriv` out-parameter of the two curve functions is bound as an optional
single-element array, following `mj_constraintUpdate`'s `cost` argument, and the optional
`gainprm` as an optional `mjNGAIN` array. No signature changed, so the C API is untouched.

Verified against the documented workaround: one `mju_mtuMuscleEquilibrate` call lands where 270
iterations of `act_dot[fiber] = (l_ce* - l_ce)/dt` land, to 6.2e-12 m in fiber length and
9.7e-8 % of `F_max` in force.

### Docs

`doc/muscle_mtu.rst` now records an independent measurement of the assembled whole-muscle force
against OpenSim — the gap that section previously listed as unverified. 38 of the 40 Rajagopal
`Millard2012EquilibriumMuscle` muscles transfer at 0.0000 % of `F_max` RMSE through the 32-slot
layout; two do not, and the cause is not established. Dropping the curve shape slots raises the
median error to 7.0 %, which is the measured justification for carrying all 32.

## v3.3.3+son4.0a2 — alpha

Supersedes `son4.0a1`, which does not build its C++ test suite: three compliant-muscle helpers
were declared in `src/engine/engine_util_misc.h` after the `extern "C"` block closed, so any C++
translation unit including both that header and `mujoco.h` saw conflicting language linkage. The
library and the published wheels were unaffected — the callers are C, and that header does not
ship — but `a1` should not be used to build from source. Everything below is otherwise unchanged
from `a1`.


Adds two muscle-tendon actuators and rebuilds all three onto MuJoCo's own state model.
**This release breaks compatibility** — see the list at the end before upgrading.

### Why alpha

Everything below is verified in the sense the "Verification" section describes: the curves match
OpenSim's own output, the state model is checked against MuJoCo's contracts, and 27 C++ plus 49
Python tests pass. What has *not* happened is use:

* no full-body model has been run with `millard_mtu` or `hyfydy_mtu` — the largest thing exercised
  is 80 Rajagopal muscles on a synthetic single-DOF rig;
* no RL or trajectory-optimisation workload has been run against them, so nothing has stressed the
  throughput or the derivative path at scale;
* the breaking changes below have not been exercised by any downstream code, and the `act` layout
  change in particular will silently alter any existing keyframe.

Later `a`/`b` releases are expected before `son4.0` final. Local version segments sort
`son4.0a1 < son4.0a2 < son4.0b1 < son4.0`, so upgrading in place works as you would expect.

### New: `millard_mtu` and `hyfydy_mtu`

| `gaintype` | Model |
| --- | --- |
| `millard_mtu` | OpenSim `Millard2012EquilibriumMuscle`, damped variant, with OpenSim's own quintic-Bézier curves |
| `hyfydy_mtu` | Hyfydy `muscle_force_m2012fast`, the polynomial curves published in the Hyfydy User Manual v1.0.2 |

Both solve the same backward-Euler fiber equilibrium as the existing `compliant_mtu`, with
fixed-width pennation, and share the residual, the analytic Jacobian and the solver. Only the four
normalized curves differ, so switching muscle models is one attribute.

**Parameters transfer from a source model unchanged.** The 32 `gainprm` slots are a 1:1 map of an
OpenSim `Millard2012EquilibriumMuscle`'s properties — including all four curve shapes, which real
models do override — with **zero in any slot meaning that property's OpenSim default**. Verified
against Gait10dof18musc, Rajagopal2016 (which retunes the active force-length curve model-wide) and
RajagopalLaiUhlrich2023 (which retunes the passive curve per muscle). See
[doc/muscle_mtu.rst](doc/muscle_mtu.rst) for the slot table.

Curves are baked to a uniform 513-knot quintic-Hermite table lazily at the first `mj_resetData`,
into a process-wide cache keyed by the shape parameters. Nothing is stored in `mjModel`, where a
table could go stale against the parameters that produced it. Muscles sharing a shape share the
table. Measured cost: 155 ns per muscle-step for `millard_mtu`, 66 ns for `hyfydy_mtu`.

### Changed: the fiber length is now an actuator activation variable

All three `*_mtu` gains previously kept the fiber length in `mjData.muscle_l_ce` and integrated it
inside `mj_fwdActuation`, with a teleport heuristic to paper over the inconsistency. Measured, that
broke three MuJoCo contracts:

* `mj_forward` was not idempotent — a second call at the same state moved the force from 101.5 N to
  193.0 N, so `mjd_transitionFD` differenced the wrong thing;
* `RK4` was wrong, because it evaluates `mj_forward` at intermediate states and each call advanced
  the fiber a full step: `qpos` +0.079 (Euler) against +0.424 (RK4);
* the fiber was in no `mjtState` element, so `mjSTATE_PHYSICS` was not a complete state.

The fiber length is now `act = [l_ce, activation]`, following MuJoCo's convention that the last
activation variable multiplies the gain while earlier ones are internal state. `act_dot` for the
fiber is the implicit step `(l_ce* - l_ce)/dt`, which keeps the unconditional stability the stiff
tendon needs while leaving `mj_forward` a pure function of the state. **Euler trajectories are
unchanged to the last digit.** `mj_forward` is now idempotent, `RK4` agrees with `Euler` to 1e-11,
and a `mjSTATE_PHYSICS` round trip into a fresh `mjData` reproduces a trajectory exactly.

`mju_mtuMuscleEquilibrate` and `mju_compliantMuscleEquilibrate` replace the teleport heuristic,
mirroring OpenSim's `Model::equilibrateMuscles`.

### Other changes

* `minimum_activation` (gainprm slot 7, default 0.01) implemented for the Millard/Hyfydy gains,
  matching OpenSim's clamp of the activation wherever it builds a force.
* The rigid-tendon fallback is now OpenSim's `ignore_tendon_compliance` path rule for rule:
  non-negative `cos(phi)`, the real tendon length reported, tendon buckling, a clamped fiber
  carrying no force, and the fiber force saturated at zero.
* `mjd_actuator_vel` gains `d(force)/d(velocity)` for the rigid-tendon path. It is exactly zero for
  a compliant tendon — the actuator force is the tendon force, a function of tendon length.
* Fixed: `compliant_mtu` read its activation from `d->act[i]` (the actuator index) instead of
  `actadr + actnum - 1`. This was wrong for any model with more than one actuator.
* `mjSTATE_MUSCLE` removed. It was declared but never implemented (`mj_stateSize` raised "invalid
  state element"); the fiber state is now covered by `mjSTATE_ACT`.
* MJX raises `NotImplementedError` for all three `*_mtu` gains, as it already did.

### Breaking changes

| Change | Effect |
| --- | --- |
| `mjNGAIN` 10 → 32 | `mjModel.actuator_gainprm` changes shape; saved binary models are incompatible |
| `*_mtu` actuators carry 2 activation variables | `na` grows; `act` layout is `[l_ce, activation]`; keyframes storing `act` need rewriting |
| `*_mtu` requires `dyntype="muscle"` | models using `dyntype="none"` on these gains no longer compile |
| `*_mtu` rejects `actrange` | a range is per-actuator and would clamp the fiber length |
| `mjSTATE_MUSCLE` removed, `mjSTATE_PLUGIN` 1<<13 → 1<<12, `mjNSTATE` 14 → 13 | serialized state specs change |
| `mjtGain` gained two values before `mjGAIN_USER` | `mjGAIN_USER` is now 6 |
| gainprm slot 7 is `minimum_activation` | was reserved-must-be-zero |
| `mju_compliantMuscleUpdate` / `mju_compliantMuscleReset` removed | replaced by `mju_compliantMuscleActDot` / `mju_compliantMuscleEquilibrate` |

### Verification

26 C++ tests (`test/engine/engine_muscle_mtu_test.cc`) and 49 Python tests
(`test_osim_muscle.py`, `test_compliant_muscle_smoothness.py`).

The curves are checked against **OpenSim itself**: `tools/opensim_curve_dump.cpp` compiles
`SmoothSegmentedFunctionFactory`, `SmoothSegmentedFunction` and `SegmentedQuinticBezierToolkit`
straight from the opensim-core sources — they need SimTK alone, not OpenSim's Object framework —
and samples them into `test/engine/testdata/millard_opensim_reference.csv`. Agreement is within
**2.1e-9** in normalized force and **4.3e-6** in slope, which is the baked table's own
interpolation error.

Also verified: the Bézier evaluator matches OpenSim's expanded polynomials to 1e-14 relative; the
curves meet the keypoint, C2-continuity and monotonicity criteria from OpenSim's
`testSmoothSegmentedFunctionFactory.cpp`; the analytic Jacobian shows quadratic Newton
convergence; the equilibrium residual stays below 1e-6 during motion.

Not run: OpenSim's whole-muscle `estimateMuscleFiberState`, which needs a constructed
`Millard2012EquilibriumMuscle` and therefore OpenSim's Object and Component framework. The
equilibrium assembly on top of the curves was transcribed from the source line by line instead.
The deliberate differences from the source models are listed in
[doc/muscle_mtu.rst](doc/muscle_mtu.rst#deliberate-differences-from-the-source-models).
