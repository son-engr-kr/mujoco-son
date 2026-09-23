# mujoco-son changelog

Changes made by this fork, on top of upstream MuJoCo. Upstream's own changelog is
[doc/changelog.rst](doc/changelog.rst) and is left untouched so it stays mergeable.

## Unreleased

### Added: `compliant_mtu` parallel element with its own slack length and reference strain

Requested for jinsimul's decision 0024, which fits the parallel element of a Geyer muscle to its
source model's passive curve instead of tying it to `W`. `gainprm[9]` is the element's slack
length `L_PE0`, in `l_opt`, and `gainprm[10]` its reference strain `E_REF_PE`:
`f_pe = ((l_ce/l_opt - L_PE0)/E_REF_PE)^2` above the slack length. Both 0 is Geyer & Herr 2010's
element, `(1, W)`; otherwise both must be positive and finite.

All five evaluations of the element (the rigid-tendon force, the isometric solve and the
backward-Euler step, each with its finite-difference probe) now go through one function,
`mju_compliantMuscleFpe0(l0, rest, e_ref)`, exported and bound in Python. At `(1, W)` it is
`mju_compliantMuscleFp0(l0, W)` operation for operation.

**A model that declares neither slot is bit-identical to `son4.0a6`.** Checked by running the same
input grid against a build of `a6`'s source and against this one: 47,020 outputs over five models
(three compliant tendons, two rigid), covering the isometric solve, the backward-Euler step from
fibers off equilibrium at five path velocities, and a 2000-step driven trajectory. 70 of the 125
passive points had the parallel element engaged. Every output matched bit for bit, on macOS arm64.
In the suite, a muscle declaring `(1, W)` is checked bit for bit against one declaring nothing on
all three branches.

The solver lands on the element it is given. At zero activation the parallel element and the
tendon are quadratic springs in series, which gives a closed form for the static force, and the
isometric solve matches it to 2e-6 `F_max` with jinsimul's `abd_r` fit (`L_PE0` 1.2376,
`E_REF_PE` 0.3815). From that equilibrium the stepping solve leaves the fiber where it is, which
it would not if it saw a different element.

### Changed: `compliant_mtu` rejects `gainprm` it does not read

Slots 9-31 used to be ignored, so `a6` runs a model with its own parallel element without an
error, and runs it wrong. A value in slots 11-31, one of slots 9-10 without the other, or a
negative or non-finite value there now fails the load, from `mj_resetData` as `millard_mtu` does.

No existing model is affected. All 3,565 `compliant_mtu` declarations across 73 MJCF files in the
neumove projects, including the MyoAssist compliant models in `mujoco-compliant-muscles`, give
exactly nine values. The 658 actuators in the files that compile on their own resolve slots 9-31
to 0 after defaults as well.

### Fixed: the Millard active force-length curve enforces OpenSim's range checks

`engine_muscle_millard_bezier.h` claimed to enforce OpenSim's admissibility conditions, but the
active curve checked only that its knots increase. OpenSim's
`createFiberActiveForceLengthCurve` also requires `min >= 0`, every knot gap wider than
`sqrt(eps)` (1.5e-8), and `shallow_ascending_slope < (1 - minimum_value)/(1 - transition)`. It now
checks all three, and the passive, tendon and force-velocity curves now check that their
curviness is in `[0, 1]`, as OpenSim's do.

This surfaced while checking jinsimul's Millard fit to `Lumbar_C_210`'s Thelen curves, which puts
`min_norm_active_fiber_length` at 3.5e-10 and the transition at 1.4e-9. OpenSim 4.6 refuses that
shape when the `ActiveForceLengthCurve` is constructed. `a6` refused it too, but by accident, at
the corner-control-point check, and it accepted nearby shapes OpenSim refuses, such as a
transition of 3.5e-9. None of the 312 fitted muscles committed in `mujoco-compliant-muscles`
violates any of the new checks.

### Measured: `min_norm_active_fiber_length` near zero

Zero in slot 8 still means OpenSim's 0.4441, like every other slot, so a fit that drives it toward
zero has to write a small positive number. The rule stays as it is and is now documented and
tested.

A value near zero works. With 3.5e-10 and an admissible transition, a muscle equilibrates and steps
without a NaN in three variants: unpennated, where the fiber clamp is 6.5e-11 m, pennated, where
`h/sin(phi_max)` is the clamp, and rigid-tendon. The sweep runs from a path half the tendon's
slack length to one that stretches the parallel element, followed by 4000 driven steps.

A value near zero is not always resolved, though. The baked table's knots are `(max - min)/512`
apart, and an ascending limb only a few knots wide is smoothed over. Measured against OpenSim
4.6's `ActiveForceLengthCurve`, the worst error in normalized force is 1.2e-8 at the default shape.
With `min` near zero it is 1.9e-8 for a transition of 0.3, 2.3e-6 for 0.1, and 0.087 for 1e-7,
in each case at the foot of the curve. For the 113 Thelen-sourced fits in
`mujoco-compliant-muscles` (`min` 0.05, transitions from 0.0975) the median is 1.5e-7 and the
worst 7.5e-5. `doc/muscle_mtu.rst` now carries these numbers in place of the unqualified 7e-9.

### Not in this release

- The Geyer 2010 residual with the buffer element (E2) and force-velocity region 3 (E3) await a
  decision.
- The damped model's forcing of `concentric_slope_at_vmax` and `eccentric_slope_at_vmax` to 0 (E5)
  is not enforced. Slots 25 and 28 are still used as given. jinsimul's Thelen fit sets them to
  0.234 and 0.126, so it is a live case rather than a hypothetical one.
- Python's `MjSpec` exposes only 10 of the 32 `gainprm` slots, because its bindings are generated
  from `introspect/structs.py`, which still declares 10. This predates this release, and it means
  `E_REF_PE` (slot 10) and the Millard curve slots cannot be set through `MjSpec` from Python.
  MJCF and `MjModel.actuator_gainprm` are unaffected.

## v3.3.3+son4.0a6 — alpha

### Fixed: `mju_mtuMuscleEquilibrate` did not reach the fiber equilibrium

Reported against `son4.0a5`, with a reproduction and OpenSim `computeEquilibrium` answers to
compare against. Two failures, with two separate causes.

**The pennation derivative was discarded at the fiber clamp.** `lce_min` is `h/sin(phi_max)`
whenever `sin(phi_opt) > sin(phi_max) * min_norm_active_fiber_length` — past 26 degrees of
pennation with OpenSim's default 0.4441, past only 14 for a muscle like `glmax3_r` that lowers it
to 0.25 — and then at the clamp `h/l_ce` lands within an ulp of `sin(phi_max)` and can fall either
side of it. The branch that fired there set `dphi` to zero, on
the reasoning that the pennation angle is frozen below the clamp — which is only right if the
fiber could go below it, and it cannot. What it produced was a Jacobian missing its dominant term
exactly where that term dominates: `dl_T/dl_ce` collapsed from -9.97 to -0.10 on `glmax3_r`, a
hundredfold error that sent the solve into a limit cycle between the clamp and a point far above
it. The derivative is now taken from the clamped sine, which is continuous and, since
`1 - sin(phi_max)^2` is 0.01, never divides by a vanishing root.

That this branch was reachable at all contradicts a comment added in `son4.0`, which argued it
was unreachable because `lce_min >= h/sin(phi_max)`. True in exact arithmetic; not in floating
point, at the one value the clamp puts the iterate on.

**Equilibration used plain Newton from a degenerate seed.** The stepping solver is warm-started a
fraction of a fiber length from the root and regularised by the fiber damping over the timestep,
and converges quadratically. Equilibration has neither: it starts from `mj_resetData`'s seed of
`optimal_fiber_length`, which is the peak of the active force-length curve, and with no timestep
the damping drops out of the Jacobian. On a muscle whose tendon is slack at that seed — `BIClong`
of MoBL-ARMS, and every zero-pennation muscle in that model — every term of `dR/dl_ce` is zero to
rounding. Newton had nothing to descend, and the step cap turned the divergence into a limit
cycle, so the fiber came back exactly where it started and the call looked like a no-op.

Equilibration now brackets the root and falls back to bisection wherever Newton leaves the
bracket. It also no longer reads the incoming fiber length: it is a function of the pose and the
activation, so two calls at the same pose agree bit for bit. Both reported muscles now match
OpenSim's `computeEquilibrium` — `BIClong` at `l_ce` 0.10016 against 0.10012, `glmax3_r` at
0.08020 against 0.08020 — and agree with relaxing the `act_dot` identity to steady state, which
was the reporter's workaround.

The stepping path is untouched: it keeps the plain Newton that was measured converging
quadratically, and the pennation change removes a branch from it rather than adding one.

## v3.3.3+son4.0a5 — alpha

### Added: `mju_millardCurveCacheClear`

The baked-curve cache never evicts, so a least-squares fit over Millard curve shapes — which
bakes a fresh set per candidate and never revisits one — exhausted the 4096-curve cap after about
eight muscles and then failed with `millard curve cache is full`. Reported against `son4.0a4`,
with measured fill rates of 300-600 entries per muscle.

`mju_millardCurveCacheClear()` drops every baked curve and returns how many it freed. Its
precondition is that every `mjData` reset since those curves were baked is reset again before
use: the clear frees what its `muscle_curve` entries point at, and `mj_resetData` re-resolves
them. Bound in Python alongside `mju_millardCurveCacheSize`.

Reproducing the reported workflow with a clear between muscles, the cache peaks at ~710 entries
per muscle and returns to 0, so 20 muscles stay well under a cap that ~14,000 entries would have
blown through three times over.

**There is deliberately no automatic eviction**, LRU included, and that is not an omission.
`mjData.muscle_curve` holds raw pointers into the cache, nothing can enumerate live `mjData`, and
there is no reference count, so no policy can know when an entry is safe to free. Dropping the
cache has to be the caller's statement that none of its `mjData` will be used unreset — a claim
only the caller can make.

## v3.3.3+son4.0a4 — alpha

Supersedes `son4.0a3`, which deadlocks the process the first time the Millard curve bake rejects
a shape. Anyone fitting `millard_mtu` curve shapes should move off `a3`; a model with fixed,
valid shapes never hits it.


### Fixed: a rejected curve shape no longer wedges the bake cache

The Millard curve bake validates the shape parameters it is handed, and a rejection reaches
Python as `FatalError`. It was raised from inside the cache's critical section, and `mju_error`
does not unwind the C++ stack, so the `lock_guard` was never destroyed and the mutex stayed
locked for the rest of the process. Every later bake then blocked forever — including a bake of
a shape that had succeeded moments earlier — at 0% CPU and ignoring SIGINT, so it presented as a
hang rather than an error. Reported against `son4.0a3`, with a reproduction.

Any parameter search over `millard_mtu` curve shapes eventually proposes a shape the bake
refuses, so any such search eventually died. `compliant_mtu` was unaffected; it bakes no curves.

The fix removes the whole class rather than the conditions that happen to fire today: **the lock
is never held across anything that can raise.** The lookup takes it, drops it, bakes outside it,
and takes it again to insert, re-checking in case another thread got there first. `Check` in the
Bézier header now throws instead of calling `mju_error`, so the stack unwinds normally, and
`engine_muscle_bake.cc` converts that to `mju_error` outside the lock with nothing else alive —
which is what the model compiler already does with `mjCError`. The cache-full path was raising
under the lock too, and no longer does.

The regression test runs the sequence in a worker thread with a deadline, so a recurrence fails
the suite instead of hanging it. Verified both ways: it fails on `a3` and passes on this build.

### Docs

`doc/muscle_mtu.rst` now states that the bake cache only grows — entries are pointed at by
`mjData` and can never be evicted — with the arithmetic for sizing a parameter search against
the 4096-curve cap.

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
