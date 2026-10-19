# mujoco-son changelog

Changes made by this fork, on top of upstream MuJoCo. Upstream's own changelog is
[doc/changelog.rst](doc/changelog.rst) and is left untouched so it stays mergeable.

## v3.3.3+son5.5

The C engine takes two upstream GJK fixes, 7f305abd7 and d33f297b5; MJX is son5.4's. At the default
`ccd_tolerance` of 1e-6 results are unchanged.

### Fixed: GJK dropped contacts and made up penetrations at small `ccd_tolerance` (7f305abd7)

Backported from upstream (google-deepmind/mujoco@7f305abd7, in 3.12.0). For a sphere or capsule, GJK
shrinks the geom to its center or axis, measures the distance to the other geom, and stops once the
Frank-Wolfe gap is below `0.5 tol²`. From 1e-9 down that is under the rounding of `|x|²`, about
`2.2e-16 |x|²` with `|x|` the radius plus the distance, so GJK did not stop once converged and took
a step onto a nearly degenerate simplex. There the iterate could move away from the origin, past the
margin, and the contact was dropped, or toward it, and a penetration was reported where there is
none. On myolegs, a tibia capsule and a foot ellipsoid 8.45e-5 m apart, inside their 1e-3 margin,
lost their contact at 1e-10 and 1e-12 whatever `ccd_iterations`, and the same pair, its pose rounded
through XML, reported -1.5e-3 m from 1e-9 down. GJK now also stops once `|x|` changes by less than
`mjMINVAL`, as upstream does; son's contact threshold on `|x|` stays `mjMINVAL`, without upstream's
330c36c9.

Over 600 capsule-ellipsoid poses at distances known from an independent closest-point computation,
315 of them within the margin, son5.4 got 4 distances wrong by up to 4.5e-4 m from 1e-9 to 1e-12,
and 1 by 4.3e-6 m at 1e-8. son5.5 gets 1 of the 600 wrong, by at most 4.3e-6 m, at every tolerance
from 1e-8 to 1e-12, as upstream 3.16 does. At 1e-6 the 600 contacts, and 1000-step rollouts of
myolegs and MyoFullBody, are bitwise those of son5.4 built with the same compiler; with both fixes,
at 1e-8 they move by up to 2.0e-5 in `qpos`. `engine_collision_gjk_test`'s two tests with the
myolegs pair, at its frames and at its XML pose, fail on son5.4; the C suite, 838 tests, passes.

### Fixed: contact normals below the default `ccd_tolerance` (upstream d33f297b5)

Backported from upstream (google-deepmind/mujoco@d33f297b5, in 3.12.0). When GJK's closest point on
a tetrahedron lay on the face opposite its fourth vertex, `S3D` kept that vertex's barycentric
coordinate from a face checked before, so the next iterate and the witness points, which give the
contact normal, still weighted that vertex. At the 600 poses above, son5.4's normals were off the
closest points' by up to 9e-4 at `ccd_tolerance` 1e-7 and 4e-4 at 1e-8, where at 1e-6 they are off
by at most 3e-5. With both fixes they are off by at most 3e-5 at every tolerance from 1e-6 to 1e-10,
and the distances are as above. `engine_collision_gjk_test`'s test with one of these poses fails
without this fix.

### Comparing with C (corrects son5.2)

son5.2 advised `ccd_tolerance` near 1e-10 for MJX-C comparisons. On son5.4 and earlier, do not go
below 1e-8: from 1e-9 down, the GJK above drops contacts inside the margin and reports penetrations
of a millimeter that are not there, and below 1e-6 capsule-ellipsoid normals are off by up to 9e-4.
On son5.5, from 1e-8 to 1e-12, 1 of the 600 poses above is off, by at most 4.3e-6 m, as in upstream
3.16, and from 1e-6 to 1e-10 the normals by at most 3e-5.

## v3.3.3+son5.4

MJX only. Reverse-mode gradients through spatial tendon wraps are finite where they were NaN, and
match C's central differences; lengths are unchanged in float64 and move by rounding in float32.

### Fixed (MJX): NaN gradients through tendon wraps

`jax.grad` through a spatial tendon's geom wrap returned NaN while the length matched C. MJX
evaluates every branch and selects with `jp.where`, and an infinite or NaN derivative in a branch
that is not selected still makes the reverse-mode gradient NaN. Four places did so:

- `support.wrap`'s cylinder correction, which also runs for spheres, took `sqrt` of zero. Where an
  inside wrap wraps, its two wrap points coincide and `sqrt(wlen² + height²)` is `sqrt(0)`; with
  no wrap, a site on the geom's z axis is at zero from its projection. These are now `math.norm`,
  whose gradient at zero is zero.
- `support.wrap_inside`, past its exits, took `asin(A z)` with `A > 1` for a site inside the
  circle, and `acos(cosG)` at `cosG = ±1`. Where an exit is taken, `A`, `B` and `cosG` are now
  replaced by values the solver is defined at; its solution is not selected there. In float32,
  `1 - mjMINVAL` rounds to 1, so `cosG = 1` now also takes C's exit for `cosG > 1 - mjMINVAL`;
  float64 is unchanged.
- `support._length_circle` took `atan2(0, 0)` where a tangent point is zero, for a site on a
  cylinder's axis.
- `_length_moment` and `tendon_dot` divided a zero-length segment by its length, below.

On the reproduction of upstream's google-deepmind/mujoco#3696 (one hinge, a side site inside the
wrap geom), son5.3's gradient of `ten_length` was NaN at q = 0 and 1 for a sphere and at q = 1 for
a cylinder. It and the gradient of `ten_J` now match C's central differences at all four angles,
in float32 and float64. On upstream 3.16.0 the `sqrt` above remains NaN with the clip #3696
proposes.

Over 800 poses each of a sphere and a cylinder, with the side site inside and outside, random and
aligned (sites on an axis, mirrored, collinear with the centre), son5.3's Jacobians of `ten_length`
and `ten_J` with respect to the site positions were non-finite at 675, 60, 664 and 66 poses in
float64 (678, 60, 682 and 66 in float32), and now at none. At the 300 random poses of each, the
gradients of `ten_length` match C's central differences to 1e-5. Elsewhere they differ only where
C's length is not differentiable: where it jumps between wrapping and not, where the side site
ties the two wrap sides, and with sites collinear with a sphere's centre, where the wrap plane is
undefined.

On MyoFullBody, at 20 states of a 1000-step rollout that falls onto the floor, son5.3's gradient
of a random combination of the 424 tendon lengths with respect to `qpos` was NaN in 64 of 129
entries at every state, and is now finite. Over 200 states of that rollout the lengths moved by
1.7e-13 mm in float64 and 1.0e-4 mm in float32, where they are 4.6e-4 mm from C.

`smooth_test`'s `test_tendon_wrap_grad`, against C, and `test_tendon_wrap_grad_finite`, at
degenerate poses, fail on son5.3.

### Fixed (MJX): safe division (upstream 0ba31e2c)

Backported from upstream (google-deepmind/mujoco@0ba31e2c, after 3.3.3): `math.safe_div`, in
`_length_moment`, `tendon_dot`, `support.wrap`'s cylinder correction, `ray` and `plane_cylinder`.
Where an inside wrap wraps, `_length_moment` divided its zero-length segment by its length, 0/0 in
a branch not selected, which made the gradient of `ten_J` NaN. son5.3 already had its own versions
of the commit's `_is_intersect`, `wrap_inside` and `muscle_dynamics_timescale` hunks, and keeps
them.

## v3.3.3+son5.3

The C engine takes upstream's tendon wrapping fix 3bc4b769; the rest is MJX. With these, every
tendon of MyoFullBody, at 200 states of a 1000-step rollout that falls onto the floor, follows C to
under 1e-4 mm in float64 and 5e-4 mm in float32, inside wraps included.

### Fixed: tendon wrapping near wrap onset and inside wraps (upstream 3bc4b769)

Backported from upstream main (google-deepmind/mujoco@3bc4b769, after 3.15.0). `wrap_inside` ran
Newton on `z` from 1 - 1e-7, where the derivative of `asin(z)` is unbounded. When the root lies
next to `z = 1`, the residual there is already positive, and C returned the midpoint of the two
sites instead of the tangent point: at 1 and 5 of 20000 random inside wraps around a cylinder and a
sphere, 0.16 to 22 mm too long, and on MyoFullBody's `EDC3` and `EDC4` sphere wraps at 4 of 200
states of a rollout, by up to 0.17 mm. Newton now runs on `phi = asin(z)` from pi/2, with the same
20 iterations and 1e-6. `is_intersect` takes its upper bounds from `p2 - p4`, which near wrap onset
do not round the way the quotients `a` and `b` do, and `length_circle` its angle by `atan2`.
`engine_util_misc_test` is now built, with upstream's cases, run through a tendon because
`mju_wrap` is not exported in 3.3.3, and an inside wrap with its root next to `z = 1`, which fails
on son5.2.

### Fixed (MJX): a tendon grazing its wrap geom took the far side in float32

When a spatial tendon only grazes a cylinder or sphere, the correct wrap has a nearly zero arc, and
its two tangent lines nearly coincide. `support._is_intersect` then decided on the edge of its
`[0, 1]` interval: float64 called the lines disjoint, float32 called them crossing, so the correct
pair was penalized and the pair around the far side, an arc of 2.70 rad, won. With a path grazing a
cylinder by micrometers and its end site moved by about 1 um, son5.2 was off C by over 1 mm at 24
of 200 poses, by 114.6 mm at worst; on myolegs millard's `semiten_r` at one state of a rollout the
path was 11.5 cm too long and the force 3.06 F_max instead of 0.47. `_is_intersect` and
`_length_circle` are now C's, above. All 200 poses agree with C to 0.05 mm, and the tendons at that
state to 0.001 mm. This is google-deepmind/mujoco_warp#1714 in MJX; sphere wraps share the test.

### Fixed (MJX): `plane_capsule`'s contact tangent follows C at every tilt

MJX swapped the tangent for a fixed axis once the capsule axis, projected on the plane, was shorter
than 0.5, that is, once the capsule stood more than 60 degrees from the plane. `mjc_PlaneCapsule`
hands `mju_makeFrame` the unit axis, which projects it however short that leaves it, so only an
axis along the normal falls back. Under a pyramidal cone, MuJoCo's default, the friction pyramid
turns with the tangent: one capsule at 65 degrees changed `qacc` by 11. The tangent and `qacc` now
match C to 1e-16 at every tilt. This is upstream's google-deepmind/mujoco#2774. Within `sqrt(eps)`
of the normal (1.5e-8 rad in float64, 3.5e-4 rad in float32) the projection is rounding error and
its direction is not orthogonal to the normal, by up to 0.95 in float32 at 1e-7 rad, so the tangent
there is `make_frame`'s, which keeps the frame orthonormal; C's is the rounding error's direction,
or (1, 0, 0) for a zero projection.

### Fixed (MJX): inside wraps used looser solver settings than C

With the side site inside the wrap geom, `support.wrap_inside` solved by Newton with `maxiter` 5,
tolerance 1e-4 and `z` from 1 - 1e-5, where C uses 20, 1e-6 and 1 - 1e-7, and it kept its last
iterate where C returns the midpoint. Over 20000 random site pairs around a cylinder and a sphere
with the side site at the centre, it was off C by over 0.1 mm at 52 and 65 pairs, by up to 5 cm, in
float32 and float64 alike; on myofullbody, `PQ_left` around `PQ2_wrap_left` was off C by 1.32 mm
and its force by 1.7 F_max. C's constants alone would not do in float32: Newton in `z` starts next
to `z = 1`, and MuJoCo Warp measured midpoint fallbacks at 2.2% of random inputs with them
(google-deepmind/mujoco_warp#1713). `wrap_inside` is now C's, above, Newton on `phi` with 20
iterations, 1e-6 and C's exits, including the midpoint when the tolerance is not met;
`wrap_inside_z_init` is gone. Over those pairs MJX agrees with C to 9e-16 m in float64 and 5e-7 m
in float32, and `PQ_left` to 0.001 mm in both.

### Fixed (MJX): Newton stopped an iteration before C

Near convergence the decrease along the search falls below the rounding of the cost, and the line
search moved only on a strictly lower cost. `CGsearch` keeps its Newton point unless the start costs
strictly less, so C took a step that MJX discarded, and the zero improvement then ended the solve.
A point whose cost ties now still counts when its slope is nearer zero. On myolegs hyfydy at the
step where a joint limit row turns on, C and MJX both run two iterations from the same warm start
and `qacc` agrees to 3e-12, where it differed by 2.6e-6 (|qacc| about 70). `solve` also writes
`solver_niter`, which kept the value `put_data` gave it.

### Added (MJX): reverse mode through several solver iterations

Reverse mode went through the constraint solve only with `opt.iterations` at 1. With
`opt.tolerance` at 0 the solve now runs all `opt.iterations` as a scan, which reverse mode
differentiates, in either solver precision, as the open upstream PR google-deepmind/mujoco#3583
does for google-deepmind/mujoco#2259. `Option.tolerance` is static as a result: changing it
recompiles, and it cannot be batched over. On a coupled-joint muscle model, the gradient through
five iterations matches central differences to 1e-5 in float64, and `solver_dtype=float64` in a
float32 model matches that to 1e-3.

### Fixed (MJX): the float32 fiber Newton of `millard_mtu` and `hyfydy_mtu` settles

Its residual stops falling near 1e-6 in float32, above the default tolerance of 1e-9, so the
convergence test never fired, and the fiber kept moving by its rounding until the iteration cap: in
a batched float32 `forward` on 500 myolegs states, 9.6% of the millard fibers and 8.9% of the
hyfydy fibers differed between `cmtu_iter` 20 and 21. The loop now also stops after a step no
larger than the rounding of the lengths it is computed from, `eps (|l_mtu| + |l_ce|)`, and none
differ. It stops within six iterations in all of 16000 solves, with the same error to the
float64 root as before (median 2.3e-8 l_opt), and the forces agree with C as before (median 3e-6
F_max, at most 2e-4). In float64 the tolerance is reached first in every one of those solves, so
double precision keeps C's iterates.

### Fixed (MJX): float32 Millard tables could read a cell's knots at the next cell's position

`_table_eval` took the cell index for its knot gather and the position within the cell from the
same `s = (x - x0) / dx`, but XLA computes `s` anew in each fusion that reads it, some copies with a
fused multiply-add, and in float32 the copies can land on either side of a knot. The knots of cell
`i` were then evaluated at the position in cell `i + 1`, off by a cell's rise: within 30 float32
steps of the knots, a jitted read of the strain formed in the same program was off the float64
value by up to 9.5e-3 (active force-length), 3.9e-3 (passive), 2.0e-3 (tendon) and 1.9e-2
(force-velocity), and a fiber Newton jitted on its own fell into a two-cycle across a tendon knot,
3.4e-5 l_opt from the root. `put_model` now bakes each cell's two knots together with its index,
one gather reads them, and the position is taken against the gathered index; the reads above are
within 1.8e-6. How often the copies disagree depends on XLA's fusion: in `forward` and in
vmapped `step` of myolegs millard on CPU, the 40000 muscle forces checked against C were the same
before and after.

### Comparing with C

`compliant_muscle_equilibrate` in float32 places some fibers up to 4e-4 l_opt from C's, on
MyoFullBody at zero activation, and this is C's tolerance rather than an error. Where the tendon is
just slack, the residual rises quadratically off zero, so `|R| < 1e-6` holds over
`sqrt(1e-6) l_slack E_REF`, 4e-4 l_opt for `fdl`; C stops anywhere in that band, and so does MJX.
Both points pass C's test in double, and the tendon forces differ by under 1e-6 F_max. float64
agrees with C to 7e-10 only because it takes C's iterates. Where the tendon is taut and stiff, the
fibers agree to one float32 step (1.2e-7 l_opt), and that step alone moves the force by up to
1.2e-4 F_max (`rect_abd`, slope 880 per l_opt), which no stopping rule can recover in float32.

### Verification

- The whole MJX suite, 435 tests, passes, and the C tests, 835 with `engine_util_misc_test`.
- New tests: grazing cylinder and sphere wraps in float32 against C, inside wraps in float32 against
  C at 2000 random site pairs, and Warp's two asymmetric inside wraps, the `plane_capsule` frame at
  random tilts against C and orthonormal next to the normal, the line search under a cost too
  large to resolve its decrease, the reported iteration count, the gradient of a five-iteration
  solve against central differences, the float32 fiber Newton holding still on a soleus-like
  muscle, float32 table reads next to every fifth knot against float64, and in C an inside wrap
  with its root next to `z = 1`. Each fails on son5.2; the table test's reads, run on son5.2's
  lookup, are off by 2e-3 to 1.9e-2 against its 1e-5 bound.

### Found, not fixed

- MyoAssist's myoLeg80 exoskeleton models are refused by `put_model`. DEPHY, HMEDI, HUMOTECH and
  OPENEXO have a 1 mm margin on the 23 capsules and 5 ellipsoids that touch the height field: the
  23 hfield-capsule pairs are refused for the margin, and the 5 hfield-ellipsoid pairs have no MJX
  collision function at all. OSL_KA adds hfield-cylinder pairs, which also have none, margin on its
  plane-mesh and hfield-mesh pairs, and two explicit ellipsoid-mesh pairs between the prosthesis and
  the foot, which have none either. Upstream main is the same. MJX's height-field functions keep
  only penetrating points, and `_capsule_convex` reports only penetration, so margin there needs a
  different algorithm; MuJoCo Warp 3.11 runs these pairs through GJK/EPA and raises the prism tops
  by the margin. In all five models the height field is flat, invisible and 5 mm below the ground
  plane, so it touches only a geom already 5 mm into the plane.

## v3.3.3+son5.2

MJX only; the C engine is son5.1's. With these, the hanging myolegs model (elliptic cone, condim=1
pairs, contacts on, zero control) follows C in float64 to 5e-7 in `qpos` over 2000 steps, and to
3e-8 with C's `ccd_tolerance` at 1e-10. son5.1 parted from C at its first capsule-capsule contact.

### Fixed (MJX): sphere and capsule contacts with a capsule

`math.closest_segment_point` divided by `|ab|^2 + 1e-6`. The regularizer is in squared meters, so it
biased the projection on short segments: against C, contacts moved by up to 25 um and normals turned
by up to 0.1 with capsules of radius 5 mm and half-length 10 mm, and by 3 um and 2.5e-3 on myolegs'
foot capsules. Only a zero-length segment is guarded now. `capsule_capsule` follows
`mjraw_CapsuleCapsule`: the closest points of the two lines, clipped first on the first axis and then
on the second, instead of clipping both independently. Parallel axes keep their one contact in the
middle of the overlap, where C makes two. Both pairs agree with C to 1e-16 m in position and 1e-14
in normal in float64.

### Fixed (MJX): ellipsoid-ellipsoid contacts

`ellipsoid_ellipsoid` used the same ten-step gradient descent on an approximate SDF that son5.1
replaced for spheres and capsules. With upstream's test ellipsoids, at 200 random poses within 2 cm
of contact, its distance was off C's by up to 74 mm and 25 contacts were missed or invented; on
myolegs' foot ellipsoids it reported +1.3 mm where C reports -2.16 mm. It now takes the largest
support gap over directions, `dpos . u - sqrt(u . a1 u) - sqrt(u . a2 u)`, which is the distance
apart and minus the penetration depth (EPA's minimum translation) in overlap, by Newton's method on
the sphere from seven starts. It agrees with a global search to 1e-16 m in float64 and 3e-8 m in
float32, and with C's distance to 1e-6 m.

### Added (MJX): condim=1 contacts under an elliptic cone

`make_data` refused them, but the solver already handled them, as `put_data`, which never refused
them, showed. The refusal is gone. The cone's six-wide slices padded only three zeros, so a model
with fewer than three constraint rows failed to trace; they pad six now, which lifts the "at least
three constraint rows" limit noted under son5.0.

### Fixed (MJX): acceleration sensors after the constraint solve

`forward` computed touch, accelerometer, force, torque, the actuator forces and the frame
accelerations before the solve, so they read the previous step's `qacc` and constraint forces. They
come after it now, as in `mj_forward`, including when there are no constraint rows. On a ball
dropped on a plane, son5.1's touch and accelerometer differed from C by up to 891 over 300 steps and
agree to 1e-6 now.

### Added (MJX): `jointlimitfrc` and `tendonlimitfrc` sensors

The force of the joint's or tendon's limit row, as `mj_sensorAcc` reads it, zero while the limit is
inactive.

### Comparing with C

C's EPA sets the floor. At the default `ccd_tolerance`, its capsule-ellipsoid normals are off the
exact ones by about 6e-6 and its ellipsoid-ellipsoid normals by up to 2e-2. For MJX-C comparisons
with these pairs, set `ccd_tolerance` to 1e-8, not lower (this advised 1e-10 until son5.5, which
explains why that drops contacts), and keep `ccd_iterations` at 170 or less: this
3.3.3 engine's `mj_geomDistance` uses a static EPA buffer of that size and crashes above it. On
MyoFullBody the humerus-thorax contacts match C to nine digits, and what remains is C's normal
(1.6e-7 off at `ccd_tolerance` 1e-10), amplified by the model.

### Verification

- The whole MJX suite, 424 tests, passes. `test_ellipsoid_cylinder` fails in float64 before and after,
  as upstream's does.
- New tests: random sphere-capsule and capsule-capsule poses against C, ellipsoid-ellipsoid against
  C's distance and the support gap over sampled directions, condim=1 under both cones and a sliding
  sphere with a condim=1 pair against C, and the two limit sensors against C. Each fails on son5.1.

### Upstream

The ellipsoid pairs are reported as google-deepmind/mujoco#3670, with PRs to follow.

### Found, not fixed

- Margin and gap on mesh and height-field pairs are still refused. MyoAssist's myoLeg22 and myoLeg26
  have margin on 14 plane-mesh, 14 hfield-mesh and 315 mesh-mesh pairs. `plane_convex` would need
  the margin passed in to pick vertices within it; the mesh and height-field functions are
  penetration-only SAT and would need a different algorithm.
- `jit(mjx.step)` on myoLeg22 compiles 402 collision groups, 340 of them mesh-mesh between the left
  and right legs (meshes of up to 5432 vertices). Without the mesh-mesh pairs it compiles in 76 s,
  measured with the margins zeroed, since `put_model` refuses them.
- The `build` workflow still fails at "Install MJX" (jax 0.4.34 from MJX's requirements.txt against
  the wheel's `jax==0.7.2`), so MJX's tests run locally only.

## v3.3.3+son5.1

### Fixed (MJX): sphere and capsule contacts with an ellipsoid

`sphere_ellipsoid` and `capsule_ellipsoid` ran ten gradient-descent steps on a clearance function
from the midpoint of the two geom centers, with an approximate ellipsoid SDF. On myolegs' foot geoms
(a capsule of radius 20 mm against an ellipsoid of semi-axes 85, 35, 40 mm), at 102 random poses with
distances from -20 to 10 mm, the distance was off by up to 25 mm and 12 contacts were missed or
invented. The exact SDF alone did not help (up to 22 mm); the optimizer was the main error. The code
is upstream's and unchanged up to 3.15.0, which has the same error.

- The closest point on an ellipsoid is exact: Newton's method on Eberly's (2013) secular equation in
  the form `1/|p(t)| = 1`, concave in `t`, so iterates started left of the root rise monotonically to
  it, as `mjc_ellipsoidOutside` does in C. 14 iterations; points outside converge in about 5.
- A sphere is exact in every regime, the sphere center inside the ellipsoid included.
- A capsule takes the axis point where the signed distance, convex along the axis, is least, found by
  30 bisections on the sign of its derivative. This is exact while the axis is outside the
  ellipsoid, that is, for penetration up to the radius. With the axis inside, it is the depth of the
  deepest axis point: equal to the penetration depth C computes (EPA) when that point is a smooth
  minimum, and less than it when the point lies on the plane of the smallest semi-axis (40 of 395
  deep random poses).
- Against the exact solution, distance, position and normal agree to 1e-10 m in float64 and 2e-8 m in
  float32, and reverse-mode gradients match central differences to 6e-8. The contact position is the
  midpoint of the two surface points, as in C.
- On the CPU in float32, 4096 capsule-ellipsoid pairs take 6.6 ms against 7.4 ms before, and compile
  in half the time. The cost on a GPU, where the 30 sequential bisections may matter more, is not
  measured.

The other pairs that use the same optimizer (`sphere_cylinder`, `capsule_cylinder`,
`ellipsoid_ellipsoid`, `ellipsoid_cylinder`, `cylinder_cylinder`) are unchanged.

### Fixed: GJK returned a stale distance from a tetrahedron simplex (upstream 50fa598d)

A backport of upstream's one-line fix in 3.3.5. When the simplex of a sphere or capsule, which GJK
shrinks to its center or axis, became a tetrahedron containing the origin, `gjk()` kept the previous
iterate's distance. A sphere center or capsule axis inside an ellipsoid, cylinder, mesh or height
field was then reported shallower than the radius, or not in contact at all, by both `mj_collision`
and `mj_geomDistance`. On 177 random capsule-ellipsoid poses within 10 mm of contact, the axis was
inside the ellipsoid in 75, and every one of those was reported shallower than the radius. With
`ccd_tolerance="1e-8"` the contact position was also off by up to 23 mm in 6 of 104 shallow contacts
(none at the default 1e-6).

The fix changes no contact-free result. A myolegs (hanging) and a MyoFullBody rollout from their
keyframes, 2000 steps with contacts on and zero control, are bit-identical with and without it.

### Verification

- `collision_driver_test.py`: the new tests compare 20 random sphere and capsule poses with C and a
  sphere and a capsule centered in an ellipsoid with the exact result; they fail on son5.0's MJX.
  41 tests pass in float32. In float64, `test_ellipsoid_cylinder` fails before and after this release.
- The whole MJX suite, 417 tests, passes (son5.0's 413 and the four new ones).
- Two C tests, capsules whose axis crosses an ellipsoid, fail without the GJK fix; the C suite, 803
  tests, passes with it.
- The MJX side is upstream main's change for the same two functions, checked there against PyPI's
  mujoco 3.15.0.

### Found, not fixed

- The `build` workflow has failed at "Install MJX" since son5.0: it installs MJX's
  `requirements.txt` (jax 0.4.34) and then the wheel, which pins `jax==0.7.2`, with `--no-index`.
  MJX's tests have therefore not run in CI since then; the ones above were run locally.
- MuJoCo C (3.3.3 and 3.15.0) reports no contact for a sphere or capsule whose center is exactly the
  center of an ellipsoid.

## v3.3.3+son5.0

### Added: MJX runs `compliant_mtu`, `millard_mtu` and `hyfydy_mtu`

`mjx.put_model` refused all three. They now run under MJX's JAX backend with the C equations,
solvers and iteration caps, in `mjx/mujoco/mjx/_src/muscle_mtu.py`, which follows
`engine_muscle_mtu.c` and the `compliant_mtu` part of `engine_util_misc.c` function by function.
`doc/muscle_mtu.rst` has a new MJX section.

- `GainType` gains `COMPLIANT_MTU`, `MILLARD_MTU` and `HYFYDY_MTU`, and `Option` gains `cmtu_iter`.
- `fwd_actuation` writes the fiber velocity `(l_ce* - l_ce)/dt` into the first activation slot and
  sets the force to `-F_mtu`; the activation dynamics drive the last slot, `actearly` included.
- `implicitfast` adds the rigid-tendon `d(force)/d(velocity)`, as `mjd_actuator_vel` does.
- `mjx.mtu_muscle_equilibrate` and `mjx.compliant_muscle_equilibrate` mirror the C functions, with
  the iteration caps as arguments.
- The Millard tables, which the engine keeps in a process-wide cache that `mjData.muscle_curve`
  points into, are copied onto the device by `put_model`, one row per distinct curve. The curve
  shape slots are therefore frozen at `put_model`, as C freezes them at `mj_resetData`; the
  mechanical slots are read every step. `put_model` checks the tables it read against
  `mju_millardCurve`, since `mjCurveTable` is not public.
- `make_data` seeds the fiber at `optimal_fiber_length`, as `mj_resetData` does; before, it would
  have left a zero fiber length. The muscle outputs travel through `put_data`/`get_data` in
  `Data._impl`.
- Loops run their C cap every time, with converged elements frozen, and branches are evaluated
  everywhere with each fed an input it is defined at, so reverse-mode gradients are finite.
  Gradients are the implicit-function derivative of the root the solver returns, not the
  derivative of its iterations, which is zero wherever the warm start already met the tolerance.

Three generic MJX paths assumed one activation variable per actuator, which no upstream actuator
MJX supports ever broke: the `act_dot` scan, `_next_activation` and the `implicitfast` derivative.
They now index each actuator's last variable. The rewritten `_next_activation` spells
`jp.clip(..., min=...)`, since jax 0.10 no longer accepts `a_min`; the two other `a_min` calls in
MJX, in `collision_driver.py` and `passive.py`, are untouched, so MJX still needs a jax that
accepts it (0.7.2 does, with a deprecation warning).

### Fixed (MJX): `muscle_dynamics` had NaN gradients at the default smoothing width

`support.muscle_dynamics_timescale` evaluated `dctrl / smoothing_width` on every element and
selected the hard switch with `jp.where` when the width is 0, the default. The unselected division
has an infinite derivative, so the reverse-mode gradient of any `dyntype="muscle"` actuator with
respect to `act` or `ctrl` was NaN. The smooth branch now sees a width of 1 wherever the hard
switch is selected. Values are unchanged.

### Fixed: `mjOption.cmtu_iter` is reachable from Python

It was missing from `MJOPTION_INTS` in `mjxmacro.h`, so `MjModel.opt` did not bind it (see "Found,
not fixed" under `son4.0a7`). MJX needs it for the Millard/Hyfydy iteration cap. `cmtu_integrator`
stays unbound: it is an enum the engine never reads. `doc/muscle_mtu.rst` gave the default as 12;
it is 20, and 12 is what zero or less means.

### Added (MJX): the constraint solve in double precision, `put_model(solver_dtype=jnp.float64)`

With `jax_enable_x64` off and the model and data in float32, `solver.solve` alone runs in float64:
it enables x64 locally, casts its inputs up and its outputs back, so nothing outside it can become
float64 and code around MJX keeps its dtypes. A `custom_vjp` runs the backward pass under the same
local x64, so the widened solve differentiates exactly where the plain one does, which is in reverse
mode with `opt.iterations = 1`; it does not support forward mode.

This is what single-precision MJX needs on these musculoskeletal models: a float32 rollout
diverges, and doing only the constraint solve in float64 keeps it with an all-float64 one (see
Precision). The first attempt, x64 on globally with float32 data, leaked float64 into about a
quarter of MJX's other operations, because MJX allocates without a dtype and numpy constants are
float64; enabling x64 only inside the solve avoids that by construction.

### Fixed (MJX): an elliptic cone with no cone rows

The elliptic-cone bookkeeping in the constraint update and the line search built index arrays
with `jp.array` of Python lists, and an empty list makes a float array, which cannot index. A model
with `cone="elliptic"`, constraints and no contact of `condim` > 1 failed to trace (myolegs, with
its contacts removed). The arrays are now `dtype=int`. The same code's fixed six-wide slices still
need at least three constraint rows; that is upstream's and unchanged.

### Added: what musclemimic calls in mujoco 3.4 and MJX 3.4

musclemimic's MJX environments (JAX backend) run on this build: `MjxMyoFullBody` and the imitation
environment built from `fullbody/conf_fullbody.yaml` with `mjx_backend: jax` reset and step. Three
things stood in the way, none needing a rebase onto 3.4:

- MJX 3.4 renamed `backend_impl` to `impl` and gave `make_data`/`put_data` Warp-only `nconmax`,
  `naconmax` and `njmax`. `put_model`, `put_data` and `make_data` accept both spellings, refuse
  them disagreeing, and accept and ignore the Warp-only budgets as 3.4 does off Warp.
- mujoco 3.4 deletes spec elements with `spec.delete(element)`; this build deletes most elements
  with `element.delete()` and bodies and defaults with `spec.detach_body`/`spec.detach_default`.
  `MjSpec.delete` now dispatches to those. On the test model a body and a sensor deleted this way
  compile to the same counts as in 3.4.0.
- MJX 3.3.3 raised on tendon armature and tendon bias with a sparse mass matrix, which MyoFullBody
  (nv = 88) gets by default. Both are backported from MJX 3.4: the armature's dense `J^T A J`
  scattered into the sparse `qM` in its own order, and the bias, which is dense anyway.

Warp is not covered: musclemimic's shipped training configs set `mjx_backend: warp`, and this build
has no Warp backend, and the muscle-tendon gains exist only in C and in MJX's JAX backend.
`musclemimic-models` requires `mujoco>=3.4.0`, so an unconstrained install replaces this wheel with
PyPI's; pin it (for uv, `override-dependencies`).

### Changed: versions, pins and the release

- mujoco and mujoco-mjx are both `3.3.3+son5.0`. mujoco-mjx requires `mujoco==3.3.3+son5.0`, which
  has the bindings it reads, and `jax==0.7.2`/`jaxlib==0.7.2`, what musclemimic pins and what it is
  tested on. The remaining `jp.clip(a_min=...)` calls in MJX (`collision_driver.py`, `passive.py`)
  are `min=` now, so 0.7.2 no longer warns.
- The release workflow builds the pure-Python mjx wheel and attaches it, and refuses a tag that
  does not match `mjx/pyproject.toml` as it already refused one that does not match
  `python/pyproject.toml`.

### Precision

Measured on three leg models converted from OpenSim to each gain, MyoAssist's myoLeg22 and myoLeg26
and myo_sim's myolegs, started from their keyframes with the `mjEQ_JOINT` dependent coordinates
derived from their polynomials and the fibres equilibrated, with gravity off, and with sensors and
contacts removed because MJX 3.3.3 cannot run them. The rollouts are 2000 steps driving one muscle
group at a time.

- **The muscles are accurate in float32.** At the rollouts' states MJX's float32 forward pass gives
  the engine's muscle force to within 1e-4 `F_max`.
- **One float32 forward pass is accurate too**: at the starting state and 100 steps in, `qacc_smooth`,
  the constraint force and `qacc` all agree with float64 to 1e-4 relative, on every model.
- **A float32 rollout is not.** On myoLeg22 and myoLeg26 it went to NaN within 1000 steps for all
  three gains, and at step 100 its force was already 0.13 to 11 `F_max` off an all-float64 run. On
  myolegs it stayed finite but 1e-2 to 7e-2 `F_max` off. It is not the muscle-tendon gains: myoLeg26
  with upstream `muscle` actuators is 2.7 off in `qpos` at step 100.
- **With `solver_dtype=float64` all nine stay finite for 2000 steps** and within 2.8e-5 `F_max` and
  6.2e-5 in `qpos` of all-float64 at step 100. Past that the gap grows as the dynamics diverge; MJX
  in float64 and the C engine part by as much themselves.
- Why a per-step difference of 1e-4 grows in float32 and not with the solve in float64 is not
  established. It is not the iteration cap: the solve takes one or two iterations in every
  precision. These models couple light via-point bodies to the joints through joint equalities,
  and their mass matrix has condition number 4.4e5 (myoLeg22/26) against 8e2 (myolegs).

MJX in float64 and the C engine agree to 1e-14 on myolegs and part by 1e-2 in `qpos` within 100
steps on myoLeg22 and myoLeg26, in the constraint solve; upstream MJX 3.4.0 differs from C the same
way on MyoFullBody, so that is MJX's and not the muscles'.

### Verification

- `mjx/mujoco/mjx/_src/muscle_mtu_test.py`, 64 tests, in double precision against the C engine:
  `fwd_actuation` on eleven muscle variants (pennated, unpennated, overridden shapes, declared and
  ratio-rigid tendons, a zero-length rigid tendon, Song's own parallel element) to 1e-9; 600 steps
  of a model mixing all three gains with filter, filterexact and `actearly` actuators, under Euler,
  RK4 and implicitfast, to 1e-7; equilibration over slack, clamped and stretched poses at three
  activations to 1e-9; the tables against `mju_millardCurve` to 1e-12; gradients against finite
  differences of the exact root to 1e-6 (1e-5 for Song, whose step stops 1e-9 m from its root).
- `solver_precision_test.py`, 6 tests: the widened solve is the float64 solve of the same float32
  inputs, nothing outside the solve is float64, gradients with one iteration, vmap, refusals.
- Regression tests for the elliptic cone with no cone rows (`solver_test.py`), the sparse tendon
  armature and bias against C on a branching tree (`smooth_test.py`), the 3.4 signatures
  (`io_test.py`) and `MjSpec.delete` (`python/mujoco/specs_test.py`).
- The whole MJX suite, 413 tests, passes, as do the Python binding tests (146).
- On the three converted leg models in all three gains: MJX on the engine's states along a
  rollout (path length to 1e-10, force and fibre to 1e-9), equilibration from the starting pose
  (1e-9), the float32 muscles (1e-4 `F_max`), 200-step rollouts with the solve in float64 against
  all-float64 (1e-3), and the float32 divergence pinned so that its disappearance is noticed.
- musclemimic's imitation environment (`fullbody/conf_fullbody.yaml`, `mjx_backend: jax`, two
  environments, a synthetic trajectory in place of the gated datasets) resets and runs ten
  vectorized steps on this build with no patches, observations finite.

## v3.3.3+son4.0a10 — alpha

Supersedes `son4.0a9`, which is left as an unpublished draft: its Linux and Windows wheels still
refuse some of the stiff tendon curves `a9` set out to accept. Everything else in `a9` is in this
release unchanged, so read the `a9` notes below as part of this one.

### Fixed: stiff tendon curves were still refused on x86 and MSVC builds

`son4.0a9`'s relaxed slope check compared each end slope with a tolerance relative to that end's
declared slope. A tendon's start slope is declared 0, so the check there stayed an absolute 1e-8.
Without fused multiply-adds, which x86 GCC/Clang and MSVC do not emit by default and arm64 Clang
does, the computed start slope of the stiff tendon (3.4e-4, 3650) is off by up to 2.4e-8, and 60 of
the 380 shapes were still refused there. The Linux and Windows `a9` wheels refuse them; the macOS
one does not. CI caught it on every Linux and Windows job. Both ends are now judged against the
curve's slope scale, the largest of 1, the end slopes and every section's chord. The full suite
passes built both ways, including with `-ffp-contract=off` on arm64, which reproduces the x86
failure without the fix.

## v3.3.3+son4.0a9 — alpha

### Added: a declared rigid tendon, OpenSim's `ignore_tendon_compliance`

The rigid-tendon path was chosen by one rule, `l_slack < 0.05 l_opt`. A muscle whose source
declares `ignore_tendon_compliance` but whose MJCF path is longer than the source's needs a longer
`l_slack` to match the path: 14-43 mm for ARMS's four lumbricals and MoBL-ARMS's PECM1, 0.12-0.71
`l_opt`. That crosses the rule, and the muscle got a compliant tendon it was never declared to have.
`ignore_tendon_compliance` is a real property of `Millard2012EquilibriumMuscle` and was missing from
the 1:1 map.

- `millard_mtu` and `hyfydy_mtu`: `gainprm[19]`, formerly reserved, is `ignore_tendon_compliance`.
  0 keeps the ratio rule and 1 selects the existing rigid-tendon path at any slack length. Anything
  else fails the load. `hyfydy_mtu` still refuses every curve-shape slot and accepts this one.
- `compliant_mtu`: `gainprm[11]`, formerly required to be 0, is the same flag. The tendon is rigid
  when it is 1 or when the ratio rule says so.

Where the ratio rule already makes a tendon rigid, declaring it rigid changes nothing, bit for bit,
for all three models. With the flag at 1 and `l_slack` = 0.7 `l_opt`, the fiber is OpenSim's
`hypot(l_MTU - l_slack, h)` and the force is its rigid-tendon fiber force, and the rigid velocity
derivative matches a central difference.

A declared rigid tendon also makes slots 25 and 28 (`concentric_slope_at_vmax`,
`eccentric_slope_at_vmax`) required to be 0, even without damping. OpenSim runs its damped-model
branch, which sets both to 0, unless the tendon is compliant and the fiber undamped
(`Millard2012EquilibriumMuscle.cpp`, `buildMuscle`). `son4.0a8` refused them only with damping.

### Fixed: a taut rigid tendon was sometimes treated as buckled

On the rigid-tendon path `l_T = l_MTU - l_ce cos(phi)` equals `l_slack` on a taut tendon only up
to roundoff. OpenSim calls the tendon buckled when `l_T < l_slack - SignificantReal`; the port
compared with `l_slack` itself, so roundoff decided. A "buckled" tendon zeroes the fiber velocity,
which drops `f_V` and the fiber damping from the force and zeroes `d(force)/d(velocity)`. On a
pennated rigid tendon (pennation 0.15, `l_slack` 0.033 `l_opt`), `son4.0a8` did this at 383 of 1000
taut, moving points, for `millard_mtu` and `hyfydy_mtu` alike. It now does it at none. The margin is
SimTK's `SignificantReal`, `eps^(7/8)` = 2.0e-14. This dates back to `son4.0`.

**What moves against `son4.0a8`.** With every flag at 0, the same grid run against `a8` and this
build agrees bit for bit for `compliant_mtu` (Geyer's soleus, an own-parallel-element fit, a
ratio-rigid tendon) and for `millard_mtu` and `hyfydy_mtu` on compliant tendons and on an unpennated
ratio-rigid tendon. The only differences are on the pennated ratio-rigid tendon, 181 of 652
outputs per model. They are in the force at points where the path moves, and in the velocity
derivative at every point that was being called buckled. Fiber lengths are unchanged.

### Fixed: stiff, short tendon curves refused by a roundoff check

The Bézier port cross-checks each curve's end slopes against the declared ones. The check was
absolute, 1e-8. A tendon with `strain_at_one_norm_force` 3.4e-4 and `stiffness_at_one_norm_force`
3650 (ARMS's `LU_RB5`, when its tendon is kept at the source's stiffness) takes its slopes from
control points 1e-4 apart at `x ~ 1`. The slopes then carry 1e-11 relative roundoff, 3.4e-7 on 3650.
Over 380 toe forces and curvinesses at that strain and stiffness, 129 were refused with
`dydx1 anchor mismatch`, and OpenSim 4.6 builds all 380. The check is now relative,
`1e-8 * max(1, |slope|)`, which still catches the O(1) error a wrong control point makes. All 380
build. The baked table reproduces the exact Bézier to 4.2e-8 over 4001 points per shape, and
OpenSim 4.6's own evaluation to 4.1e-10 over 39.

## v3.3.3+son4.0a8 — alpha

**`compliant_mtu` dynamics change in this release.** Every `compliant_mtu` model moves, not just
the ones declaring new parameters: the force balance, the slack tendon and the fiber clamps are all
different, so results are not comparable with `son4.0a7` or earlier. An isometric equilibrium with
the tendon taut and the fiber above the buffer element's rest length solves the same equation as
before. Over the grid below, the 34,674 such points agree with `a7` to 1.9e-6 `F_max` in force,
except two where `a7`'s solve had not converged (trunk `IL_L2_r` at activation 0.5 and 1, residual
-9e-3 and -4e-2), which move by up to 3.8% `F_max`. Of the other 24,741 points, where the tendon is
slack or the buffer element engaged, 21,286 move by more than 1e-3 `l_opt` in fiber length.

### Changed: `compliant_mtu` solves Geyer & Herr 2010's force balance

Geyer & Herr 2010, Appendix II: `F_m = F_se = F_ce + F_pe - F_be`, with
`F_pe = F_max f_pe f_v(v_ce)` and a buffer element
`F_be = F_max ((l_min - l_ce)/(l_opt eps_be))^2`, `l_min = l_opt - w`, `eps_be = w/2`. Song's
reference implementation writes the same thing as `f_vce0 = (f_se0 + f_be0)/(f_pe0 + A*f_lce0)`
(`seungmoon_muscle.py`). The engine solved `f_se = f_pe + A f_l f_v` instead, which is Geyer
2003's structure, with no buffer element. The residual is now

    R = f_se + f_be - f_v(v0) (f_pe + A f_l)

in both the backward-Euler step and the isometric solve (`f_v(0) = 1`). The parallel element keeps
`son4.0a7`'s own slack length and reference strain when declared; the buffer element stays tied to
`W`, as the paper fixes it.

Four things follow from it:

- **The tendon may go slack.** Both solvers used to project every iterate back to
  `l_se >= l_slack`, which held a slack tendon at exactly its slack length: a tendon pushing, and
  the only thing that kept an active fiber from collapsing. The projection is gone, and the buffer
  element does that job, as the paper's Fig. 6 says it does.
- **The 1 mm fiber clamps are gone.** They were absolute, and a `W` fitted to a Thelen-sourced curve
  (0.92-0.99) puts the buffer element's rest length at 0.01-0.08 `l_opt`, under a millimetre on a
  short fiber, so the clamp could engage before the element. Fiber and tendon lengths are now only
  kept positive, by a guard at `1e-6 l_opt`.
- **The rigid-tendon path** computes `F_max max(0, f_v (f_pe + A f_l) - f_be)`, floored at zero
  because a tendon cannot push, as `millard_mtu`'s rigid path saturates. Its velocity derivative for
  the implicit integrators now includes the parallel element, and is zero on the floor.
- **Equilibration brackets the root.** With the projection gone, the damped Newton it used failed
  on 18% of a grid (below) from `mj_resetData`'s seed, the peak of the active curve, where a slack
  tendon leaves no slope. It is now a safeguarded Newton-bisection on
  `[1e-6 l_opt, l_mtu - 1e-6 l_opt]`, as `mju_mtuMuscleEquilibrate` became in `a6`. It no longer
  reads the incoming fiber length, so it is a function of the pose and the activation. At zero
  activation, where a band of lengths is in equilibrium, it returns the shortest, which is the limit
  as the activation goes to zero.

**Measured** over 239 muscles: jinsimul's 232 own-parallel-element fits (arm, full body, h1622,
myoleg26, trunk; `W` 0.59-1.002) and Geyer & Herr's seven Table II muscles. The residual is rebuilt
outside the engine from the public curves.

- Isometric, 59,415 points (51 path lengths from `0.3 l_slack` to `l_slack + 2.2 l_opt`, five
  activations): no failures. 54,478 converge; 4,479 sit on the guard, all with `W >= 0.93` and a
  non-zero activation, or `W >= 1`, which is where the buffer element cannot hold the fiber (at full
  activation the threshold is `W` = 0.929). 458 sit at the path's length. The damped Newton that
  preceded the bracket, same residual, failed on 10,573 of these points.
- Backward-Euler step, 139,800 points (10 path lengths, five activations, six starting fibers from
  0.2 to 1.7 `l_opt`, `dt` 1e-3 and 2e-3): 60,380 of 60,382 converge where the step ends at
  `v0 <= 1`, and 79,414 of 79,418 where it ends past `v_max`. The six failures are the region-3
  question below.

A fiber carrying no force now keeps its length. At zero activation, with the tendon slack and both
the parallel and buffer elements unloaded, nothing acts on it, and the stepping solve leaves it
where it is instead of snapping it to `l_mtu - l_slack`.

Where the buffer element cannot hold an active fiber (`W` above the threshold, or `W >= 1`), the
fiber goes to the guard and the slack tendon carries nothing. jinsimul lets the fiber go slightly
negative instead (decision 0024 reports -0.3% `l_opt` at the collapse edge); this build does not.

### Measured, not decided: `f_v` region 3 (E3)

Song's `f_v` has a third region past `v_max`, `N + 100 (v0 - 1)`, and with the 2010 residual it now
scales the parallel element too. It stays in this release. What it does, on the grid above and
compared with a build that extends region 2 past `v0 = 1` instead (jinsimul's choice, which
saturates at `N + 0.027`):

| | region 3 (this release) | region 2 extended |
| --- | --- | --- |
| step failures, 139,800 points | 6 | 0 |
| worst unconverged residual | 1,278 `F_max` | none |
| failures within 0.01 of `v0 = 1` | 2, residual up to 3.1e-4 | none |
| failures deep in region 3 | 4, at `v0` 7-18, residual 467-1,278 `F_max` | none |
| converged steps past `v_max`: median / p99 `v0` | 7.6 / 98 | 37 / 159 |

The four deep failures are the fiber landing where `f_v` is in the hundreds to thousands, a force no
tendon balances. That matches jinsimul's measurement on its own engine (6 of 600 points at
1,391 `F_max`). The kink at `v0 = 1`, where the slope jumps from 0.026 to 100, also shows up in an
ordinary driven trajectory. A 50 kg mass held by Geyer's soleus under a sinusoidal activation
spends 39 of 4,000 steps within 0.01 of the kink, and the step stops at up to 1.4e-5 there, over its
1e-5 tolerance. With region 2 extended the same trajectory stays within 1e-5.

The grid starts fibers far from equilibrium on purpose, so its `v0` distribution says how each
curve behaves when pushed, not how often real motion goes past `v_max`.

### Changed: `millard_mtu` refuses at-vmax slopes under fiber damping (E5)

With fiber damping, OpenSim's `Millard2012EquilibriumMuscle` sets `concentric_slope_at_vmax` and
`eccentric_slope_at_vmax` to 0 whatever the file declares, and only logs it
(`Millard2012EquilibriumMuscle.cpp`, the damped-model branch of `buildMuscle`). The engine used them
as given. A non-zero slot 25 or 28 with damping now fails the load; the undamped model (negative
`gainprm[5]`) keeps them. None of 312 muscles fitted from OpenSim models sets either.
jinsimul's Thelen fit (0.234 and 0.126) is the case this catches.

### Fixed: Python `MjSpec` exposes all 32 `gainprm` slots

The spec bindings are generated from `introspect/structs.py`, which still declared `mjsActuator`'s
`gainprm` and `biasprm` with upstream's 10 slots after this fork widened `mjNGAIN` to 32. From
Python, `E_REF_PE` (slot 10) and the Millard curve slots could not be set through `MjSpec`. Both are
now the full 32-wide array. A shorter sequence, such as upstream's ten, replaces the whole array and
zero-fills the rest, through the property and through `add_actuator`. More than 32 is an error. Code
written against upstream's size keeps working, which matters for `myoassist_mjspec` and
`myochallenge2026`, which pass ten.

`specs_test.py::test_actuator_shortname`, which failed on `son4.0a7`, passes again, but the slot
count was not why it failed. It passes `gainprm` as a `(10, 1)` array, and NumPy 2.4 made `float()`
of a one-element array an error, which broke the element-by-element conversion. The `gainprm` and
`biasprm` setters now read the value as an Eigen vector, which accepts that shape. The other array
keyword arguments still convert element by element, so a `(n, 1)` array passed to one of them fails
the same way on NumPy 2.4.

### Found, not fixed

`mjOption.cmtu_iter`, `mjOption.cmtu_integrator` and the `mjtCMTUIntegrator` enum are not reachable
from Python, through `MjModel.opt` or `MjSpec.option`. `mjxmacro.h` and the introspect data predate
them. They can still be set in MJCF.

## v3.3.3+son4.0a7 — alpha

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

No existing model is affected. All 3,565 `compliant_mtu` declarations across 73 MJCF files checked,
including MyoAssist compliant models, give exactly nine values. The 658 actuators in the files that compile on their own resolve slots 9-31
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
transition of 3.5e-9. None of 312 muscles fitted from OpenSim models violates any of the new
checks.

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
in each case at the foot of the curve. For 113 Thelen-sourced fits (`min` 0.05, transitions from
0.0975) the median is 1.5e-7 and the worst 7.5e-5. `doc/muscle_mtu.rst` now carries these numbers in place of the unqualified 7e-9.

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
