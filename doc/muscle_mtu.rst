=====================
Muscle-tendon units
=====================

.. _mtuOverview:

Overview
--------

Alongside MuJoCo's built-in ``muscle`` gain, this fork ships three Hill-type muscle-tendon units
with a **compliant tendon** and a fiber-length state:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``gaintype``
     - Model
   * - ``compliant_mtu``
     - Seungmoon Song's compliant MTU (Geyer/Song neuromuscular model).
   * - ``millard_mtu``
     - OpenSim ``Millard2012EquilibriumMuscle``, damped variant, with OpenSim's own quintic-Bézier
       curves.
   * - ``hyfydy_mtu``
     - Hyfydy's ``muscle_force_m2012fast``. The Hyfydy User Manual states that it implements
       Millard et al. (2013) but replaces the Bézier splines with polynomials, so its curves
       "differ slightly from the curves used in the OpenSim implementation".

All three carry the fiber length ``l_ce`` as a per-actuator state and advance it by one implicit
(backward-Euler) step per ``mj_step``, solving

.. math::

   R(l_{ce}) = f_T\!\left(\frac{l_T}{l_{slack}}\right)
             - \cos\phi \left[ a\, f_L(\hat l)\, f_V(\hat v) + f_P(\hat l) + \beta \hat v \right] = 0

with :math:`\hat l = l_{ce}/l_{opt}`, :math:`\hat v = \dot l_{ce}/(v_{max} l_{opt})`,
:math:`l_T = l_{MTU} - l_{ce}\cos\phi`, and fixed-width pennation
:math:`\phi = \arcsin(h/l_{ce})`, :math:`h = l_{opt}\sin\phi_{opt}`. The actuator force is the
tendon force :math:`F = F_{max} f_T`.

``millard_mtu`` and ``hyfydy_mtu`` differ **only** in the four normalized curves. They share the
residual, the analytic Jacobian, the pennation algebra and the solver, so switching between them
is a one-attribute change rather than a different code path.

State is exposed in :ref:`mjData` as ``muscle_l_ce``, ``muscle_l_se``, ``muscle_v_ce`` and
``muscle_F_mtu``, each ``nu x 1``.

.. _mtuParameters:

Parameters
----------

``millard_mtu`` and ``hyfydy_mtu`` read all 32 ``gainprm`` slots. The layout is a 1:1 map of the
properties an OpenSim ``Millard2012EquilibriumMuscle`` (or a Hyfydy ``.hfd`` muscle) declares, in
the same units, so a model's parameters transfer without reinterpretation.

**Zero in any slot means that property's OpenSim default.** A hand-written model therefore fills
in the first six slots and leaves the rest blank; a converted ``.osim`` fills in whatever it
overrode. The rule is complete rather than merely convenient: the only properties whose OpenSim
default is itself zero are ones for which zero is also the value you would want, and
``strain_at_zero_force`` — the one property real models set to something neither zero nor positive
— is negative and so still expressible.

.. list-table::
   :header-rows: 1
   :widths: 8 34 34 24

   * - Slot
     - OpenSim property
     - Hyfydy ``.hfd``
     - Default
   * - 0
     - ``max_isometric_force``
     - ``max_isometric_force``
     - — (required)
   * - 1
     - ``optimal_fiber_length``
     - ``optimal_fiber_length``
     - — (required)
   * - 2
     - ``tendon_slack_length``
     - ``tendon_slack_length``
     - — (required)
   * - 3
     - ``max_contraction_velocity``
     - ``v_max``
     - 10
   * - 4
     - ``pennation_angle_at_optimal``
     - ``pennation_angle``
     - 0
   * - 5
     - ``fiber_damping``
     - ``xi``
     - 0.1
   * - 6
     - fiber Newton residual tolerance
     - same
     - 1e-9
   * - 7
     - reserved, must be 0
     -
     -
   * - 8-11
     - ``ActiveForceLengthCurve``: ``min_norm_active_fiber_length``,
       ``transition_norm_fiber_length``, ``max_norm_active_fiber_length``,
       ``shallow_ascending_slope``
     - n/a
     - 0.4441, 0.73, 1.8123, 0.8616
   * - 12-13
     - reserved, must be 0 (slot 12 is ``minimum_value``, which the damped Millard model forces
       to 0)
     -
     -
   * - 14-18
     - ``FiberForceLengthCurve``: ``strain_at_zero_force``, ``strain_at_one_norm_force``,
       ``stiffness_at_low_force``, ``stiffness_at_one_norm_force``, ``curviness``
     - n/a
     - 0, 0.7, 0.2, 2/(e1-e0), 0.75
   * - 19
     - reserved, must be 0
     -
     -
   * - 20-23
     - ``TendonForceLengthCurve``: ``strain_at_one_norm_force``,
       ``stiffness_at_one_norm_force``, ``norm_force_at_toe_end``, ``curviness``
     - n/a
     - 0.049, 1.375/e1, 2/3, 0.5
   * - 24-31
     - ``ForceVelocityCurve``: ``max_eccentric_velocity_force_multiplier``,
       ``concentric_slope_at_vmax``, ``concentric_slope_near_vmax``, ``isometric_slope``,
       ``eccentric_slope_at_vmax``, ``eccentric_slope_near_vmax``, ``concentric_curviness``,
       ``eccentric_curviness``
     - n/a
     - 1.4, 0, 0.25, 5, 0, 0.15, 0.6, 0.9

Slots 8-31 are curve **shape** parameters and apply to ``millard_mtu`` only. Hyfydy's curves are
published polynomials with no per-muscle shape, so ``hyfydy_mtu`` rejects a non-zero value there
rather than silently ignoring one that was meant to have an effect.

Notes on individual parameters:

``max_contraction_velocity`` (slot 3)
   In **optimal fiber lengths per second**, not m/s: the normalized fiber velocity is
   :math:`\dot l_{ce}/(v_{max} l_{opt})`. This is OpenSim's convention, Hyfydy's, and MuJoCo's own.

``fiber_damping`` (slot 5)
   Defaults to OpenSim's 0.1. Pass a **negative** value to ask for exactly zero, since 0 itself
   means "take the default". With the damped Millard curves the at-``vmax`` force-velocity slopes
   are zero, so outside :math:`|\hat v| \le 1` the force-velocity term contributes nothing to
   :math:`dR/dl_{ce}` and the damping is the only thing keeping the Newton Jacobian bounded away
   from zero. It is a well-posedness condition, not a decoration. It is also what lets an inactive
   muscle be a damper: the term is not scaled by activation.

Activation dynamics are MuJoCo's own: set ``dyntype="muscle"`` with
``dynprm="<activation_time_constant> <deactivation_time_constant>"`` (OpenSim's defaults are
0.01 and 0.04). An actuator with no activation state is driven by ``ctrl`` directly, which is what
``dyntype="none"`` means for a muscle: excitation applied without a lag.

.. code-block:: xml

   <general name="soleus_r" tendon="soleus_r" gaintype="millard_mtu" biastype="none"
            dyntype="muscle" dynprm="0.01 0.04" ctrllimited="true" ctrlrange="0 1"
            gainprm="3549 0.05 0.25 10 0.4887 0.1"/>

.. _mtuCurves:

Curves and where they come from
-------------------------------

OpenSim's four Millard curves are **parametric**: :math:`x(u)` and :math:`y(u)` are separate
quintic Béziers, so evaluating :math:`y` at a given :math:`x` means first solving :math:`x(u)=x`
for :math:`u` by Newton iteration. Doing that inside the per-muscle fiber Newton would nest Newton
inside Newton with data-dependent trip counts at both levels.

Instead, each curve is **baked**: sampled once onto a uniform 513-knot grid of
:math:`(y, dy/dx, d^2y/dx^2)` and evaluated at runtime as a quintic Hermite polynomial — an index,
a clamp and a polynomial, with no iteration and no section scan. Value and slope come from the
same interpolant, so the Jacobian differentiates exactly what the residual evaluates. Worst
deviation from the exact Bézier curve is below :math:`7\times10^{-9}` in normalized force.

The bake happens **lazily at runtime**, at the first ``mj_resetData`` that sees the muscle, into a
process-wide cache keyed by the shape parameters. Nothing is stored in ``mjModel``: a table written
into a model file could go stale against the parameters that produced it. Muscles that share a
curve shape share the table, so a model whose muscles differ only mechanically bakes four curves
in total, and one that tunes the passive curve per muscle (RajagopalLaiUhlrich2023, for instance)
bakes three plus one per distinct passive shape. ``mjData.muscle_curve`` holds the resolved
pointers, so the stepping path never touches the cache.

Hyfydy's curves need no bake at all: they are two cubics, two rationals and a quadratic, with no
transcendental, which makes ``hyfydy_mtu`` the cheapest of the three models.

``mju_millardCurve`` and ``mju_hyfydyCurve`` expose the curves directly, for checking a model
against OpenSim.

.. admonition:: Changing shape parameters at runtime
   :class: note

   The mechanical parameters (slots 0-7) are re-read every step, so writing to
   ``model.actuator_gainprm`` takes effect immediately — which is what randomizing muscle strength
   or fiber length in a training loop needs. The curve **shape** parameters (slots 8-31) are
   resolved to a baked table at ``mj_resetData``, so changing one takes effect at the next reset.

   Curve shapes come from the source ``.osim`` and are not the kind of thing a training loop
   varies per step; making them per-step would mean a cache lookup inside the stepping path, which
   costs more than the muscle solve itself.

.. _mtuNumerics:

Solver and edge cases
---------------------

The fiber Newton is warm-started from the previous step's ``l_ce``, uses an **analytic** Jacobian,
and stops on the residual, so a muscle in a smooth regime converges in one to three iterations. The
iteration cap is ``option/cmtu_iter`` (default 12).

**Fiber length clamp.** ``l_ce`` is clamped from below at
``max(min_norm_active_fiber_length * l_opt, h/sin(phi_max))``, ``phi_max = acos(0.1)``. Below the
active curve's left foot the active curve is identically zero, so nothing is lost, and the
pennation term :math:`1/\sqrt{1-(h/l_{ce})^2}` diverges at :math:`l_{ce}=h`. OpenSim clamps in the
same place for the same reason.

**Rigid-tendon fallback.** When ``tendon_slack_length < 0.05 * optimal_fiber_length`` the
series-elastic normalization is ill-conditioned for no physical gain, and the tendon is treated as
inextensible: ``l_ce`` follows the path algebraically and the fiber force is transmitted directly.
The Hyfydy manual documents no rigid-tendon variant — its tendon is always the compliant quadratic
— so for ``hyfydy_mtu`` this path evaluates Hyfydy's curves inside OpenSim's rigid-tendon
formulation. It exists so a short-tendon muscle degrades gracefully rather than stalling the
solver; a model that cares about Hyfydy parity should not be in this regime.

**Re-seeding after a reset or a teleport.** If the path length moved by something other than what
the reported path velocity predicts, the state was not reached by integration (a reset, a ``qpos``
edit, a static sweep) and there is no meaningful previous fiber length to difference against. The
muscle re-seeds from the isometric equilibrium instead of letting a phantom fiber velocity drive
the force.

**Smoothness.** The Millard force is a C1 function of the path length, which is what makes the
muscle usable under finite-difference derivatives. The Hyfydy force is only C0 where the fiber
leaves the active curve's support: its active force-length curve is published as a polynomial with
a hard cut, "0 for l <= r1", and the polynomial reaches zero at ``r1`` with slope 4.19 rather than
0. That slope discontinuity is Hyfydy's, and reproducing it is the point of shipping the model.
