"""Tests for the millard_mtu (OpenSim Millard2012EquilibriumMuscle) and hyfydy_mtu
(Hyfydy muscle_force_m2012fast) actuators.

These are the Python-level counterparts of test/engine/engine_muscle_mtu_test.cc, which
covers the curves themselves; what is checked here is the behaviour a model author sees:
that a muscle carries its load, that .osim curve overrides reach the force, that the state
stays finite and smooth, and that a reset is repeatable.

Run:  pytest test_osim_muscle.py -v
"""
import numpy as np
import pytest
import mujoco

# gainprm slots (see doc/muscle_mtu.rst). Zero in any slot means the OpenSim default, so a
# muscle with default curves needs only the first six.
FMAX, LOPT, LSLACK, VMAX, PENNATION, BETA, TOL = range(7)
AFL_MIN, AFL_TRANS, AFL_MAX, AFL_SLOPE = 8, 9, 10, 11
PFL_E0, PFL_E1, PFL_KLOW, PFL_KISO, PFL_CURV = 14, 15, 16, 17, 18
TFL_E1, TFL_KISO, TFL_FTOE, TFL_CURV = 20, 21, 22, 23
(FV_FMAXE, FV_DYDXC, FV_DYDXNEARC, FV_DYDXISO,
 FV_DYDXE, FV_DYDXNEARE, FV_CONCCURV, FV_ECCCURV) = range(24, 32)

_MODELS = ('millard_mtu', 'hyfydy_mtu')
_DT = 0.001
_LOAD_MASS = 20.0
_HANG = 0.30          # anchor height, so the path length at qpos=0 is 0.30 m


def _gainprm(overrides=None):
    """A default muscle, with `overrides` a {slot: value} dict. Slots are ints, so this takes a
    dict rather than keyword arguments."""
    prm = [0.0]*32
    prm[FMAX], prm[LOPT], prm[LSLACK] = 3000.0, 0.15, 0.15
    prm[VMAX], prm[PENNATION], prm[BETA] = 10.0, 0.15, 0.1
    for slot, value in (overrides or {}).items():
        prm[slot] = value
    return ' '.join(repr(v) for v in prm)


def _model_xml(gaintype, prm):
    return f"""
<mujoco>
  <option timestep="{_DT}" gravity="0 0 -9.81"/>
  <worldbody>
    <site name="anchor" pos="0 0 {_HANG}" size="0.005"/>
    <body name="load">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <site name="ins" size="0.005"/>
      <geom type="sphere" size="0.01" mass="{_LOAD_MASS}"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t"><site site="anchor"/><site site="ins"/></spatial>
  </tendon>
  <actuator>
    <general name="m" tendon="t" gaintype="{gaintype}" biastype="none"
             dyntype="muscle" dynprm="0.01 0.04" ctrllimited="true" ctrlrange="0 1"
             gainprm="{prm}"/>
  </actuator>
</mujoco>"""


def _settle(gaintype, prm, ctrl, steps=6000):
    model = mujoco.MjModel.from_xml_string(_model_xml(gaintype, prm))
    data = mujoco.MjData(model)
    data.ctrl[0] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return model, data


@pytest.mark.parametrize('gaintype', _MODELS)
@pytest.mark.parametrize('ctrl', [0.0, 0.5, 1.0])
def test_muscle_carries_the_load(gaintype, ctrl):
    """At rest the tendon force must equal the weight, whatever the activation."""
    _, data = _settle(gaintype, _gainprm(), ctrl)
    assert data.muscle_F_mtu[0] == pytest.approx(_LOAD_MASS*9.81, abs=1e-2)
    assert abs(data.qvel[0]) < 1e-4


@pytest.mark.parametrize('gaintype', _MODELS)
def test_activation_shortens_the_muscle(gaintype):
    """More activation must pull the load higher and shorten the fiber."""
    _, off = _settle(gaintype, _gainprm(), 0.0)
    _, on = _settle(gaintype, _gainprm(), 1.0)
    assert on.qpos[0] > off.qpos[0] + 0.05
    assert on.muscle_l_ce[0] < off.muscle_l_ce[0]


@pytest.mark.parametrize('gaintype', _MODELS)
def test_path_closes_through_pennation(gaintype):
    """l_MTU = l_ce cos(phi) + l_T, with fixed-width pennation."""
    model, data = _settle(gaintype, _gainprm(), 1.0, steps=2000)
    prm = model.actuator_gainprm[0]
    h = prm[LOPT]*np.sin(prm[PENNATION])
    l_ce = data.muscle_l_ce[0]
    cos_phi = np.sqrt(1.0 - (h/l_ce)**2)
    assert l_ce*cos_phi + data.muscle_l_se[0] == pytest.approx(data.actuator_length[0], abs=1e-12)


@pytest.mark.parametrize('gaintype', _MODELS)
def test_state_stays_finite_while_driven(gaintype):
    """A 1 Hz excitation must not produce a non-finite state or an unbounded fiber."""
    model = mujoco.MjModel.from_xml_string(_model_xml(gaintype, _gainprm()))
    data = mujoco.MjData(model)
    for i in range(4000):
        data.ctrl[0] = 0.5 + 0.5*np.sin(2*np.pi*i*_DT)
        mujoco.mj_step(model, data)
        for name in ('muscle_l_ce', 'muscle_v_ce', 'muscle_l_se', 'muscle_F_mtu'):
            assert np.isfinite(getattr(data, name)[0]), f'{name} at step {i}'
        assert 0.0 < data.muscle_l_ce[0] < 10*model.actuator_gainprm[0][LOPT]


@pytest.mark.parametrize('gaintype', _MODELS)
def test_fiber_length_is_an_activation_variable(gaintype):
    """act = [l_ce, activation]: the fiber is state, so mjSTATE_PHYSICS is a complete state."""
    model = mujoco.MjModel.from_xml_string(_model_xml(gaintype, _gainprm()))
    assert model.na == 2
    ref = mujoco.MjData(model)
    ref.ctrl[0] = 1.0
    for _ in range(300):
        mujoco.mj_step(model, ref)
    assert ref.act[0] == pytest.approx(ref.muscle_l_ce[0])   # act[0] is the fiber length

    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_PHYSICS))
    mujoco.mj_getState(model, ref, state, mujoco.mjtState.mjSTATE_PHYSICS)
    for _ in range(200):
        mujoco.mj_step(model, ref)

    restored = mujoco.MjData(model)
    restored.ctrl[0] = 1.0
    mujoco.mj_setState(model, restored, state, mujoco.mjtState.mjSTATE_PHYSICS)
    for _ in range(200):
        mujoco.mj_step(model, restored)
    assert restored.qpos[0] == ref.qpos[0]
    assert restored.muscle_F_mtu[0] == ref.muscle_F_mtu[0]


@pytest.mark.parametrize('gaintype', _MODELS)
def test_forward_is_idempotent(gaintype):
    """mj_forward is a pure function of the state: calling it twice must change nothing."""
    model = mujoco.MjModel.from_xml_string(_model_xml(gaintype, _gainprm()))
    data = mujoco.MjData(model)
    data.ctrl[0] = 1.0
    for _ in range(300):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    first = (data.muscle_F_mtu[0], data.act[0])
    mujoco.mj_forward(model, data)
    assert (data.muscle_F_mtu[0], data.act[0]) == first


@pytest.mark.parametrize('gaintype', _MODELS)
def test_rk4_matches_euler(gaintype):
    """RK4 evaluates mj_forward at intermediate states; the fiber must integrate with them."""
    q = []
    for integrator in ('Euler', 'RK4'):
        xml = _model_xml(gaintype, _gainprm()).replace(
            '<option ', f'<option integrator="{integrator}" ')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.ctrl[0] = 1.0
        for _ in range(2000):
            mujoco.mj_step(model, data)
        q.append(data.qpos[0])
    assert q[1] == pytest.approx(q[0], abs=1e-6)


@pytest.mark.parametrize('gaintype', _MODELS)
def test_reset_is_repeatable(gaintype):
    """mj_resetData must put the muscle back where it started, bake cache and all."""
    model = mujoco.MjModel.from_xml_string(_model_xml(gaintype, _gainprm()))
    data = mujoco.MjData(model)

    def run():
        mujoco.mj_resetData(model, data)
        data.ctrl[0] = 0.7
        for _ in range(1000):
            mujoco.mj_step(model, data)
        return np.array([data.qpos[0], data.muscle_l_ce[0], data.muscle_F_mtu[0]])

    np.testing.assert_array_equal(run(), run())


def test_rigid_tendon_fallback():
    """A tendon far shorter than the fiber switches to the rigid path and still holds."""
    for gaintype in _MODELS:
        _, data = _settle(gaintype, _gainprm({LSLACK: 0.005}), 0.5)
        assert data.muscle_F_mtu[0] == pytest.approx(_LOAD_MASS*9.81, abs=1e-1)
        assert data.muscle_l_se[0] == pytest.approx(0.005)


# ---------------------------------------------------------------------------------------
# .osim parameter fidelity: the curve shape slots must actually reach the force.

def test_tendon_strain_override_changes_compliance():
    """A more compliant tendon (larger strain_at_one_norm_force) must stretch further."""
    _, stiff = _settle('millard_mtu', _gainprm({TFL_E1: 0.033}), 0.5)
    _, soft = _settle('millard_mtu', _gainprm({TFL_E1: 0.10}), 0.5)
    # same load, so the same tendon force, reached at a larger tendon length
    assert soft.muscle_F_mtu[0] == pytest.approx(stiff.muscle_F_mtu[0], abs=1e-2)
    assert soft.muscle_l_se[0] > stiff.muscle_l_se[0]


def test_passive_curve_override_changes_passive_force():
    """RajagopalLaiUhlrich2023 tunes the passive curve per muscle; that must have an effect.

    strain_at_zero_force below zero means passive force starts before the optimal fiber
    length, so a fully inactive muscle sits at a shorter fiber for the same load.
    """
    default = _gainprm()
    tuned = _gainprm({PFL_E0: -0.18444538749, PFL_E1: 0.50595953637, PFL_KISO: 2.896858})
    _, a = _settle('millard_mtu', default, 0.0)
    _, b = _settle('millard_mtu', tuned, 0.0)
    assert a.muscle_F_mtu[0] == pytest.approx(b.muscle_F_mtu[0], abs=1e-2)
    assert b.muscle_l_ce[0] < a.muscle_l_ce[0] - 1e-3


def test_zero_means_opensim_default():
    """Writing OpenSim's defaults out explicitly must change nothing."""
    explicit = _gainprm({
        AFL_MIN: 0.4441, AFL_TRANS: 0.73, AFL_MAX: 1.8123, AFL_SLOPE: 0.8616,
        PFL_E1: 0.7, PFL_KLOW: 0.2, PFL_KISO: 2.0/0.7, PFL_CURV: 0.75,
        TFL_E1: 0.049, TFL_KISO: 1.375/0.049, TFL_FTOE: 2.0/3.0, TFL_CURV: 0.5,
        FV_FMAXE: 1.4, FV_DYDXNEARC: 0.25, FV_DYDXISO: 5.0,
        FV_DYDXNEARE: 0.15, FV_CONCCURV: 0.6, FV_ECCCURV: 0.9})
    _, a = _settle('millard_mtu', _gainprm(), 0.6, steps=2000)
    _, b = _settle('millard_mtu', explicit, 0.6, steps=2000)
    assert b.qpos[0] == pytest.approx(a.qpos[0], abs=1e-12)
    assert b.muscle_F_mtu[0] == pytest.approx(a.muscle_F_mtu[0], abs=1e-9)


# ---------------------------------------------------------------------------------------
# The helper bindings. doc/muscle_mtu.rst tells users to call these, so they have to be
# reachable from Python; they were exported from the C library but unbound until now.

@pytest.mark.parametrize('gaintype', _MODELS)
def test_equilibrate_binding_reaches_the_isometric_equilibrium(gaintype):
    """The sequence doc/muscle_mtu.rst prescribes, run from Python."""
    model = mujoco.MjModel.from_xml_string(_model_xml(gaintype, _gainprm()))
    data = mujoco.MjData(model)
    data.qpos[0] = 0.02
    data.act[1] = 0.5
    data.ctrl[0] = 0.5

    mujoco.mj_forward(model, data)
    seeded = data.act[0]
    mujoco.mju_mtuMuscleEquilibrate(model, data)
    mujoco.mj_forward(model, data)

    # the fiber left its mj_resetData seed and is now at rest
    assert data.act[0] != seeded
    assert data.act_dot[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(data.muscle_F_mtu[0])


def test_equilibrate_binding_matches_the_act_dot_iteration():
    """One call must land where iterating the documented act_dot identity lands.

    Before the binding existed the only way to equilibrate from Python was to iterate
    act_dot[fiber] = (l_ce* - l_ce)/dt to a fixed point, at a step size forced to be
    model.opt.timestep. That is what this compares against: same answer, bounded cost.
    """
    model = mujoco.MjModel.from_xml_string(_model_xml('millard_mtu', _gainprm()))

    direct = mujoco.MjData(model)
    iterated = mujoco.MjData(model)
    for d in (direct, iterated):
        d.qpos[0] = 0.02
        d.act[1] = 0.5
        d.ctrl[0] = 0.5

    mujoco.mj_forward(model, direct)
    mujoco.mju_mtuMuscleEquilibrate(model, direct)
    mujoco.mj_forward(model, direct)

    steps = 0
    while steps < 5000:
        mujoco.mj_forward(model, iterated)
        previous = iterated.act[0]
        iterated.act[0] += model.opt.timestep * iterated.act_dot[0]
        iterated.act[1] = 0.5
        steps += 1
        if abs(iterated.act[0] - previous) <= 1e-13 * max(1.0, abs(previous)):
            break
    mujoco.mj_forward(model, iterated)

    assert steps > 1, 'the iteration must actually have had work to do'
    assert direct.act[0] == pytest.approx(iterated.act[0], abs=1e-9)
    f_max = model.actuator_gainprm[0][FMAX]
    assert direct.muscle_F_mtu[0] == pytest.approx(iterated.muscle_F_mtu[0], abs=1e-6*f_max)


def test_compliant_muscle_equilibrate_binding():
    prm = '3000 0.15 0.15 10 0.56 -2.995732274 1.5 5.0 0.04'
    model = mujoco.MjModel.from_xml_string(_model_xml('compliant_mtu', prm))
    data = mujoco.MjData(model)
    data.qpos[0] = 0.02
    data.act[1] = 0.5
    data.ctrl[0] = 0.5
    mujoco.mj_forward(model, data)
    mujoco.mju_compliantMuscleEquilibrate(model, data)
    mujoco.mj_forward(model, data)
    assert data.act_dot[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(data.muscle_F_mtu[0])


def test_curve_bindings_return_opensim_landmarks():
    """mju_millardCurve / mju_hyfydyCurve, the comparison doc/muscle_mtu.rst points at."""
    deriv = np.zeros(1)

    # value and slope at each curve's defining point
    assert mujoco.mju_millardCurve(0, 1.0, None, deriv) == pytest.approx(1.0, abs=1e-8)
    assert mujoco.mju_millardCurve(1, 1.7, None, deriv) == pytest.approx(1.0, abs=1e-8)
    assert deriv[0] == pytest.approx(2.0/0.7, abs=1e-6)     # stiffness_at_one_norm_force
    assert mujoco.mju_millardCurve(2, 1.049, None, deriv) == pytest.approx(1.0, abs=1e-8)
    assert deriv[0] == pytest.approx(1.375/0.049, abs=1e-6)
    assert mujoco.mju_millardCurve(3, 0.0, None, deriv) == pytest.approx(1.0, abs=1e-8)
    assert deriv[0] == pytest.approx(5.0, abs=1e-6)         # isometric_slope
    assert mujoco.mju_millardCurve(3, 1.0, None, None) == pytest.approx(1.4, abs=1e-8)

    assert mujoco.mju_hyfydyCurve(0, 1.0, None) == pytest.approx(1.0, abs=1e-12)
    assert mujoco.mju_hyfydyCurve(3, 0.0, deriv) == pytest.approx(1.0, abs=1e-12)
    assert deriv[0] == pytest.approx(0.11*(1.6 - 1)/0.11**2, rel=1e-9)


def test_millard_curve_binding_honours_gainprm():
    """A gainprm block must reach the curve, and None must mean OpenSim's defaults."""
    tuned = np.zeros(mujoco.mjNGAIN)
    tuned[PFL_E1] = 0.5                                     # strain_at_one_norm_force
    # by definition the passive curve is 1.0 at 1 + strain_at_one_norm_force
    assert mujoco.mju_millardCurve(1, 1.5, tuned, None) == pytest.approx(1.0, abs=1e-8)
    assert mujoco.mju_millardCurve(1, 1.7, None, None) == pytest.approx(1.0, abs=1e-8)
    assert mujoco.mju_millardCurve(1, 1.5, None, None) < 0.9


def test_curve_binding_errors_are_python_exceptions():
    with pytest.raises(Exception, match='unknown curve'):
        mujoco.mju_millardCurve(99, 1.0, None, None)
    with pytest.raises(TypeError, match='gainprm'):
        mujoco.mju_millardCurve(0, 1.0, np.zeros(5), None)


def test_every_public_muscle_helper_is_reachable():
    """The whole muscle surface of mujoco.h, not a subset.

    Partial exposure is what made this a bug report: the symbols were exported from the C
    library and absent from the module, and six of them were declared in mujoco.h without
    being exported at all.
    """
    for name in ('mju_compliantMuscleInvFvce0', 'mju_compliantMuscleFlce0',
                 'mju_compliantMuscleFp0', 'mju_compliantMuscleFp0Ext',
                 'mju_compliantMuscleInit', 'mju_compliantMuscleActDot',
                 'mju_compliantMuscleEquilibrate', 'mju_compliantMuscleForceVel',
                 'mju_compliantMuscleECC', 'mju_mtuMuscleInit', 'mju_mtuMuscleActDot',
                 'mju_mtuMuscleEquilibrate', 'mju_mtuMuscleForceVel',
                 'mju_millardCurve', 'mju_hyfydyCurve', 'mju_millardCurveCacheSize'):
        assert hasattr(mujoco, name), f'{name} is not bound'


def test_hyfydy_rejects_curve_shape_parameters():
    """Hyfydy's curves are fixed polynomials, so a shape parameter is a modelling mistake."""
    with pytest.raises(ValueError, match='fixed polynomials'):
        mujoco.MjModel.from_xml_string(
            _model_xml('hyfydy_mtu', _gainprm({PFL_E1: 0.3})))
