"""Smoothness tests for the compliant_mtu (compliant muscle-tendon) actuator.

Hangs a load from a compliant muscle, drops it, and checks that the muscle
state trajectories (l_ce, v_ce, l_se, F_mtu) evolve smoothly -- finite, no
discontinuous jumps -- across a range of muscle properties and activation
patterns (constant on, constant off, 1 Hz sinusoid).

Run:  pytest test_compliant_muscle_smoothness.py -v
"""
import numpy as np
import pytest
import mujoco

# compliant_mtu gainprm order: F_max, l_opt, l_slack, v_max, W, C, N, K, E_REF
_DEFAULT = dict(F_max=1500.0, l_opt=0.15, l_slack=0.10, v_max=12.0,
                W=0.56, C=-2.995732274, N=1.5, K=5.0, E_REF=0.04)

# Muscle property variants (override the defaults). Chosen to span the
# rigid-tendon regime, long compliant tendons, strength, contraction speed
# and fiber size.
_PROPERTIES = {
    'baseline':    {},
    'tendonless':  dict(l_slack=0.003),   # l_slack/l_opt ~ 0.02 -> rigid-tendon path
    'long_tendon': dict(l_slack=0.30),
    'stiff':       dict(F_max=6000.0),
    'weak':        dict(F_max=250.0),
    'fast':        dict(v_max=24.0),
    'slow':        dict(v_max=6.0),
    'short_fiber': dict(l_opt=0.08),
}
_PATTERNS = ('on', 'off', 'sine')

_DT = 0.001          # integrator timestep [s]
_DURATION = 3.0      # simulated time [s]
_LOAD_MASS = 5.0     # hanging load [kg]
_REST_HANG = 0.30    # qpos at which the tendon sits at its natural length
_DROP = 0.03         # load starts this much higher -> tendon slack, then drops


def _model_xml(props):
    """Build a 1-DOF "load hung from a compliant muscle" model."""
    g = dict(_DEFAULT, **props)
    gainprm = ' '.join(str(g[k]) for k in
                       ('F_max', 'l_opt', 'l_slack', 'v_max',
                        'W', 'C', 'N', 'K', 'E_REF'))
    # Anchor placed so the tendon equals its natural length at qpos=_REST_HANG.
    anchor_z = g['l_opt'] + g['l_slack'] + _REST_HANG
    return f"""
<mujoco model="compliant_muscle_drop">
  <option timestep="{_DT}" gravity="0 0 -9.81" integrator="Euler"/>
  <worldbody>
    <site name="anchor" pos="0 0 {anchor_z}"/>
    <body name="load" pos="0 0 0">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.02" mass="{_LOAD_MASS}"/>
      <site name="load_site" pos="0 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="mtu_tendon">
      <site site="anchor"/>
      <site site="load_site"/>
    </spatial>
  </tendon>
  <actuator>
    <general name="mtu" tendon="mtu_tendon" gaintype="compliant_mtu"
             biastype="none" biasprm="0"
             dyntype="muscle" dynprm="0.01 0.04"
             ctrllimited="true" ctrlrange="0 1"
             gainprm="{gainprm}"/>
  </actuator>
</mujoco>"""


def _activation(pattern, t):
    """Control signal for a given activation pattern at time t."""
    if pattern == 'on':
        return 1.0
    if pattern == 'off':
        return 0.0
    if pattern == 'sine':
        return 0.5 + 0.5 * np.sin(2.0 * np.pi * 1.0 * t)   # 1 Hz, range [0, 1]
    raise ValueError(f'unknown pattern: {pattern}')


def _simulate(props, pattern):
    """Drop the load and record the muscle state trajectory."""
    model = mujoco.MjModel.from_xml_string(_model_xml(props))
    data = mujoco.MjData(model)
    data.qpos[0] = _REST_HANG + _DROP        # release the load from up high

    n = int(_DURATION / _DT)
    rec = {k: np.zeros(n) for k in ('l_ce', 'v_ce', 'l_se', 'F_mtu', 'qpos')}
    for i in range(n):
        data.ctrl[0] = _activation(pattern, data.time)
        mujoco.mj_step(model, data)
        rec['l_ce'][i] = data.muscle_l_ce[0]
        rec['v_ce'][i] = data.muscle_v_ce[0]
        rec['l_se'][i] = data.muscle_l_se[0]
        rec['F_mtu'][i] = data.muscle_F_mtu[0]
        rec['qpos'][i] = data.qpos[0]
    return rec


@pytest.mark.parametrize('prop_name', list(_PROPERTIES))
@pytest.mark.parametrize('pattern', _PATTERNS)
def test_compliant_muscle_smoothness(prop_name, pattern):
    """A dropped load on a compliant muscle yields smooth, finite state."""
    g = dict(_DEFAULT, **_PROPERTIES[prop_name])
    rec = _simulate(_PROPERTIES[prop_name], pattern)

    # 1. no NaN / Inf anywhere in the trajectory
    for name, sig in rec.items():
        assert np.all(np.isfinite(sig)), f'{name}: non-finite value'

    # 2. physical: fiber length stays positive, force does not blow up
    assert np.all(rec['l_ce'] > 0.0), 'l_ce became non-positive'
    assert np.all(np.abs(rec['F_mtu']) < 100.0 * g['F_max']), 'F_mtu blew up'

    # 3. smoothness: a single step cannot move the fiber further than it can
    #    physically travel, |dl_ce| <= (l_opt * v_max) * dt, with a 5x margin.
    #    A solver stall / teleport would jump far beyond this bound.
    max_step = (g['l_opt'] * g['v_max']) * _DT * 5.0
    dl_ce = np.abs(np.diff(rec['l_ce']))
    assert dl_ce.max() <= max_step, (
        f'l_ce discontinuity: max|dl_ce|={dl_ce.max():.3e} > {max_step:.3e}')

    # 4. the drop actually happened (the simulation is not frozen)
    assert np.ptp(rec['qpos']) > 1e-4, 'load did not move'
