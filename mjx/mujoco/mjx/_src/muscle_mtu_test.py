# Copyright 2025 Neumove
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the muscle-tendon unit gains, against the C engine."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
from jax.experimental import enable_x64
import mujoco
from mujoco import mjx
from mujoco.mjx._src import muscle_mtu
import numpy as np


# A point mass hung from one muscle; engine_muscle_mtu_test.cc's HangingMuscle.
def _hanging(gain: str, prm: str, extra: str = '') -> str:
  return f"""
  <mujoco>
    <option timestep="0.001" gravity="0 0 -9.81"/>
    <worldbody>
      <site name="anchor" pos="0 0 0.3" size="0.005"/>
      <body name="m">
        <joint name="s" type="slide" axis="0 0 1"/>
        <site name="ins" size="0.005"/>
        <geom type="sphere" size="0.01" mass="20" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="t"><site site="anchor"/><site site="ins"/></spatial>
    </tendon>
    <actuator>
      <general name="a" tendon="t" gaintype="{gain}" biastype="none"
               dyntype="muscle" dynprm="0.01 0.04" ctrllimited="true"
               ctrlrange="0 1" gainprm="{prm}" {extra}/>
    </actuator>
  </mujoco>"""


_MECH = '3000 0.15 0.15 10 0.15 0.1'  # kDefaultPrm: F_max l_opt l_slack v_max phi beta
_SONG = '4000 0.04 0.26 6 0.56 -2.995732273553991 1.5 5 0.04'  # kSol


def _rigid_flag(mech: str) -> str:
  """gainprm with ignore_tendon_compliance (slot 19) set."""
  return mech + ' 0' * (19 - len(mech.split())) + ' 1'


# gain, gainprm, the path length the load settles near
_HANGING = {
    'millard': ('millard_mtu', _MECH),
    'hyfydy': ('hyfydy_mtu', _MECH),
    'millard_unpennated': ('millard_mtu', '3000 0.15 0.15 10 0 0.1'),
    # afl transition_norm_fiber_length, pfl strain_at_one_norm_force, tfl
    # strain_at_one_norm_force: slots 9, 15 and 20
    'millard_shape': (
        'millard_mtu',
        _MECH + ' 0 0 0 0.6 0 0 0 0 0 0.5 0 0 0 0 0.03',
    ),
    'millard_rigid': ('millard_mtu', _rigid_flag(_MECH)),
    'hyfydy_rigid': ('hyfydy_mtu', _rigid_flag(_MECH)),
    'millard_short_tendon': ('millard_mtu', '3000 0.15 0.005 10 0.15 0.1'),
    'compliant': ('compliant_mtu', _SONG),
    'compliant_pe': ('compliant_mtu', _SONG + ' 0.9 0.5'),
    'compliant_rigid': ('compliant_mtu', _SONG + ' 0 0 1'),
    # rigid by the ratio rule with no tendon at all, which the compliant
    # residual, evaluated but not selected, would divide by
    'compliant_rigid_zero_slack': (
        'compliant_mtu',
        '4000 0.04 0 6 0.56 -2.995732273553991 1.5 5 0.04',
    ),
}


# Every gain side by side with ordinary stateful actuators whose activations
# sit between the muscles' two slots, so every act address is exercised.
_MIXED = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <default>
    <geom type="sphere" size="0.01" mass="10" contype="0" conaffinity="0"/>
    <joint type="slide" axis="0 0 1" damping="1"/>
    <general dyntype="muscle" dynprm="0.01 0.04" biastype="none"
             ctrllimited="true" ctrlrange="0 1"/>
  </default>
  <worldbody>
    <body name="arm" pos="0 1 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
    <site name="a0" pos="0 0 0.3"/>
    <site name="a1" pos="0.2 0 0.3"/>
    <site name="a2" pos="0.4 0 0.3"/>
    <site name="a3" pos="0.6 0 0.3"/>
    <site name="a4" pos="0.8 0 0.3"/>
    <site name="a5" pos="1.0 0 0.3"/>
    <body name="b0" pos="0 0 0"><joint name="s0"/><geom/><site name="i0"/></body>
    <body name="b1" pos="0.2 0 0"><joint name="s1"/><geom/><site name="i1"/></body>
    <body name="b2" pos="0.4 0 0"><joint name="s2"/><geom/><site name="i2"/></body>
    <body name="b3" pos="0.6 0 0.18"><joint name="s3"/><geom/><site name="i3"/></body>
    <body name="b4" pos="0.8 0 0"><joint name="s4"/><geom/><site name="i4"/></body>
    <body name="b5" pos="1.0 0 0"><joint name="s5"/><geom/><site name="i5"/></body>
  </worldbody>
  <tendon>
    <spatial name="t0"><site site="a0"/><site site="i0"/></spatial>
    <spatial name="t1"><site site="a1"/><site site="i1"/></spatial>
    <spatial name="t2"><site site="a2"/><site site="i2"/></spatial>
    <spatial name="t3"><site site="a3"/><site site="i3"/></spatial>
    <spatial name="t4"><site site="a4"/><site site="i4"/></spatial>
    <spatial name="t5"><site site="a5"/><site site="i5"/></spatial>
  </tendon>
  <actuator>
    <motor joint="hinge" gear="2"/>
    <general name="filter" joint="hinge" dyntype="filter" dynprm="0.05"
             gaintype="fixed" gainprm="3" ctrlrange="-1 1"/>
    <general name="millard" tendon="t0" gaintype="millard_mtu"
             gainprm="1500 0.15 0.15 10 0.15 0.1"/>
    <general name="compliant" tendon="t1" gaintype="compliant_mtu"
             gainprm="2000 0.04 0.26 6 0.56 -2.995732273553991 1.5 5 0.04"/>
    <general name="filterexact" joint="hinge" dyntype="filterexact"
             dynprm="0.02" gaintype="fixed" gainprm="2" ctrlrange="-1 1"/>
    <general name="hyfydy_early" tendon="t2" gaintype="hyfydy_mtu"
             actearly="true" gainprm="1500 0.15 0.15 10 0.3 0.1"/>
    <general name="millard_rigid" tendon="t3" gaintype="millard_mtu"
             gainprm="1500 0.1 0.004 10 0.1 0.1"/>
    <general name="millard_shared" tendon="t4" gaintype="millard_mtu"
             gainprm="1000 0.15 0.15 10 0 0.1"/>
    <general name="millard_shape" tendon="t5" gaintype="millard_mtu"
             gainprm="1500 0.15 0.15 10 0.15 0.1 0 0 0 0 0 0 0 0 0 0.5"/>
  </actuator>
</mujoco>
"""


def _assert_close(expected, actual, name, rtol, atol):
  np.testing.assert_allclose(
      np.asarray(actual), expected, rtol=rtol, atol=atol, err_msg=name
  )


def _put(m, d):
  return mjx.put_model(m), mjx.put_data(m, d)


def _muscle(dx, name):
  return np.asarray(getattr(dx._impl, name))


def _exact(mx, dq, l_prev, isometric):
  """Fiber length and force of actuator 0 at the exact root of its residual.

  The solvers stop at a tolerance, and Song's also steps with a finite-difference
  Jacobian, so the derivative of what they return is not the derivative of the
  root: for compliant_mtu the two differ by 2e-5 relative, which is the Jacobian's
  own error. Polishing with exact Newton steps removes that, so finite
  differences of this are a reference for the root's derivative.

  Args:
    mx: the model
    dq: data after forward, at the pose being differentiated
    l_prev: the fiber length the step starts from
    isometric: the equilibrium's residual rather than the step's

  Returns:
    fiber length and actuator force
  """
  gain = mujoco.mjtGain(int(mx.actuator_gaintype[0]))
  prm = mx.actuator_gainprm[:1]
  l_mtu = dq._impl.actuator_length[:1]
  x = dq._impl.muscle_l_ce[:1]
  dt = mx.opt.timestep
  if gain == mujoco.mjtGain.mjGAIN_COMPLIANT_MTU:
    p = muscle_mtu._cmtu_params(prm)
    a = jp.clip(dq.act[1:], 0, 1)
    dtv = dt * p.l_opt * p.v_max

    def residual(x):
      v = jp.zeros(1) if isometric else (x - l_prev) / dtv
      return muscle_mtu._cmtu_residual(p, a, x, l_mtu, v)[0]

    def force(x):
      return p.F_max * muscle_mtu._cmtu_fp0((l_mtu - x) / p.l_slack, p.E_REF)

  else:
    hyfydy = gain == mujoco.mjtGain.mjGAIN_HYFYDY_MTU
    p = muscle_mtu._mtu_params(prm, hyfydy)
    curves = muscle_mtu._curves(mx, gain, np.array([0]))
    a = jp.clip(dq.act[1:], p.min_act, 1)
    dtv = jp.zeros(1) if isometric else dt * p.v_max_ms
    prev = x if isometric else l_prev

    def residual(x):
      return muscle_mtu._mtu_eval(curves, p, x, l_mtu, prev, a, dtv).R[0]

    def force(x):
      e = muscle_mtu._mtu_eval(curves, p, x, l_mtu, prev, a, dtv)
      return p.F_max * e.f_T

  if bool(p.rigid[0]):
    return x[0], dq.actuator_force[0]
  for _ in range(6):
    r, dr = jax.value_and_grad(lambda x: residual(x[None]))(x[0])
    x = x - r / dr
  return x[0], -force(x)[0]


class MtuTest(parameterized.TestCase):

  def _random_state(self, m, d, rng):
    """A taut or slack, moving, partly activated state reached by stepping."""
    mujoco.mj_resetData(m, d)
    for i in range(400):
      d.ctrl[:] = 0.5 + 0.5 * np.sin(0.02 * i + np.arange(m.nu))
      mujoco.mj_step(m, d)
    d.qvel[:] += 0.2 * rng.standard_normal(m.nv)
    d.ctrl[:] = rng.uniform(0, 1, m.nu)

  @parameterized.named_parameters(*[(k, *v) for k, v in _HANGING.items()])
  def test_fwd_actuation(self, gain, prm):
    """act_dot, force and the muscle outputs match mj_fwdActuation."""
    m = mujoco.MjModel.from_xml_string(_hanging(gain, prm))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    with enable_x64():
      fwd = jax.jit(mjx.forward)
      for trial in range(8):
        self._random_state(m, d, rng)
        # perturb the fiber off its step's solution, so Newton has work to do
        d.act[0] *= 1 + 0.05 * rng.standard_normal()
        d.act[1] = rng.uniform(0, 1)
        mx, dx = _put(m, d)
        mujoco.mj_forward(m, d)
        dx = fwd(mx, dx)
        for f in ('act_dot', 'actuator_force', 'qfrc_actuator', 'qacc'):
          _assert_close(getattr(d, f), getattr(dx, f), f'{f} {trial}', 1e-9, 1e-9)
        for f in ('muscle_l_ce', 'muscle_l_se', 'muscle_v_ce', 'muscle_F_mtu'):
          _assert_close(getattr(d, f), _muscle(dx, f), f'{f} {trial}', 1e-9, 1e-9)

  @parameterized.parameters('euler', 'rk4', 'implicitfast')
  def test_step_mixed(self, integrator):
    """A mixed model steps as mj_step does, under each integrator."""
    m = mujoco.MjModel.from_xml_string(_MIXED)
    m.opt.integrator = {
        'euler': mujoco.mjtIntegrator.mjINT_EULER,
        'rk4': mujoco.mjtIntegrator.mjINT_RK4,
        'implicitfast': mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }[integrator]
    d = mujoco.MjData(m)
    with enable_x64():
      mx, dx = _put(m, d)
      step = jax.jit(mjx.step)
      for i in range(600):
        ctrl = 0.5 + 0.5 * np.sin(0.02 * i + np.arange(m.nu))
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        dx = step(mx, dx.replace(ctrl=ctrl))
      for f in ('qpos', 'qvel', 'act', 'act_dot', 'actuator_force'):
        _assert_close(getattr(d, f), getattr(dx, f), f, 1e-7, 1e-9)
      for f in ('muscle_l_ce', 'muscle_l_se', 'muscle_v_ce', 'muscle_F_mtu'):
        _assert_close(getattr(d, f), _muscle(dx, f), f, 1e-7, 1e-9)

  def test_mixed_model_exercises_every_path(self):
    """The mixed model really has shared, distinct and rigid Millard curves."""
    m = mujoco.MjModel.from_xml_string(_MIXED)
    fields = muscle_mtu.model_fields(m)
    curve = fields['actuator_mtu_curve']
    ids = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
           for i in range(m.nu)}
    # a shape override bakes a table of its own, only for the curve it changes
    np.testing.assert_array_equal(curve[ids['millard']], curve[ids['millard_shared']])
    self.assertNotEqual(curve[ids['millard']][1], curve[ids['millard_shape']][1])
    self.assertEqual(curve[ids['millard']][0], curve[ids['millard_shape']][0])
    self.assertTrue((curve[ids['compliant']] == -1).all())
    self.assertTrue((curve[ids['hyfydy_early']] == -1).all())
    self.assertEqual(fields['mtu_curve_knot'].shape[0], 5)

  @parameterized.named_parameters(*[(k, *v) for k, v in _HANGING.items()])
  def test_equilibrate(self, gain, prm):
    """Equilibration matches mju_*MuscleEquilibrate over poses and activations."""
    m = mujoco.MjModel.from_xml_string(_hanging(gain, prm))
    d = mujoco.MjData(m)
    compliant = gain == 'compliant_mtu'
    c_fn = (mujoco.mju_compliantMuscleEquilibrate if compliant
            else mujoco.mju_mtuMuscleEquilibrate)
    with enable_x64():
      mjx_fn = jax.jit(mjx.compliant_muscle_equilibrate if compliant
                       else mjx.mtu_muscle_equilibrate)
      fwd = jax.jit(mjx.forward)
      # from slack, through the fiber's clamp, to far past optimal length
      for qpos in np.linspace(-0.25, 0.12, 12):
        for act in (0.0, 0.3, 1.0):
          mujoco.mj_resetData(m, d)
          d.qpos[0] = qpos
          d.act[1] = act
          d.act[0] *= 1.3  # equilibration must not depend on the incoming fiber
          mujoco.mj_forward(m, d)
          mx, dx = _put(m, d)
          c_fn(m, d)
          dx = mjx_fn(mx, fwd(mx, dx))
          name = f'qpos={qpos} act={act}'
          _assert_close(d.act, dx.act, f'act {name}', 1e-9, 1e-12)
          _assert_close(d.act_dot, dx.act_dot, f'act_dot {name}', 0, 1e-12)
          for f in ('muscle_l_ce', 'muscle_l_se', 'muscle_v_ce', 'muscle_F_mtu'):
            _assert_close(getattr(d, f), _muscle(dx, f), f'{f} {name}', 1e-9, 1e-9)

  def test_equilibrate_mixed(self):
    """Each equilibration touches only its own gains' fibers."""
    m = mujoco.MjModel.from_xml_string(_MIXED)
    d = mujoco.MjData(m)
    d.qpos[:] = np.linspace(-0.05, 0.05, m.nq)
    d.act[:] = np.linspace(0.02, 0.9, m.na)
    mujoco.mj_forward(m, d)
    with enable_x64():
      mx, dx = _put(m, d)
      dx = jax.jit(mjx.forward)(mx, dx)
      dx = jax.jit(mjx.mtu_muscle_equilibrate)(mx, dx)
      dx = jax.jit(mjx.compliant_muscle_equilibrate)(mx, dx)
    mujoco.mju_mtuMuscleEquilibrate(m, d)
    mujoco.mju_compliantMuscleEquilibrate(m, d)
    _assert_close(d.act, dx.act, 'act', 1e-9, 1e-12)
    _assert_close(d.muscle_F_mtu, _muscle(dx, 'muscle_F_mtu'), 'F', 1e-9, 1e-9)

  @parameterized.named_parameters(
      *[(k, *v) for k, v in _HANGING.items() if 'rigid' in k or 'short' in k]
  )
  def test_force_vel(self, gain, prm):
    """d(force)/d(velocity) matches mju_*MuscleForceVel on the rigid path."""
    m = mujoco.MjModel.from_xml_string(_hanging(gain, prm))
    d = mujoco.MjData(m)
    c_fn = (mujoco.mju_compliantMuscleForceVel if gain == 'compliant_mtu'
            else mujoco.mju_mtuMuscleForceVel)
    rng = np.random.default_rng(1)
    with enable_x64():
      fn = jax.jit(muscle_mtu.force_vel)
      nonzero = 0
      for _ in range(10):
        self._random_state(m, d, rng)
        mujoco.mj_forward(m, d)
        mx, dx = _put(m, d)
        want = c_fn(m, d, 0)
        _assert_close([want], fn(mx, dx), 'force_vel', 1e-9, 1e-9)
        nonzero += want != 0
      self.assertGreater(nonzero, 0)

  def test_make_data_seeds_the_fiber(self):
    """make_data seeds act and the muscle outputs as mj_resetData does."""
    m = mujoco.MjModel.from_xml_string(_MIXED)
    d = mujoco.MjData(m)
    dx = mjx.make_data(m)
    _assert_close(d.act, dx.act, 'act', 0, 1e-7)
    for f in ('muscle_l_ce', 'muscle_l_se', 'muscle_v_ce', 'muscle_F_mtu'):
      _assert_close(getattr(d, f), _muscle(dx, f), f, 0, 1e-7)

  def test_get_data_carries_the_muscle(self):
    m = mujoco.MjModel.from_xml_string(_MIXED)
    mx, dx = mjx.put_model(m), mjx.make_data(m)
    dx = jax.jit(mjx.step)(mx, dx.replace(ctrl=jp.full(m.nu, 0.7)))
    d = mjx.get_data(m, dx)
    _assert_close(d.muscle_F_mtu, _muscle(dx, 'muscle_F_mtu'), 'F', 0, 0)
    _assert_close(d.act, dx.act, 'act', 0, 0)

  def test_millard_tables(self):
    """The device tables evaluate as mju_millardCurve, inside and outside."""
    m = mujoco.MjModel.from_xml_string(_MIXED)
    fields = muscle_mtu.model_fields(m)
    ranges = ((0.2, 2.2), (0.8, 2.0), (0.9, 1.2), (-1.5, 1.5))
    with enable_x64():
      for i in np.nonzero(fields['actuator_mtu_curve'][:, 0] >= 0)[0]:
        for c, (lo, hi) in enumerate(ranges):
          x = np.linspace(lo, hi, 997)
          tid = np.full(x.shape, fields['actuator_mtu_curve'][i, c])
          y, dy = muscle_mtu._table_eval(
              jp.asarray(fields['mtu_curve_knot']),
              jp.asarray(fields['mtu_curve_meta']), tid, jp.asarray(x))
          want = np.zeros((2, x.size))
          for k, xk in enumerate(x):
            deriv = np.zeros(1)
            want[0, k] = mujoco.mju_millardCurve(c, xk, m.actuator_gainprm[i], deriv)
            want[1, k] = deriv[0]
          _assert_close(want[0], y, f'curve {c} of {i}', 1e-12, 1e-13)
          _assert_close(want[1], dy, f'slope {c} of {i}', 1e-10, 1e-11)

  def test_hyfydy_curves(self):
    fns = (muscle_mtu._hyfydy_fl, muscle_mtu._hyfydy_fp,
           muscle_mtu._hyfydy_ft, muscle_mtu._hyfydy_fv)
    ranges = ((0.2, 2.2), (0.5, 2.0), (0.9, 1.2), (-1.5, 1.5))
    with enable_x64():
      for c, (fn, (lo, hi)) in enumerate(zip(fns, ranges)):
        x = np.linspace(lo, hi, 1001)
        y, dy = fn(jp.asarray(x))
        want = np.zeros((2, x.size))
        for k, xk in enumerate(x):
          deriv = np.zeros(1)
          want[0, k] = mujoco.mju_hyfydyCurve(c, xk, deriv)
          want[1, k] = deriv[0]
        _assert_close(want[0], y, f'curve {c}', 1e-14, 1e-14)
        _assert_close(want[1], dy, f'slope {c}', 1e-14, 1e-14)

  @parameterized.named_parameters(*[(k, *v) for k, v in _HANGING.items()])
  def test_force_gradient(self, gain, prm):
    """d(force)/d(qpos) is the derivative of the force at the step's root.

    The fiber starts off its equilibrium, so the step moves it, and for
    millard_mtu also on it: there the stepping Newton's warm start already meets
    its tolerance and no iteration runs, and differentiating the iterations
    would report the tendon stiffness alone. Only millard_mtu is taken there,
    because a step that does not move the fiber has v_ce = 0, where Hyfydy's
    force-velocity slope jumps by 0.9% and Song's by 2.4x, so neither is
    differentiable at that point.
    """
    m = mujoco.MjModel.from_xml_string(_hanging(gain, prm))
    d = mujoco.MjData(m)
    compliant = gain == 'compliant_mtu'
    offsets = (1.0, 1.01, 0.97) if gain == 'millard_mtu' else (1.01, 0.97)
    with enable_x64():
      mx = mjx.put_model(m)
      for (qpos, act), offset in zip(
          ((0.013, 0.5), (-0.021, 1.0), (0.034, 0.1)), offsets
      ):
        mujoco.mj_resetData(m, d)
        d.qpos[0], d.act[1] = qpos, act
        mujoco.mj_forward(m, d)
        (mujoco.mju_compliantMuscleEquilibrate if compliant
         else mujoco.mju_mtuMuscleEquilibrate)(m, d)
        d.act[0] *= offset
        dx = mjx.put_data(m, d)

        def force(q, dx=dx):
          return mjx.forward(mx, dx.replace(qpos=q)).actuator_force[0]

        def exact(q, dx=dx):
          dq = mjx.forward(mx, dx.replace(qpos=q))
          return _exact(mx, dq, dx.act[:1], isometric=False)[1]

        q = jp.asarray(d.qpos)
        grad = jax.jit(jax.grad(force))(q)[0]
        eps = 1e-6
        fd = (exact(q + eps) - exact(q - eps)) / (2 * eps)
        self.assertTrue(np.isfinite(grad))
        # Song's step stops at a residual of 1e-5, about 1e-9 m from the root,
        # and the implicit derivative is taken where it stops: the curvature of
        # the force balance over that distance is a 1e-6 relative error
        rtol = 1e-5 if compliant else 1e-6
        _assert_close(fd, grad, f'qpos={qpos} act={act}', rtol, 1e-6)

  @parameterized.named_parameters(*[(k, *v) for k, v in _HANGING.items()])
  def test_equilibrium_gradient(self, gain, prm):
    """d(equilibrium fiber length)/d(qpos) is the root's derivative."""
    m = mujoco.MjModel.from_xml_string(_hanging(gain, prm))
    d = mujoco.MjData(m)
    compliant = gain == 'compliant_mtu'
    eq = (mjx.compliant_muscle_equilibrate if compliant
          else mjx.mtu_muscle_equilibrate)
    with enable_x64():
      mx = mjx.put_model(m)
      mujoco.mj_resetData(m, d)
      d.act[1] = 0.6
      dx = mjx.put_data(m, d)

      def fiber(q):
        dq = mjx.forward(mx, dx.replace(qpos=q))
        return eq(mx, dq).act[0]

      def exact(q):
        dq = eq(mx, mjx.forward(mx, dx.replace(qpos=q)))
        return _exact(mx, dq, dq.act[:1], isometric=True)[0]

      for qpos in (-0.031, 0.007, 0.043):
        q = jp.asarray([qpos])
        grad = jax.jit(jax.grad(fiber))(q)[0]
        eps = 1e-6
        fd = (exact(q + eps) - exact(q - eps)) / (2 * eps)
        _assert_close(fd, grad, f'qpos={qpos}', 1e-6, 1e-9)

  def test_rollout_gradient(self):
    """Reverse mode through a rollout of the mixed model is finite and right."""
    m = mujoco.MjModel.from_xml_string(_MIXED)
    with enable_x64():
      mx, dx0 = mjx.put_model(m), mjx.make_data(m)
      # make_data's int32 wrap indices come back int64 from a step under x64,
      # which a scan carry refuses; one step first gives the carry its types
      dx0 = jax.jit(mjx.step)(mx, dx0)

      def final_height(ctrl):
        def body(dx, _):
          return mjx.step(mx, dx.replace(ctrl=ctrl)), None

        dx, _ = jax.lax.scan(body, dx0, None, length=100)
        return jp.sum(dx.qpos)

      ctrl = jp.full(m.nu, 0.6)
      grad = jax.jit(jax.grad(final_height))(ctrl)
      self.assertTrue(np.isfinite(grad).all())
      eps = 1e-6
      for i in range(m.nu):
        e = jp.zeros(m.nu).at[i].set(eps)
        fd = (final_height(ctrl + e) - final_height(ctrl - e)) / (2 * eps)
        # finite differences see the solver, the gradient sees the root. Song's
        # step is a Newton damped by 0.8 that stops at a residual of 1e-5, so
        # after k iterations its sensitivity still carries 0.2^k of the warm
        # start's: 1e-4 relative here, where the undamped Newtons agree to 1e-6
        compliant = m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_COMPLIANT_MTU
        _assert_close(fd, grad[i], f'ctrl {i}', 1e-3 if compliant else 1e-4, 1e-7)

  def test_vmap(self):
    m = mujoco.MjModel.from_xml_string(_MIXED)
    mx, dx = mjx.put_model(m), mjx.make_data(m)
    ctrl = jp.linspace(0, 1, 4)[:, None] * jp.ones(m.nu)
    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    batch = jax.vmap(lambda c: dx.replace(ctrl=c))(ctrl)
    for _ in range(20):
      batch = step(mx, batch)
    for i in range(4):
      one = dx.replace(ctrl=ctrl[i])
      for _ in range(20):
        one = jax.jit(mjx.step)(mx, one)
      _assert_close(one.qpos, batch.qpos[i], f'qpos {i}', 1e-6, 1e-6)
      _assert_close(one.act, batch.act[i], f'act {i}', 1e-6, 1e-6)

  def test_float32(self):
    """Single precision stays finite and close to double precision."""
    m = mujoco.MjModel.from_xml_string(_MIXED)
    d = mujoco.MjData(m)
    for i in range(500):
      d.ctrl[:] = 0.5 + 0.5 * np.sin(0.02 * i + np.arange(m.nu))
      mujoco.mj_step(m, d)
    mx, dx = mjx.put_model(m), mjx.make_data(m)
    step = jax.jit(mjx.step)
    for i in range(500):
      ctrl = 0.5 + 0.5 * np.sin(0.02 * i + np.arange(m.nu))
      dx = step(mx, dx.replace(ctrl=ctrl))
    self.assertEqual(dx.qpos.dtype, jp.float32)
    self.assertTrue(np.isfinite(dx.qpos).all() and np.isfinite(dx.act).all())
    _assert_close(d.qpos, dx.qpos, 'qpos', 1e-3, 1e-4)
    _assert_close(d.act, dx.act, 'act', 1e-3, 1e-4)
    _assert_close(d.muscle_F_mtu, _muscle(dx, 'muscle_F_mtu'), 'F', 1e-3, 1.0)

  def test_path_shorter_than_the_guards(self):
    """A path under 2e-6 l_opt puts Song's lower guard above its upper one.

    mju_clip then returns the upper bound, and the step must keep it rather
    than mistake it for the lower one.
    """
    m = mujoco.MjModel.from_xml_string(_hanging('compliant_mtu', _SONG))
    d = mujoco.MjData(m)
    d.qpos[0] = 0.3 - 1e-8
    d.act[1] = 0.5
    mujoco.mj_forward(m, d)
    with enable_x64():
      # fwd_actuation on C's path length: MJX's tendon length rounds a 1e-8 m
      # spatial tendon to 0, which is upstream's and not the muscle's
      mx, dx = _put(m, d)
      dx = jax.jit(mjx.fwd_actuation)(mx, dx)
    self.assertLess(d.actuator_length[0], 2e-6 * 0.04)
    _assert_close(d.act_dot, dx.act_dot, 'act_dot', 1e-12, 0)
    _assert_close(d.muscle_l_ce, _muscle(dx, 'muscle_l_ce'), 'l_ce', 1e-12, 0)

  def test_significant_real(self):
    """The buckling margin is C's in double and scales with the precision."""
    self.assertEqual(muscle_mtu._significant_real(np.float64), 2.009718347115232e-14)
    self.assertAlmostEqual(
        muscle_mtu._significant_real(np.float32),
        float(np.finfo(np.float32).eps) ** 0.875,
        delta=1e-12,
    )

  def test_rejects_nonpositive_vmax(self):
    m = mujoco.MjModel.from_xml_string(_hanging('millard_mtu', _MECH))
    m.actuator_gainprm[0, 3] = -1
    with self.assertRaisesRegex(NotImplementedError, 'max_contraction_velocity'):
      mjx.put_model(m)


if __name__ == '__main__':
  absltest.main()
