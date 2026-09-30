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
"""Tests for the constraint solve in double precision, put_model(solver_dtype)."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
from jax.experimental import enable_x64
import mujoco
from mujoco import mjx
from mujoco.mjx._src import solver
from mujoco.mjx._src import test_util
import numpy as np


# A heavy link whose hinge drives light via-point bodies through joint
# equalities, as OpenSim's coupled coordinates arrive in a musculoskeletal MJCF,
# with a muscle routed over them.
_COUPLED = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="thigh">
      <joint name="hip" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="5"/>
      <body name="shank" pos="0 0 -0.4">
        <joint name="knee" type="hinge" axis="0 1 0" range="-2 0" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="3"/>
        <body name="patella" pos="0.05 0 0">
          <joint name="px" type="slide" axis="1 0 0"/>
          <joint name="pz" type="slide" axis="0 0 1"/>
          <geom size="0.01" mass="1e-3"/>
          <site name="p"/>
        </body>
        <body name="via" pos="0.03 0 -0.1">
          <joint name="vx" type="slide" axis="1 0 0"/>
          <joint name="vz" type="slide" axis="0 0 1"/>
          <geom size="0.005" mass="1e-3"/>
          <site name="v"/>
        </body>
      </body>
    </body>
    <site name="origin" pos="0.05 0 0.1"/>
  </worldbody>
  <equality>
    <joint joint1="px" joint2="knee" polycoef="0 0.02 -0.01 0 0"/>
    <joint joint1="pz" joint2="knee" polycoef="0 0.01 0.005 0 0"/>
    <joint joint1="vx" joint2="knee" polycoef="0 -0.01 0.002 0 0"/>
    <joint joint1="vz" joint2="knee" polycoef="0 0.004 0 0 0"/>
  </equality>
  <tendon>
    <spatial name="t"><site site="origin"/><site site="p"/><site site="v"/></spatial>
  </tendon>
  <actuator>
    <general tendon="t" gaintype="millard_mtu" biastype="none" dyntype="muscle"
             dynprm="0.01 0.04" ctrllimited="true" ctrlrange="0 1"
             gainprm="300 0.1 0.35 10 0.1 0.1"/>
    <motor joint="hip" gear="20" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def _ctrl(nu):
  return lambda i: 0.5 + 0.5 * np.sin(0.01 * i + np.arange(nu))


class SolverPrecisionTest(parameterized.TestCase):

  @parameterized.parameters('coupled', 'constraints.xml')
  def test_solves_the_single_precision_inputs_in_double(self, name):
    """The widened solve is the double solve of the same f32 inputs, cast back.

    Whether that is worth it depends on the model's conditioning, and is measured
    on the musculoskeletal models it is for; here the plumbing is: which inputs,
    which precision, which outputs.
    """
    if name == 'coupled':
      m = mujoco.MjModel.from_xml_string(_COUPLED)
    else:
      m = test_util.load_test_file(name)
    d = mujoco.MjData(m)
    for i in range(50):
      d.ctrl[:] = _ctrl(m.nu)(i)
      mujoco.mj_step(m, d)
    mx32 = mjx.put_model(m)
    mxw = mjx.put_model(m, solver_dtype=jp.float64)
    dx = mjx.put_data(m, d)
    for fn in (mjx.fwd_position, mjx.fwd_velocity, mjx.fwd_actuation,
               mjx.fwd_acceleration):
      dx = jax.jit(fn)(mx32, dx)

    out = jax.jit(solver.solve)(mxw, dx)
    with enable_x64():
      ref = jax.jit(solver._solve)(solver._cast(mx32, jp.float64),
                                   solver._cast(dx, jp.float64))
    # the same computation compiled in two programs, so equal up to fusion:
    # elements that are roundoff around zero may differ by 1e-17
    def check(a, b, name):
      self.assertEqual(a.dtype, jp.float32, name)
      b = np.asarray(b).astype(np.float32)
      scale = max(np.abs(b).max(), 1e-30)
      np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-10 * scale, err_msg=name)

    for f in ('qacc', 'qacc_warmstart', 'qfrc_constraint'):
      check(getattr(out, f), getattr(ref, f), f)
    check(out._impl.efc_force, ref._impl.efc_force, 'efc_force')

  def test_nothing_outside_the_solve_is_double(self):
    m = mujoco.MjModel.from_xml_string(_COUPLED)
    mx = mjx.put_model(m, solver_dtype='float64')
    dx = mjx.make_data(m)
    closed = jax.make_jaxpr(mjx.step)(mx, dx)

    inside = outside = 0

    # a sub-jaxpr's name stack is relative to the equation that holds it, so
    # being inside the solve is inherited rather than read off each equation
    def walk(jaxpr, in_solve):
      nonlocal inside, outside
      for e in jaxpr.eqns:
        here = in_solve or 'solve' in str(e.source_info.name_stack)
        dtypes = [str(getattr(o.aval, 'dtype', '')) for o in e.outvars]
        if 'float64' in dtypes:
          if here:
            inside += 1
          else:
            outside += 1
        for p in e.params.values():
          for sub in p if isinstance(p, (list, tuple)) else [p]:
            if hasattr(sub, 'jaxpr') and hasattr(sub.jaxpr, 'eqns'):
              walk(sub.jaxpr, here)
            elif hasattr(sub, 'eqns'):
              walk(sub, here)

    walk(closed.jaxpr, False)
    self.assertGreater(inside, 0)
    self.assertEqual(outside, 0)

  def test_gradient_with_one_iteration(self):
    """Reverse mode goes through the widened solve where the plain one does."""
    m = mujoco.MjModel.from_xml_string(_COUPLED)
    m.opt.iterations = 1
    dx = mjx.put_data(m, mujoco.MjData(m))

    def height(mx, ctrl):
      return jp.sum(mjx.step(mx, dx.replace(ctrl=ctrl)).qvel)

    ctrl = jp.full(m.nu, 0.3, dtype=jp.float32)
    g_mix = jax.jit(jax.grad(height, argnums=1))(
        mjx.put_model(m, solver_dtype=jp.float64), ctrl
    )
    self.assertEqual(g_mix.dtype, jp.float32)
    with enable_x64():
      g_64 = jax.jit(jax.grad(height, argnums=1))(
          mjx.put_model(m), jp.asarray(ctrl, dtype=jp.float64)
      )
    np.testing.assert_allclose(g_mix, g_64, rtol=1e-3, atol=1e-4)

  def test_vmap(self):
    m = mujoco.MjModel.from_xml_string(_COUPLED)
    mx = mjx.put_model(m, solver_dtype=jp.float64)
    dx = mjx.make_data(m)
    ctrl = jp.linspace(0, 1, 3)[:, None] * jp.ones(m.nu)
    batch = jax.jit(jax.vmap(lambda c: mjx.step(mx, dx.replace(ctrl=c))))(ctrl)
    for i in range(3):
      one = jax.jit(mjx.step)(mx, dx.replace(ctrl=ctrl[i]))
      np.testing.assert_allclose(one.qacc, batch.qacc[i], rtol=1e-5, atol=1e-5)

  def test_rejects_what_it_cannot_do(self):
    m = mujoco.MjModel.from_xml_string(_COUPLED)
    with self.assertRaisesRegex(ValueError, 'float64 or None'):
      mjx.put_model(m, solver_dtype=jp.float32)
    with self.assertRaisesRegex(ValueError, 'JAX backend'):
      mjx.put_model(m, backend_impl='c', solver_dtype=jp.float64)


if __name__ == '__main__':
  absltest.main()
