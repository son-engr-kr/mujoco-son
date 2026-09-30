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
"""Muscle-tendon units: gaintype compliant_mtu, millard_mtu and hyfydy_mtu.

A port of src/engine/engine_muscle_mtu.c (millard_mtu, hyfydy_mtu) and of the
compliant_mtu section of src/engine/engine_util_misc.c. The C sources carry the
derivations and the reasons behind every rule. This file keeps their structure
function for function, so the two read side by side, and comments only where
JAX forces a difference. There are three, and a fourth in single precision,
where the rigid tendon's buckling margin is taken at the simulated precision
(_significant_real):

  * Every loop runs a fixed trip count. An element that has converged is frozen
    with jp.where, so it ends on exactly the value the C loop breaks with, and
    the trip count is the C loop's own cap.
  * Every branch is evaluated on every element and selected with jp.where. A
    branch that is not taken must still be finite, or reverse-mode gradients
    are NaN, so each branch is fed an input it is defined at.
  * Gradients do not go through the solver iterations. The loops run on
    stop_gradient'ed inputs, and _implicit attaches the implicit-function
    derivative of the root they return. That is the derivative of the root
    itself, where differentiating the iterations would give the derivative of
    the algorithm: zero whenever the warm start already met the tolerance.
"""

import ctypes
from typing import Dict, NamedTuple, Tuple

import jax
from jax import numpy as jp
import mujoco
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import GainType
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np


MTU_GAINS = (GainType.COMPLIANT_MTU, GainType.MILLARD_MTU, GainType.HYFYDY_MTU)

# gainprm slots, engine_muscle_mtu.h
_FMAX, _LOPT, _LSLACK, _VMAX, _PENNATION, _BETA, _TOL, _MINACT = range(8)
_AFL_MIN = 8
_IGNORE_TENDON_COMPLIANCE = 19

# mjtMuscleCurve: the column of Model.actuator_mtu_curve
_ACTIVE_FL, _PASSIVE_FL, _TENDON_FL, _FV = range(4)
_NCURVE = 4

# engine_muscle_mtu.c
_SINPHIMAX = 0.9949874371066201
_RIGID_RATIO = 0.05
_NITER_DEFAULT = 12  # Newton iterations when opt.cmtu_iter <= 0

# Hyfydy User Manual v1.0.2, "Actuator Forces"; see engine_muscle_mtu.c for
# why the passive curve's second coefficient is read as c_P2
_HYF_CT1 = 260.972
_HYF_CT2 = 7.9706
_HYF_CL1 = 1.5
_HYF_CL2 = -2.75
_HYF_R1 = 0.46899
_HYF_R2 = 1.80528
_HYF_CV1 = 0.227
_HYF_CV2 = 0.110
_HYF_FVMAX = 1.6
_HYF_CP1 = 1.08027
_HYF_CP2 = 1.27368

# engine_util_misc.c, compliant_mtu
_CMTU_RIGID_RATIO = 0.05
_CMTU_FIBER_GUARD = 1e-6
_CMTU_NEWTON_ITER = 50
_CMTU_NEWTON_TOL = 1e-5
_CMTU_STEADY_TOL = 1e-6


def _significant_real(dtype) -> float:
  """SimTK::SignificantReal, eps^(7/8), at the precision being simulated.

  C uses the double value, 2.0e-14, as its margin for calling a rigid tendon
  buckled, and this reproduces it bit for bit in double. SimTK defines the
  constant per precision, and in single precision the double value is far below
  the roundoff of l_MTU - l_ce cos(phi), which would call a taut tendon buckled
  at random.

  Args:
    dtype: the floating point type

  Returns:
    the margin
  """
  eps = np.finfo(dtype).eps
  return float(eps / np.sqrt(np.sqrt(np.sqrt(eps))))


def _clip(x: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
  """mju_clip, including its order of tests when lo > hi."""
  return jp.where(x < lo, lo, jp.where(x > hi, hi, x))


def _implicit(
    x: jax.Array,
    r: jax.Array,
    dr: jax.Array,
    pinned: jax.Array,
    pinned_value: jax.Array,
) -> jax.Array:
  """x unchanged in value, carrying the derivative of the root of r.

  `x` is a root found on stop_gradient'ed inputs and `r` the residual evaluated
  there with live ones, so r - stop_gradient(r) is exactly zero and carries
  dr/dtheta. Dividing by dr/dx gives dx/dtheta = -(dr/dtheta)/(dr/dx), the
  implicit function theorem. Where the solver ended on a bound instead of a root
  (`pinned`), x is the bound and its derivative is the bound's.

  Args:
    x: the solver's answer
    r: residual at x, with gradients flowing through everything but x
    dr: dr/dx at x
    pinned: where x is a bound rather than a root
    pinned_value: the bound, live

  Returns:
    x, with the root's derivative attached
  """
  inv = jp.where(dr != 0, 1 / jp.where(dr != 0, dr, 1), 0)
  inv = jp.where(jp.isfinite(inv), inv, 0)
  sg = jax.lax.stop_gradient
  x = x - (r - sg(r)) * sg(inv)
  return jp.where(pinned, pinned_value, x)


def _stop(tree):
  return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


# ------------------------------ Millard2012 curve table ------------------------------


def _table_eval(
    knot: jax.Array, meta: jax.Array, tid: np.ndarray, x: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """mju_curveTableEval, for one curve of each actuator in a group.

  Args:
    knot: (ntable, n, 3) knots, (y, dy/dx, d2y/dx2)
    meta: (ntable, 8) x0, x1, dx, inv_dx, a_y0, a_d0, a_y1, a_d1
    tid: (k,) table row of each actuator
    x: (k,) abscissa

  Returns:
    value and slope at x
  """
  mt = meta[tid]
  x0, x1, dx, inv_dx = mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3]
  a_y0, a_d0, a_y1, a_d1 = mt[:, 4], mt[:, 5], mt[:, 6], mt[:, 7]
  n = knot.shape[1]

  # the interior formula is evaluated outside [x0, x1] too, so s is bounded to
  # keep it finite there; inside, s < n - 1 + roundoff and the bound never binds
  s = jp.clip((x - x0) * inv_dx, 0, n)
  i = jp.minimum(jp.floor(s).astype(jp.int32), n - 2)
  u = s - i
  u2 = u * u
  u3 = u2 * u
  u4 = u3 * u
  u5 = u4 * u
  h = dx
  k_i, k_j = knot[tid, i], knot[tid, i + 1]
  y_i, d_i, e_i = k_i[:, 0], k_i[:, 1], k_i[:, 2]
  y_j, d_j, e_j = k_j[:, 0], k_j[:, 1], k_j[:, 2]

  # quintic Hermite basis on the unit cell
  h0 = 1 - 10 * u3 + 15 * u4 - 6 * u5
  h1 = u - 6 * u3 + 8 * u4 - 3 * u5
  h2 = 0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5
  g0 = 10 * u3 - 15 * u4 + 6 * u5
  g1 = -4 * u3 + 7 * u4 - 3 * u5
  g2 = 0.5 * u3 - u4 + 0.5 * u5
  y = (
      h0 * y_i
      + h1 * h * d_i
      + h2 * h * h * e_i
      + g0 * y_j
      + g1 * h * d_j
      + g2 * h * h * e_j
  )

  p0 = -30 * u2 + 60 * u3 - 30 * u4
  p1 = 1 - 18 * u2 + 32 * u3 - 15 * u4
  p2 = u - 4.5 * u2 + 6 * u3 - 2.5 * u4
  q0 = 30 * u2 - 60 * u3 + 30 * u4
  q1 = -12 * u2 + 28 * u3 - 15 * u4
  q2 = 1.5 * u2 - 4 * u3 + 2.5 * u4
  dydx = (
      (p0 * y_i + q0 * y_j) * inv_dx
      + p1 * d_i
      + q1 * d_j
      + (p2 * e_i + q2 * e_j) * h
  )

  below, above = x <= x0, x >= x1
  y = jp.where(below, a_y0 + a_d0 * (x - x0), jp.where(above, a_y1 + a_d1 * (x - x1), y))
  dydx = jp.where(below, a_d0, jp.where(above, a_d1, dydx))
  return y, dydx


# ------------------------------ Hyfydy muscle_force_m2012fast curves ------------------------------


def _hyfydy_fl(l: jax.Array) -> Tuple[jax.Array, jax.Array]:
  inside = (l > _HYF_R1) & (l < _HYF_R2)
  u = l - 1
  val = _HYF_CL1 * u * u * u + _HYF_CL2 * u * u + 1
  dval = 3 * _HYF_CL1 * u * u + 2 * _HYF_CL2 * u
  return jp.where(inside, val, 0), jp.where(inside, dval, 0)


def _hyfydy_fp(l: jax.Array) -> Tuple[jax.Array, jax.Array]:
  stretched = l > 1
  u = l - 1
  val = _HYF_CP1 * u * u * u + _HYF_CP2 * u * u
  dval = 3 * _HYF_CP1 * u * u + 2 * _HYF_CP2 * u
  return jp.where(stretched, val, 0), jp.where(stretched, dval, 0)


def _hyfydy_fv(v: jax.Array) -> Tuple[jax.Array, jax.Array]:
  shortening = (v > -1) & (v < 0)
  lengthening = v >= 0

  # each rational has a pole on the other branch's side, so each is evaluated
  # at a point it is defined at wherever it is not selected
  vn = jp.where(shortening, v, -0.5)
  den = _HYF_CV1 - vn
  val_n = _HYF_CV1 * (vn + 1) / den
  dval_n = _HYF_CV1 * (_HYF_CV1 + 1) / (den * den)

  vp = jp.where(lengthening, v, 0)
  den = _HYF_CV2 + vp
  val_p = (_HYF_FVMAX * vp + _HYF_CV2) / den
  dval_p = _HYF_CV2 * (_HYF_FVMAX - 1) / (den * den)

  val = jp.where(lengthening, val_p, jp.where(shortening, val_n, 0))
  dval = jp.where(lengthening, dval_p, jp.where(shortening, dval_n, 0))
  return val, dval


def _hyfydy_ft(x: jax.Array) -> Tuple[jax.Array, jax.Array]:
  e = x - 1
  taut = e > 0
  val = _HYF_CT1 * e * e + _HYF_CT2 * e
  dval = 2 * _HYF_CT1 * e + _HYF_CT2
  return jp.where(taut, val, 0), jp.where(taut, dval, 0)


class _Curves:
  """The four normalized curves of one group of millard_mtu or hyfydy_mtu."""

  def __init__(
      self,
      hyfydy: bool,
      tid: np.ndarray = None,
      knot: jax.Array = None,
      meta: jax.Array = None,
  ):
    self.hyfydy = hyfydy
    self.tid = tid  # (k, 4), row of each actuator's table for each curve
    self.knot = knot
    self.meta = meta

  def stop_gradient(self) -> '_Curves':
    if self.hyfydy:
      return self
    return _Curves(False, self.tid, *_stop((self.knot, self.meta)))

  def _table(self, curve: int, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    return _table_eval(self.knot, self.meta, self.tid[:, curve], x)

  def active(self, l0):
    return _hyfydy_fl(l0) if self.hyfydy else self._table(_ACTIVE_FL, l0)

  def passive(self, l0):
    return _hyfydy_fp(l0) if self.hyfydy else self._table(_PASSIVE_FL, l0)

  def tendon(self, x):
    return _hyfydy_ft(x) if self.hyfydy else self._table(_TENDON_FL, x)

  def fv(self, v0):
    return _hyfydy_fv(v0) if self.hyfydy else self._table(_FV, v0)


# ------------------------------ shared parameters and equilibrium ------------------------------


class _MtuParams(NamedTuple):
  """mjMtuParams, one entry per actuator of a group."""

  F_max: jax.Array  # pylint: disable=invalid-name
  l_opt: jax.Array
  l_slack: jax.Array
  v_max_ms: jax.Array
  pen_h: jax.Array
  beta: jax.Array
  min_act: jax.Array
  tol: jax.Array
  lce_min: jax.Array
  rigid: jax.Array  # mtuIsRigid


def _vmax(prm: jax.Array) -> jax.Array:
  return jp.where(prm[:, _VMAX] == 0, 10, prm[:, _VMAX])


def _mtu_params(prm: jax.Array, hyfydy: bool) -> _MtuParams:
  """mtuResolveDefaults followed by mtuGetParams, for the slots the solver reads."""
  l_opt, l_slack = prm[:, _LOPT], prm[:, _LSLACK]
  beta = jp.where(prm[:, _BETA] == 0, 0.1, prm[:, _BETA])
  beta = jp.where(beta < 0, 0, beta)
  min_act = jp.where(prm[:, _MINACT] == 0, 0.01, prm[:, _MINACT])
  min_act = jp.where(min_act < 0, 0, min_act)
  tol = jp.where(prm[:, _TOL] == 0, 1e-9, prm[:, _TOL])

  phi_opt = prm[:, _PENNATION]
  pen_h = jp.where(phi_opt != 0, l_opt * jp.sin(phi_opt), 0)

  if hyfydy:
    lce_min_norm = _HYF_R1
  else:
    afl_min = prm[:, _AFL_MIN]
    lce_min_norm = jp.where(afl_min == 0, 0.47 - 0.0259, afl_min)
  lce_min = jp.maximum(lce_min_norm * l_opt, pen_h / _SINPHIMAX)

  rigid_declared = prm[:, _IGNORE_TENDON_COMPLIANCE] == 1
  return _MtuParams(
      F_max=prm[:, _FMAX],
      l_opt=l_opt,
      l_slack=l_slack,
      v_max_ms=_vmax(prm) * l_opt,
      pen_h=pen_h,
      beta=beta,
      min_act=min_act,
      tol=tol,
      lce_min=lce_min,
      rigid=rigid_declared | (l_slack < _RIGID_RATIO * l_opt),
  )


class _MtuEval(NamedTuple):
  """The parts of mjMtuEval that callers read."""

  l_T: jax.Array  # pylint: disable=invalid-name
  f_T: jax.Array  # pylint: disable=invalid-name
  R: jax.Array  # pylint: disable=invalid-name
  dR: jax.Array  # pylint: disable=invalid-name


def _mtu_eval(
    curves: _Curves,
    p: _MtuParams,
    l_ce: jax.Array,
    l_mtu: jax.Array,
    l_ce_prev: jax.Array,
    a: jax.Array,
    dtv: jax.Array,
) -> _MtuEval:
  """mtuEval: residual and analytic Jacobian at l_ce. dtv <= 0 is isometric."""
  l0 = l_ce / p.l_opt
  moving = dtv > 0
  dtv_safe = jp.where(moving, dtv, 1)
  v0 = jp.where(moving, (l_ce - l_ce_prev) / dtv_safe, 0)

  # fixed-width pennation, with dphi taken from the clamped sin(phi); for h == 0
  # this gives cos(phi) = 1 and dphi = 0, but the branch is kept explicit
  pennated = p.pen_h > 0
  sp = jp.minimum(p.pen_h / l_ce, _SINPHIMAX)
  r = 1 - sp * sp
  dphi = jp.where(pennated, -(sp / l_ce) / jp.sqrt(r), 0)
  sin_phi = jp.where(pennated, sp, 0)
  cos_phi = jp.where(pennated, jp.sqrt(r), 1)

  l_t = l_mtu - l_ce * cos_phi
  dlt = l_ce * sin_phi * dphi - cos_phi
  dcos = -sin_phi * dphi

  f_t, d_t = curves.tendon(l_t / p.l_slack)
  f_l, d_l = curves.active(l0)
  f_p, d_p = curves.passive(l0)
  f_v, d_v = curves.fv(v0)

  inv_dtv = jp.where(moving, 1 / dtv_safe, 0)
  fib = a * f_l * f_v + f_p + p.beta * v0
  d_fib = (
      a * (d_l * f_v / p.l_opt + f_l * d_v * inv_dtv)
      + d_p / p.l_opt
      + p.beta * inv_dtv
  )

  res = f_t - cos_phi * fib
  dres = d_t * (dlt / p.l_slack) - (dcos * fib + cos_phi * d_fib)
  return _MtuEval(l_T=l_t, f_T=f_t, R=res, dR=dres)


def _mtu_newton(
    curves: _Curves,
    p: _MtuParams,
    l_ce: jax.Array,
    l_mtu: jax.Array,
    a: jax.Array,
    dtv: jax.Array,
    niter: int,
) -> jax.Array:
  """mtuSolve: Newton on R(l_ce) = 0, warm-started at l_ce."""
  max_step = 0.5 * p.l_opt

  def body(_, carry):
    x, done = carry
    e = _mtu_eval(curves, p, x, l_mtu, l_ce, a, dtv)
    done = done | (jp.abs(e.R) < p.tol) | (e.dR == 0)
    step = jp.clip(-e.R / jp.where(e.dR == 0, 1, e.dR), -max_step, max_step)
    x = jp.where(done, x, jp.maximum(x + step, p.lce_min))
    return x, done

  x = jp.maximum(l_ce, p.lce_min)
  x, _ = jax.lax.fori_loop(0, niter, body, (x, jp.zeros(x.shape, bool)))
  return x


def _mtu_isometric(
    curves: _Curves,
    p: _MtuParams,
    l_mtu: jax.Array,
    a: jax.Array,
    bracket_iter: int,
    refine_iter: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """mtuSolveIsometric: the bracketed isometric root-find.

  Args:
    curves: the group's curves
    p: the group's parameters
    l_mtu: path length
    a: activation
    bracket_iter: cap on the bracket walk, 64 in C
    refine_iter: cap on the Newton/bisection refinement, 100 in C

  Returns:
    x: the fiber length mtuSolveIsometric returns
    x_eval: where it last evaluated the residual, which is x except when the
      bracket walk ran out: C then returns the next point up without evaluating
      it, and reports the state at the last point it did evaluate
    on_clamp: the fiber sits on its lower clamp
    bracketed: the walk found a sign change, so x is a root
  """
  zero = jp.zeros_like(l_mtu)

  def residual(x):
    return _mtu_eval(curves, p, x, l_mtu, x, a, zero)

  lo = p.lce_min
  on_clamp = residual(lo).R <= 0

  def walk(_, carry):
    lo, hi, hi_eval, found = carry
    stop = found | on_clamp
    negative = residual(hi).R < 0
    advance = ~stop & ~negative
    hi_eval = jp.where(stop, hi_eval, hi)
    lo = jp.where(advance, hi, lo)
    hi = jp.where(advance, hi + p.l_opt, hi)
    return lo, hi, hi_eval, found | (~stop & negative)

  hi = lo + p.l_opt
  init = (lo, hi, hi, jp.zeros(lo.shape, bool))
  lo, hi, hi_eval, bracketed = jax.lax.fori_loop(0, bracket_iter, walk, init)

  def refine(_, carry):
    lo, hi, x, done = carry
    e = residual(x)
    stop = done | (jp.abs(e.R) < p.tol)
    positive = e.R > 0
    lo_next = jp.where(positive, x, lo)
    hi_next = jp.where(positive, hi, x)
    nxt = jp.where(e.dR != 0, x - e.R / jp.where(e.dR != 0, e.dR, 1), x)
    inside = (nxt > lo_next) & (nxt < hi_next)
    nxt = jp.where(inside, nxt, 0.5 * (lo_next + hi_next))
    pinned = nxt == x
    lo = jp.where(stop, lo, lo_next)
    hi = jp.where(stop, hi, hi_next)
    x = jp.where(stop | pinned, x, nxt)
    return lo, hi, x, stop | pinned

  init = (lo, hi, 0.5 * (lo + hi), on_clamp | ~bracketed)
  _, _, x, _ = jax.lax.fori_loop(0, refine_iter, refine, init)

  x_eval = jp.where(on_clamp, p.lce_min, jp.where(bracketed, x, hi_eval))
  x = jp.where(on_clamp, p.lce_min, jp.where(bracketed, x, hi))
  return x, x_eval, on_clamp, bracketed


# ------------------------------ rigid tendon ------------------------------


class _Rigid(NamedTuple):
  l_ce: jax.Array
  l_T: jax.Array  # pylint: disable=invalid-name
  v_ce: jax.Array
  F: jax.Array  # pylint: disable=invalid-name
  dF_dvmtu: jax.Array  # pylint: disable=invalid-name


def _mtu_rigid(
    curves: _Curves,
    p: _MtuParams,
    a: jax.Array,
    l_mtu: jax.Array,
    v_mtu: jax.Array,
) -> _Rigid:
  """mtuRigidForce: OpenSim's ignore_tendon_compliance path."""
  along = l_mtu - p.l_slack
  pennated = p.pen_h > 0
  # sqrt(along^2 + 0) is |along| exactly; the unpennated case takes abs so its
  # derivative is defined where the path is exactly at slack length
  hyp = jp.where(pennated, along * along + p.pen_h * p.pen_h, 1)
  l_ce = jp.where(pennated, jp.sqrt(hyp), jp.abs(along))
  l_ce = jp.maximum(l_ce, p.lce_min)

  sp = jp.minimum(jp.where(pennated, p.pen_h / l_ce, 0), _SINPHIMAX)
  cos_phi = jp.sqrt(1 - sp * sp)
  l_t = l_mtu - l_ce * cos_phi

  buckled = l_t < p.l_slack - _significant_real(l_t.dtype)
  v_ce = jp.where(buckled, 0, v_mtu * cos_phi)
  clamped = (l_ce < p.lce_min) | ((l_ce <= p.lce_min) & (v_ce <= 0))

  l0 = l_ce / p.l_opt
  v0 = v_ce / p.v_max_ms
  f_l, _ = curves.active(l0)
  f_p, _ = curves.passive(l0)
  f_v, d_v = curves.fv(v0)
  f_v = jp.where(buckled, 1, f_v)
  d_v = jp.where(buckled, 0, d_v)

  f_fiber = a * f_l * f_v + f_p + p.beta * v0
  slack = clamped | (f_fiber <= 0)
  force = jp.where(slack, 0, p.F_max * f_fiber * cos_phi)
  df_dvmtu = jp.where(
      slack | buckled,
      0,
      p.F_max * cos_phi * cos_phi * (a * f_l * d_v + p.beta) / p.v_max_ms,
  )
  return _Rigid(l_ce=l_ce, l_T=l_t, v_ce=v_ce, F=force, dF_dvmtu=df_dvmtu)


# ------------------------------ compliant_mtu (Song) ------------------------------


class _CmtuParams(NamedTuple):
  """mjCompliantMuscleParams."""

  F_max: jax.Array  # pylint: disable=invalid-name
  l_opt: jax.Array
  l_slack: jax.Array
  v_max: jax.Array
  W: jax.Array  # pylint: disable=invalid-name
  C: jax.Array  # pylint: disable=invalid-name
  N: jax.Array  # pylint: disable=invalid-name
  K: jax.Array  # pylint: disable=invalid-name
  E_REF: jax.Array  # pylint: disable=invalid-name
  L_PE0: jax.Array  # pylint: disable=invalid-name
  E_REF_PE: jax.Array  # pylint: disable=invalid-name
  rigid: jax.Array  # mju_compliantMuscleIsRigid


def _cmtu_params(prm: jax.Array) -> _CmtuParams:
  """mju_compliantMuscleExtractParams."""
  geyer_pe = (prm[:, 9] == 0) & (prm[:, 10] == 0)
  l_opt, l_slack = prm[:, 1], prm[:, 2]
  return _CmtuParams(
      F_max=prm[:, 0],
      l_opt=l_opt,
      l_slack=l_slack,
      v_max=prm[:, 3],
      W=prm[:, 4],
      C=prm[:, 5],
      N=prm[:, 6],
      K=prm[:, 7],
      E_REF=prm[:, 8],
      L_PE0=jp.where(geyer_pe, 1.0, prm[:, 9]),
      E_REF_PE=jp.where(geyer_pe, prm[:, 4], prm[:, 10]),
      rigid=(prm[:, 11] == 1) | (l_slack < _CMTU_RIGID_RATIO * l_opt),
  )


def _cmtu_fp0(l0: jax.Array, e_ref: jax.Array) -> jax.Array:
  x = (l0 - 1) / e_ref
  return jp.where(l0 > 1, x * x, 0)


def _cmtu_fpe0(l0: jax.Array, rest: jax.Array, e_ref: jax.Array) -> jax.Array:
  x = (l0 - rest) / e_ref
  return jp.where(l0 > rest, x * x, 0)


def _cmtu_fp0_ext(l0: jax.Array, e_ref: jax.Array, e_ref2: jax.Array) -> jax.Array:
  x = (l0 - e_ref2) / e_ref
  return jp.where(l0 < e_ref2, x * x, 0)


def _cmtu_flce0(l_ce0: jax.Array, w: jax.Array, c: jax.Array) -> jax.Array:
  x = jp.abs((l_ce0 - 1) / w)
  return jp.exp(c * x * x * x)


def _cmtu_fvce0(v_ce0: jax.Array, k: jax.Array, n: jax.Array) -> jax.Array:
  """mju_compliantMuscleForwardVce0."""
  region1 = v_ce0 <= 0
  region2 = ~region1 & (v_ce0 <= 1)

  # each region's rational is evaluated at a point it is defined at wherever
  # that region is not selected
  v1 = jp.where(region1, v_ce0, 0)
  denom = 1 - k * v1
  denom = jp.where(denom == 0, 1e-8, denom)
  f1 = (1 + v1) / denom

  v2 = jp.where(region2, v_ce0, 0.5)
  denom_t = 7.56 * k * v2 + 1
  denom_t = jp.where(denom_t == 0, 1e-8, denom_t)
  t = (v2 - 1) / denom_t
  denom_f = t - 1
  denom_f = jp.where(denom_f == 0, 1e-8, denom_f)
  f2 = n - t / denom_f

  f3 = n + 100 * (v_ce0 - 1)
  return jp.where(region1, f1, jp.where(region2, f2, f3))


def _cmtu_fvce0_deriv(v_ce0: jax.Array, k: jax.Array, n: jax.Array) -> jax.Array:
  """mju_compliantMuscleForwardVce0Deriv."""
  del n  # the slope does not depend on it
  region1 = v_ce0 <= 0
  region2 = ~region1 & (v_ce0 <= 1)

  v1 = jp.where(region1, v_ce0, 0)
  denom = 1 - k * v1
  denom = jp.where(denom == 0, 1e-8, denom)
  d1 = (1 + k) / (denom * denom)

  v2 = jp.where(region2, v_ce0, 0.5)
  denom_t = 7.56 * k * v2 + 1
  denom_t = jp.where(denom_t == 0, 1e-8, denom_t)
  t = (v2 - 1) / denom_t
  dt = (1 + 7.56 * k) / (denom_t * denom_t)
  denom_f = t - 1
  denom_f = jp.where(denom_f == 0, -1e-8, denom_f)
  d2 = dt / (denom_f * denom_f)

  return jp.where(region1, d1, jp.where(region2, d2, 100))


def _cmtu_residual(
    p: _CmtuParams,
    a: jax.Array,
    l_ce: jax.Array,
    l_mtu: jax.Array,
    v_ce0: jax.Array,
) -> jax.Array:
  """mju_compliantMuscleResidual: Geyer & Herr 2010's force balance."""
  l_ce0 = l_ce / p.l_opt
  f_se0 = _cmtu_fp0((l_mtu - l_ce) / p.l_slack, p.E_REF)
  f_be0 = _cmtu_fp0_ext(l_ce0, 0.5 * p.W, 1 - p.W)
  f_pe0 = _cmtu_fpe0(l_ce0, p.L_PE0, p.E_REF_PE)
  f_lce0 = _cmtu_flce0(l_ce0, p.W, p.C)
  f_vce0 = _cmtu_fvce0(v_ce0, p.K, p.N)
  return f_se0 + f_be0 - f_vce0 * (f_pe0 + a * f_lce0)


def _cmtu_residual_dx(
    p: _CmtuParams,
    a: jax.Array,
    l_ce: jax.Array,
    l_mtu: jax.Array,
    v_ce0,
) -> Tuple[jax.Array, jax.Array]:
  """The residual and its exact d/dl_ce, for _implicit.

  Args:
    p: the group's parameters
    a: activation
    l_ce: fiber length
    l_mtu: path length
    v_ce0: the normalized fiber velocity as a function of l_ce

  Returns:
    R and dR/dl_ce, elementwise
  """
  fn = lambda x: _cmtu_residual(p, a, x, l_mtu, v_ce0(x))
  return jax.jvp(fn, (l_ce,), (jp.ones_like(l_ce),))


def _cmtu_bounds(p: _CmtuParams, l_mtu: jax.Array) -> Tuple[jax.Array, jax.Array]:
  return _CMTU_FIBER_GUARD * p.l_opt, l_mtu - _CMTU_FIBER_GUARD * p.l_opt


def _cmtu_newton(
    p: _CmtuParams,
    a: jax.Array,
    l_ce: jax.Array,
    l_mtu: jax.Array,
    dt: jax.Array,
) -> jax.Array:
  """mju_compliantMuscleNewtonStep, compliant tendon."""
  lo, hi = _cmtu_bounds(p, l_mtu)
  eps = 1e-5 * p.l_opt
  dtv = dt * p.l_opt * p.v_max

  def body(_, carry):
    x, done = carry
    res = _cmtu_residual(p, a, x, l_mtu, (x - l_ce) / dtv)
    stop = done | (jp.abs(res) < _CMTU_NEWTON_TOL)
    res_p = _cmtu_residual(p, a, x + eps, l_mtu, (x + eps - l_ce) / dtv)
    jac = (res_p - res) / eps
    jac = jp.where(jp.abs(jac) < 1e-6, jp.where(jac < 0, -1e-6, 1e-6), jac)
    nxt = _clip(x - 0.8 * res / jac, lo, hi)
    pinned = nxt == x
    x = jp.where(stop | pinned, x, nxt)
    return x, stop | pinned

  x = _clip(l_ce, lo, hi)
  x, _ = jax.lax.fori_loop(
      0, _CMTU_NEWTON_ITER, body, (x, jp.zeros(x.shape, bool))
  )
  return x


def _cmtu_steady_state(
    p: _CmtuParams,
    a: jax.Array,
    l_mtu: jax.Array,
    iterations: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """mju_compliantMuscleSolveSteadyState, compliant tendon.

  Args:
    p: the group's parameters
    a: activation
    l_mtu: path length
    iterations: cap on the refinement, 200 in C

  Returns:
    the fiber length, and where it sits on the lower or the upper bound
  """
  lo, hi = _cmtu_bounds(p, l_mtu)
  eps = 1e-5 * p.l_opt
  zero = jp.zeros_like(l_mtu)

  def residual(x):
    return _cmtu_residual(p, a, x, l_mtu, zero)

  at_lo = residual(lo) <= 0
  at_hi = ~at_lo & (residual(hi) > 0)

  def body(_, carry):
    lo, hi, x, step, done = carry
    res = residual(x)
    below = residual(x - 1e-8 * p.l_opt)
    converged = (jp.abs(res) < _CMTU_STEADY_TOL) & ((res > 0) | (below > 0))
    stop = done | converged
    positive = res > 0
    lo_next = jp.where(positive, x, lo)
    hi_next = jp.where(positive, hi, x)

    dres = (residual(x + eps) - res) / eps
    newton = jp.where(dres != 0, -res / jp.where(dres != 0, dres, 1), 0)
    nxt = x + newton
    bisect = (
        (dres == 0)
        | ~((nxt > lo_next) & (nxt < hi_next))
        | (jp.abs(newton) > 0.25 * jp.abs(step))
    )
    nxt = jp.where(bisect, 0.5 * (lo_next + hi_next), nxt)
    pinned = nxt == x
    update = ~stop & ~pinned
    lo = jp.where(stop, lo, lo_next)
    hi = jp.where(stop, hi, hi_next)
    step = jp.where(update, nxt - x, step)
    x = jp.where(update, nxt, x)
    return lo, hi, x, step, stop | pinned

  init = (lo, hi, 0.5 * (lo + hi), hi - lo, at_lo | at_hi)
  _, _, x, _, _ = jax.lax.fori_loop(0, iterations, body, init)
  x = jp.where(at_lo, lo, jp.where(at_hi, hi, x))
  return x, at_lo, at_hi


class _CmtuRigid(NamedTuple):
  l_ce: jax.Array
  v_ce: jax.Array
  F: jax.Array  # pylint: disable=invalid-name


def _cmtu_rigid(
    p: _CmtuParams, a: jax.Array, l_mtu: jax.Array, v_mtu: jax.Array
) -> _CmtuRigid:
  """mju_compliantMuscleRigidForce."""
  l_ce = jp.maximum(_CMTU_FIBER_GUARD * p.l_opt, l_mtu - p.l_slack)
  v_ce = v_mtu
  l_ce0 = l_ce / p.l_opt
  v_ce0 = v_ce / (p.l_opt * p.v_max)

  f_be0 = _cmtu_fp0_ext(l_ce0, 0.5 * p.W, 1 - p.W)
  f_pe0 = _cmtu_fpe0(l_ce0, p.L_PE0, p.E_REF_PE)
  f_lce0 = _cmtu_flce0(l_ce0, p.W, p.C)
  f_vce0 = _cmtu_fvce0(v_ce0, p.K, p.N)
  force = p.F_max * jp.maximum(0, f_vce0 * (f_pe0 + a * f_lce0) - f_be0)
  return _CmtuRigid(l_ce=l_ce, v_ce=v_ce, F=force)


# ------------------------------ group solves ------------------------------


class _Out(NamedTuple):
  """What mtuSolveAndReport / mju_compliantMuscleSolve write, per actuator."""

  l_ce: jax.Array
  l_se: jax.Array
  v_ce: jax.Array
  F: jax.Array  # pylint: disable=invalid-name


def _mtu_step(
    curves: _Curves,
    p: _MtuParams,
    a: jax.Array,
    l_ce: jax.Array,
    l_mtu: jax.Array,
    v_mtu: jax.Array,
    dt: jax.Array,
    niter: int,
) -> _Out:
  """mtuSolveAndReport with dtv > 0: one backward-Euler step of the fiber."""
  dtv = dt * p.v_max_ms
  rigid = _mtu_rigid(curves, p, a, l_mtu, v_mtu)

  x = _mtu_newton(
      curves.stop_gradient(), *_stop((p, l_ce, l_mtu, a, dtv)), niter
  )
  e = _mtu_eval(curves, p, x, l_mtu, l_ce, a, dtv)
  on_clamp = x <= jax.lax.stop_gradient(p.lce_min)
  x = _implicit(x, e.R, e.dR, on_clamp, p.lce_min)
  e = _mtu_eval(curves, p, x, l_mtu, l_ce, a, dtv)

  return _Out(
      l_ce=jp.where(p.rigid, rigid.l_ce, x),
      l_se=jp.where(p.rigid, rigid.l_T, e.l_T),
      v_ce=jp.where(p.rigid, rigid.v_ce, (x - l_ce) / dt),
      F=jp.where(p.rigid, rigid.F, p.F_max * e.f_T),
  )


def _mtu_equilibrium(
    curves: _Curves,
    p: _MtuParams,
    a: jax.Array,
    l_mtu: jax.Array,
    bracket_iter: int,
    refine_iter: int,
) -> _Out:
  """mtuSolveAndReport with dtv = 0: the isometric equilibrium."""
  zero = jp.zeros_like(l_mtu)
  rigid = _mtu_rigid(curves, p, a, l_mtu, zero)

  x, x_eval, on_clamp, bracketed = _mtu_isometric(
      curves.stop_gradient(),
      *_stop((p, l_mtu, a)),
      bracket_iter,
      refine_iter,
  )
  e = _mtu_eval(curves, p, x, l_mtu, x, a, zero)
  x = _implicit(x, e.R, e.dR, ~bracketed, jp.where(on_clamp, p.lce_min, x))
  x_eval = jp.where(bracketed | on_clamp, x, x_eval)
  e = _mtu_eval(curves, p, x_eval, l_mtu, x_eval, a, zero)

  return _Out(
      l_ce=jp.where(p.rigid, rigid.l_ce, x),
      l_se=jp.where(p.rigid, rigid.l_T, e.l_T),
      v_ce=jp.where(p.rigid, rigid.v_ce, 0),
      F=jp.where(p.rigid, rigid.F, p.F_max * e.f_T),
  )


def _cmtu_compliant(p: _CmtuParams) -> _CmtuParams:
  """p for the compliant-tendon branch, which a rigid tendon evaluates too.

  C never reaches the compliant solve for a rigid tendon, and a rigid tendon may
  have l_slack = 0, which the compliant residual divides by. Where the branch is
  not selected it sees l_slack = 1 instead, so its gradient stays finite.

  Args:
    p: the group's parameters

  Returns:
    p with a finite l_slack wherever the tendon is rigid
  """
  return p._replace(l_slack=jp.where(p.rigid, 1.0, p.l_slack))


def _cmtu_report(
    p: _CmtuParams,
    rigid: _CmtuRigid,
    x: jax.Array,
    l_mtu: jax.Array,
    v_ce: jax.Array,
) -> _Out:
  """The outputs of mju_compliantMuscleSolve at fiber length x."""
  l_se = l_mtu - x
  f_se0 = _cmtu_fp0(l_se / _cmtu_compliant(p).l_slack, p.E_REF)
  return _Out(
      l_ce=jp.where(p.rigid, rigid.l_ce, x),
      l_se=jp.where(p.rigid, p.l_slack, l_se),
      v_ce=jp.where(p.rigid, rigid.v_ce, v_ce),
      F=jp.where(p.rigid, rigid.F, p.F_max * f_se0),
  )


def _cmtu_step(
    p: _CmtuParams,
    a: jax.Array,
    l_ce: jax.Array,
    l_mtu: jax.Array,
    v_mtu: jax.Array,
    dt: jax.Array,
) -> _Out:
  """mju_compliantMuscleSolve with dt > 0."""
  rigid = _cmtu_rigid(p, a, l_mtu, v_mtu)
  pc = _cmtu_compliant(p)

  x = _cmtu_newton(*_stop((pc, a, l_ce, l_mtu, dt)))
  lo, hi = _cmtu_bounds(p, l_mtu)
  dtv = dt * p.l_opt * p.v_max
  res, dres = _cmtu_residual_dx(pc, a, x, l_mtu, lambda x: (x - l_ce) / dtv)
  # pinned means clipped onto a bound, which mju_clip does exactly; an ordering
  # test would pick the wrong bound when a path shorter than the guards puts
  # lo above hi
  at_lo = x == jax.lax.stop_gradient(lo)
  at_hi = x == jax.lax.stop_gradient(hi)
  x = _implicit(x, res, dres, at_lo | at_hi, jp.where(at_lo, lo, hi))

  return _cmtu_report(p, rigid, x, l_mtu, (x - l_ce) / dt)


def _cmtu_equilibrium(
    p: _CmtuParams, a: jax.Array, l_mtu: jax.Array, iterations: int
) -> _Out:
  """mju_compliantMuscleSolve with dt = 0."""
  zero = jp.zeros_like(l_mtu)
  rigid = _cmtu_rigid(p, a, l_mtu, zero)

  pc = _cmtu_compliant(p)
  x, at_lo, at_hi = _cmtu_steady_state(*_stop((pc, a, l_mtu)), iterations)
  lo, hi = _cmtu_bounds(p, l_mtu)
  res, dres = _cmtu_residual_dx(pc, a, x, l_mtu, lambda x: zero)
  x = _implicit(x, res, dres, at_lo | at_hi, jp.where(at_lo, lo, hi))

  return _cmtu_report(p, rigid, x, l_mtu, zero)


# ------------------------------ entry points ------------------------------


def _groups(m: Model) -> Dict[GainType, np.ndarray]:
  """The ids of the actuators of each muscle-tendon gain, in actuator order."""
  groups = {}
  for gain in MTU_GAINS:
    (ids,) = np.nonzero(m.actuator_gaintype == gain)
    if ids.size:
      groups[gain] = ids
  return groups


def _curves(m: Model, gain: GainType, ids: np.ndarray) -> _Curves:
  if gain == GainType.HYFYDY_MTU:
    return _Curves(hyfydy=True)
  impl = m._impl  # pytype: disable=attribute-error
  return _Curves(
      False, impl.actuator_mtu_curve[ids], impl.mtu_curve_knot, impl.mtu_curve_meta
  )


def _act_adr(m: Model, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """act = [l_ce, activation]: the fiber is first, the activation last."""
  fiber = m.actuator_actadr[ids]
  return fiber, fiber + m.actuator_actnum[ids] - 1


def _write_muscle(d: Data, ids: np.ndarray, out: _Out) -> Data:
  impl = d._impl  # pytype: disable=attribute-error
  return d.tree_replace({
      '_impl.muscle_l_ce': impl.muscle_l_ce.at[ids].set(out.l_ce),
      '_impl.muscle_l_se': impl.muscle_l_se.at[ids].set(out.l_se),
      '_impl.muscle_v_ce': impl.muscle_v_ce.at[ids].set(out.v_ce),
      '_impl.muscle_F_mtu': impl.muscle_F_mtu.at[ids].set(out.F),
  })


def fwd_actuation(
    m: Model, d: Data, act_dot: jax.Array
) -> Tuple[Data, jax.Array, jax.Array]:
  """The muscle-tendon pass of mj_fwdActuation.

  Solves every muscle-tendon unit's fiber equilibrium at the current state,
  writes the fiber velocity into its first activation slot, as
  mju_mtuMuscleActDot and mju_compliantMuscleActDot do, and returns the tendon
  force for the force pass to negate.

  Args:
    m: the model
    d: the data, after fwd_velocity
    act_dot: act_dot with the activation slots already filled, which actearly
      reads

  Returns:
    d with its muscle outputs written, act_dot with the fiber slots filled, and
    the (nu,) muscle-tendon force, zero for every other actuator
  """
  dt = m.opt.timestep
  impl = d._impl  # pytype: disable=attribute-error
  f_mtu = jp.zeros(m.nu, dtype=act_dot.dtype)

  for gain, ids in _groups(m).items():
    fiber, activation = _act_adr(m, ids)
    prm = m.actuator_gainprm[ids]
    a = d.act[activation]
    early = m.actuator_actearly[ids]
    if early.any():
      a = jp.where(early, a + dt * act_dot[activation], a)
    l_ce = d.act[fiber]
    l_mtu = impl.actuator_length[ids]
    v_mtu = impl.actuator_velocity[ids]

    if gain == GainType.COMPLIANT_MTU:
      p = _cmtu_params(prm)
      out = _cmtu_step(p, _clip(a, 0, 1), l_ce, l_mtu, v_mtu, dt)
    else:
      p = _mtu_params(prm, hyfydy=gain == GainType.HYFYDY_MTU)
      niter = m.opt.cmtu_iter if m.opt.cmtu_iter > 0 else _NITER_DEFAULT
      out = _mtu_step(
          _curves(m, gain, ids),
          p,
          _clip(a, p.min_act, 1),
          l_ce,
          l_mtu,
          v_mtu,
          dt,
          niter,
      )

    # the backward-Euler step reported as a velocity; see mju_mtuMuscleActDot
    act_dot = act_dot.at[fiber].set((out.l_ce - l_ce) / dt)
    f_mtu = f_mtu.at[ids].set(out.F)
    d = _write_muscle(d, ids, out)

  return d, act_dot, f_mtu


def force_vel(m: Model, d: Data) -> jax.Array:
  """d(actuator_force)/d(actuator_velocity) of every muscle-tendon unit.

  mju_mtuMuscleForceVel and mju_compliantMuscleForceVel: zero for a compliant
  tendon, and nonzero only on the rigid-tendon path.

  Args:
    m: the model
    d: the data, after fwd_velocity

  Returns:
    (nu,) array, zero for every other actuator
  """
  impl = d._impl  # pytype: disable=attribute-error
  vel = jp.zeros(m.nu, dtype=d.act.dtype)

  for gain, ids in _groups(m).items():
    _, activation = _act_adr(m, ids)
    prm = m.actuator_gainprm[ids]
    a = d.act[activation]
    l_mtu = impl.actuator_length[ids]
    v_mtu = impl.actuator_velocity[ids]

    if gain == GainType.COMPLIANT_MTU:
      p = _cmtu_params(prm)
      a = _clip(a, 0, 1)
      l_ce = jp.maximum(_CMTU_FIBER_GUARD * p.l_opt, l_mtu - p.l_slack)
      l_ce0 = l_ce / p.l_opt
      v_ce0 = v_mtu / (p.l_opt * p.v_max)
      f_be0 = _cmtu_fp0_ext(l_ce0, 0.5 * p.W, 1 - p.W)
      f_pe0 = _cmtu_fpe0(l_ce0, p.L_PE0, p.E_REF_PE)
      f_lce0 = _cmtu_flce0(l_ce0, p.W, p.C)
      f_vce0 = _cmtu_fvce0(v_ce0, p.K, p.N)
      pulling = f_vce0 * (f_pe0 + a * f_lce0) - f_be0 > 0
      dfv = _cmtu_fvce0_deriv(v_ce0, p.K, p.N)
      dvel = -p.F_max * (f_pe0 + a * f_lce0) * dfv / (p.l_opt * p.v_max)
      dvel = jp.where(p.rigid & pulling, dvel, 0)
    else:
      p = _mtu_params(prm, hyfydy=gain == GainType.HYFYDY_MTU)
      a = _clip(a, p.min_act, 1)
      rigid = _mtu_rigid(_curves(m, gain, ids), p, a, l_mtu, v_mtu)
      dvel = jp.where(p.rigid, -rigid.dF_dvmtu, 0)

    vel = vel.at[ids].set(dvel)

  return vel


def mtu_muscle_equilibrate(
    m: Model, d: Data, bracket_iter: int = 64, refine_iter: int = 100
) -> Data:
  """Puts every millard_mtu / hyfydy_mtu fiber at its isometric equilibrium.

  mju_mtuMuscleEquilibrate. Call it after forward (which fills
  actuator_length) whenever the pose was set rather than integrated: a reset,
  a keyframe, a qpos edit.

  The solve is bracketed and runs a fixed trip count: bracket_iter steps of the
  bracket walk, then refine_iter of Newton-or-bisection. The defaults are the C
  caps, so the result matches mju_mtuMuscleEquilibrate. A typical muscle needs
  a handful of each; lower caps trade that guarantee for speed, which matters
  when this runs every step inside a vmapped auto-reset.

  Args:
    m: the model
    d: the data, after forward
    bracket_iter: cap on the bracket walk
    refine_iter: cap on the refinement

  Returns:
    d with the fiber lengths, their act_dot and the muscle outputs updated
  """
  act, act_dot = d.act, d.act_dot
  impl = d._impl  # pytype: disable=attribute-error

  for gain, ids in _groups(m).items():
    if gain == GainType.COMPLIANT_MTU:
      continue
    fiber, activation = _act_adr(m, ids)
    p = _mtu_params(m.actuator_gainprm[ids], hyfydy=gain == GainType.HYFYDY_MTU)
    a = _clip(d.act[activation], p.min_act, 1)
    out = _mtu_equilibrium(
        _curves(m, gain, ids),
        p,
        a,
        impl.actuator_length[ids],
        bracket_iter,
        refine_iter,
    )
    act = act.at[fiber].set(out.l_ce)
    act_dot = act_dot.at[fiber].set(0)
    d = _write_muscle(d, ids, out)

  return d.replace(act=act, act_dot=act_dot)


def compliant_muscle_equilibrate(
    m: Model, d: Data, iterations: int = 200
) -> Data:
  """Puts every compliant_mtu fiber at its isometric equilibrium.

  mju_compliantMuscleEquilibrate. Call it after forward whenever the pose was
  set rather than integrated. `iterations` caps the bracketed refinement at a
  fixed trip count; the default is the C cap.

  Args:
    m: the model
    d: the data, after forward
    iterations: cap on the refinement

  Returns:
    d with the fiber lengths, their act_dot and the muscle outputs updated
  """
  ids = _groups(m).get(GainType.COMPLIANT_MTU)
  if ids is None:
    return d

  impl = d._impl  # pytype: disable=attribute-error
  fiber, activation = _act_adr(m, ids)
  p = _cmtu_params(m.actuator_gainprm[ids])
  a = _clip(d.act[activation], 0, 1)
  out = _cmtu_equilibrium(p, a, impl.actuator_length[ids], iterations)
  d = _write_muscle(d, ids, out)
  return d.replace(
      act=d.act.at[fiber].set(out.l_ce), act_dot=d.act_dot.at[fiber].set(0)
  )


# ------------------------------ put_model / make_data ------------------------------


class _CurveTable(ctypes.Structure):
  """mjCurveTable, engine_muscle_mtu.h."""

  _fields_ = [
      ('knot', ctypes.POINTER(ctypes.c_double)),
      ('n', ctypes.c_int),
      ('x0', ctypes.c_double),
      ('x1', ctypes.c_double),
      ('dx', ctypes.c_double),
      ('inv_dx', ctypes.c_double),
      ('a_y0', ctypes.c_double),
      ('a_d0', ctypes.c_double),
      ('a_y1', ctypes.c_double),
      ('a_d1', ctypes.c_double),
  ]


def _check_table(
    knot: np.ndarray, meta: np.ndarray, curve: int, gainprm: np.ndarray, name: str
) -> None:
  """Checks a table read through ctypes against mju_millardCurve.

  mjCurveTable is not public API, so its layout is read on trust. At a knot the
  quintic Hermite interpolant is the knot's own value, and outside the grid it
  is the anchor line, so comparing those against the public mju_millardCurve
  catches a layout that has drifted from _CurveTable.

  Args:
    knot: (n, 3) knots
    meta: (8,) x0, x1, dx, inv_dx, a_y0, a_d0, a_y1, a_d1
    curve: which curve
    gainprm: the actuator's gainprm row
    name: the actuator, for the message
  """
  x0, x1, dx, _, a_y0, a_d0, a_y1, a_d1 = meta
  n = knot.shape[0]
  expect = []
  for k in (1, n // 3, n - 2):
    expect.append((x0 + k * dx, knot[k, 0], knot[k, 1]))
  span = x1 - x0
  expect.append((x0 - 0.25 * span, a_y0 - 0.25 * span * a_d0, a_d0))
  expect.append((x1 + 0.25 * span, a_y1 + 0.25 * span * a_d1, a_d1))
  for x, y, dydx in expect:
    deriv = np.zeros(1)
    got = mujoco.mju_millardCurve(curve, x, gainprm, deriv)
    ok = np.allclose([got, deriv[0]], [y, dydx], rtol=1e-9, atol=1e-12)
    assert ok, (
        f'millard_mtu actuator {name}: baked curve {curve} read back as'
        f' ({got}, {deriv[0]}) at x={x} against the table\'s ({y}, {dydx}).'
        ' mjCurveTable no longer matches its ctypes mirror in muscle_mtu.py.'
    )


def _bake_curves(
    m: mujoco.MjModel, d: mujoco.MjData, ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The baked Millard curve tables of the millard_mtu actuators `ids`.

  Args:
    m: the model
    d: data made from m, whose mj_resetData baked the tables
    ids: the millard_mtu actuators

  Returns:
    knot: (ntable, n, 3) knots
    meta: (ntable, 8) x0, x1, dx, inv_dx, a_y0, a_d0, a_y1, a_d1
    curve: (nu, 4) row of each actuator's table for each curve, -1 for an
      actuator that is not millard_mtu
  """
  curve = np.full((m.nu, _NCURVE), -1, dtype=np.int32)
  if not ids.size:
    return np.zeros((0, 2, 3)), np.zeros((0, 8)), curve

  rows = {}
  knots, metas = [], []
  for i in ids:
    for c in range(_NCURVE):
      addr = int(d.muscle_curve[i, c])
      assert addr, f'millard_mtu actuator {i}: curve {c} was not baked'
      if addr not in rows:
        t = _CurveTable.from_address(addr)
        knot = np.ctypeslib.as_array(t.knot, shape=(t.n, 3)).copy()
        meta = np.array(
            [t.x0, t.x1, t.dx, t.inv_dx, t.a_y0, t.a_d0, t.a_y1, t.a_d1]
        )
        _check_table(knot, meta, c, m.actuator_gainprm[i], str(i))
        rows[addr] = len(knots)
        knots.append(knot)
        metas.append(meta)
      curve[i, c] = rows[addr]

  sizes = {k.shape[0] for k in knots}
  assert len(sizes) == 1, f'baked curves differ in knot count: {sizes}'
  return np.stack(knots), np.stack(metas), curve


def model_fields(m: mujoco.MjModel) -> Dict[str, np.ndarray]:
  """The ModelJAX fields of the muscle-tendon units.

  The engine bakes the Millard tables in mj_resetData, into a process-wide
  cache, and hands each mjData raw pointers into it (mjData.muscle_curve).
  Nothing about that crosses to a device, so this makes an mjData, whose reset
  also validates every muscle as mj_resetData always does, and copies the
  tables it points at. Actuators that share a curve shape share a row, as they
  share a table in C.

  The consequence is the C contract, one step later: the mechanical slots 0..7
  of actuator_gainprm are read every step, and the curve shape slots 8..31 are
  frozen here, at put_model, as C freezes them at mj_resetData. A shape changed
  on an mjx.Model after put_model is ignored; put the MjModel again.

  Args:
    m: the model

  Returns:
    mtu_curve_knot, mtu_curve_meta and actuator_mtu_curve
  """
  groups = _groups(m)
  if groups and not hasattr(m.opt, 'cmtu_iter'):
    raise NotImplementedError(
        'muscle-tendon units need mjOption.cmtu_iter, which this mujoco does'
        ' not bind; install the mujoco-son release this mujoco-mjx pins'
    )
  for gain, ids in groups.items():
    if gain == GainType.COMPLIANT_MTU:
      continue
    # mtuSolveAndReport switches to the isometric solve when dt*v_max*l_opt is
    # not positive; MJX runs the stepping solve unconditionally
    vmax = m.actuator_gainprm[ids, _VMAX]
    bad = ids[~(np.where(vmax == 0, 10, vmax) > 0)]
    if bad.size:
      raise NotImplementedError(
          f'{GainType(gain).name.lower()} actuators {bad.tolist()}: MJX'
          ' requires a positive gainprm[3] (max_contraction_velocity)'
      )

  millard = groups.get(GainType.MILLARD_MTU, np.zeros(0, dtype=int))
  d = mujoco.MjData(m) if groups else None
  knot, meta, curve = _bake_curves(m, d, millard)
  return {
      'mtu_curve_knot': knot,
      'mtu_curve_meta': meta,
      'actuator_mtu_curve': curve,
  }


def init_data(m: Model, act: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
  """mju_mtuMuscleInit / mju_compliantMuscleInit: seed the fiber at l_opt.

  A zero fiber length is not a physical state, and it is what make_data would
  otherwise leave there.

  Args:
    m: the model
    act: the zeroed act

  Returns:
    act, and the four muscle output fields
  """
  float_ = act.dtype
  muscle = {
      f'muscle_{k}': jp.zeros(m.nu, dtype=float_)
      for k in ('l_ce', 'l_se', 'v_ce', 'F_mtu')
  }
  for _, ids in _groups(m).items():
    fiber, activation = _act_adr(m, ids)
    l_opt = jp.asarray(m.actuator_gainprm[ids, _LOPT], dtype=float_)
    l_slack = jp.asarray(m.actuator_gainprm[ids, _LSLACK], dtype=float_)
    act = act.at[fiber].set(l_opt).at[activation].set(0)
    muscle['muscle_l_ce'] = muscle['muscle_l_ce'].at[ids].set(l_opt)
    muscle['muscle_l_se'] = muscle['muscle_l_se'].at[ids].set(l_slack)
  return act, muscle
