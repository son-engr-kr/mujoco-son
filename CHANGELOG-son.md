# mujoco-son changelog

Changes made by this fork, on top of upstream MuJoCo. Upstream's own changelog is
[doc/changelog.rst](doc/changelog.rst) and is left untouched so it stays mergeable.

The son6 line is based on upstream 3.16.0. The son5 line (3.3.3, with `compliant_mtu`,
`millard_mtu`, `hyfydy_mtu` and son's MJX) stays on the `son-build` branch and its releases.

## v3.16.0+son6.1

The activation of DiffNeuSim's muscles can now be written on the shortcuts, so a harness no longer
has to adjust the control to match it. DiffNeuSim runs MuJoCo's `mju_muscleDynamics` with a smooth
switch between the time constants (`tausmooth`, 0.1 by default there), no excitation floor, and the
fiber solved with the next activation, for every muscle model.

- `<millard2012 tausmooth>`: width of the smooth switch between the activation and deactivation
  time constants, as on `<muscle>` and `<geyer2010>`. The default of 0 is OpenSim's abrupt switch.
  It is stored in `gainprm[9]` (`mjMILLARD_TAUSMOOTH`), the free slot, because dynprm is full.
- `actearly` is accepted on `<millard2012>` (default false, as OpenSim integrates activation and
  fiber together) and `<geyer2010>` (default true, Song's order).
- `mjs_setToMillard2012` and `set_to_millard2012` take `tausmooth` after `timeconst`, as
  `mjs_setToGeyer2010` does. C callers of son6.0's setter must add the argument; Python keyword
  callers are unaffected.

For DiffNeuSim's activation on a Millard2012 muscle: `tausmooth="0.1" minactivation="0"
actearly="true"`. The C suite passes (2011 tests).

These options are in son6 only, not in google-deepmind/mujoco#3943.

## v3.16.0+son6.0

Upstream MuJoCo 3.16.0 with the compliant-tendon muscle actuators proposed upstream in
google-deepmind/mujoco#3943, backported from that pull request's commit (6e718c6ec). The only change
to the muscle code is an include of `engine_util_blas.h`, where 3.16.0 defines `mju_abs` and the
other math macros:

- `<millard2012>`: OpenSim's `Millard2012EquilibriumMuscle`, with OpenSim's parameters and curves.
- `<geyer2010>`: the muscle of Geyer and Herr (2010) as DiffNeuSim implements it (jinsimul-cuda
  `core/mtu`); force within 3.7e-6 F_max of DiffNeuSim's on all 356 muscles of its body.
- `act = [fiber, activation]` per muscle, with fiber = l_ce/l_opt - 1; `mj_muscleEquilibrium` sets
  every fiber to its isometric equilibrium after a pose is set rather than integrated.

The son5 muscle gaintypes are not in son6, so models that use `gaintype="compliant_mtu"`,
`millard_mtu` or `hyfydy_mtu` do not load. To port a `compliant_mtu` muscle to `<geyer2010>`, write
the active curve's residual as `c = exp(C)`, set `tausmooth="0.1"`, mark the tendon
`rigidtendon="true"` where l_slack < 0.05 l_opt (DiffNeuSim's rule), and set `fiberdamping="0"` on
rigid muscles for DiffNeuSim's force.

There is no MJX wheel: son6's MJX is upstream's, which has no compliant-tendon muscles. Wheels are
built for CPython 3.11 and 3.12 on Linux x86-64, macOS arm64 and Windows x86-64.

The C suite passes (2009 tests).
