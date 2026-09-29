# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""UMiTDS: native UMi plus three independently gated augmentation knobs.

T (tilt)     -- per-drop BS downtilt, acts in `set_topology` on `bs_orientations`.
D (delay)    -- per-link shift on the DS log-mean, acts in the scenario's LSP hook.
S (scramble) -- per-link phase on the RX antenna axis, acts in `__call__`.

Each knob is (flag, gate probability, law). The gate supplies the atom at native
that a continuous range cannot. With every flag off the class is inert: no
augmented scenario, no RNG, no extra op, no RNG draw -- output is bit-exact
native UMi.

`lsp_shift` reuses D's hook at other LSP indices (ungated, for calibration).
`fix_ray_offsets` is hygiene, not a knob: it corrects the TR 38.901 Table 7.5-3
entry that upstream Sionna carries as -0.1481 instead of -1.1481.
"""

import numpy as np
import tensorflow as tf

from sionna import config
from .umi import UMi
from .umi_scenario import UMiScenario


LSP_NAMES = ("DS", "ASD", "ASA", "SF", "K", "ZSA", "ZSD")
SIG_CAP = 3.0           # rad; caps sigma so lambda -> 1 stays finite
BAD_OFFSET_IDX = 15     # -0.1481 where Table 7.5-3 has -1.1481


def _check_range(name, r, lo=None, hi=None):
    if not (isinstance(r, (tuple, list)) and len(r) == 2):
        raise ValueError(f"{name} must be a (lo, hi) pair, got {r!r}")
    a, b = float(r[0]), float(r[1])
    if a > b:
        raise ValueError(f"{name} needs lo <= hi, got {r!r}")
    if lo is not None and a < lo:
        raise ValueError(f"{name} lower bound must be >= {lo}, got {r!r}")
    if hi is not None and b > hi:
        raise ValueError(f"{name} upper bound must be <= {hi}, got {r!r}")
    return a, b


def _check_prob(name, p):
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")
    return p


class UMiTDSScenario(UMiScenario):
    """UMi scenario whose LSP log-means accept a per-link additive shift."""

    _lsp_shift = None   # [batch, num_bs, num_ut, 7] stashed by UMiTDS.set_topology

    def _compute_lsp_log_mean_std(self):
        super()._compute_lsp_log_mean_std()
        if self._lsp_shift is not None:
            self._lsp_log_mean = self._lsp_log_mean + self._lsp_shift


class UMiTDS(UMi):
    """UMi with the T/D/S augmentation knobs. All flags off == native UMi."""

    def __init__(self, carrier_frequency, o2i_model, ut_array, bs_array,
        direction, enable_pathloss=True, enable_shadow_fading=True,
        always_generate_lsp=False,
        random_num_clusters=False, random_num_rays=False, mask_doa=False,
        num_rays=None, dtype=tf.complex64, *,
        tilt_aug=False, tilt_prob=0.3, tilt_range_deg=(-25.0, 15.0),
        ds_aug=False, ds_prob=0.3, ds_shift_range=(0.0, 0.4),
        lsp_shift=None,
        rx_scramble=False, rx_scramble_prob=0.3, lam_range=(0.0, 1.0),
        fix_ray_offsets=False, aug_seed=None):

        self._tilt_aug = bool(tilt_aug)
        self._ds_aug = bool(ds_aug)
        self._rx_scramble = bool(rx_scramble)

        tlo, thi = _check_range("tilt_range_deg", tilt_range_deg, -90.0, 90.0)
        self._tilt_lo = tlo*np.pi/180.0
        self._tilt_hi = thi*np.pi/180.0
        self._tilt_prob = _check_prob("tilt_prob", tilt_prob)

        self._ds_lo, self._ds_hi = _check_range("ds_shift_range", ds_shift_range)
        self._ds_prob = _check_prob("ds_prob", ds_prob)

        self._scr_prob = _check_prob("rx_scramble_prob", rx_scramble_prob)
        self._lam_lo, self._lam_hi = _check_range("lam_range", lam_range, 0.0, 1.0)

        self._shift_idx = {}
        for key, rng in (lsp_shift or {}).items():
            if key not in LSP_NAMES:
                raise ValueError(f"lsp_shift key {key!r} not in {LSP_NAMES}")
            if key == "DS" and self._ds_aug:
                raise ValueError("ds_aug and lsp_shift['DS'] both set; pick one")
            self._shift_idx[LSP_NAMES.index(key)] = _check_range(
                f"lsp_shift[{key!r}]", rng)

        self._shift_on = self._ds_aug or bool(self._shift_idx)
        self._aug_on = self._tilt_aug or self._shift_on or self._rx_scramble

        # One sub-stream per knob, so toggling one does not shift another's draws.
        if self._aug_on:
            seed = aug_seed
            if seed is None and config.seed is not None:
                seed = int(config.seed) ^ 0x5544D5
            root = (tf.random.Generator.from_non_deterministic_state()
                    if seed is None else tf.random.Generator.from_seed(seed))
            self._rng_t, self._rng_d, self._rng_s = root.split(3)

        if self._shift_on:
            self._scenario_cls = UMiTDSScenario

        super().__init__(carrier_frequency, o2i_model, ut_array, bs_array,
                         direction, enable_pathloss, enable_shadow_fading,
                         always_generate_lsp, random_num_clusters,
                         random_num_rays, mask_doa, num_rays, dtype)

        if fix_ray_offsets:
            off = self._ray_sampler._ray_offsets
            self._ray_sampler._ray_offsets = tf.tensor_scatter_nd_update(
                off, [[BAD_OFFSET_IDX]], tf.constant([-1.1481], off.dtype))

    def _draw_shift(self, shape, rdt):
        cols = []
        for i in range(len(LSP_NAMES)):
            if i == 0 and self._ds_aug:
                on = tf.cast(self._rng_d.uniform(shape, dtype=rdt)
                             < self._ds_prob, rdt)
                cols.append(on*self._rng_d.uniform(shape, self._ds_lo,
                                                   self._ds_hi, dtype=rdt))
            elif i in self._shift_idx:
                lo, hi = self._shift_idx[i]
                cols.append(self._rng_d.uniform(shape, lo, hi, dtype=rdt))
            else:
                cols.append(tf.zeros(shape, rdt))
        return tf.stack(cols, axis=-1)

    def set_topology(self, ut_loc=None, bs_loc=None, ut_orientations=None,
                     bs_orientations=None, ut_velocities=None, in_state=None,
                     los=None):

        # knob T: gated per-drop downtilt; the off branch is the native element
        if self._tilt_aug and bs_orientations is not None:
            rdt = bs_orientations.dtype
            shape = tf.shape(bs_orientations)[:2]
            on = self._rng_t.uniform(shape, dtype=rdt) < self._tilt_prob
            beta = self._rng_t.uniform(shape, self._tilt_lo, self._tilt_hi,
                                       dtype=rdt)
            bs_orientations = tf.stack([bs_orientations[..., 0],
                                        tf.where(on, beta,
                                                 bs_orientations[..., 1]),
                                        bs_orientations[..., 2]], axis=-1)

        # knob D (+ lsp_shift): stash the per-link shift the LSP hook adds
        if self._shift_on:
            rdt = self._scenario.dtype.real_dtype
            if ut_loc is not None and bs_loc is not None:
                shape = tf.stack([tf.shape(ut_loc)[0], tf.shape(bs_loc)[1],
                                  tf.shape(ut_loc)[1]])
            else:
                shape = tf.stack([self._scenario.batch_size,
                                  self._scenario.num_bs,
                                  self._scenario.num_ut])
            self._scenario._lsp_shift = self._draw_shift(shape, rdt)

        super().set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                             ut_velocities, in_state, los)

    def __call__(self, num_time_samples, sampling_frequency, foo=None):
        h, delays = super().__call__(num_time_samples, sampling_frequency, foo)

        # knob S: gated per-link phase on the RX antenna axis (axis 2)
        if not self._rx_scramble:
            return h, delays
        rdt = h.dtype.real_dtype
        shp = tf.shape(h)
        on = tf.cast(self._rng_s.uniform([shp[0]], dtype=rdt)
                     < self._scr_prob, rdt)
        lam = on*self._rng_s.uniform([shp[0]], self._lam_lo, self._lam_hi,
                                     dtype=rdt)
        floor = tf.exp(-tf.constant(SIG_CAP*SIG_CAP, rdt))
        sig = tf.sqrt(-tf.math.log(tf.maximum(1.0 - lam, floor)))
        sig = tf.reshape(sig, [-1, 1, 1, 1, 1, 1, 1])
        phi = sig*self._rng_s.normal(
            tf.stack([shp[0], shp[1], shp[2], 1, 1, shp[5], 1]), dtype=rdt)
        return h*tf.complex(tf.cos(phi), tf.sin(phi)), delays
