# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Generic CDL (GCDL): a parameterized CDL-family channel model.

Builds a CDL-schema parameter set (same keys as the ``models/CDL-*.json``
files) from a compact knob vector, following TR 38.901's own stochastic
recipe (Sec. 7.5 steps 5-7) made deterministic:

* delays  : exponential grid, normalized to unit power-weighted RMS
  (the pipeline then scales by the runtime delay spread);
* powers  : log-linear decay over normalized delay, total span given by
  ``decay_db_total``; optional specular row 0 (LoS mode, as in CDL-D/E);
* cluster centre angles : the spec's power-to-angle mapping (sqrt(-ln p)
  for azimuth, -ln p for zenith) with alternating signs instead of random
  ones, then re-centered and re-scaled EXACTLY to the requested
  power-weighted mean and RMS spread;
* intra-cluster spreads : ``cASD/cZSD/cASA/cZSA`` knobs verbatim
  (consumed by the standard 20-ray Table 7.5-3 offsets in ``CDL``).

Angle-knob naming uses the DOWNLINK table convention of the JSON files:
``aod/zod`` are the BS side and ``aoa/zoa`` the UT side; with
``direction='uplink'`` the ``CDL`` base class swaps them as usual, so the
``*_bs`` knobs control what a BS receiver sees.

Reference knob settings that approximate the real CDL models (BS side,
computed from the shipped JSONs): see ``anchor_knobs()``.

Knobs (all angles in degrees, spreads are power-weighted RMS of cluster
centres):

* ``num_clusters``   : number of diffuse clusters (rows in the tables)
* ``los``/``k_db``   : LoS flag and specular-to-diffuse power ratio [dB]
* ``mu_az_bs``/``sig_az_bs``/``c_asd`` : BS azimuth centre/inter/intra
* ``mu_ze_bs``/``sig_ze_bs``/``c_zsd`` : BS zenith centre/inter/intra
* ``mu_az_ut``/``sig_az_ut``/``c_asa`` : UT azimuth centre/inter/intra
* ``mu_ze_ut``/``sig_ze_ut``/``c_zsa`` : UT zenith centre/inter/intra
* ``decay_db_total`` : power drop [dB] from first to last cluster (0=flat)
* ``xpr_db``         : cross-polarization ratio [dB]
"""

import json

import numpy as np
import tensorflow as tf
from importlib_resources import files

from . import models
from .cdl import CDL


def _wrap_deg(x):
    """Wrap angles to (-180, 180]."""
    return (np.asarray(x, dtype=np.float64) + 180.) % 360. - 180.


def _pw_circ_mean_deg(ang_deg, w):
    """Power-weighted circular mean [deg]."""
    a = np.deg2rad(np.asarray(ang_deg, dtype=np.float64))
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    return np.rad2deg(np.angle((w * np.exp(1j * a)).sum()))


def _pw_stats(ang_deg, w, circular):
    """Power-weighted (mean, RMS-about-mean) of angles [deg]."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    if circular:
        mu = _pw_circ_mean_deg(ang_deg, w)
        d = _wrap_deg(np.asarray(ang_deg, dtype=np.float64) - mu)
    else:
        mu = float((w * np.asarray(ang_deg, dtype=np.float64)).sum())
        d = np.asarray(ang_deg, dtype=np.float64) - mu
    return float(mu), float(np.sqrt((w * d**2).sum()))


def _centre_angles(p_lin, mu, sig, circular):
    """Cluster centre angles from cluster powers (spec step-7 style).

    Shape from the power profile (sqrt(-ln p) for azimuth, -ln p for
    zenith), alternating signs, then exactly re-centered to ``mu`` and
    re-scaled to power-weighted RMS ``sig``.
    """
    p = np.asarray(p_lin, dtype=np.float64)
    z = np.clip(p / p.max(), 1e-9, 1.0)
    shape = np.sqrt(-np.log(z)) if circular else -np.log(z)
    shape = shape * (-1.) ** np.arange(len(p))
    if sig == 0.:
        return np.full(len(p), float(mu))
    w = p / p.sum()
    m = (w * shape).sum()
    rms = np.sqrt((w * (shape - m) ** 2).sum())
    if rms < 1e-12:  # degenerate profile (e.g. flat powers)
        shape = (-1.) ** np.arange(len(p)) * np.linspace(0.5, 1., len(p))
        m = (w * shape).sum()
        rms = np.sqrt((w * (shape - m) ** 2).sum())
    return mu + (shape - m) * (sig / rms)


def generate_gcdl_params(num_clusters=23,
                         los=False,
                         k_db=9.0,
                         mu_az_bs=0., sig_az_bs=40., c_asd=10.,
                         mu_ze_bs=105., sig_ze_bs=5., c_zsd=3.,
                         mu_az_ut=180., sig_az_ut=55., c_asa=22.,
                         mu_ze_ut=67., sig_ze_ut=8., c_zsa=7.,
                         decay_db_total=-15.,
                         tail_ratio=None,
                         xpr_db=8.):
    """Build a CDL-schema parameter dict from the GCDL knob vector.

    Returns a dict with the exact keys of the ``models/CDL-*.json`` files
    (powers in dB, angles in degrees, delays normalized to unit
    power-weighted RMS delay spread). For ``los=True`` the first row is
    the specular component with power set so that
    K = P_spec / sum(P_diffuse) = ``k_db`` (the convention the ``CDL``
    loader expects, cf. its step-11 comment).
    """
    n = int(num_clusters)

    # Delays: deterministic exponential grid -> ascending, first at 0
    u = (np.arange(n, dtype=np.float64) + 1.) / (n + 1.)
    tau = np.sort(-np.log(u))
    tau -= tau[0]

    # Powers: log-linear decay over delay, total span decay_db_total
    span = tau[-1] if tau[-1] > 0 else 1.
    p_db = decay_db_total * tau / span
    p_lin = 10. ** (p_db / 10.)

    # Normalize delays to unit power-weighted RMS delay spread
    def _unit_rms(t):
        w = p_lin / p_lin.sum()
        rms = np.sqrt((w * (t - (w * t).sum()) ** 2).sum())
        return t / rms if rms > 0 else t

    tau = _unit_rms(tau)

    # Optional tail stretch/compress: bend delays (tau -> tau**gamma, then
    # re-normalized to unit RMS) until max(tau)/rms(tau) hits tail_ratio.
    # Power profile is left untouched, so this isolates the delay-tail
    # EXTENT from both the power decay and the RMS delay spread.
    # (CDL-B has max/rms ~ 4.8; CDL-D ~ 12.5; CDL-E ~ 20.6.)
    if tail_ratio is not None and tau[-1] > 0:
        lo, hi = 0.2, 5.0
        for _ in range(60):
            gam = 0.5 * (lo + hi)
            ratio = _unit_rms(tau ** gam)[-1]
            if ratio < tail_ratio:
                lo = gam
            else:
                hi = gam
        tau = _unit_rms(tau ** (0.5 * (lo + hi)))

    # Cluster centre angles from the power profile
    aod = _wrap_deg(_centre_angles(p_lin, mu_az_bs, sig_az_bs, True))
    aoa = _wrap_deg(_centre_angles(p_lin, mu_az_ut, sig_az_ut, True))
    zod = _centre_angles(p_lin, mu_ze_bs, sig_ze_bs, False)
    zoa = _centre_angles(p_lin, mu_ze_ut, sig_ze_ut, False)
    zod = np.clip(zod, 0.5, 179.5)
    zoa = np.clip(zoa, 0.5, 179.5)

    delays = tau.tolist()
    powers = p_db.tolist()
    aod_l, aoa_l = aod.tolist(), aoa.tolist()
    zod_l, zoa_l = zod.tolist(), zoa.tolist()

    if los:
        # Specular row 0: zero delay, centre angles, power k_db above the
        # total diffuse power (K = P_spec / sum P_diffuse).
        p_spec_db = float(k_db + 10. * np.log10(p_lin.sum()))
        delays = [0.] + delays
        powers = [p_spec_db] + powers
        aod_l = [float(_wrap_deg(mu_az_bs))] + aod_l
        aoa_l = [float(_wrap_deg(mu_az_ut))] + aoa_l
        zod_l = [float(np.clip(mu_ze_bs, 0.5, 179.5))] + zod_l
        zoa_l = [float(np.clip(mu_ze_ut, 0.5, 179.5))] + zoa_l

    return {
        "los": 1 if los else 0,
        "num_clusters": n,
        "cASD": float(c_asd), "cASA": float(c_asa),
        "cZSD": float(c_zsd), "cZSA": float(c_zsa),
        "xpr": float(xpr_db),
        "delays": delays,
        "powers": powers,
        "aod": aod_l, "aoa": aoa_l,
        "zod": zod_l, "zoa": zoa_l,
    }


def measure_gcdl_params(params):
    """Realized power-weighted stats of a CDL-schema dict (self-check).

    Diffuse clusters only (specular row excluded for LoS sets), matching
    how the generator's ``mu/sig`` knobs are defined.
    """
    p = 10. ** (np.asarray(params["powers"], dtype=np.float64) / 10.)
    ang = {k: np.asarray(params[k], dtype=np.float64)
           for k in ("aod", "aoa", "zod", "zoa")}
    if params["los"]:
        p_diff = p[1:]
        k_db = 10. * np.log10(p[0] / p_diff.sum())
        ang = {k: v[1:] for k, v in ang.items()}
        p = p_diff
    else:
        k_db = None
    out = {"k_db": k_db, "num_clusters": params["num_clusters"]}
    for key, circ in (("aod", True), ("aoa", True),
                      ("zod", False), ("zoa", False)):
        mu, sig = _pw_stats(ang[key], p, circ)
        out[f"mu_{key}"] = mu
        out[f"sig_{key}"] = sig
    w = p / p.sum()
    tau = np.asarray(params["delays"], dtype=np.float64)
    tau = tau[1:] if params["los"] else tau
    m = (w * tau).sum()
    out["rms_delay"] = float(np.sqrt((w * (tau - m) ** 2).sum()))
    return out


def anchor_knobs(model):
    """Knob vector approximating a real CDL model, from its shipped JSON.

    ``model`` is one of "A".."E". Centres/spreads are the power-weighted
    stats of the tabulated cluster angles (diffuse clusters only for the
    LoS models D/E), ``decay_db_total`` is a least-squares fit of the
    power-vs-delay slope, and the intra-cluster spreads/XPR are copied
    verbatim. ``generate_gcdl_params(**anchor_knobs(X))`` therefore
    matches CDL-X's summary statistics, not its exact cluster layout.
    """
    assert model in ("A", "B", "C", "D", "E"), "model must be A..E"
    with open(files(models).joinpath(f"CDL-{model}.json")) as f:
        params = json.load(f)
    m = measure_gcdl_params(params)
    tau = np.asarray(params["delays"], dtype=np.float64)
    p_db = np.asarray(params["powers"], dtype=np.float64)
    if params["los"]:
        tau, p_db = tau[1:], p_db[1:]
    slope = np.polyfit(tau, p_db, 1)[0]
    return {
        "num_clusters": int(params["num_clusters"]),
        "los": bool(params["los"]),
        "k_db": float(m["k_db"]) if params["los"] else 9.0,
        "mu_az_bs": m["mu_aod"], "sig_az_bs": m["sig_aod"],
        "c_asd": float(params["cASD"]),
        "mu_ze_bs": m["mu_zod"], "sig_ze_bs": m["sig_zod"],
        "c_zsd": float(params["cZSD"]),
        "mu_az_ut": m["mu_aoa"], "sig_az_ut": m["sig_aoa"],
        "c_asa": float(params["cASA"]),
        "mu_ze_ut": m["mu_zoa"], "sig_ze_ut": m["sig_zoa"],
        "c_zsa": float(params["cZSA"]),
        "decay_db_total": float(slope * (tau.max() - tau.min())),
        "xpr_db": float(params["xpr"]),
    }


def save_gcdl_json(path, **knobs):
    """Generate a GCDL parameter set and write it as a CDL-schema JSON."""
    params = generate_gcdl_params(**knobs)
    with open(path, "w") as f:
        json.dump(params, f, indent=1)
    return params


class GCDL(CDL):
    # pylint: disable=line-too-long
    r"""GCDL(gcdl_knobs, delay_spread, carrier_frequency, ut_array, bs_array, direction, ...)

    Generic CDL channel model built from a knob vector instead of a
    tabulated model letter. Accepts the same constructor arguments as
    :class:`~sionna.channel.tr38901.CDL` except that ``model`` is replaced
    by ``gcdl_knobs`` — either a knob dict for
    :func:`generate_gcdl_params`, or a ready CDL-schema parameter dict
    (as returned by that function / loaded from a JSON).
    """

    def __init__(self, gcdl_knobs, delay_spread, carrier_frequency,
                 ut_array, bs_array, direction, ut_orientation=None,
                 bs_orientation=None, min_speed=0., max_speed=None,
                 dtype=tf.complex64):
        if isinstance(gcdl_knobs, dict) and "delays" in gcdl_knobs:
            params = gcdl_knobs          # ready CDL-schema dict
        else:
            params = generate_gcdl_params(**(gcdl_knobs or {}))
        self._gcdl_raw_params = params
        # "B" is only a carrier for the base-class assert; its JSON is
        # never read because _load_parameters is overridden below.
        super().__init__("B", delay_spread, carrier_frequency, ut_array,
                         bs_array, direction, ut_orientation,
                         bs_orientation, min_speed, max_speed, dtype)

    def _load_parameters(self, fname):
        self._process_parameters(self._gcdl_raw_params)
