# SPDX-FileCopyrightText: Copyright (c) 2026 Mahdi Abdollahpour <mahdi.abdollahpour@unibo.it>
"""SRS pilot pattern."""

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from sionna.ofdm import PilotPattern
from .srs_config import SRSConfig


class SRSPilotPattern(PilotPattern):
    # pylint: disable=line-too-long
    r"""Pilot pattern of the SRS of one or several transmitters.

    One :class:`~sionna.nr.SRSConfig` per transmitter; every SRS port is a
    stream. The mask is the union of the SRS resource elements of all
    transmitters, so that all streams have the same pilot positions; the
    pilots of a port are zero where it does not transmit.

    Parameters
    ----------
    srs_configs : instance or list of :class:`~sionna.nr.SRSConfig`
        SRS configurations, one per transmitter.

    dtype : tf.Dtype
        Datatype. Defaults to `tf.complex64`.
    """
    def __init__(self, srs_configs, dtype=tf.complex64):
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        assert isinstance(srs_configs, Sequence) and \
            all(isinstance(c, SRSConfig) for c in srs_configs), \
            "srs_configs must be a list of SRSConfig"
        num_ports = srs_configs[0].num_ports
        shape = srs_configs[0].srs_mask.shape
        for c in srs_configs:
            assert c.num_ports == num_ports, "All SRS configs must have the same number of ports"
            assert c.srs_mask.shape == shape, "All SRS configs must have the same carrier grid"

        union = np.any([c.srs_mask for c in srs_configs], axis=0).T     # [num_symbols, num_sc]
        num_tx = len(srs_configs)
        mask = np.broadcast_to(union, [num_tx, num_ports] + list(union.shape)).copy()
        pilots = np.zeros([num_tx, num_ports, int(np.sum(union))], complex)
        for i, c in enumerate(srs_configs):
            grid = c.srs_grid
            for j in range(num_ports):
                pilots[i, j] = grid[j].T[np.where(union)]

        super().__init__(mask, pilots, trainable=False, normalize=False, dtype=dtype)
