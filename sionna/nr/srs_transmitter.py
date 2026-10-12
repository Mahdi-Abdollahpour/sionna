# SPDX-FileCopyrightText: Copyright (c) 2026 Mahdi Abdollahpour <mahdi.abdollahpour@unibo.it>
"""SRS transmitter."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from sionna.ofdm import ResourceGrid, ResourceGridMapper
from .srs_config import SRSConfig
from .srs_pilot_pattern import SRSPilotPattern


class SRSTransmitter(Layer):
    # pylint: disable=line-too-long
    r"""SRS of one or several transmitters on the slot resource grid.

    Parameters
    ----------
    srs_configs : instance or list of :class:`~sionna.nr.SRSConfig`
        SRS configurations, one per transmitter. They must share the carrier.

    dtype : tf.Dtype
        Datatype. Defaults to `tf.complex64`.

    Input
    -----
    batch_size : int

    Output
    ------
    x : [batch_size, num_tx, num_ports, num_symbols_per_slot, 12 * n_size_grid], tf.complex
        Frequency-domain resource grid; only the SRS resource elements are nonzero.
    """
    def __init__(self, srs_configs, dtype=tf.complex64, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if isinstance(srs_configs, SRSConfig):
            srs_configs = [srs_configs]
        self._srs_configs = srs_configs
        carrier = srs_configs[0].carrier
        self._pilot_pattern = SRSPilotPattern(srs_configs, dtype=dtype)
        bandwidth = 12*carrier.n_size_grid*carrier.subcarrier_spacing*1e3
        self._resource_grid = ResourceGrid(
            num_ofdm_symbols=carrier.num_symbols_per_slot,
            fft_size=12*carrier.n_size_grid,
            subcarrier_spacing=carrier.subcarrier_spacing*1e3,
            num_tx=len(srs_configs),
            num_streams_per_tx=srs_configs[0].num_ports,
            cyclic_prefix_length=int(np.ceil(carrier.cyclic_prefix_length*bandwidth)),
            pilot_pattern=self._pilot_pattern,
            dtype=dtype)
        self._mapper = ResourceGridMapper(self._resource_grid, dtype=dtype)

    @property
    def resource_grid(self):
        """:class:`~sionna.ofdm.ResourceGrid` of the SRS"""
        return self._resource_grid

    @property
    def pilot_pattern(self):
        """:class:`~sionna.nr.SRSPilotPattern`"""
        return self._pilot_pattern

    @property
    def srs_configs(self):
        """List of :class:`~sionna.nr.SRSConfig`"""
        return self._srs_configs

    def call(self, batch_size):
        rg = self._resource_grid
        data = tf.zeros([batch_size, rg.num_tx, rg.num_streams_per_tx, rg.num_data_symbols],
                        self.dtype)
        return self._mapper(data)
