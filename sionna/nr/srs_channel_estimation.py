# SPDX-FileCopyrightText: Copyright (c) 2026 Mahdi Abdollahpour <mahdi.abdollahpour@unibo.it>
"""LS channel estimation for the SRS."""

import numpy as np
import tensorflow as tf

from sionna.ofdm import BaseChannelEstimator
from sionna.utils import expand_to_rank


def _block_length(own, others):
    """Smallest L dividing len(own) such that the pilots of every other port
    sum to zero against own over every L consecutive pilots."""
    m = len(own)
    for length in range(1, m + 1):
        if m % length:
            continue
        ok = True
        for o in others:
            s = np.sum((own*np.conj(o)).reshape(-1, length), axis=-1)
            if np.max(np.abs(s)) > 1e-6*np.sum(np.abs(own)**2)/m*length:
                ok = False
                break
        if ok:
            return length
    return None


class SRSLSChannelEstimator(BaseChannelEstimator):
    # pylint: disable=line-too-long
    r"""LS channel estimation for the SRS, as
    :class:`~sionna.nr.PUSCHLSChannelEstimator` for the DMRS.

    The LS estimates of a port are averaged over the L consecutive pilots of
    an OFDM symbol on which the pilots of all other ports sharing its resource
    elements cancel (cyclic shifts; L = 2 for the DMRS FD-OCC), and held on
    those L pilots. The interpolation is that of
    :class:`~sionna.ofdm.BaseChannelEstimator`. L is derived from the pilot
    pattern, so the estimator also applies to single-symbol DMRS patterns.

    Parameters
    ----------
    resource_grid : :class:`~sionna.ofdm.ResourceGrid`
        Resource grid, e.g. that of :class:`~sionna.nr.SRSTransmitter`.

    interpolation_type : "nn" | "lin" (default) | "lin_time_avg"
        Interpolation method.

    interpolator : :class:`~sionna.ofdm.BaseChannelInterpolator` or `None`
        Custom interpolator; overrides ``interpolation_type``.

    dtype : tf.Dtype
        Datatype. Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
    err_var : Same shape as ``h_hat``, tf.float
    """
    def __init__(self, resource_grid, interpolation_type="lin", interpolator=None,
                 dtype=tf.complex64, **kwargs):
        super().__init__(resource_grid, interpolation_type, interpolator, dtype, **kwargs)
        pilots = self._pilot_pattern.pilots.numpy()                 # [T, S, P]
        mask = self._pilot_pattern.mask.numpy()
        num_tx, num_streams, num_pilots = pilots.shape
        # Row-major pilot indices, as the mapper; the argsort of the base
        # class does not keep this order on every device
        flat = mask.reshape(num_tx, num_streams, -1)
        self._pilot_ind = tf.constant(np.stack([[np.nonzero(flat[t, s])[0]
                                                 for s in range(num_streams)]
                                                for t in range(num_tx)]), tf.int32)
        symbol = np.nonzero(mask[0, 0])[0]                          # OFDM symbol of every pilot
        streams = [(t, s) for t in range(num_tx) for s in range(num_streams)]
        own = {ts: np.abs(pilots[ts]) > 0 for ts in streams}

        blocks = {ts: [] for ts in streams}
        for t, s in streams:
            for l in np.unique(symbol[own[(t, s)]]):
                idx = np.nonzero(own[(t, s)] & (symbol == l))[0]
                others = []
                for ts in streams:
                    if ts == (t, s) or not np.any(own[ts][idx]):
                        continue
                    assert np.array_equal(own[ts][idx], np.ones(len(idx), bool)) and \
                        np.sum(own[ts] & (symbol == l)) == len(idx), \
                        "Ports sharing resource elements must share all of them"
                    others.append(pilots[ts][idx])
                length = _block_length(pilots[(t, s)][idx], others)
                assert length is not None, \
                    "Ports sharing resource elements cannot be separated within one OFDM symbol"
                blocks[(t, s)] += list(idx.reshape(-1, length))

        num_blocks = max(len(b) for b in blocks.values())
        max_len = max(len(b) for v in blocks.values() for b in v)
        index = np.full([num_tx, num_streams, num_blocks, max_len], num_pilots)
        weight = np.zeros([num_tx, num_streams, num_blocks, max_len])
        assign = np.full([num_tx, num_streams, num_pilots], num_blocks)
        length = np.zeros([num_tx, num_streams, num_pilots])
        for (t, s), bl in blocks.items():
            for b, idx in enumerate(bl):
                index[t, s, b, :len(idx)] = idx
                weight[t, s, b, :len(idx)] = 1/len(idx)
                assign[t, s, idx] = b
                length[t, s, idx] = len(idx)
        self._index = tf.constant(index.reshape(num_tx, num_streams, -1), tf.int32)
        self._weight = tf.constant(weight, dtype)
        self._assign = tf.constant(assign, tf.int32)
        self._length = tf.constant(length, dtype.real_dtype)
        self._num_blocks, self._max_len = num_blocks, max_len

    @property
    def averaging_length(self):
        """[num_tx, num_streams_per_tx, num_pilots], int : Number of pilots
        averaged into the estimate at every pilot (0 where a stream has no pilot)"""
        return self._length.numpy().astype(int)

    def estimate_at_pilot_locations(self, y_pilots, no):
        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilots]
        h_ls = tf.math.divide_no_nan(y_pilots, self._pilot_pattern.pilots)

        # Average every port over its blocks and hold the mean on the block
        x = tf.transpose(h_ls, [3, 4, 0, 1, 2, 5])                  # [T, S, B, R, RA, P]
        x = tf.concat([x, tf.zeros_like(x[..., :1])], axis=-1)
        x = tf.gather(x, self._index, axis=5, batch_dims=2)
        x = tf.reshape(x, tf.concat([tf.shape(x)[:5], [self._num_blocks, self._max_len]], 0))
        x = tf.reduce_sum(x*self._weight[:, :, None, None, None], axis=-1)
        x = tf.concat([x, tf.zeros_like(x[..., :1])], axis=-1)
        x = tf.gather(x, self._assign, axis=5, batch_dims=2)
        h_hat = tf.transpose(x, [2, 3, 4, 0, 1, 5])

        no = expand_to_rank(no, tf.rank(h_ls), -1)
        pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0)
        err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2*self._length)
        return h_hat, err_var
