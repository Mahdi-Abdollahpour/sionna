
# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2025

import tensorflow as tf
from sionna import PI, SPEED_OF_LIGHT


class ChannelParams:
    """A class to store channel parametric data (per deep_echo), including randomly generated data for training & debug """

    def __init__(self, 
                carrier_frequency,
                tx_array, rx_array,
                subclustering,
                dtype=tf.complex64):
        

        
        self.carrier_frequency = carrier_frequency
        self.tx_array = tx_array
        self.rx_array = rx_array
        self.subclustering = subclustering
        self.dtype = dtype
        # ----------------------
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT/carrier_frequency,
            self.dtype.real_dtype)

        # initial phases at step 10
        # [batch_size, num_tx, num_rx, num_clusters, num_rays, 4]
        self.phi = None 

        # # cross-polarization & exp applied to initial random phases 
        # # [batch size, num TXs, num RXs, num clusters, num rays, 2, 2]
        # self._phase_matrix = phase_matrix
        self.strongest_clusters = None
        self.delays_ind = None
        # ------------------------------------------------
        self.rays = None
        self.num_time_samples = None
        self.sampling_frequency = None
        self.k_factor = None
        self.rays = None
        self.topology = None
        self.c_ds = None
        self.sf = None
        self.pl_db = None
        self.shadow_fading_enabled = None
        self.direction = None
        # -----------------------Delete These Params--------------------------

        self.f_rx = None # debug only
        self.f_tx = None
        self.phi_mat = None
        self.doppler_shifts = None


        # -------------------------------------------------
        self.h = None
        self.delays = None


    # @property
    # def phi(self):
    #     return self._phi

    # @phi.setter
    # def phi(self, value):
    #     self._phi = value

    # @property
    # def rays(self):
    #     return self._rays

    # @rays.setter
    # def rays(self, value):
    #     self._rays = value 

    # @property
    # def num_time_samples(self):
    #     return self._num_time_samples

    # @num_time_samples.setter
    # def num_time_samples(self, value):
    #     self._num_time_samples = value

    # @property
    # def sampling_frequency(self):
    #     return self._sampling_frequency

    # @sampling_frequency.setter
    # def sampling_frequency(self, value):
    #     self._sampling_frequency = value

    # @property
    # def k_factor(self):
    #     return self._k_factor

    # @k_factor.setter
    # def k_factor(self, value):
    #     self._k_factor = value

    # @property
    # def topology(self):
    #     return self._topology

    # @topology.setter
    # def topology(self, value):
    #     self._topology = value

    # @property
    # def c_ds(self):
    #     return self._c_ds

    # @c_ds.setter
    # def c_ds(self, value):
    #     self._c_ds = value

    # @property
    # def h(self):
    #     return self._h

    # @h.setter
    # def h(self, value):
    #     self._h = value

    # @property
    # def delays(self):
    #     return self._delays

    # @delays.setter
    # def delays(self, value):
    #     self._delays = value

