#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file Modified to generate batches
# of channel delay/coefficients with randomized number of [clusters & rays]
# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2025

"""Base class for implementing system level channel models from 3GPP TR38.901
specification"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from . import LSPGenerator
from . import RaysGenerator
from . import Topology, ChannelCoefficientsGenerator
from sionna.channel import ChannelModel
from sionna.channel.utils import deg_2_rad


from .channel_params import ChannelParams


class SystemLevelChannel(ChannelModel):
    # pylint: disable=line-too-long
    r"""
    Baseclass for implementing 3GPP system level channel models, such as UMi,
    UMa, and RMa.

    Parameters
    -----------
    scenario : SystemLevelScenario
        Scenario for the channel simulation

    always_generate_lsp : bool
        If `True`, new large scale parameters (LSPs) are generated for every
        new generation of channel impulse responses. Otherwise, always reuse
        the same LSPs, except if the topology is changed. Defaults to
        `False`.

    Input
    -----

    num_time_samples : int
        Number of time samples

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
        a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_samples], tf.complex
            Path coefficients

        tau : [batch size, num_rx, num_tx, num_paths], tf.float
            Path delays [s]
    """

    def __init__(self, scenario, always_generate_lsp=False, 
        random_num_clusters=False, random_num_rays=False, 
        mask_doa=False):



        self._random_num_clusters = random_num_clusters

        self._scenario = scenario
        self._lsp_sampler = LSPGenerator(scenario)
        self._ray_sampler = RaysGenerator(scenario, random_num_clusters, random_num_rays)
        self._set_topology_called = False

        if scenario.direction == "uplink":
            tx_array = scenario.ut_array
            rx_array = scenario.bs_array
        else: # "downlink"
            tx_array = scenario.bs_array
            rx_array = scenario.ut_array


        # Channel Params
        self.channel_params = ChannelParams(   scenario.carrier_frequency,
                                            tx_array, rx_array,
                                            subclustering=True,
                                            dtype = scenario.dtype)

        self._cir_sampler = ChannelCoefficientsGenerator(
                                            scenario.carrier_frequency,
                                            tx_array, rx_array,
                                            subclustering=True,
                                            mask_doa=mask_doa,
                                            dtype = scenario.dtype)

        # Are new LSPs needed
        self._always_generate_lsp = always_generate_lsp





    def set_topology(self, ut_loc=None, bs_loc=None, ut_orientations=None,
        bs_orientations=None, ut_velocities=None, in_state=None, los=None):
        r"""
        Set the network topology.

        It is possible to set up a different network topology for each batch
        example. The batch size used when setting up the network topology
        is used for the link simulations.

        When calling this function, not specifying a parameter leads to the
        reuse of the previously given value. Not specifying a value that was not
        set at a former call rises an error.

        Input
        ------
            ut_loc : [batch size,num_ut, 3], tf.float
                Locations of the UTs

            bs_loc : [batch size,num_bs, 3], tf.float
                Locations of BSs

            ut_orientations : [batch size,num_ut, 3], tf.float
                Orientations of the UTs arrays [radian]

            bs_orientations : [batch size,num_bs, 3], tf.float
                Orientations of the BSs arrays [radian]

            ut_velocities : [batch size,num_ut, 3], tf.float
                Velocity vectors of UTs

            in_state : [batch size,num_ut], tf.bool
                Indoor/outdoor state of UTs. `True` means indoor and `False`
                means outdoor.

            los : tf.bool or `None`
                If not `None` (default value), all UTs located outdoor are
                forced to be in LoS if ``los`` is set to `True`, or in NLoS
                if it is set to `False`. If set to `None`, the LoS/NLoS states
                of UTs is set following 3GPP specification [TR38901]_.

        Note
        ----
        If you want to use this function in Graph mode with XLA, i.e., within
        a function that is decorated with ``@tf.function(jit_compile=True)``,
        you must set ``sionna.Config.xla_compat=true``.
        See :py:attr:`~sionna.Config.xla_compat`.
        """

        # Update the scenario topology
        need_for_update = self._scenario.set_topology(  ut_loc,
                                                        bs_loc,
                                                        ut_orientations,
                                                        bs_orientations,
                                                        ut_velocities,
                                                        in_state,
                                                        los)

        if need_for_update:
            # Update the LSP sampler
            self._lsp_sampler.topology_updated_callback()

            # Update the ray sampler
            self._ray_sampler.topology_updated_callback()

            # Sample LSPs if no need to generate them everytime
            if not self._always_generate_lsp:
                self._lsp = self._lsp_sampler()

        if not self._set_topology_called:
            self._set_topology_called = True

    def __call__(self, num_time_samples, sampling_frequency, foo=None):

        # Some channel layers (GenerateOFDMChannel and GenerateTimeChannel)
        # give as input (batch_size, num_time_samples, sampling_frequency)
        # instead of (num_time_samples, sampling_frequency), as specified
        # in the ChannelModel interface.
        # With this model, the batch size is ignored, and only the required
        # parameters are kept.
        if foo is not None:
            # batch_size = num_time_samples
            num_time_samples = sampling_frequency
            sampling_frequency = foo
        #             if ( (batch_size is not None)
        #                     and tf.not_equal(batch_size,self._scenario.batch_size) ):
        #                 tf.print("Warning: The value of `batch_size` specified when \
        # calling the channel model is different from the one previously configured for \
        # the topology. The value specified when calling is ignored.")

        # tf.print(f"sampling_frequency:{sampling_frequency}, sampling_time:{1./sampling_frequency}")

        # Sample LSPs if required
        if self._always_generate_lsp:
            lsp = self._lsp_sampler()
        else:
            lsp = self._lsp

        # Sample rays
        rays = self._ray_sampler(lsp)

        # Sample channel responses
        # First we need to create a topology
        # Indicates which end of the channel is moving: TX or RX
        if self._scenario.direction == 'downlink':
            moving_end = 'rx'
            tx_orientations = self._scenario.bs_orientations
            rx_orientations = self._scenario.ut_orientations
        else : # 'uplink'
            moving_end = 'tx'
            tx_orientations = self._scenario.ut_orientations
            rx_orientations = self._scenario.bs_orientations
        topology = Topology(    velocities=self._scenario.ut_velocities,
                                moving_end=moving_end,
                                los_aoa=deg_2_rad(self._scenario.los_aoa),
                                los_aod=deg_2_rad(self._scenario.los_aod),
                                los_zoa=deg_2_rad(self._scenario.los_zoa),
                                los_zod=deg_2_rad(self._scenario.los_zod),
                                los=self._scenario.los,
                                distance_3d=self._scenario.distance_3d,
                                tx_orientations=tx_orientations,
                                rx_orientations=rx_orientations)

        # The channel coefficient needs the cluster delay spread parameter in ns
        c_ds = self._scenario.get_param("cDS")*1e-9


        # According to the link direction, we need to specify which from BS
        # and UT is uplink, and which is downlink.
        # Default is downlink, so we need to do some tranpose to switch tx and
        # rx and to switch angle of arrivals and departure if direction is set
        # to uplink. Nothing needs to be done if direction is downlink
        if self._scenario.direction == "uplink":
            aoa = rays.aoa
            zoa = rays.zoa
            aod = rays.aod
            zod = rays.zod
            rays.aod = tf.transpose(aoa, [0, 2, 1, 3, 4])
            rays.zod = tf.transpose(zoa, [0, 2, 1, 3, 4])
            rays.aoa = tf.transpose(aod, [0, 2, 1, 3, 4])
            rays.zoa = tf.transpose(zod, [0, 2, 1, 3, 4])
            rays.powers = tf.transpose(rays.powers, [0, 2, 1, 3])
            rays.delays = tf.transpose(rays.delays, [0, 2, 1, 3])
            rays.xpr = tf.transpose(rays.xpr, [0, 2, 1, 3, 4])
            los_aod = topology.los_aod
            los_aoa = topology.los_aoa
            los_zod = topology.los_zod
            los_zoa = topology.los_zoa
            topology.los_aoa = tf.transpose(los_aod, [0, 2, 1])
            topology.los_aod = tf.transpose(los_aoa, [0, 2, 1])
            topology.los_zoa = tf.transpose(los_zod, [0, 2, 1])
            topology.los_zod = tf.transpose(los_zoa, [0, 2, 1])
            topology.los = tf.transpose(topology.los, [0, 2, 1])
            c_ds = tf.transpose(c_ds, [0, 2, 1])
            topology.distance_3d = tf.transpose(topology.distance_3d, [0, 2, 1])
            # Concerning LSPs, only these two are used.
            # We do not transpose the others to reduce complexity
            k_factor = tf.transpose(lsp.k_factor, [0, 2, 1])
            sf = tf.transpose(lsp.sf, [0, 2, 1])


            # transpose rays mask
            # [batch_size, num_of_BSs, num_of_UTs, max_number_of_clusters, num_rays]
            # To [batch_size, num_of_UTs, num_of_BSs, max_number_of_clusters, num_rays]
            rays.ray_mask = tf.transpose(rays.ray_mask, [0, 2, 1, 3, 4])


        else:
            k_factor = lsp.k_factor
            sf = lsp.sf



        

        # # pylint: disable=unbalanced-tuple-unpacking
        # # h[batch size, num_tx, num_rx, num_paths, num_rx_ant, num_tx_ant, num_time_samples]
        # # delays_nlos[batch_size, num_tx, num_rx, num_paths]
        # h, delays = self._cir_sampler(num_time_samples, sampling_frequency,
        #                               k_factor, rays, topology, c_ds)

        # pylint: disable=unbalanced-tuple-unpacking
        # h[batch size, num_tx, num_rx, num_paths, num_rx_ant, num_tx_ant, num_time_samples]
        # delays_nlos[batch_size, num_tx, num_rx, num_paths]
        # debug mode needed to get phi
        h, delays, phi, sample_times, strongest_clusters, delays_ind = self._cir_sampler(num_time_samples, sampling_frequency,
                                      k_factor, rays, topology, c_ds, debug=True)

        # # --------------Plot tau,a------------------
        # # Plot tau,a
        # # Pick one sample
        # h_sample = tf.squeeze(h[:, 0, 0,:,0,0,0])                  # [batch_size,num_paths]
        # delays_ = tf.squeeze(delays[:, 0, 0,:])          # [batch_size,num_paths]

        # plot_h_vs_delay(h_sample, delays_, num_samples=1, outfile='tau_a_umi.png', x_limit_ns=800)


        # # Close to free memory

        # print("Press any key or mouse button in the plot window to close it...")
        # # plt.waitforbuttonpress()  # pauses until a key or mouse click
        # input("[sys.ch] Press Enter to continue...")
        # # --------------------------------

        # # --------------Plot Topology & Antenna Array------------------
        # self._scenario.bs_array.show()
        # # self.show_topology()
        # input("Press Enter to continue...")
        # # --------------------------------

        # store channel data
        self.channel_params.rays = rays
        self.channel_params.num_time_samples = num_time_samples
        self.channel_params.sampling_frequency = sampling_frequency
        self.channel_params.k_factor = k_factor
        self.channel_params.topology = topology
        self.channel_params.c_ds = c_ds
        self.channel_params.phi = phi
        self.channel_params.sf = sf
        self.channel_params.strongest_clusters = strongest_clusters
        self.channel_params.delays_ind = delays_ind
        # --------- end store


        # Step 12
        h = self._step_12(h, sf)

        # Reshaping to match the expected output
        h = tf.transpose(h, [0, 2, 4, 1, 5, 3, 6])
        delays = tf.transpose(delays, [0, 2, 1, 3])

        # Stop gadients to avoid useless backpropagation
        h = tf.stop_gradient(h)
        delays = tf.stop_gradient(delays)


        return h, delays

    def show_topology(self, bs_index=0, batch_index=0):
        r"""
        Shows the network topology of the batch example with index
        ``batch_index``.

        The ``bs_index`` parameter specifies with respect to which BS the
        LoS/NLoS state of UTs is indicated.

        Input
        -------
        bs_index : int
            BS index with respect to which the LoS/NLoS state of UTs is
            indicated. Defaults to 0.

        batch_index : int
            Batch example for which the topology is shown. Defaults to 0.
        """

        def draw_coordinate_system(ax, loc, ort, delta):
            # This function draw the coordinate system x-y-z, represented by
            # three lines with colors red-green-blue (rgb), to show the
            # orientation of the array (LCS) in the GCS.
            # To always draw a visible and not too big axes, we scale them
            # according to the spread of the network in each direction.

            a = ort[0]
            b = ort[1]
            c = ort[2]

            arrow_ratio_size = 0.1

            x_ = np.array([ np.cos(a)*np.cos(b),
                            np.sin(a)*np.cos(b),
                            -np.sin(b) ])
            scale_x = arrow_ratio_size/np.sqrt(np.sum(np.square(x_/delta)))
            x_ = x_*scale_x

            y_ = np.array([ np.cos(a)*np.sin(b)*np.sin(c)-np.sin(a)*np.cos(c),
                            np.sin(a)*np.sin(b)*np.sin(c)+np.cos(a)*np.cos(c),
                            np.cos(b)*np.sin(c) ])
            scale_y = arrow_ratio_size/np.sqrt(np.sum(np.square(y_/delta)))
            y_ = y_*scale_y

            z_ = np.array([ np.cos(a)*np.sin(b)*np.cos(c)+np.sin(a)*np.sin(c),
                            np.sin(a)*np.sin(b)*np.cos(c)-np.cos(a)*np.sin(c),
                            np.cos(b)*np.cos(c)])
            scale_z = arrow_ratio_size/np.sqrt(np.sum(np.square(z_/delta)))
            z_ = z_*scale_z

            ax.plot([loc[0], loc[0] + x_[0]],
                    [loc[1], loc[1] + x_[1]],
                    [loc[2], loc[2] + x_[2]], c='r')
            ax.plot([loc[0], loc[0] + y_[0]],
                    [loc[1], loc[1] + y_[1]],
                    [loc[2], loc[2] + y_[2]], c='g')
            ax.plot([loc[0], loc[0] + z_[0]],
                    [loc[1], loc[1] + z_[1]],
                    [loc[2], loc[2] + z_[2]], c='b')

        indoor = self._scenario.indoor.numpy()[batch_index]
        los = self._scenario.los.numpy()[batch_index,bs_index]

        indoor_indices = np.where(indoor)
        los_indices = np.where(los)
        nlos_indices = np.where(np.logical_and(np.logical_not(indoor),
                                np.logical_not(los)))


        ut_loc = self._scenario.ut_loc.numpy()[batch_index]
        bs_loc = self._scenario.bs_loc.numpy()[batch_index]
        ut_orientations = self._scenario.ut_orientations.numpy()[batch_index]
        bs_orientations = self._scenario.bs_orientations.numpy()[batch_index]

        delta_x = np.max(np.concatenate([ut_loc[:,0], bs_loc[:,0]]))\
            - np.min(np.concatenate([ut_loc[:,0], bs_loc[:,0]]))
        delta_y = np.max(np.concatenate([ut_loc[:,1], bs_loc[:,1]]))\
            - np.min(np.concatenate([ut_loc[:,1], bs_loc[:,1]]))
        delta_z = np.max(np.concatenate([ut_loc[:,2], bs_loc[:,2]]))\
            - np.min(np.concatenate([ut_loc[:,2], bs_loc[:,2]]))
        delta = np.array([delta_x, delta_y, delta_z])

        indoor_ut_loc = ut_loc[indoor_indices]
        los_ut_loc = ut_loc[los_indices]
        nlos_ut_loc = ut_loc[nlos_indices]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # Showing BS
        ax.scatter( bs_loc[:,0], bs_loc[:,1], bs_loc[:,2], c='k', label='BS',
                    depthshade=False)
        # Showing BS indices and orientations
        for u, loc in enumerate(bs_loc):
            ax.text(loc[0], loc[1], loc[2], f'{u}')
            draw_coordinate_system(ax, loc, bs_orientations[u], delta)
        # Showing UTs
        ax.scatter(indoor_ut_loc[:,0], indoor_ut_loc[:,1], indoor_ut_loc[:,2],
                    c='b', label='UT Indoor', depthshade=False)
        ax.scatter(los_ut_loc[:,0], los_ut_loc[:,1], los_ut_loc[:,2],
                    c='r', label='UT LoS', depthshade=False)
        ax.scatter(nlos_ut_loc[:,0], nlos_ut_loc[:,1], nlos_ut_loc[:,2],
                    c='y', label='UT NLoS', depthshade=False)
        # Showing UT indices and orientations
        for u, loc in enumerate(ut_loc):
            ax.text(loc[0], loc[1], loc[2], f'{u}')
            draw_coordinate_system(ax, loc, ut_orientations[u], delta)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.legend()
        plt.tight_layout()

        plt.savefig("topology.png", dpi=300)

    #####################################################
    # Internal utility methods
    #####################################################

    def _step_12(self, h, sf):
        # pylint: disable=line-too-long
        """Apply path loss and shadow fading ``sf`` to paths coefficients ``h``.

        Input
        ------
        h : [batch size, num_tx, num_rx, num_paths, num_rx_ant, num_tx_ant, num_time_samples], tf.complex
            Paths coefficients

        sf : [batch size, num_tx, num_rx]
            Shadow fading
        """

        self.channel_params.pathloss_enabled = self._scenario.pathloss_enabled
        self.channel_params.shadow_fading_enabled = self._scenario.shadow_fading_enabled
        self.channel_params.direction = self._scenario.direction

        if self._scenario.pathloss_enabled:
            pl_db = self._lsp_sampler.sample_pathloss()
            # store 
            self.channel_params.pl_db = pl_db
            if self._scenario.direction == 'uplink':
                pl_db = tf.transpose(pl_db, [0,2,1])
        else:
            pl_db = tf.constant(0.0, self._scenario.dtype.real_dtype)

        if not self._scenario.shadow_fading_enabled:
            sf = tf.ones_like(sf)

        # [batch size, num_tx, num_rx]
        gain = tf.math.pow(tf.constant(10., self._scenario.dtype.real_dtype),
            -(pl_db)/20.)*tf.sqrt(sf)
        # [batch size, num_tx, num_rx, 1, 1, ...]
        gain = tf.reshape(gain, tf.concat([tf.shape(gain),
            tf.ones([tf.rank(h)-tf.rank(gain)], tf.int32)],0))

        h *= tf.complex(gain, tf.constant(0., self._scenario.dtype.real_dtype))

        return h





from matplotlib import cm
from matplotlib.ticker import MultipleLocator
def plot_h_vs_delay(h_sample, delays_, num_samples=5, 
                    outfile='h_vs_delay.png', x_limit_ns=700):
    """
    h_sample: tf.Tensor of shape [batch_size, num_paths]
    delays_:  tf.Tensor of shape [batch_size, num_paths]
    num_samples: how many batch‑rows to plot
    outfile: filename for the saved PNG
    x_limit_ns: max for x‑axis (nanoseconds)
    """
    # Convert to numpy
    h_np      = tf.abs(h_sample).numpy()
    delays_np = delays_.numpy()*1e9
    
    batch_size = h_np.shape[0]
    N = min(num_samples, batch_size)
    
    # Choose N distinct colors from a colormap
    colors = cm.get_cmap('tab10', N)
    
    plt.figure(figsize=(12, 4))
    for i in range(N):
        # stem returns (markerline, stemlines, baseline)

        markerline, stemlines, baseline = plt.stem(
            delays_np[i], h_np[i],
            linefmt='-', markerfmt='o', basefmt='k-', 
            label=f'Sample {i}', 
        )
        # override color
        col = colors(i)
        markerline.set_color(col)
        markerline.set_markerfacecolor(col)
        stemlines.set_color(col)
        # leave baseline in black or grey
    
    plt.xlabel('Delay [ns]')
    plt.ylabel(r'$|h|$')
    plt.title(f'H magnitude vs. Delay (first {N} samples)')
    plt.xlim(0, x_limit_ns)
    plt.legend(loc='upper right')
    ax = plt.gca()

    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    plt.minorticks_on()
    ax.tick_params(axis='x', which='minor', length=4, bottom=True)



    plt.minorticks_on()
    plt.tight_layout()
    
    # Save to file instead of showing
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved plot to {outfile}")

