# SPDX-FileCopyrightText: Copyright (c) 2026 Mahdi Abdollahpour <mahdi.abdollahpour@unibo.it>
"""SRS configuration and sequences (3GPP TS 38.211 Sec. 5.2.2 and 6.4.1.4)."""

import numpy as np

from .carrier_config import CarrierConfig
from .config import Config
from .srs_tables import NCS_MAX, PHI, SRS_BANDWIDTH
from .utils import generate_prng_seq


def _largest_prime_below(m):
    for p in range(m - 1, 1, -1):
        if all(p % d for d in range(2, int(p ** 0.5) + 1)):
            return p
    raise ValueError(f"No prime below {m}")


def low_papr_sequence(length, u, v=0):
    r"""Low-PAPR type-1 base sequence :math:`\bar{r}_{u,v}(n)`, TS 38.211 5.2.2.

    Input
    -----
    length : int, :math:`M_\text{ZC}` (6, 12, 18, 24, 30 or >= 36)
    u : int, group number in [0, 29]
    v : int, base sequence number (0, or 1 for lengths >= 72)

    Output
    ------
    : [length], complex
    """
    assert 0 <= u < 30 and v in (0, 1), "Invalid group or sequence number"
    n = np.arange(length)
    if length in PHI:
        assert v == 0, f"Only v=0 exists for length {length}"
        return np.exp(1j*np.pi/4*np.asarray(PHI[length][u]))
    if length == 30:
        assert v == 0, "Only v=0 exists for length 30"
        return np.exp(-1j*np.pi*(u+1)*(n+1)*(n+2)/31)
    assert length >= 36, f"No low-PAPR sequence of length {length}"
    assert length >= 72 or v == 0, f"Only v=0 exists for length {length}"
    n_zc = _largest_prime_below(length)
    a = n_zc*(u+1)
    # q = floor(q_bar + 1/2) + v (-1)^floor(2 q_bar), q_bar = a / 31, in integers
    q = (2*a + 31)//62 + v*(-1)**((2*a)//31)
    m = n % n_zc
    return np.exp(-1j*np.pi*q*((m*(m+1)) % (2*n_zc))/n_zc)


def srs_group_sequence_number(hopping, sequence_id, slot, l, length,
                              num_symbols_per_slot=14):
    r"""Group and sequence number :math:`(u, v)` of the SRS in OFDM symbol
    ``l`` of slot ``slot``, TS 38.211 6.4.1.4.2."""
    if hopping == "neither":
        return sequence_id % 30, 0
    h = num_symbols_per_slot*slot + l
    if hopping == "group":
        c = generate_prng_seq(8*h + 8, sequence_id).astype(int)
        f_gh = int(np.sum(c[8*h:8*h+8] << np.arange(8))) % 30
        return (f_gh + sequence_id) % 30, 0
    assert hopping == "sequence", f"Unknown hopping mode {hopping}"
    v = int(generate_prng_seq(h + 1, sequence_id)[h]) if length >= 72 else 0
    return sequence_id % 30, v


class SRSConfig(Config):
    # pylint: disable=line-too-long
    r"""
    SRS configuration of one UE (one ``SRS-Resource``), TS 38.211 6.4.1.4.

    The subcarrier index is relative to the first subcarrier of the carrier
    grid (BWP start at CRB 0). Frequency hopping is not supported.

    Parameters
    ----------
    carrier_config : :class:`~sionna.nr.CarrierConfig` or `None`
        Carrier of the SRS. If `None`, a default one is created.

    num_ports : int, 1 (default) | 2 | 4
        Number of SRS ports :math:`N_\text{ap}^\text{SRS}` (``nrofSRS-Ports``).

    comb : int, 2 (default) | 4 | 8
        Comb size :math:`K_\text{TC}`.

    comb_offset : int, 0 (default)
        Comb offset :math:`\bar{k}_\text{TC}`.

    cyclic_shift : int, 0 (default)
        Cyclic shift :math:`n_\text{SRS}^\text{cs}`.

    start_position : int, 0 (default)
        :math:`l_\text{offset}`, counted back from the last OFDM symbol of the slot.

    num_symbols : int, 1 (default) | 2 | 4
        Number of consecutive SRS OFDM symbols.

    repetition : int, 1 (default) | 2 | 4
        Repetition factor :math:`R`.

    c_srs, b_srs, b_hop : int
        Bandwidth configuration :math:`C_\text{SRS}` (0 by default),
        :math:`B_\text{SRS}` (0) and :math:`b_\text{hop}` (3). Frequency
        hopping (:math:`b_\text{hop} < B_\text{SRS}`) is not supported.

    freq_position, freq_shift : int, 0 (default)
        :math:`n_\text{RRC}` and :math:`n_\text{shift}` [RB].

    sequence_id : int, 0 (default)
        :math:`n_\text{ID}^\text{SRS}`.

    hopping : str, "neither" (default) | "group" | "sequence"
        Group or sequence hopping.

    beta : float or `None`
        Amplitude scaling :math:`\beta_\text{SRS}`. Every port is scaled by
        :math:`\beta_\text{SRS}/\sqrt{N_\text{ap}}`. `None` (default) gives
        :math:`\sqrt{K_\text{TC}}`: unit average energy per resource element
        and SRS symbol over the sounded band, summed over the ports, as for
        a PUSCH resource grid.
    """
    def __init__(self, carrier_config=None, **kwargs):
        super().__init__(**kwargs)
        self._name = "SRS Configuration"
        self.carrier = carrier_config
        self.check_config()

    #-----------------------------#
    #---Configurable parameters---#
    #-----------------------------#

    @property
    def carrier(self):
        """:class:`~sionna.nr.CarrierConfig` : Carrier configuration"""
        return self._carrier

    @carrier.setter
    def carrier(self, value):
        if value is None:
            value = CarrierConfig()
        assert isinstance(value, CarrierConfig), "carrier must be a CarrierConfig"
        self._carrier = value

    def _setter(self, name, value, valid):
        assert valid(value), f"Invalid value {value} for {name}"
        setattr(self, f"_{name}", value)

    @property
    def num_ports(self):
        """int, 1 (default) | 2 | 4 : Number of SRS ports"""
        self._ifndef("num_ports", 1)
        return self._num_ports

    @num_ports.setter
    def num_ports(self, value):
        self._setter("num_ports", value, lambda x: x in (1, 2, 4))

    @property
    def comb(self):
        """int, 2 (default) | 4 | 8 : Comb size"""
        self._ifndef("comb", 2)
        return self._comb

    @comb.setter
    def comb(self, value):
        self._setter("comb", value, lambda x: x in NCS_MAX)

    @property
    def comb_offset(self):
        """int, 0 (default) : Comb offset"""
        self._ifndef("comb_offset", 0)
        return self._comb_offset

    @comb_offset.setter
    def comb_offset(self, value):
        self._setter("comb_offset", value, lambda x: 0 <= x < 8)

    @property
    def cyclic_shift(self):
        """int, 0 (default) : Cyclic shift"""
        self._ifndef("cyclic_shift", 0)
        return self._cyclic_shift

    @cyclic_shift.setter
    def cyclic_shift(self, value):
        self._setter("cyclic_shift", value, lambda x: 0 <= x < 12)

    @property
    def start_position(self):
        """int, 0 (default) : OFDM symbol offset counted back from the end of the slot"""
        self._ifndef("start_position", 0)
        return self._start_position

    @start_position.setter
    def start_position(self, value):
        self._setter("start_position", value, lambda x: 0 <= x < 14)

    @property
    def num_symbols(self):
        """int, 1 (default) | 2 | 4 : Number of SRS OFDM symbols"""
        self._ifndef("num_symbols", 1)
        return self._num_symbols

    @num_symbols.setter
    def num_symbols(self, value):
        self._setter("num_symbols", value, lambda x: x in (1, 2, 4))

    @property
    def repetition(self):
        """int, 1 (default) | 2 | 4 : Repetition factor"""
        self._ifndef("repetition", 1)
        return self._repetition

    @repetition.setter
    def repetition(self, value):
        self._setter("repetition", value, lambda x: x in (1, 2, 4))

    @property
    def c_srs(self):
        """int, 0 (default) : Bandwidth configuration index"""
        self._ifndef("c_srs", 0)
        return self._c_srs

    @c_srs.setter
    def c_srs(self, value):
        self._setter("c_srs", value, lambda x: 0 <= x < 64)

    @property
    def b_srs(self):
        """int, 0 (default) : Bandwidth index"""
        self._ifndef("b_srs", 0)
        return self._b_srs

    @b_srs.setter
    def b_srs(self, value):
        self._setter("b_srs", value, lambda x: 0 <= x < 4)

    @property
    def b_hop(self):
        """int, 3 (default) : Frequency hopping index"""
        self._ifndef("b_hop", 3)
        return self._b_hop

    @b_hop.setter
    def b_hop(self, value):
        self._setter("b_hop", value, lambda x: 0 <= x < 4)

    @property
    def freq_position(self):
        r"""int, 0 (default) : :math:`n_\text{RRC}`"""
        self._ifndef("freq_position", 0)
        return self._freq_position

    @freq_position.setter
    def freq_position(self, value):
        self._setter("freq_position", value, lambda x: 0 <= x < 68)

    @property
    def freq_shift(self):
        r"""int, 0 (default) : :math:`n_\text{shift}` [RB]"""
        self._ifndef("freq_shift", 0)
        return self._freq_shift

    @freq_shift.setter
    def freq_shift(self, value):
        self._setter("freq_shift", value, lambda x: 0 <= x < 269)

    @property
    def sequence_id(self):
        r"""int, 0 (default) : :math:`n_\text{ID}^\text{SRS}`"""
        self._ifndef("sequence_id", 0)
        return self._sequence_id

    @sequence_id.setter
    def sequence_id(self, value):
        self._setter("sequence_id", value, lambda x: 0 <= x < 1024)

    @property
    def hopping(self):
        """str, "neither" (default) | "group" | "sequence" : Group or sequence hopping"""
        self._ifndef("hopping", "neither")
        return self._hopping

    @hopping.setter
    def hopping(self, value):
        self._setter("hopping", value, lambda x: x in ("neither", "group", "sequence"))

    @property
    def beta(self):
        r"""float : Amplitude scaling :math:`\beta_\text{SRS}`, :math:`\sqrt{K_\text{TC}}` by default"""
        self._ifndef("beta", None)
        return np.sqrt(self.comb) if self._beta is None else self._beta

    @beta.setter
    def beta(self, value):
        self._setter("beta", value, lambda x: x is None or x > 0)

    #-----------------------------#
    #---Read-only parameters------#
    #-----------------------------#

    @property
    def ncs_max(self):
        r"""int, read-only : :math:`n_\text{SRS}^\text{cs,max}`"""
        return NCS_MAX[self.comb]

    @property
    def m_srs(self):
        r"""int, read-only : Sounded bandwidth :math:`m_{\text{SRS},B_\text{SRS}}` [RB]"""
        return SRS_BANDWIDTH[self.c_srs][self.b_srs][0]

    @property
    def length(self):
        r"""int, read-only : Sequence length :math:`M_{\text{sc},b}^\text{SRS}`"""
        return 12*self.m_srs//self.comb

    @property
    def l0(self):
        """int, read-only : First SRS OFDM symbol"""
        return self.carrier.num_symbols_per_slot - 1 - self.start_position

    @property
    def symbols(self):
        """list, read-only : SRS OFDM symbols of the slot"""
        return list(range(self.l0, self.l0 + self.num_symbols))

    @property
    def rb_start(self):
        """int, read-only : First RB of the sounded band"""
        start = self.freq_shift
        for b in range(self.b_srs + 1):
            m_b, n_b = SRS_BANDWIDTH[self.c_srs][b]
            start += m_b*((4*self.freq_position//m_b) % n_b)
        return start

    @property
    def amplitude(self):
        r"""float, read-only : Amplitude of every port, :math:`\beta_\text{SRS}/\sqrt{N_\text{ap}}`"""
        return self.beta/np.sqrt(self.num_ports)

    def port_cyclic_shift(self, i):
        r"""Cyclic shift :math:`n_\text{SRS}^{\text{cs},i}` of port ``i``"""
        if self.num_ports == 4 and self.ncs_max == 6:
            return (self.cyclic_shift + 3*(i//2)) % 6
        return (self.cyclic_shift + self.ncs_max*i//self.num_ports) % self.ncs_max

    def port_comb_offset(self, i):
        r"""Comb offset :math:`k_\text{TC}^{(p_i)}` of port ``i``"""
        split = self.ncs_max == 6 or self.cyclic_shift >= self.ncs_max//2
        if self.num_ports == 4 and i in (1, 3) and split:
            return (self.comb_offset + self.comb//2) % self.comb
        return self.comb_offset

    def subcarriers(self, i):
        """[length], int : Subcarriers of port ``i``"""
        k0 = 12*self.rb_start + self.port_comb_offset(i)
        return k0 + self.comb*np.arange(self.length)

    def sequence_numbers(self, slot=None):
        """List of the :math:`(u, v)` of every SRS symbol"""
        slot = self.carrier.slot_number if slot is None else slot
        return [srs_group_sequence_number(self.hopping, self.sequence_id, slot, l,
                                          self.length, self.carrier.num_symbols_per_slot)
                for l in self.symbols]

    @property
    def srs_grid(self):
        r"""[num_ports, num_subcarriers, num_symbols_per_slot], complex, read-only :
        SRS of every port on the carrier grid, for ``carrier.slot_number``"""
        self.check_config()
        num_sc = 12*self.carrier.n_size_grid
        grid = np.zeros([self.num_ports, num_sc, self.carrier.num_symbols_per_slot], complex)
        n = np.arange(self.length)
        for l, (u, v) in zip(self.symbols, self.sequence_numbers()):
            r = low_papr_sequence(self.length, u, v)
            for i in range(self.num_ports):
                alpha = 2*np.pi*self.port_cyclic_shift(i)/self.ncs_max
                grid[i, self.subcarriers(i), l] = self.amplitude*np.exp(1j*alpha*n)*r
        return grid

    @property
    def srs_mask(self):
        """[num_subcarriers, num_symbols_per_slot], bool, read-only :
        Resource elements carrying the SRS of at least one port"""
        num_sc = 12*self.carrier.n_size_grid
        mask = np.zeros([num_sc, self.carrier.num_symbols_per_slot], bool)
        for i in range(self.num_ports):
            mask[np.ix_(self.subcarriers(i), self.symbols)] = True
        return mask

    def check_config(self):
        """Check that the configuration is consistent"""
        assert self.comb_offset < self.comb, "comb_offset must be smaller than comb"
        assert self.cyclic_shift < self.ncs_max, f"cyclic_shift must be smaller than {self.ncs_max}"
        assert self.num_symbols - 1 <= self.start_position, \
            "The SRS symbols do not fit in the slot"
        assert self.repetition <= self.num_symbols, "repetition must be <= num_symbols"
        assert self.b_hop >= self.b_srs, "SRS frequency hopping is not supported"
        assert self.rb_start + self.m_srs <= self.carrier.n_size_grid, \
            f"The {self.m_srs}-RB SRS starting at RB {self.rb_start} exceeds the " \
            f"{self.carrier.n_size_grid}-RB carrier"
