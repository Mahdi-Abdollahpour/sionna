
# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2025



class ChannelData:
    """A class to store channel parametric data"""

    def __init__(self, powers=None, doppler_shifts=None, alpha=None,
                 theta=None, phi=None, r=None, d=None):
        self._powers = powers
        self._doppler_shifts = doppler_shifts
        self._alpha = alpha
        self._theta = theta
        self._phi = phi
        self._r = r
        self._d = d

    # Properties for powers
    @property
    def powers(self):
        return self._powers

    @powers.setter
    def powers(self, value):
        self._powers = value

    # Properties for doppler_shifts
    @property
    def doppler_shifts(self):
        return self._doppler_shifts

    @doppler_shifts.setter
    def doppler_shifts(self, value):
        self._doppler_shifts = value

    # Properties for alpha
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    # Properties for theta
    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value

    # Properties for phi
    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        self._phi = value

    # Properties for r
    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._r = value

    # Properties for d
    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = value
    



