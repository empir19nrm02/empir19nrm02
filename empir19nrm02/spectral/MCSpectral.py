import numpy as np
import luxpy as lx

# Ich mache es trotzdem :-(, Danke für den Hinweis
# https://stackoverflow.com/questions/15454285/numpy-array-of-class-instances/15455053

# mod. version of drawValues() --> draw_values_gum()
from empir19nrm02.tools import *

__all__ = ['McSpectrumX']

class McSpectrumX(object):
    """
    Spectrum class for MC simulations
    An object holds a spectrum (class luxpy.SPD) with only one wavelength scale and one value array

    With different

    Example:

    Default Values:
    """

    def __init__(self, spd=None):
        if spd is None:
            self.spd = lx.SPD(spd=lx.cie_interp(lx._CIE_ILLUMINANTS['A'], lx.getwlr(), kind='S'), wl=lx.getwlr(),
                              negative_values_allowed=True)
        else:
            self.spd = lx.SPD(spd=spd, negative_values_allowed=True)
        # to remember the number of elements
        self.wlElements = len(self.spd.wl)

    def add_wl_noise_nc(self, ref, mean=0., stddev=1., distribution='normal'):
        self.spd.wl = ref.spd.wl + draw_values_gum(mean, stddev, draws=self.wlElements, distribution=distribution)
        return self.spd.wl

    def add_wl_noise_c(self, ref, mean=0., stddev=1., distribution='normal'):
        self.spd.wl = draw_values_gum(mean, stddev, draws=1, distribution=distribution)[0] + ref.spd.wl
        return self.spd.wl

    def add_value_noise_nc(self, ref, mean=0., stddev=1., distribution='normal'):
        self.spd.value = ref.spd.value + draw_values_gum(mean, stddev, draws=self.wlElements, distribution=distribution)
        return self.spd.value

    def add_value_noise_c(self, ref, mean=0., stddev=1., distribution='normal'):
        self.spd.value = draw_values_gum(mean, stddev, draws=1, distribution=distribution)[0] + ref.spd.value
        return self.spd.value
