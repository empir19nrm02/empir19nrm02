import numpy as np
import luxpy as lx

# Ich mache es trotzdem :-(, Danke für den Hinweis
# https://stackoverflow.com/questions/15454285/numpy-array-of-class-instances/15455053

# mod. version of drawValues() --> draw_values_gum()
from empir19nrm02.tools import *
import math
__all__ = ['McSpectrumX', "generate_FourierMC0"]

def py_getBaseFunctions( number, wl, phaseVector):
    lambda1 = wl[0]
    deltaLambda = wl[wl.size-1]-lambda1
    baseFunctions = np.zeros((number+1, wl.size))
    for i in range(number+1):
        if i==0:
            singleBase = np.ones(wl.size)
        else:
            singleBase = math.sqrt(2)*np.sin(i*(2*math.pi*((wl-lambda1)/(deltaLambda))+phaseVector[i]))
        baseFunctions[i,:] = singleBase.transpose()
    return baseFunctions.transpose()

def py_getGammai( number):
    Yi = stats.norm.rvs(size=number+1)
    QSum=np.sum( Yi**2)
    return Yi/math.sqrt(QSum)

def generate_FourierMC0( number, wl, uValue):
    rGammai = np.random.normal(size=(number+1))
    QSumSqrt = np.sqrt(np.sum(rGammai**2))
    rGammaiN = rGammai / QSumSqrt
    # Correction for the correlated contribution : Here we do not need the normalization
    rGammaiN[0] = rGammai[0]
    rPhasei = np.random.uniform(low = 0, high = 2*math.pi, size = (number+1))
    baseFunctions = py_getBaseFunctions( number, wl, rPhasei)
    rMatrix = np.dot(baseFunctions, rGammaiN)
    rMatrixSPD = rMatrix*uValue
    return rMatrixSPD

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

    def add_wl_noise_nc(self, mean=0., stddev=1., distribution='normal'):
        self.spd.wl = self.spd.wl + draw_values_gum(mean, stddev, draws=self.wlElements, distribution=distribution)
        return self.spd.wl

    def add_wl_noise_c(self, mean=0., stddev=1., distribution='normal'):
        self.spd.wl = draw_values_gum(mean, stddev, draws=1, distribution=distribution)[0] + self.spd.wl
        return self.spd.wl

    def add_wl_fourier_noise(self, ref, number, stddev=1.):
        self.spd.wl = generate_FourierMC0( number, ref.spd.wl, stddev) + self.spd.wl
        return self.spd.value

    def add_value_noise_nc(self, mean=0., stddev=1., distribution='normal'):
        self.spd.value = self.spd.value + draw_values_gum(mean, stddev, draws=self.wlElements, distribution=distribution)
        return self.spd.value

    def add_value_noise_c(self, mean=0., stddev=1., distribution='normal'):
        self.spd.value = draw_values_gum(mean, stddev, draws=1, distribution=distribution)[0] + self.spd.value
        return self.spd.value

    def add_value_fourier_noise(self, ref, number, stddev=1.):
        self.spd.value = (1+generate_FourierMC0( number, ref.spd.wl, stddev)) * self.spd.value
        return self.spd.value
