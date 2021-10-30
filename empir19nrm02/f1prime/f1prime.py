########################################################################
# <f1Prime: a Python module for general f1Prime calculations.>
# Copyright (C) <2021>  <Udo Krueger> (udo.krueger at technoteam.de)
# All basic photometric/colorimetric stuff from luxpy
#########################################################################

from pyxll import xl_func
import luxpy as lx
import numpy as np
import math
from scipy.optimize import minimize_scalar
from scipy.fft import fft, ifft, fftfreq

__all__ = ['py_f1Prime', 'py_f1PrimeGlx', 'py_f1PrimeG']

@xl_func("numpy_array<float> wlScale, numpy_array<float> srData: float")
def py_f1Prime( wlScale, srData):
    """
    Calculate the standard f1Prime value according to (ISO/CIE 19476:2014-06, 2014).

    Args:
        :wlScale:
            | wavelength scale (ndarray, .shape(n,))
        :srData:
            | spectral responsivity data at the points of wlScale (ndarray, .shape(n,))
    Returns:
        :returns:
            | float f1Prime value

    Note:
        Only one wlScale and one spectral responsivity is supported.
    """
    return py_f1PrimeG( wlScale, srData, strObserver='1931_2', iObserverOffset=1, strWeighting='A')


@xl_func("numpy_array<float> srDataWithWlScale, string strObserver, int iObserverOffset, \
          string strWeighting, int iMin, float dCutOff, float dBandWidth: float", auto_resize=True)
def py_f1PrimeGlx( srDataWithWlScale, strObserver='1931_2', iObserverOffset = 1, strWeighting='A', iMin=0, \
                 dCutOff=0., dBandWidth=0.):
    f1p = np.zeros(srDataWithWlScale.shape[0]-1)
    for iNumber in range(f1p.size):
        [f1p[iNumber], _]=py_f1PrimeG(srDataWithWlScale[0,:], srDataWithWlScale[iNumber+1,:], \
                                 strObserver=strObserver, iObserverOffset=iObserverOffset, \
                                 iMin=iMin, dCutOff=dCutOff, dBandWidth=dBandWidth, strWeighting=strWeighting)
    return f1p


def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return math.ceil(math.log(number,2))

def PadRight(arr):
    nextPower = NextPowerOfTwo(len(arr))
    deficit = int(math.pow(2, nextPower) - len(arr))
    arr = np.concatenate((arr, np.zeros(deficit, dtype=arr.dtype)))
    return arr


useAFOrgImplementation = False

@xl_func("numpy_array<float> wlScale, numpy_array<float> srData, string strObserver, int iObserverOffset, \
          string strWeighting, int iMin, float dCutOff, float dBandWidth: float", auto_resize=True)
def py_f1PrimeG( wlScale, srData, strObserver='1931_2', iObserverOffset = 1, strWeighting='A', iMin=0, \
                 dCutOff=0., dBandWidth=0.):
    """
    Calculate the general f1Prime value with very different versions of target functions, weightings and
    other ideas from literature. 

    Args:
        :wlScale:
            | wavelength scale (ndarray, .shape=(n,))
        :srData:
            | spectral responsivity data at the point of wlScale (ndarray, .shape=(n,))
        :strObserver:
            | Name of the Observer used for the target functions
            | All Observers (color matching functions implemented in  luypy (see lx._CMF.keys()) 
            | are supported.
        :iObserverOffset:
            | 0 ... xBar
            | 1 ... yBar (V(Lambda), ...)
            | 2 ... zBar 
        :strWeighting:
            | Weighting function to scale the srData to the V(Lambda)/Target function
            | All illuminants from luypy are supported (see lx._CIE_ILLUMINANTS.keys()).
            | Examples:
            | 'E' ...  No weighting at all
            | 'A' ...  Weighting with standard illuminant A (the standard weighting)
            | 'LED_B3' Weighting with illuminant L the future standard illuminant L
        :iMin:
            | 0 ... Use the selected weighting
            | 1 ... Calculate the minimal f1Prime value while changing the weighting factor
        :dCutOff:
            | dCutOff > 0 ... Use the fourier method of Alejandro Ferrero (https://doi.org/10.1364/OE.26.018633)
            | dCutOff < 0 ... same as dCutOff>0 but use the invers fourier transformation after applying the CutOff
            |                 to get comparible f1Prime values
            | The CutOff-frequency should be given in 1/nm
        :dBandWidth:
            | dBandWidth > 0 (dBandWidth in nm)
            | Convolution of the difference function with an symmetric LED model of dBandWidth FWHM

    Returns:
        :returns:
            | float f1Prime value

    Examples:

    Note:
        There is no need that the wlScale is monotone or equidistant. The calculation is done with the trapz
        integration on the wlScale of the caller. Only the target and weighting functions are interpolated.
    """
    res = wlScale.size
    wlScale = wlScale.reshape(res)
    srData = srData.reshape(res)
    # calculate the mean step for the data (assume a non equidistant wlScale)
    deltaLambda = np.mean(np.diff(wlScale))

    # Get CMF from lx
    lxCmf = lx._CMF[strObserver]
    # Get the weighting function from lx
    lxWeighting = lx._CIE_ILLUMINANTS[strWeighting]

    # interpolate to srData
    iCmf = lx.cie_interp(lxCmf['bar'], wlScale, kind='linear')
    iWeighting = lx.cie_interp(lxWeighting, wlScale, kind='linear')

    # calculate some temporary sums
    sObserver = np.trapz(iCmf[iObserverOffset + 1], wlScale)
    sProduct = np.trapz(iCmf[iObserverOffset + 1] * iWeighting[1], wlScale)
    sDenominator = np.trapz( iWeighting[1] * srData, wlScale)

    sNorm = sProduct / sDenominator

    # we minimize the usual f1Prime equation varying the dNorm value only
    def dstFunction( dNorm):
        return np.trapz(abs((srData * dNorm).T - iCmf[iObserverOffset + 1]), wlScale) / sObserver

    # Use the Min f1Prime value according to Alejandro Ferrero
    if iMin > 0:
        # call optimization (usually no start value and no boundaries required
        dNormMinOpt = minimize_scalar(dstFunction)
        # take over the weighting factor for the minimal f1Prime value
        sNorm = dNormMinOpt.x

    # calculate the difference vector, normalized to the observer integral
    deltaVector = (srData * sNorm - iCmf[iObserverOffset + 1]) / sObserver
    # Use the fourier method of Alejandro Ferrero (https://doi.org/10.1364/OE.26.018633)
    # For the original method strWeighting should be used with 'E'
    # FYI: dCutOff>0 can be combined with iMin>0!
    if dCutOff == 0:
        if dBandWidth > 0:
            # get a gaussian SPD of a LED
            led = lx.toolboxes.spdbuild.gaussian_spd( \
                peakwl=0, fwhm=dBandWidth, wl=[-3 * dBandWidth, 3 * dBandWidth, deltaLambda], with_wl=False)
            led = led.reshape(led.size)/np.sum(led)
            # make the convolution of the deltaVector with the gaussian SPD
            # assuming the wlScale is approximately equidistant :-(
            deltaVector = np.convolve(deltaVector, led, mode='same')
        else:
            pass
    else:
        if dCutOff > 0:
            # original method from AF
            # calculate the abs value of the fft (squared)
            # Original implementation in python based on matlab source code from AF
#            #if useAFOrgImplementation:
#                ldo0=wlScale
#                Vlambda=iCmf[iObserverOffset + 1]
#                Responsividad=srData
#                normali = np.sum((ldo0[2] - ldo0[1]) * Vlambda) / np.sum((ldo0[2] - ldo0[1]) * Responsividad)
#                Dif_fpirma_s = (Responsividad * normali - Vlambda) / np.sum((ldo0[2] - ldo0[1]) * Vlambda)
#                L = ldo0.shape[0]
#                Fs = 1 / (ldo0[2] - ldo0[1])
#                NFFT = int(math.pow(2, NextPowerOfTwo(L)))
#                Dif_fpirma_net=np.zeros(NFFT)
#                #Dif_fpirma_net[L + 1: NFFT]=0
#                Dif_fpirma_net[:L]= Dif_fpirma_s - np.mean(Dif_fpirma_s)
#                PSD = (1 / (L * Fs)) * np.power(np.abs(fft(Dif_fpirma_net)), 2)
#                f = Fs / 2 * np.linspace(0, 1, int(NFFT / 2 + 1))
#                PSD_ss = np.abs(PSD[0:int(NFFT / 2 + 1)])
#                f_inter = Fs * lx.getwlr([0, 0.5, 0.0001])
#                PSD_ss_int = np.interp(f_inter, f, PSD_ss)
#                indices = np.where(np.logical_and(f_inter >= 0, f_inter < dCutOff))
#                f1PrimeGValue1 = np.sqrt(2 * np.sum((f_inter[2] - f_inter[1]) * PSD_ss_int[indices]))
#                deltaVector= Dif_fpirma_s
            if useAFOrgImplementation:
                # modified version with identical behavior
                deltaVectorZeroPadding=PadRight(deltaVector)
                deltaVectorZeroPadding[:res]=deltaVectorZeroPadding[:res]-np.mean(deltaVectorZeroPadding[:res])
                resZeroPadding = deltaVectorZeroPadding.shape[0]
                resOrgData = deltaVector.shape[0]
                deltaVectorFFTZeroPadding = deltaLambda/resOrgData*np.power(np.abs(fft(deltaVectorZeroPadding)), 2)

                # get the frequency list from the FFT scale
                wlFrequenciesZeroPadding = 1 / (2*deltaLambda) * np.linspace(0, 1, resZeroPadding // 2 +1)

                wlFrequenciesInterool = 1/deltaLambda*lx.getwlr([0, 0.5, 0.0001])
                deltaVectorFFTZeroPaddingInterpol = np.interp( wlFrequenciesInterool, wlFrequenciesZeroPadding, deltaVectorFFTZeroPadding[:resZeroPadding // 2+1])

                intIndexZeroPaddingInterpol = np.where(np.logical_and(wlFrequenciesInterool >= 0, wlFrequenciesInterool < dCutOff))
                # attention this value gives total different numbers compared with f1Prime
                f1PrimeGValue = np.sqrt(2 * np.sum(deltaVectorFFTZeroPaddingInterpol[intIndexZeroPaddingInterpol]) * (wlFrequenciesInterool[2]-wlFrequenciesInterool[1]))
            else:
                # different Implementation cf AF
                # - No ZeroPadding
                # - No Interpolation in the frequency space
                # - using the right frequencies in the frequency domain

                deltaVectorOffset = deltaVector - np.mean(deltaVector)
                deltaVectorOffsetFFT =np.power(np.abs(fft(deltaVectorOffset)), 2)

                #var1 = np.std(deltaVectorOffset)
                #var2 = math.sqrt(2/(res*res)*np.sum(deltaVectorOffsetFFT[:res//2]))
                #print( var1, var2, var1/var2, var1-var2)
                # get the frequency list from the FFT scale
                wlFrequencies = fftfreq(res, deltaLambda)[:res // 2]

                intIndex = np.where( np.logical_and(wlFrequencies >= 0, wlFrequencies < dCutOff))
                # attention this value gives total different numbers compared with f1Prime
                f1PrimeGValue = np.sqrt(2/(res*res) * np.sum(deltaVectorOffsetFFT[intIndex]))

        else:
            # modified version with back transfer after applying the cutoff
            # calculate the abs value of the fft (squared)
            deltaVectorFFT = fft(deltaVector)
            # get the frequency list from the FFT scale
            wlFrequencies = fftfreq(res, deltaLambda)
            intIndex = np.where(abs(wlFrequencies) >= abs(dCutOff))
            deltaVectorFFT[intIndex] = 0
            # modification of the delta vector only
            deltaVector = ifft( deltaVectorFFT)
    if dCutOff <= 0:
        f1PrimeGValue = np.trapz(abs(deltaVector), wlScale)

    return [f1PrimeGValue, deltaVector]

@xl_func("numpy_array<float> wlScale, numpy_array<float> srData, string strObserver, int iObserverOffset, \
          string strWeighting, int iMin, float dCutOff, float dBandWidth: numpy_array<float>", auto_resize=True)
def py_f1PrimeGTestFreq( wlScale, srData, strObserver='1931_2', iObserverOffset = 1, strWeighting='A', iMin=0, \
                 dCutOff=0., dBandWidth=0.):
    res = wlScale.size
    wlScale = wlScale.reshape(res)
    srData = srData.reshape(res)
    # calculate the mean step for the data (assume a non equidistant wlScale)
    deltaLambda = np.mean(np.diff(wlScale))

    # Get CMF from lx
    lxCmf = lx._CMF[strObserver]
    # Get the weighting function from lx
    lxWeighting = lx._CIE_ILLUMINANTS[strWeighting]

    # interpolate to srData
    iCmf = lx.cie_interp(lxCmf['bar'], wlScale, kind='linear')
    iWeighting = lx.cie_interp(lxWeighting, wlScale, kind='linear')

    # calculate some temporary sums
    sObserver = np.trapz(iCmf[iObserverOffset + 1], wlScale)
    sProduct = np.trapz(iCmf[iObserverOffset + 1] * iWeighting[1], wlScale)
    sDenominator = np.trapz( iWeighting[1] * srData, wlScale)

    sNorm = sProduct / sDenominator

    # we minimize the usual f1Prime equation varying the dNorm value only
    def dstFunction( dNorm):
        return np.trapz(abs((srData * dNorm).T - iCmf[iObserverOffset + 1]), wlScale) / sObserver

    # Use the Min f1Prime value according to Alejandro Ferrero
    if iMin > 0:
        # call optimization (usually no start value and no boundaries required
        dNormMinOpt = minimize_scalar(dstFunction)
        # take over the weighting factor for the minimal f1Prime value
        sNorm = dNormMinOpt.x

    # calculate the difference vector, normalized to the observer integral
    deltaVector = (srData * sNorm - iCmf[iObserverOffset + 1]) / sObserver
    # Use the fourier method of Alejandro Ferrero (https://doi.org/10.1364/OE.26.018633)
    # For the original method strWeighting should be used with 'E'
    # FYI: dCutOff>0 can be combined with iMin>0!
    if dCutOff == 0:
        if dBandWidth > 0:
            # get a gaussian SPD of a LED
            led = lx.toolboxes.spdbuild.gaussian_spd( \
                peakwl=0, fwhm=dBandWidth, wl=[-3 * dBandWidth, 3 * dBandWidth, deltaLambda], with_wl=False)
            led = led.reshape(led.size)/np.sum(led)
            # make the convolution of the deltaVector with the gaussian SPD
            # assuming the wlScale is approximately equidistant :-(
            deltaVector = np.convolve(deltaVector, led, mode='same')
        else:
            pass
    else:
        if dCutOff > 0:
            # original method from AF
            # calculate the abs value of the fft (squared)
            if iMin > 0:
                deltaVector1 = np.zeros(iMin)
                deltaVector1[0:res] = deltaVector
                res1 = deltaVector1.size
            else:
                 deltaVector1 = deltaVector
                 res1 = res
            deltaVectorFFT = np.power(np.abs(fft(deltaVector1)), 2)
            # get the frequency list from the FFT scale
            wlFrequencies = fftfreq(res1, deltaLambda)[:res1 // 2]
            intIndex = np.where(wlFrequencies < dCutOff)
            return np.vstack((wlFrequencies, deltaVectorFFT[:res1//2]))
            # attention this value gives total different numbers compared with f1Prime
            f1PrimeGValue = math.sqrt(2 * np.trapz(deltaVectorFFT[intIndex], wlFrequencies[intIndex]))
        else:
            # modified version with back transfer after applying the cutoff
            # calculate the abs value of the fft (squared)
            deltaVectorFFT = fft(deltaVector)
            # get the frequency list from the FFT scale
            wlFrequencies = fftfreq(res, deltaLambda)
            intIndex = np.where(abs(wlFrequencies) >= abs(dCutOff))
            deltaVectorFFT[intIndex] = 0
            # modification of the delta vector only
            deltaVector = ifft( deltaVectorFFT)
    if dCutOff <= 0:
        f1PrimeGValue = np.trapz(abs(deltaVector), wlScale)

    return f1PrimeGValue