########################################################################
# <ValueCorrection: a Python module for correcting wavelength and value scales.>
# Copyright (C) <2022>  <Udo Krueger> (udo.krueger at technoteam.de)
#########################################################################

"""
Module for class functionality for ValueCorrections
===========================================================

.. codeauthor:: UK
"""



import numpy as np
import luxpy as lx
import scipy
import pickle

from matplotlib import pyplot
from numpy import ndarray
from numpy.polynomial import Polynomial

from empir19nrm02.tools import plotSPDs
from empir19nrm02.tools.help import label_font_size, save_fig

__all__ = ['q_param_org','ValueCorrection','ValueCorrectionNonLinearity','WavelengthCalib']

def q_param_org(x_prime:ndarray, x:ndarray):
    """
    Calculate a quality metric for the fit for further use and display.
    Form the x' and x values a difference function is calculated.
    For the difference values the standard deviation and the maximum absolute value of the difference values is calculated
    :param x_prime: measurement values
    :param x: reference values
    :return: difference values, std. deviation, abs(max. difference)
    """
    _diff = x-x_prime
    qs = np.std(_diff)
    qmm = np.max(np.abs(_diff))
    return x-x_prime, qs, qmm


class ValueCorrection(object):
    """
    Class for correcting small errors (e.g. for wavelength or non-linearity corrections)
    """
    def __init__(self, n:int=3, domain=None):
        """
        Constructor
        :param n: degree of the polynom for regression
        :param domain: domain for the polynom regression (https://numpy.org/devdocs/reference/generated/numpy.polynomial.polynomial.Polynomial.html#numpy.polynomial.polynomial.Polynomial)
        """
        self.n = n
        self.poly = Polynomial(np.zeros(self.n), domain=domain, window=[-1,1])
        self.series = None
        self.x_prime = None
        self.x = None
    def getInputValues(self):
        """
        Return all measurement and reference values valid for the regression (under the overload)
        :return:
        """
        return self.x_prime, self.x
    def fit(self, x_prime:ndarray, x:ndarray, domain=[]):
        """
        calculate a polynomial fit for the values (x', x)
        :param x_prime: measurement values (current knowlege)
        :param x: reference values
        :param domain: See abouve. [] means using the settings from the constructor.
        :return:
        """
        self.x_prime = x_prime.copy()
        self.x = x.copy()
        self.series=self.poly.fit(self.x_prime, self.x, deg=self.n, domain=domain)
        return self.series
    def p(self, x_prime):
        """
        Calculating the nominal value from the measurement value based on the polynom fit.
        :param x_prime: measurement values
        :return: corrected values
        """
        return self.series(x_prime)
    def p_diff(self, x_prime):
        """
        Calculating the difference of the nominal value from the measurement value based on the polynom fit to the measurement value.
        :param x_prime: measurement values
        :return: p(x')-x
        """
        return self.p(x_prime)-x_prime

    def q_param_corr(self, x_prime, x):
        """
        Calculate the quality params for the corrected values.
        :param x_prime:measurement values
        :param x:reference values
        :return:quality parameters (see q_param_org)
        """
        x_corr = self.p(x_prime)
        return q_param_org(x_corr, x)

class ValueCorrectionNonLinearity(ValueCorrection):
    """
    Class for correcting the non-linearity
    """
    def __init__(self, n:int=3, domain=None, value_ref:float = 3500, value_max:float = 3900):
        """
        Correction for non-linearity issues
        :param n: polynom degree
        :param domain: see above (None mean automatic setting for the domain)
        :param value_ref:reference value for the non-linearity correction
        :param value_max:maximal value for the data which will be used in the non-linearity correction
        """
        super().__init__(n, domain=domain)
        self.value_ref = value_ref
        self.value_max = value_max
        self.x_ref = None

    def getInputValues(self):
        """
        Return all measurement and reference values valid for the regression (under the overload)
        :return:
        """
        return self.x_prime[self.x_prime < self.value_max], self.x_ref[self.x_prime < self.value_max]

    def fit(self, x_prime:ndarray, x:ndarray, domain=[]):
        """
        calculating the polynomial fit
        :param x_prime: measurement values (grey values)
        :param x: reference values (in this case usually integration times --> will be converted to grey values)
        :param domain: see above ([] means using the original setting of the contractor)
        :return: result of the polynomial regression
        """
        # store the original input data
        self.x_prime = x_prime.copy()
        self.x = x.copy()
        # generate a reference scale usually from the integration times
        time_interp = scipy.interpolate.interp1d(x_prime, x)
        time_ref = time_interp( self.value_ref)
        self.x_ref = x * self.value_ref/time_ref

        # select the input values below saturation
        x_prime_t, x_t = self.getInputValues()
        # make a regression
        self.series=self.poly.fit(x_prime_t, x_t, deg=self.n, domain=domain)
        return self.series

    def pickle(self, filename = None):
        if filename is None:
            filename = 'ValueCorrectionNonLinearity.pkl'
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickle(filename=None):
        if filename is None:
            filename = 'ValueCorrectionNonLinearity.pkl'
        with open(filename, 'rb') as f:
            return pickle.load(f)


class WavelengthCalib(object):
    """
    Class for wavelength calibration
    """
    def __init__(self, type = 'Hg'):
        self.type = type
        self.n = 0
        self.S = None
        match type:
            case 'Hg':
                self.allocS(3)
                i_start = 0
                #Hg; https://physics.nist.gov/PhysRefData/Handbook/Tables/mercurytable2_a.htm
                self.S[0,i_start]=404.6563
                self.S[0,i_start+1]=435.8328
                self.S[0,i_start+2]=546.0735
            case 'Kr':
                self.allocS(4)
                #Kr; https://physics.nist.gov/PhysRefData/Handbook/Tables/kryptontable2_a.htm
                i_start = 0
                self.S[0,i_start]=557.02894
                self.S[0,i_start+1]=587.09160
                self.S[0,i_start+2]=760.15457
                self.S[0,i_start+3]=769.45401
            case _:
                print(type + 'not implemented')
    def allocS(self, _n):
        self.n = _n
        self.S = np.zeros((2,self.n))
    def getPairs(self, sd, verbosity=1):
        # plot the original data
        if verbosity == 1:
            plotSPDs(sd)
            pyplot.title( self.type, fontsize=label_font_size)
            pyplot.xlabel(r'$\lambda^{´}$' + '/ nm', fontsize=label_font_size)
            pyplot.ylabel('Signal', fontsize=label_font_size)
            save_fig(dir='Poly', filename=self.type + 'Measurement')
        # detect the peaks
        res = lx.spectrum.detect_peakwl(sd[0:2], n=self.n, verbosity=verbosity)
        # plot the detected peaks
        if verbosity == 1:
            pyplot.title(self.type, fontsize=label_font_size)
            pyplot.xlabel(r'$\lambda^{´}$' + '/ nm', fontsize=label_font_size)
            pyplot.ylabel('Signal', fontsize=label_font_size)
            save_fig(dir='Poly', filename=self.type + 'Peak')
        # collect the peaks
        i_start = 0
        for i in range(self.n):
            self.S[1,i+i_start]=res[0]['fwhms_mid'][i]
        return self.S
