# -*- coding: utf-8 -*-
"""
MC_Tools_PTB V1.2 (NMISA)
Created on Mon Feb  1 12:07:30 2021

MC_Tools_PTB summarizes functions from Scipy and Numpy to new functions useful for monte carlo simulations.
An additional function for summarizing the output of a monte carlo simulation for a single output quantity is given.

V 1.0 PTB internal version with specific functions for data handling in TDMS and evaluating LSA spectra
V 1.1 PTB internal version with full english commentary
V 1.2 (NMISA) Removed PTB specific data handling and spectral evaluation
---------------------------------
Overview:

    -drawValues         Draws values from a given distribution returning a numpy array of values
    -drawFromArray      Draws values from an array of measurement data according to a Student-T-distribution returning a numpy array of values
    -sumMC              Summarizes the monte carlo draws for a single quantity
    -drawMultiVariate   Draws values from a multivariate distribution taking a correlation matrix into account
    -correlation        Calculates the correlation matrix
    -corrPlot           Plots the values of a correlation matrix

---------------------------------


@author: schnei19
@modified: UK
"""

import numpy as np
import pandas
import scipy.stats as stats
import math

__all__ = ['draw_values_gum','sumMC', 'sumMCV']

def draw_values_gum(mean=0, stddev=1, draws=1000, distribution="normal"):
    """
    UK/210807 mod. version from drawValues
    Attention: Without T-distribution
    Generating different distributions with mean and stddev in the result.

    Generates draws number of values from several usually used distribution types using stats.
    As stddev the absolute standard measurement uncertainty has to be given.

    If called with Numpy-Arrays of mean and stddev a Matrix is returned
    with len(mean) rows and draws columns (shape=(len(mean), draws)).

    Example:
        draw_values_gum(mean=1, stddev=0.5, draws=1000, distribution="normal")
        draw_values_gum(mean=np.array([0.,3.]), stddev=np.array([1.,10.]), draws=1000, \
            distribution="triangle")

    Defaultvalues:
        mean = 0.
        stddev = 1.
        draws = 1000
        distribution = "normal"

    Implemented are these distribution types: "normal", "uniform" and "triangle"

    """
    try:
        size = (draws, len(mean))
    except (TypeError, AttributeError) as e:
        size = (draws)

    if distribution == "normal":
        samples = stats.norm.rvs(loc=mean, scale=stddev, size=size)
        return samples.T
    if distribution == "uniform":
        w = math.sqrt(3) * stddev
        a = mean - w
        b = mean + w
        samples = stats.uniform.rvs(loc=a, scale=b - a, size=size)
        return samples.T
    if distribution == "triangle":
        w = math.sqrt(6) * stddev
        a = mean - w
        b = mean + w
        m = (b + a) / 2.
        samples = stats.triang.rvs(loc=a, scale=b - a, c=w / (2 * w), size=size)
        return samples.T
    return 0


def sumMC(InputValues, Coverage=0.95):
    """
    Based on InputValues for one quantity and the given coverage the measurement uncertainty based on montecarlo results is calculated.

    Output is returned as: [[Mean, absolute Standarduncertainty],[lower coverage boundary, upper coverage boundary]]

    Example:    sumMC([Numpy array], Coverage = 0.99)

    Defaultvalue:
        Coverage = 0.95 (k=2 for normal distributions)
    """
    # Sorting of the input values
    Ys = np.sort(InputValues, axis=None)
    # Calculating the number of draws
    Ylen = len(Ys)
    # Calculating the number of draws covering the given coverage
    q = int(Ylen * Coverage)
    # Calculating the draw representing the lower coverage intervall boundary
    r = int(0.5 * (Ylen - q))
    # Calculating the mean of the input values
    ymean = np.mean(Ys)
    # Calculating standard deviation of the input values as absolute standard uncertainty
    yunc = np.std(Ys)
    # Summarizing mean and uncertainty
    values = [ymean, yunc]
    # Calculating the values of the draws for lower and upper boundary of the coverage intervall
    ylow = Ys[r]
    yhigh = Ys[r + q]
    # Summarizing the coverage intervall
    interval = [ylow, yhigh]
    # Summarizing the total output
    output = [values, interval]
    # Returns the output values
    return (output)

def sumMCV(InputValues, Coverage=0.95):
    """
    Based on sumMC
    Based on InputValues as an Array (trials, wavelength) for one quantity and the given coverage the measurement uncertainty
    based on montecarlo results is calculated.

    Output is returned as: [[Mean, absolute Standard uncertainty],[lower coverage boundary, upper coverage boundary]] as vectors

    Example:    sumMCV([Numpy array], Coverage = 0.99)

    Defaultvalue:
        Coverage = 0.95 (k=2 for normal distributions)
    """
    # Sorting of the input values
    Ys = np.sort(InputValues, axis=0)
    # Calculating the number of draws
    Ylen = Ys.shape[0]
    # Calculating the number of draws covering the given coverage
    q = int(Ylen * Coverage)
    # Calculating the draw representing the lower coverage intervall boundary
    r = int(0.5 * (Ylen - q))
    # Calculating the mean of the input values
    ymean = np.mean(Ys, axis=0)
    # Calculating standard deviation of the input values as absolute standard uncertainty
    yunc = np.std(Ys, axis=0)
    # Summarizing mean and uncertainty
    values = [ymean, yunc]
    # Calculating the values of the draws for lower and upper boundary of the coverage intervall
    ylow = Ys[r,:]
    yhigh = Ys[r + q,:]
    # Summarizing the coverage intervall
    interval = [ylow, yhigh]
    # Summarizing the total output
    output = [values, interval]
    return (output)

def main():
    # Test the default values
    res1=draw_values_gum()
    [data1, interval1] = sumMC( res1)
    print ('Mean1:', data1[0], 'StdDev:', data1[1], 'Interval:', interval1[0], interval1[1])

    # test a 1D Array for mean / stddev
    mean = np.array([1, 2])
    stddev = np.array([2,3])
    draws=1000
    dist = 'normal'
    res2=draw_values_gum(mean=mean, stddev=stddev, draws=draws, distribution=dist)
    [data2, interval2] = sumMC( res2[0])
    print ('Mean2:', data2[0], 'StdDev:', data2[1], 'Interval:', interval2[0], interval2[1])
    [data3, interval3] = sumMC( res2[1])
    print ('Mean3:', data3[0], 'StdDev:', data3[1], 'Interval:', interval3[0], interval3[1])


if __name__ == '__main__':
    main()
