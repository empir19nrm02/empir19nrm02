########################################################################
# <FourierNoise: a Python module to generate base function noise for MC simulations.>
# Original Idea (Fourier noise):  https://doi.org/10.1088/1681-7575/aa7b39
# Other base functions (chebyshev): https://doi.org/10.5194/amt-11-3595-2018
# Copyright (C) <2023>  <Udo Krueger> (udo.krueger at technoteam.de)
#########################################################################

"""
Module for class functionality for MC Simulations
===========================================================

.. codeauthor:: UK
"""

from scipy.stats import stats
import math
from numpy import ndarray
import numpy as np

__all__ = ['generate_base_functions','py_getBaseFunctions']

def py_getBaseFunctions( number:int, base_function_size:int, phaseVector:ndarray, single_function = False)->ndarray:
    """
    Generate a set of base functions according to https://doi.org/10.1088/1681-7575/aa7b39.
    equation (15)

    Args:
        :number:
            | number of base functions
        :base_function_size:
            | length of the base functions (the number of wavelength points in the original article)
        :phaseVector:
            | vector of phase information (according to ())

    Returns:
        :returns:
            | ndarray with base number functions

    Note:
    """
    wl = np.arange(0., base_function_size, 1.)/base_function_size
    if single_function:
        baseFunctions = np.zeros((1, base_function_size))
        i = number
        if i==0:
            singleBase = np.ones(base_function_size)
        else:
            singleBase = math.sqrt(2)*np.sin(i*(2*math.pi*wl+phaseVector[i]))
        baseFunctions[0,:] = singleBase.transpose()
    else:
        baseFunctions = np.zeros((number+1, base_function_size))
        for i in range(number+1):
            if i==0:
                singleBase = np.ones(base_function_size)
            else:
                singleBase = math.sqrt(2)*np.sin(i*(2*math.pi*wl+phaseVector[i]))
            baseFunctions[i,:] = singleBase.transpose()
    return baseFunctions.transpose()

def py_getGammai( number:int)->ndarray:
    """
    Generate a set of gamma values according to https://doi.org/10.1088/1681-7575/aa7b39.
    equation (13)

    Args:
        :number:
            | number of base functions

    Returns:
        :returns:
            | ndarray with set of normalized gamma values

    Note:
    """
    Yi = stats.norm.rvs(size=number+1)
    QSum=np.sum( Yi**2)
    return Yi/math.sqrt(QSum)

def generate_FourierMC0( number:int, base_function_size:int, uValue:float, org_function = True, single_function = False, use_gamma_phi=True)->ndarray:
    """
    Generate a set of fourier based functions according to https://doi.org/10.1088/1681-7575/aa7b39.
    equation (9)

    Args:
        :number:
            | number of base functions
        :base_function_size:
            | length of the base functions (the number of wavelength points in the original article)
        :uValue:
            | uncertainty for the base functions (u_c() in equation (8))
        :single_function:
            | use the maximum order only
        : use_gamma_phi:
            | use gamma and phi

    Returns:
        :returns:
            | ndarray with set of base functions with weighting

    Note:
    """
    rGammai = np.random.normal(size=(number+1))
    QSumSqrt = np.sqrt(np.sum(rGammai**2))
    rGammaiN = rGammai / QSumSqrt
    # Correction for the correlated contribution : Here we do not need the normalization
    # this is different to the original article
    if not org_function:
        rGammaiN[0] = rGammai[0]
    rPhasei = np.random.uniform(low = 0, high = 2*math.pi, size = (number+1))
    if not use_gamma_phi:
        rPhasei[:] = 0
    baseFunctions = py_getBaseFunctions( number, base_function_size, rPhasei, single_function = single_function)
    if not use_gamma_phi:
        rGammaiN[:] = 1.
    if single_function:
        rMatrix = np.squeeze(np.dot(baseFunctions, rGammaiN[0]))
    else:
        rMatrix = np.dot(baseFunctions, rGammaiN)
    rMatrixSPD = rMatrix*uValue
    return rMatrixSPD

def g_order(base_function_size, order=0):
    wl = np.linspace(0, base_function_size-1, num=base_function_size, endpoint=True, dtype=float)
    T_order = np.cos(order * np.arccos((2*wl-base_function_size)/(base_function_size)))
    return T_order/np.std(T_order)

def f_order(base_function_size, phase, order=0, single_function=False):
    if order == 0:
        return np.ones((1,base_function_size)).transpose()

    if single_function:
        baseFunctions = np.zeros((1, base_function_size))
    else:
        baseFunctions = np.zeros((order+1, base_function_size))
    if single_function:
        i = order
        if i == 0:
            singleBase = np.ones(base_function_size)
        else:
            singleBase = np.cos(phase[i]) * g_order(base_function_size, order=2 * i - 1) + \
                         np.sin(phase[i]) * g_order(base_function_size, order=2 * i)
        baseFunctions[0, :] = singleBase.transpose()
    else:
        for i in range(order+1):
            if i==0:
                singleBase = np.ones(base_function_size)
            else:
                singleBase = np.cos(phase[i]) * g_order(base_function_size, order = 2*i-1) + \
                             np.sin(phase[i]) * g_order(base_function_size, order=2 * i)
            baseFunctions[i,:] = singleBase.transpose()
    return baseFunctions.transpose()
def generate_chebyshevMC0( number:int, base_function_size:int, uValue:float, org_function = True, single_function = False, use_gamma_phi = True)->ndarray:
    """
    Generate a set of chebyshev based functions according to https://doi.org/10.5194/amt-11-3595-2018.
    equation (9)

    Args:
        :number:
            | number of base functions
        :base_function_size:
            | length of the base functions (the number of wavelength points in the original article)
        :uValue:
            | uncertainty for the base functions (u_c() in equation (8))
        :single_function:
            | use the maximum order only
        : use_gamma:
            | use gamma for normalization
    Returns:
        :returns:
            | ndarray with set of base functions with weighting

    Note:
    """
    rGammai = np.random.normal(size=(number+1))
    QSumSqrt = np.sqrt(np.sum(rGammai**2))
    rGammaiN = rGammai / QSumSqrt
    # Correction for the correlated contribution : Here we do not need the normalization
    # this is different to the original article
    if not org_function:
        rGammaiN[0] = rGammai[0]
    rPhasei = np.random.uniform(low = 0, high = 2*math.pi, size = (number+1))
    if not use_gamma_phi:
        rPhasei[:] = 0.
    baseFunctions = f_order( base_function_size, rPhasei, order=number, single_function=single_function)
    if not use_gamma_phi:
        rGammaiN[:] = 1.
    if single_function:
        rMatrix = np.squeeze(np.dot(baseFunctions, rGammaiN[0]))
    else:
        rMatrix = np.dot(baseFunctions, rGammaiN)
    rMatrixSPD = rMatrix*uValue
    return rMatrixSPD

def generate_base_functions( param:str, base_function_size:int, uValue:float)->ndarray:
    """
    Generate a set of based functions according to
    Fourier: https://doi.org/10.1088/1681-7575/aa7b39 or
    Chebyshev: https://doi.org/10.5194/amt-11-3595-2018.

    Args:
        :param:
            | string to describe the properties of the base functions
            | structure 'f 7 s o'
            | first element: f (Fourier) or c (Chebyshev)
            | second element: number of base functions (N)
            | third element: m (optional) use a combination with N base functions, s use only the N-th base function
            | fourth element: o (optional) use the original normalization for the 0th order function
            | fifth element: ng (optional) use no gamma phi
        :number:
            | number of base functions
        :base_function_size:
            | length of the base functions (the number of wavelength points in the original article)
        :uValue:
            | uncertainty for the base functions (u_c() in equation (8))

    Returns:
        :returns:
            | ndarray with set of base functions with weighting

    Note:
    """
    working = str.split(param.lower())
    # use the original normalization for the 0th order function or not
    if 'o' in param.lower():
        org_function = True
    else:
        org_function = False
    # use only the maximum base function (single=True) or all base functions (single=False)
    if 's' in param.lower():
        single_function = True
    else:
        single_function = False
    if 'ng' in param.lower():
        use_gamma_phi = False
    else:
        use_gamma_phi = True
    noise = None
    match working[0]:
        case 'f':noise = generate_FourierMC0(number=int(working[1]),
                                            base_function_size=base_function_size, uValue=uValue, org_function=org_function, single_function=single_function, use_gamma_phi=use_gamma_phi)
        case 'c':noise = generate_chebyshevMC0(number=int(working[1]),
                                            base_function_size=base_function_size, uValue=uValue, org_function=org_function, single_function=single_function, use_gamma_phi=use_gamma_phi)
        case _: print( 'Base functions from type ' + working[0] + ' are not supported.')
    return noise