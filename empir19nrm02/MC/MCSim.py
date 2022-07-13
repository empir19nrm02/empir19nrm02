########################################################################
# <MCSim: a Python module for MC simulations.>
# Copyright (C) <2022>  <Udo Krueger> (udo.krueger at technoteam.de)
#########################################################################

"""
Module for class functionality for MC Simulations
===========================================================

.. codeauthor:: UK
"""

from dataclasses import dataclass
import copy
from luxpy import SPD
from scipy.stats import stats
import math
import traceback
from abc import ABC
from empir19nrm02.tools import draw_values_gum, sumMC, make_symm, nearcorr, sumMCV
import numpy as np
import luxpy as lx

__all__ = ['DistributionParam','NameUnit','McVar','McInputVar','McOutputVar', 'McSpectrumVar','McSim']

default_trials:int = 10000

@dataclass
class   DistributionParam(object):
    def __init__(self, mean:float = 0, stddev:float=1, distribution:str = 'normal', add_params = None):
        self.mean = mean
        self.stddev = stddev
        self.distribution = distribution
        self.add_params = add_params

@dataclass
class   NameUnit(object):
    name: str = 'Name'
    unit: str = 'Unit'

class McVar(ABC):
    def __init__(self, name:NameUnit = None):
        self.name:NameUnit = name
        self.trials:int = 0
        self.step:int = 0
        self.file:str = None
        self.val = None
    def __getitem__(self, item):
        return self.val[item]
    def __setitem__(self, key, value):
        self.val[key] = value
    def generate_numbers(self, trials:int = default_trials, step:int = 0, file:str = None):
        self.trials = trials
        self.step = step
        self.file = file

    def print_stat(self):
        # with help from https://schurpf.com/python-save-a-class/
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
        def_name = text[:text.find('=')].strip()
        [values, interval] = sumMC(self.val)
        print ('Name:', def_name, self.name, 'Values:', values, 'Interval:', interval)

class McInputVar(McVar):
    def __init__(self,  name:NameUnit = None, distribution: DistributionParam = DistributionParam()):
        super().__init__(name = name)
        self.distribution = distribution

    def generate_numbers(self, trials:int = default_trials, step:int = 0, file:str = None):
        super().generate_numbers( trials, step, file)
        if file is None:
            self.val=draw_values_gum(mean=self.distribution.mean, stddev=self.distribution.stddev, draws=trials, distribution=self.distribution.distribution)
            # store the first value as reference
            self.val[0] = self.distribution.mean
        else:
            #Load data from file
            print('Not yet implemented')

class McOutputVar(McVar):
    def __init__(self, name:NameUnit = None):
        super().__init__(name = name)

    def allocate(self, trials:int = default_trials, step:int = 0):
        self.val = np.zeros(trials)

class MCVectorVar(McVar):
    def __init__(self,name: NameUnit = None, elements:int = 2):
        super().__init__(name = name)
        self.elements:int = elements
        self.cov_matrix = None
        self.corr_matrix = None
        self.v_std = None
        self.v_mean = None

    def set_vector_param(self, v_mean, v_std = None, corr = None, cov = None):
        if v_mean.shape[0] != self.elements:
            raise TypeError("v_mean should have the rigth shape", self.elements)
        self.v_mean = v_mean.copy()
        if v_std is None:
            if cov is None:
                raise TypeError("v_std and cov are None")
            else:
                if cov.shape[0] != self.elements | cov.shape[1] != self.elements:
                    raise ValueError("Shape of Mean and Cov do not fit", self.elements, cov.shape)
                self.cov_matrix = cov.copy()
                self.calc_corr_matrix()
        else:
            if corr is None:
                self.corr_matrix = np.eye(self.elements, dtype=float)
            else:
                self.corr_matrix = corr.copy()
            self.v_std = v_std.copy()
            self.calc_cov_matrix()

    def allocate(self, trials:int = default_trials, step:int = 0):
        super().generate_numbers(trials, step)
        self.val = np.zeros((trials, self.elements))

    def generate_numbers(self, trials:int = default_trials, step:int = 0, file:str = None):
        super().generate_numbers(trials, step, file)
        if file is None:
            self.val = np.random.default_rng().multivariate_normal(self.v_mean, self.cov_matrix, self.trials)
            # store the first value as reference
            self.val[0] = self.v_mean
        else:
            #Load data from file
            print('Not yet implemented')

    def print_stat(self):
         # with help from https://schurpf.com/python-save-a-class/
         (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
         def_name = text[:text.find('=')].strip()
         [values, interval] = sumMCV(self.val)
         print ('Name:', def_name, self.name, 'Values:', values, 'Interval:', interval)

    # calc a correlation matrix from covariance matrix
    # We need the covariance matrix first!
    # THX: https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    def calc_corr_matrix(self):
        self.v_std = np.sqrt(np.diag(self.cov_matrix))
        outer_v = np.outer(self.v_std, self.v_std)
        self.corr_matrix = self.cov_matrix / outer_v
        self.corr_matrix[self.cov_matrix == 0] = 0
        return self.corr_matrix

    # calc a covariance matrix from correlation matrix and standard deviation vector
    # We need the correlation matrix and standard deviation vector first!
    def calc_cov_matrix(self):
        D = np.diag(self.v_std)
        self.cov_matrix = np.matmul(np.matmul(D, self.corr_matrix),D)
        return self.cov_matrix

    # improve the covariance matrix for further calculations or transfers
    def improve_covariance(self, mode='nearcorr', ratio=0.01):
        # oDo
        # Some information about:
        # https://github.com/GGiecold-zz/pyRMT
        # https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
        # https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb

        if mode == 'nearcorr':
            # or for the coorlation matrix only:
            X=make_symm(self.corr_matrix)
            X_n=nearcorr(X)
            self.corr_matrix = X_n.copy()
            self.calc_cov_matrix()
            print("MaxDiff nearcorr:", np.max(np.abs(X-X_n)))
        if mode == 'eigenvalues':
            # THX: https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
            # GUM S2, 3.20/3.21
            X=make_symm(self.corr_matrix)
            eigenValues, eigenVectors = np.linalg.eig(X)
            idx = eigenValues.argsort()[::-1]
            eigenValues = eigenValues[idx]
            eigenVectors = eigenVectors[:,idx]
            k = np.where(eigenValues < ratio*eigenValues[0])
            eigenValues[k[0][0]:] = eigenValues[k[0][0]]
            X_n=eigenVectors @ np.diag(eigenValues) @ np.linalg.inv(eigenVectors)
            self.corr_matrix = X_n.copy()
            self.calc_cov_matrix()
            print("MaxDiff eigenvalues:", np.max(np.abs(X-X_n)))

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

noise_list_default = {'wl_nc': DistributionParam(),
                      'wl_c': DistributionParam(),
                      'wl_f': DistributionParam(),
                      'v_nc': DistributionParam(),
                      'v_c': DistributionParam(),
                      'v_f': DistributionParam(),
                      'all': None,}

class McSpectrumVar(McVar):
    """
    Spectrum class for MC simulations
    An object holds a spectrum (class luxpy.SPD) with only one wavelength scale and one value array

    With different

    Example:

    Default Values:
    """

    def __init__(self, name:NameUnit = None, spd=None, noise_list:dict = None):
        super().__init__(name = name)
        if spd is None:
            self.spd = lx.SPD(spd=lx.cie_interp(lx._CIE_ILLUMINANTS['A'], lx.getwlr(), kind='S'), wl=lx.getwlr(),
                              negative_values_allowed=True)
        else:
            newSPD = np.vstack((spd.wl, spd.value))
            self.spd = lx.SPD(spd=newSPD, negative_values_allowed=True)
        # to remember the number of elements
        self.wlElements = len(self.spd.wl)
        if noise_list is None:
            self.noise_list = dict()
        else:
            self.noise_list = noise_list

    # interpolate all data in self.values to the nominal wavelength scale
    # Attention: No update of statistical information here
    def normalize_wl_scale(self):
        for i in range(1, self.trials):
            self.val[i].cie_interp(self.val[0].wl, kind = 'linear', negative_values_allowed=True)
        return

    def generate_numbers(self, trials:int = default_trials, step:int = 0, file:str = None):
        super().generate_numbers(trials, step, file)
        self.val = np.empty(self.trials, dtype=object)
        # This handles the 0 index object automatically
        newSPD = np.vstack((self.spd.wl, self.spd.value))
        for i in range(self.trials):
            self.val[i] = lx.SPD(spd=newSPD, negative_values_allowed=True)
        if file is None:
            # start from 1 to hold the first item as reference
            for i in range(1, self.trials):
                for noise, params in self.noise_list.items():
                    match noise:
                        case 'wl_nc':
                            #print( 'wl_nc', noise)
                            self.add_wl_noise_nc( self.val[i], params)
                        case 'wl_c':
                            #print( 'wl_c', noise)
                            self.add_wl_noise_c( self.val[i], params)
                        case 'wl_f':
                            #print( 'wl_f', noise)
                            self.add_wl_fourier_noise( self.val[i], params)
                        case 'v_nc':
                            #print( 'v_nc', noise)
                            self.add_value_noise_nc( self.val[i], params)
                        case 'v_c':
                            #print( 'v_c', noise)
                            self.add_value_noise_c( self.val[i], params)
                        case 'v_f':
                            #print( 'v_f', noise)
                            self.add_value_fourier_noise( self.val[i], params)
                        case 'all':
                            if 'wl_nc' in self.noise_list:
                                #print( 'wl_nc->all', noise)
                                self.add_wl_noise_nc( self.val[i], self.noise_list['wl_nc'])
                            if 'wl_c' in self.noise_list:
                                #print( 'wl_c->all', noise)
                                self.add_wl_noise_c( self.val[i], self.noise_list['wl_c'])
                            if 'wl_f' in self.noise_list:
                                #print( 'wl_f->all', noise)
                                self.add_wl_fourier_noise( self.val[i], self.noise_list['wl_f'])
                            if 'v_nc' in self.noise_list:
                                #print( 'v_nc->all', noise)
                                self.add_value_noise_nc( self.val[i], self.noise_list['v_nc'])
                            if 'v_c' in self.noise_list:
                                #print( 'v_c->all', noise)
                                self.add_value_noise_c( self.val[i], self.noise_list['v_c'])
                            if 'v_f' in self.noise_list:
                                #print( 'v_f->all', noise)
                                self.add_value_fourier_noise( self.val[i], self.noise_list['v_f'])
        else:
            #Load data from file
            print('Not yet implemented')
        self.normalize_wl_scale()

    def add_wl_noise_nc(self, spd_tmp:SPD, params:DistributionParam)->None:
        spd_tmp.wl = spd_tmp.wl + draw_values_gum(mean=params.mean, stddev=params.stddev, draws=self.wlElements, distribution=params.distribution)

    def add_wl_noise_c(self, spd_tmp:SPD, params:DistributionParam)->None:
        spd_tmp.wl = draw_values_gum(mean=params.mean, stddev=params.stddev, draws=1, distribution=params.distribution)[0] + spd_tmp.wl

    def add_wl_fourier_noise(self, spd_tmp:SPD, params:DistributionParam)->None:
        spd_tmp.wl = generate_FourierMC0( params.add_params, self.spd.wl, params.stddev) + spd_tmp.wl

    def add_value_noise_nc(self, spd_tmp:SPD, params:DistributionParam)->None:
        spd_tmp.value = spd_tmp.value + draw_values_gum(mean=params.mean, stddev=params.stddev, draws=self.wlElements, distribution=params.distribution)

    def add_value_noise_c(self, spd_tmp:SPD, params:DistributionParam)->None:
        spd_tmp.value = draw_values_gum(mean=params.mean, stddev=params.stddev, draws=1, distribution=params.distribution)[0] + spd_tmp.value

    def add_value_fourier_noise(self, spd_tmp:SPD, params:DistributionParam)->None:
        #print('add_value_fourier_noise:', params.add_params, params.stddev, np.min(self.spd.wl), np.max(self.spd.wl),np.min(spd_tmp.value), np.max(spd_tmp.value) )
        spd_tmp.value = (1+generate_FourierMC0( params.add_params, self.spd.wl, params.stddev)) * spd_tmp.value
        #print('add_value_fourier_noise2:', np.min(spd_tmp.wl), np.max(spd_tmp.wl),np.min(spd_tmp.value), np.max(spd_tmp.value) )

#%%
class McSim(object):
    def __init__(self, trials = default_trials):
        self.trials = trials
        self.input_var = None
        self.in_elements = 0
        self.output_var = None

    def set_input_var(self, input_var: list):
        # Create the array for the OAT approach
        self.in_elements = len(input_var) + 1
        self.input_var = input_var

    def set_output_var(self, output_var:list):
        self.output_var = np.empty(self.in_elements, dtype=object)
        for i in range(self.in_elements):
            self.output_var[i] = copy.deepcopy(output_var)

    def generate(self):
        for var in self.input_var:
            var.generate_numbers(self.trials)
        for i in range(0,self.in_elements):
            for var in self.output_var[i]:
                var.allocate(self.trials)

    def calculate_model(self, model):
        x0 =[var[0] for var in self.input_var]
        for i in range(self.trials):
            for k in range(self.in_elements):
                if i == 0:
                    x = x0.copy()
                else:
                    if k == self.in_elements - 1:
                        x = [var[i] for var in self.input_var]
                    else:
                        x = x0.copy()
                        x[k] = self.input_var[k][i]
                #print ( 'x before model (i,k):', i, k, x)
                res = model(*x)
                for j, var in enumerate(self.output_var[k]):
                    var[i] = res[j]
