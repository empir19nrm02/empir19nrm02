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
import pandas as pd
import traceback
from empir19nrm02.tools import draw_values_gum, sumMC, make_symm, nearcorr, sumMCV
from empir19nrm02.tools import  plot_2D, plotHistScales
from empir19nrm02.MC import  generate_base_functions
import numpy as np
from numpy import ndarray
import luxpy as lx
import pickle


__all__ = ['DistributionParam', 'NameUnit', 'MCVar', 'MCVectorVar', 'MCSimulation', 'noise_list_default', 'pickle_copy', 'StatDataVector']

default_trials:int = 10000

pickle_copy = lambda obj: pickle.loads(pickle.dumps(obj))


@dataclass
class   DistributionParam(object):
    def __init__(self, mean:float = 0, stddev:float=1, distribution:str = 'normal', add_params = None):
        self.mean = mean
        self.stddev = stddev
        self.distribution = distribution
        # additional control for the base function approach with 'f 7 s' (fourier N=7 single base function)
        self.add_params = add_params
    def __str__(self):
        return 'Distribution: Mean:{0:.4f}, StdDev: {1:.4f}, Dist: {2}, Add_Param: {3}'.format(self.mean, self.stddev, self.distribution, self.add_params)
@dataclass
class   NameUnit(object):
    name:list['Name']
    unit:list ['Unit']
    def get_name_unit(self, i:int=0):
        if isinstance(self.name, str):
            return self.name, self.unit
        if isinstance(self.name, list) and i < len(self.name):
            return self.name[i], self.unit[i]
@dataclass
class   SetParams(object):
    trials: int = default_trials
    step:int = 0
    distribution: DistributionParam = DistributionParam()
    file:str = None

class MCVar(object):
    def __init__(self, name:NameUnit = None, distribution: DistributionParam = None, file: str = None):
        # description of the properties
        self.name = name
        self.setParam = SetParams()
        self.setParam.distribution = distribution
        self.setParam.file = file
        # samples
        self.val = None

    def __getitem__(self, item):
        return self.val[item]
    def __setitem__(self, key, value):
        self.val[key] = value
    def allocate(self, trials:int = default_trials, step:int = 0):
        self.setParam.trials = trials
        self.setParam.step = step
        self.val = np.zeros(trials)
    def generate_numbers(self, trials:int = default_trials, step:int = 0, file:str = None):
        self.setParam.trials = trials
        self.setParam.step = step

        if file is None:
            self.val=draw_values_gum(mean=self.setParam.distribution.mean,
                                     stddev=self.setParam.distribution.stddev,
                                     draws=self.setParam.trials,
                                     distribution=self.setParam.distribution.distribution)
            # store the first value as reference
            self.val[0] = self.setParam.distribution.mean
        else:
            # load data from file
            if '.pkl' in file:
                tmpVar = MCVar()
                tmpVar.unpickle(file)
                # do not change the value name and unit here
                self.val = tmpVar.val.copy()
                self.setParam.trials = tmpVar.setParam.trials
                self.setParam.step = tmpVar.setParam.step
                self.setParam.distribution = tmpVar.setParam.distribution
                self.setParam.file = tmpVar.setParam.file

        self.setParam.file = file
    def pickle(self, filename = None):
        if filename is None:
            (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
            filename = text[:text.find('.')].strip()
        if not '.pkl' in filename:
            filename = filename + '.pkl'
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def unpickle(self, filename=None):
        if filename is None:
            (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
            filename = text[:text.find('.')].strip()
        if not 'pkl' in filename:
            filename = filename + '.pkl'
        with open(filename, 'rb') as f:
            dataPickle = f.read()
            f.close()
        self.__dict__ = pickle.loads(dataPickle)

    def print_stat(self, out_all = False):
        [values, interval] = sumMC(self.val)
        (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
        def_name = text[:text.find('.')].strip()
        print('Name:', def_name, self.name, 'Values:', values, (values[1] / values[0]), 'Interval:', interval)
        if out_all:
            print('    trials:', self.setParam.trials, 'step:', self.setParam.step, 'Distribution:', self.setParam.distribution)

# Some tests for McVar
def McVar_test():
    var1 = MCVar(name=NameUnit('var1', 'Unit_var1'), distribution=DistributionParam())
    var1.generate_numbers()
    var1.print_stat()
    var1.pickle()

    var1Load = MCVar(name=NameUnit('var1Load', 'Unit_var1Load'))
    var1Load.unpickle('var1')
    var1Load.print_stat(out_all=True)

    var2 = MCVar(name=NameUnit('var2', 'Unit_var2'),
                 distribution=DistributionParam(mean=1, stddev=2, distribution='triangle', add_params='f 4'))
    var2.generate_numbers()
    var2.print_stat(out_all=True)
    var2.pickle()

    var2Load = MCVar()
    var2Load.unpickle('var2')
    var2Load.print_stat(out_all=True)

    var2Load.generate_numbers(file='var1')
    var2Load.print_stat(out_all=True)

noise_list_default = {'nc_add': DistributionParam(),
                      'f_add': DistributionParam(),
                      'nc_mul': DistributionParam(),
                      'f_mul': DistributionParam(),
                    }
@dataclass
class   StatDataVector(object):
    v_mean:ndarray = None
    v_std:ndarray = None
    corr_matrix:ndarray = None
    cov_matrix:ndarray = None

class MCVectorVar(MCVar):
    def __init__(self,name: NameUnit = None, elements:int = 2, noise_list:dict = None):
        super().__init__(name = name)
        self.elements = elements
        self.setData = StatDataVector()
        self.runData = StatDataVector()
        if noise_list is None:
            self.noise_list = dict()
        else:
            self.noise_list = noise_list
    def set_vector_param(self, v_mean, v_std = None, corr = None, cov = None):
        if v_mean.shape[0] != self.elements:
            raise TypeError("v_mean should have the rigth shape", self.elements)
        self.setData.v_mean = v_mean.copy()
        if not cov is None:
            if cov.shape[0] != self.elements | cov.shape[1] != self.elements:
                raise ValueError("Shape of Mean and Cov do not fit", self.elements, cov.shape)
            self.setData.cov_matrix = cov.copy()
            self.calc_corr_matrix_set()
        else:
            if not v_std is None:
                if corr is None:
                    self.setData.corr_matrix = np.eye(self.elements, dtype=float)
                else:
                    self.setData.corr_matrix = corr.copy()
                self.setData.v_std = v_std.copy()
                self.calc_cov_matrix_set()
            else:
                self.setData.v_std = np.zeros_like(self.setData.v_mean)
                self.setData.cov_matrix = np.zeros((self.elements, self.elements))
                self.setData.corr_matrix = np.zeros((self.elements, self.elements))
        self.setParam.distribution = DistributionParam(mean = np.mean(self.setData.v_mean), stddev=np.mean(self.setData.v_std), distribution='normal')

    def allocate(self, trials:int = default_trials, step:int = 0):
        self.setParam.trials = trials
        self.val = np.zeros((trials, self.elements))

    def generate_numbers(self, trials:int = default_trials, step:int = 0, file:str = None):
        self.setParam.trials = trials
        self.setParam.step = step

        if file is None:
            if not self.noise_list:
                try:
                    self.val = np.random.default_rng().multivariate_normal(self.setData.v_mean, self.setData.cov_matrix, self.setParam.trials)
                except np.linalg.LinAlgError:
                    print('SVD LinAlgError:  improvement of cov. Matrix required.')
                    self.improve_covariance(mode='eigenvalues', ratio=0.001)
                    # try again
                    self.val = np.random.default_rng().multivariate_normal(self.setData.v_mean, self.setData.cov_matrix, self.setParam.trials)
            else:
                self.val = np.zeros((self.setParam.trials, self.elements))
                for i in range(1, self.setParam.trials):
                    self.val[i] = self.setData.v_mean.copy()
                    for noise, params in self.noise_list.items():
                        match noise:
                            case 'nc_add':
                                self.val[i] = self.add_noise_nc_add(self.val[i], params).copy()
                            case 'f_add':
                                noise = self.add_base_function_noise_add(self.val[i], params)
                                self.val[i] = noise.copy()
                            case 'nc_mul':
                                self.val[i] = self.add_noise_nc_mul(self.val[i], params).copy()
                            case 'f_mul':
                                self.val[i] = self.add_base_function_noise_mul(self.val[i], params).copy()
                            case _: print( noise, ' Not implemented')
            # store the first value as reference
            self.calc_cov_matrix_from_data()
            # using the mean value from the generated data
            # v_std is using the std. deviation from the generated data too
            self.val[0] = self.runData.v_mean
        else:
            # load data from file
            if '.pkl' in file:
                tmpVar = MCVectorVar()
                tmpVar.unpickle(file)
                # do not change the value name and unit here
                self.setParam = tmpVar.setParam
                self.elements = tmpVar.elements
                self.setData = tmpVar.setData
                self.runData = tmpVar.runData
                self.noise_list = tmpVar.noise_list
                self.val = tmpVar.val
            if '.xls' in file:
                # ToDo: Read from XLS
                tmpVar = MCVectorVar()

    def add_noise_nc_add(self, tmpData:ndarray, params:DistributionParam)->ndarray:
        noise = draw_values_gum(mean=params.mean, stddev=params.stddev, draws=self.elements, distribution=params.distribution)
        return tmpData + noise

    def add_base_function_noise_add(self, tmpData:ndarray, params:DistributionParam)->ndarray:
        noise = generate_base_functions( params.add_params, self.elements, params.stddev)
        return noise + tmpData

    def add_noise_nc_mul(self, tmpData:ndarray, params:DistributionParam)->ndarray:
        noise = draw_values_gum(mean=params.mean, stddev=params.stddev, draws=self.elements, distribution=params.distribution)
        return tmpData * (1. + noise)

    def add_base_function_noise_mul(self, tmpData:ndarray, params:DistributionParam)->ndarray:
        noise = (1+generate_base_functions( params.add_params, self.elements, params.stddev))
        return noise * tmpData

    def print_stat(self):
         # with help from https://schurpf.com/python-save-a-class/
         (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
         def_name = text[:text.find('.')].strip()
         [values, interval] = sumMCV(self.val)
         print ('Name:', def_name, self.name, 'Values:', values, (values[1]/values[0]), 'Interval:', interval)
         print ('Set: Corr:', self.setData.corr_matrix, 'Cov:', self.setData.cov_matrix)
         print ('Data: Corr:', self.runData.corr_matrix, 'Cov:', self.runData.cov_matrix)

    # calc a correlation matrix from covariance matrix
    # We need the covariance matrix first!
    # THX: https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b

    @staticmethod
    def calc_corr_matrix_static(cov_matrix:ndarray):
        v_std = np.sqrt(np.diag(cov_matrix))
        outer_v = np.outer(v_std, v_std)
        corr_matrix = np.divide(cov_matrix, outer_v, out=np.zeros_like(cov_matrix), where=outer_v!=0)
        corr_matrix[cov_matrix == 0] = 0
        return v_std, corr_matrix

    def calc_corr_matrix_set(self):
        self.setData.v_std, self.setData.corr_matrix = MCVectorVar.calc_corr_matrix_static(self.setData.cov_matrix)
        return self.setData.corr_matrix
    def calc_corr_matrix_data(self):
        self.runData.v_std, self.runData.corr_matrix = MCVectorVar.calc_corr_matrix_static(self.runData.cov_matrix)
        return self.runData.corr_matrix

    # calc a covariance matrix from correlation matrix and standard deviation vector
    # We need the correlation matrix and standard deviation vector first!
    def calc_cov_matrix_set(self):
        D = np.diag(self.setData.v_std)
        self.setData.cov_matrix = np.matmul(np.matmul(D, self.setData.corr_matrix), D)
        self.calc_corr_matrix_set()
        return self.setData.cov_matrix
    def calc_cov_matrix_data(self):
        D = np.diag(self.runData.v_std)
        self.runData.cov_matrix = np.matmul(np.matmul(D, self.runData.corr_matrix), D)
        self.calc_corr_matrix_data()
        return self.runData.cov_matrix

    # calc a covariance matrix from current data and update corr_matrix too
    def calc_cov_matrix_from_data(self):
        self.runData.v_mean = np.mean(self.val, axis=0)
        self.runData.cov_matrix = np.cov(self.val.T)
        self.calc_corr_matrix_data()
        return self.runData.cov_matrix

    def interpolate(self, wl_new, wl_current, kind='S', trials:int = None):
        if wl_current.shape[0] != self.elements:
            raise ValueError( f'Elements {self.elements} and shape of wl_current {wl_current.shape[0]} do not match')
        if self.val is None or self.val.shape[0] != self.setParam.trials:
            if trials:
                self.generate_numbers(trials=trials)
            else:
                self.generate_numbers(self.setParam.trials)
        valPlus = np.vstack((wl_current, self.val))
        self.val = lx.cie_interp(valPlus, wl_new, kind=kind)[1:]
        self.elements = wl_new.shape[0]
        self.calc_cov_matrix_from_data()
        self.setData.v_mean = self.runData.v_mean
        self.setData.v_std = self.runData.v_std
        self.setData.cov_matrix = self.runData.cov_matrix
        self.setData.corr_matrix = self.runData.corr_matrix

    # improve the covariance matrix for further calculations or transfers
    # ToDo: Test, did not work at the moment
    def improve_covariance(self, mode='nearcorr', ratio=0.01):
        # oDo
        # Some information about:
        # https://github.com/GGiecold-zz/pyRMT
        # https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
        # https://nbviewer.org/github/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb

        if mode == 'nearcorr':
            # or for the coorlation matrix only:
            X=make_symm(self.setData.corr_matrix)
            X_n=nearcorr(X, tol=[1e-4])
            self.setData.corr_matrix = X_n.copy()
            self.calc_cov_matrix_set()
            print("MaxDiff nearcorr:", np.max(np.abs(X-X_n)))
        if mode == 'eigenvalues':
            # THX: https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
            # GUM S2, 3.20/3.21
            X=make_symm(self.setData.corr_matrix)
            eigenValues, eigenVectors = np.linalg.eig(X)
            idx = eigenValues.argsort()[::-1]
            eigenValues = eigenValues[idx]
            eigenVectors = eigenVectors[:,idx]
            k = np.where(eigenValues < ratio*eigenValues[0])
            eigenValues[k[0][0]:] = eigenValues[k[0][0]]
            X_n=eigenVectors @ np.diag(eigenValues) @ np.linalg.inv(eigenVectors)
            # to avoid complex matrices, the img. part is usually quite small
            self.setData.corr_matrix = X_n.real.copy()
            self.calc_cov_matrix_set()
            print("MaxDiff eigenvalues:", np.max(np.abs(X-X_n)))

def MCVector_test():
    var1V = MCVectorVar(name=NameUnit('var1', 'Unit_var1'))

    v_mean = np.zeros((2))
    v_std = np.ones((2)) * 2
    v_std[0] = 0.5

    var1V.set_vector_param(v_mean, v_std)
    var1V.generate_numbers()
    var1V.print_stat()
    var1V.pickle()

    var2V = MCVectorVar(name=NameUnit('var2', 'Unit_var2'))
    var2V.generate_numbers(file='var1V')
    var2V.print_stat()
    plot_2D(var2V, number=1000)

    var3V = var1V
    v_mean[0] = 2
    v_mean[1] = 3
    corr = np.eye(2, dtype=float)
    corr[0, 1] = -0.7
    corr[1, 0] = -0.7
    var3V.set_vector_param(v_mean=v_mean, v_std=v_std, corr=corr)
    var3V.generate_numbers()
    var3V.print_stat()
    plot_2D(var3V, number=1000)


#class MCSpectrumVar(MCVar):
#    """
#    Spectrum class for MC simulations
#    An object holds a spectrum (class luxpy.SPD) with only one wavelength scale and one value array
#
#    *** obsolate ****
#
#    With different
#
#    Example:
#
#    Default Values:
#    """
#
#    def __init__(self, name:NameUnit = None, spd=None, noise_list:dict = None):
#        super().__init__(name = name)
#        if spd is None:
#            self.spd = lx.SPD(spd=lx.cie_interp(lx._CIE_ILLUMINANTS['A'], lx.getwlr(), kind='S'), wl=lx.getwlr(),
#                              negative_values_allowed=True)
#        else:
#            newSPD = np.vstack((spd.wl, spd.value))
#            self.spd = lx.SPD(spd=newSPD, negative_values_allowed=True)
#        # to remember the number of elements
#        self.wlElements = len(self.spd.wl)
#
#
#    # interpolate all data in self.values to the nominal wavelength scale
#    # Attention: No update of statistical information here
#    def normalize_wl_scale(self):
#        for i in range(1, self.trials):
#            self.val[i].cie_interp(self.val[0].wl, kind = 'linear', negative_values_allowed=True)
#        return
#
#    def generate_numbers(self, trials:int = default_trials, step:int = 0, file:str = None):
#        #print('generate_numbers:', self.name)
#        super().generate_numbers(trials, step, file)
#        self.val = np.empty(self.trials, dtype=object)
#        # This handles the 0 index object automatically
#        newSPD = np.vstack((self.spd.wl, self.spd.value))
#        for i in range(self.trials):
#            self.val[i] = lx.SPD(spd=newSPD, negative_values_allowed=True)
#        if file is None:
#            # start from 1 to hold the first item as reference
#            for i in range(1, self.trials):
#                for noise, params in self.noise_list.items():
#                    match noise:
#                        case 'wl_nc':
#                            #print( 'wl_nc', noise)
#                            self.add_wl_noise_nc( self.val[i], params)
#                        case 'wl_c':
#                            #print( 'wl_c', noise)
#                            self.add_wl_noise_c( self.val[i], params)
#                        case 'wl_f':
#                            #print( 'wl_f', noise)
#                            self.add_wl_fourier_noise( self.val[i], params)
#                        case 'v_nc':
#                            #print( 'v_nc', noise)
#                            self.add_value_noise_nc( self.val[i], params)
#                        case 'v_c':
#                            #print( 'v_c', noise)
#                            self.add_value_noise_c( self.val[i], params)
#                        case 'v_f':
#                            #print( 'v_f', noise)
#                            self.add_value_fourier_noise( self.val[i], params)
#                        case 'all':
#                            if 'wl_nc' in self.noise_list:
#                                #print( 'wl_nc->all', noise)
#                                self.add_wl_noise_nc( self.val[i], self.noise_list['wl_nc'])
#                            if 'wl_c' in self.noise_list:
#                                #print( 'wl_c->all', noise)
#                                self.add_wl_noise_c( self.val[i], self.noise_list['wl_c'])
#                            if 'wl_f' in self.noise_list:
#                                #print( 'wl_f->all', noise)
#                                self.add_wl_fourier_noise( self.val[i], self.noise_list['wl_f'])
#                            if 'v_nc' in self.noise_list:
#                                #print( 'v_nc->all', noise)
#                                self.add_value_noise_nc( self.val[i], self.noise_list['v_nc'])
#                            if 'v_c' in self.noise_list:
#                                #print( 'v_c->all', noise)
#                                self.add_value_noise_c( self.val[i], self.noise_list['v_c'])
#                            if 'v_f' in self.noise_list:
#                                #print( 'v_f->all', noise)
#                                self.add_value_fourier_noise( self.val[i], self.noise_list['v_f'])
#        else:
#            #Load data from file
#            print('Not yet implemented')
#        self.normalize_wl_scale()
#
#    def add_wl_noise_nc(self, spd_tmp:SPD, params:DistributionParam)->None:
#        spd_tmp.wl = spd_tmp.wl + draw_values_gum(mean=params.mean, stddev=params.stddev, draws=self.wlElements, distribution=params.distribution)
#
#    def add_wl_noise_c(self, spd_tmp:SPD, params:DistributionParam)->None:
#        spd_tmp.wl = draw_values_gum(mean=params.mean, stddev=params.stddev, draws=1, distribution=params.distribution)[0] + spd_tmp.wl
#
#    def add_wl_fourier_noise(self, spd_tmp:SPD, params:DistributionParam)->None:
#        spd_tmp.wl = generate_FourierMC0( params.add_params, self.spd.wl, params.stddev) + spd_tmp.wl
#
#    def add_value_noise_nc(self, spd_tmp:SPD, params:DistributionParam)->None:
#        spd_tmp.value = spd_tmp.value + draw_values_gum(mean=params.mean, stddev=params.stddev, draws=self.wlElements, distribution=params.distribution)
#
#    def add_value_noise_c(self, spd_tmp:SPD, params:DistributionParam)->None:
#        spd_tmp.value = draw_values_gum(mean=params.mean, stddev=params.stddev, draws=1, distribution=params.distribution)[0] + spd_tmp.value
#
#    def add_value_fourier_noise(self, spd_tmp:SPD, params:DistributionParam)->None:
#        #print('add_value_fourier_noise:', params.add_params, params.stddev, np.min(self.spd.wl), np.max(self.spd.wl),np.min(spd_tmp.value), np.max(spd_tmp.value) )
#        spd_tmp.value = (1+generate_FourierMC0( params.add_params, self.spd.wl, params.stddev)) * spd_tmp.value
#        #print('add_value_fourier_noise2:', np.min(spd_tmp.wl), np.max(spd_tmp.wl),np.min(spd_tmp.value), np.max(spd_tmp.value) )
#
#%%
class MCSimulation(object):
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
        print( 'Generate:')
        for var in self.input_var:
            print( var)
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
                res = model(*x)
                for j, var_out in enumerate(self.output_var[k]):
                    var_out[i] = res[j]

    def get_result_db(self):
        res_data = None
        line_data = {
            'Input':'',
            'Unit': '',
            'Mean':0.,
            'StdDev': 0,
            'Distr': 0,
            }
        for i in range(self.in_elements):
            if i < len(self.input_var):
                if self.input_var[i].name:
                    line_data['Input'] = self.input_var[i].name.get_name_unit()[0]
                    line_data['Unit'] = self.input_var[i].name.get_name_unit()[1]
                else:
                    line_data['Input'] = 'Input#' + str(i)
                    line_data['Unit'] = 'Unit#' + str(i)
                if isinstance(self.input_var[i], MCVectorVar) and self.input_var[i].noise_list:
                    for key in self.input_var[i].noise_list:
                        line_data['Mean'] = self.input_var[i].noise_list[key].mean
                        line_data['StdDev'] = self.input_var[i].noise_list[key].stddev
                        line_data['Distr'] = self.input_var[i].noise_list[key].distribution + ' ' + self.input_var[i].noise_list[key].add_params
                else:
                    line_data['Mean'] = self.input_var[i].setParam.distribution.mean
                    line_data['StdDev'] = self.input_var[i].setParam.distribution.stddev
                    line_data['Distr'] = self.input_var[i].setParam.distribution.distribution
            else:
                line_data['Input'] = 'All'
                line_data['Unit'] = ''
                line_data['Mean'] = ''
                line_data['StdDev'] = ''
                line_data['Distr'] = ''

            for k in range( len( self.output_var[0])):
                var = self.output_var[i][k]
                if isinstance(var, MCVectorVar):
                    if var.elements < 5:
                        var.val[:,0]=var.val[:,0]/var.val[0,0]
                        [values, interval] = sumMCV(var.val)
                        for l in range(var.elements):
                            line_data[var.name.get_name_unit(l)[0]] = values[0][l]
                            line_data['u('+var.name.get_name_unit(l)[0]+')'] = values[1][l]
                else:
                    [values, interval] = sumMC(var.val)
                    line_data[var.name.get_name_unit()[0]] = values[0]
                    line_data['u(' + var.name.get_name_unit()[0] + ')'] = values[1]
            if res_data is None:
                res_data = pd.DataFrame( [line_data])
            else:
                res_data = pd.concat( [res_data, pd.DataFrame( [line_data])])
        pd.options.display.float_format = '{:,.6f}'.format
        return res_data

def McSim_main():
    #McVar_test()
    #MCVector_test()
    mc1 = MCVar(name=NameUnit('V1', 'U1'), distribution=DistributionParam(mean=1.,stddev=2., distribution='normal'))
    mc2 = MCVar(name=NameUnit('V2', 'U2'), distribution=DistributionParam(mean=2.,stddev=4., distribution='normal'))

    # define some output data (as MCVar to make an analysis after the run)
    mcOut = MCVar(name=NameUnit('Plus','UnitPlus'))
    def model_easy( a, b):
        return a+b

    input = [mc1, mc2]
    output = [mcOut]

    sim = MCSimulation(trials=100)
    # set the input and output data
    sim.set_input_var(input)
    sim.set_output_var(output)
    # generate/load the random numbers
    sim.generate()
    # calculate the model
    sim.calculate_model(model_easy)

    res_data = sim.get_result_db()
    print(res_data)


if __name__ == '__main__':
    print( 'MCSim Test Start')
    McSim_main()
    print( 'MCSim Test End')
