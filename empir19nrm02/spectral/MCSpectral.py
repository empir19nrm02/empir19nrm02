import pickle
import traceback

import numpy as np
from scipy.stats import stats
import seaborn as sns
import pandas as pd

import luxpy as lx

# Ich mache es trotzdem :-(, Danke für den Hinweis
# https://stackoverflow.com/questions/15454285/numpy-array-of-class-instances/15455053

# mod. version of drawValues() --> draw_values_gum()
from empir19nrm02.f1prime import py_f1PrimeG
from empir19nrm02.tools import *
import math
import pandas as pd

__all__ = ['McSpectrumX', 'MCSpectrumResults','MCSpectrumSamples',"generate_FourierMC0"]

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

    def add_value_noise(self, noise):
        self.spd.value = self.spd.value + noise
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

# oDo: Add methods for flexible use of w@ and v@ data
_RESULT_TYPE = ['f1p','f1pE','f1pL','f1pMin','f1pp','f1pBW','Y','Yrel','x','y','CCT', 'v@450', 'w@450', 'v@550', 'w@550', 'v@650', 'w@650']
_RESULT_Name = ['f1p','f1pE','f1pL','f1pMin','f1pp','f1pBW','Y','Yrel', 'x','y', 'CCT', 'v@450', 'w@450', 'v@550', 'w@550', 'v@650', 'w@650']
_RESULT_Unit = ['1','1','1','1','1','1','cd/m²','1', '1','1', 'K', '1', 'nm', '1', 'nm', '1', 'nm', '1', 'nm' ]

# calculate some color data from the samples and make some statistics
class MCSpectrumResults(object):
    def __init__(self):
        self.trials = 0
        self.result = {'types': _RESULT_TYPE}
        for i, result_type in enumerate(_RESULT_TYPE): # store all in single nested dict
            self.result[result_type]  = {'data':  []}
            self.result[result_type] ['Name']= _RESULT_Name[i]
            self.result[result_type] ['Unit']= _RESULT_Unit[i]
            self.result[result_type] ['Mean'] = 0
            self.result[result_type] ['Std']= 0
            self.result[result_type] ['p95Min'] = 0
            self.result[result_type] ['p95Max'] = 0

    def calculate(self, samples, to_calculate):
        self.trials=samples.trials_current
        # Array containing the integral results (here Yxy and CCT for a more-dimensional  output)
        if any(x in to_calculate for x in ['Y', 'Yrel', 'x', 'y', 'CCT']):
            tmp_result = np.zeros( (4, self.trials))
            for i in range(0,self.trials):
                tmp_result[0:3,i] = lx.xyz_to_Yxy(samples.values[i].spd.to_xyz(relative=False).value)
                tmp_result[3,i] = lx.xyz_to_cct(samples.values[i].spd.to_xyz(relative=False).value)
            if 'Y' in to_calculate:
                self.result['Y']['data']=tmp_result[0,:]
            if 'Yrel' in to_calculate:
                self.result['Yrel']['data']=tmp_result[0,:]/tmp_result[0,0]
            if 'x' in to_calculate:
                self.result['x']['data']=tmp_result[1,:]
            if 'y' in to_calculate:
                self.result['y']['data']=tmp_result[2,:]
            if 'CCT' in to_calculate:
                self.result['CCT']['data']=tmp_result[3,:]
        if any(x in to_calculate for x in ['f1p', 'f1pE', 'f1pL', 'f1pMin', 'f1pp', 'f1pBW']):
            tmp_resultf1p = np.zeros( (6, self.trials))
            for i in range(0,self.trials):
                if 'f1p' in to_calculate:
                    [tmp_resultf1p[0, i], _] = py_f1PrimeG(samples.values[i].spd.wl, samples.values[i].spd.value, strWeighting='A')
                if 'f1pE' in to_calculate:
                    [tmp_resultf1p[1, i], _] = py_f1PrimeG(samples.values[i].spd.wl, samples.values[i].spd.value, strWeighting='E')
                if 'f1pL' in to_calculate:
                    [tmp_resultf1p[2, i], _] = py_f1PrimeG(samples.values[i].spd.wl, samples.values[i].spd.value, strWeighting='LED_B3')
                if 'f1pMin' in to_calculate:
                    [tmp_resultf1p[3, i], _] = py_f1PrimeG(samples.values[i].spd.wl, samples.values[i].spd.value, strWeighting='A', iMin=True)
                if 'f1pp' in to_calculate:
                    [tmp_resultf1p[4, i], _] = py_f1PrimeG(samples.values[i].spd.wl, samples.values[i].spd.value, strWeighting='E', dCutOff=-0.003)
                if 'f1pBW' in to_calculate:
                    [tmp_resultf1p[5, i], _] = py_f1PrimeG(samples.values[i].spd.wl, samples.values[i].spd.value, strWeighting='A', dBandWidth=20.)
            if 'f1p' in to_calculate:
                self.result['f1p']['data']=tmp_resultf1p[0,:]
            if 'f1pE' in to_calculate:
                self.result['f1pE']['data']=tmp_resultf1p[1,:]
            if 'f1pL' in to_calculate:
                self.result['f1pL']['data']=tmp_resultf1p[2,:]
            if 'f1pMin' in to_calculate:
                self.result['f1pMin']['data']=tmp_resultf1p[3,:]
            if 'f1pp' in to_calculate:
                self.result['f1pp']['data']=tmp_resultf1p[4,:]
            if 'f1pBW' in to_calculate:
                self.result['f1pBW']['data']=tmp_resultf1p[5,:]
        for x in to_calculate:
            if '@' in x:
                ref_wl = float(x[2:])
                index = np.argmin( np.abs(samples.v_wl_ref-ref_wl))
                self.result[x]['data'] = np.zeros((self.trials))
                if 'v' in x:
                    for i in range(0, self.trials):
                        # oDo interpolation
                        # oDo [0][index]? better [index]--> check creation
                        #print(samples.values[i].spd.value.shape)
                        self.result[x]['data'][i] = samples.values[i].spd.value[0][index]
                if 'w' in x:
                    for i in range(0, self.trials):
                        # oDo interpolation
                        self.result[x]['data'][i] = samples.values[i].spd.wl[index]

    def clear_result_stat(self):
        for i, result_type in enumerate(_RESULT_TYPE): # store all in single nested dict
            self.result[result_type] ['data']= np.zeros(0)
            self.result[result_type] ['Mean']= float("nan")
            self.result[result_type] ['Std']= float("nan")
            self.result[result_type] ['p95Min']= float("nan")
            self.result[result_type] ['p95Max']= float("nan")

    def calc_result_stat(self, quantil = 0.95):
        for i, result_type in enumerate(_RESULT_TYPE): # store all in single nested dict
            if self.result[result_type] ['data'].size > 1:
                self.result[result_type] ['Mean']= np.mean(self.result[result_type] ['data'])
                self.result[result_type] ['Std']= np.std(self.result[result_type] ['data'])
                self.result[result_type] ['p95Min']= np.quantile(self.result[result_type] ['data'], quantil/2)
                self.result[result_type] ['p95Max']= np.quantile(self.result[result_type] ['data'], 1-quantil/2)
            else:
                self.result[result_type] ['Mean']= float("nan")
                self.result[result_type] ['Std']= float("nan")
                self.result[result_type] ['p95Min']= float("nan")
                self.result[result_type] ['p95Max']= float("nan")

    def replace_nan(self, corr, replace_with = 0):
        list = np.where(corr!=corr)
        for i in range(len(list[0])):
            corr[list[0][i]][list[1][i]]=replace_with
        return corr

    def get_table_correlation(self, whished_results):
        disp_array_count = self.result[whished_results[0]]['data'].shape[0]
        number_data = len(whished_results)
        disp_array = np.zeros((number_data, disp_array_count))
        for j in range(number_data):
            disp_array[j] = self.result[whished_results[j]]['data']
        corr = np.corrcoef(disp_array)
        return self.replace_nan(corr)

    def get_table_covariane(self, whished_results):
        disp_array_count = self.result[whished_results[0]]['data'].shape[0]
        number_data = len(whished_results)
        disp_array = np.zeros((number_data, disp_array_count))
        for j in range(number_data):
            disp_array[j] = self.result[whished_results[j]]['data']
        corr = np.cov(disp_array)
        return self.replace_nan(corr)

    def seaborn_plot(self, whished_results, filename=None, title = None, whished_results_str = None):
        disp_array_count, step = get_data_step(self.result[whished_results[0]]['data'].shape[0])
        number_data = len(whished_results)
        disp_array = np.zeros((number_data, disp_array_count - 1))
        for j in range(number_data):
            for i in range(disp_array_count - 1):
                disp_array[j, i] = self.result[whished_results[j]]['data'][i * step + 1]

        sns.set_theme(style="ticks")
        if whished_results_str is None:
            whished_results_str = whished_results
        df = pd.DataFrame(data=disp_array.T, columns=whished_results_str)
        grid = sns.pairplot(df, corner=True)
        plotTitle = title
        grid.fig.suptitle(plotTitle.format())
        if filename is not None:
            grid.fig.savefig(filename)

# Class to manage the MC Samples for a MC simulation
# Store/load/save the sample data
# load/save statistical data from samples and restore the samples later on
# calculate some statistical information from samples
class MCSpectrumSamples(object):
    # init the spectral array with the reference distribution to add some noise later on
    def __init__(self, spd=None, trials=1000):
        # with help from https://schurpf.com/python-save-a-class/
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
        def_name = text[:text.find('=')].strip()
        self.name = def_name
        # make a different between trials and trials_current to support adaptive MCS later on
        self.trials=trials
        self.trials_current = 0
        self.spd = spd
        self.init_values()
        # reference wavelength scale
        self.v_wl_ref = self.values[0].spd.wl
        # number of wavelength elements
        self.res = len(self.v_wl_ref)
        # nominal wavelength scale
        self.v_wl = np.zeros((self.res))
        # difference of the mean value to the initial reference value
        self.v_wl_diff= np.zeros((self.res))
        # mean values after simulation
        self.v_mean = np.zeros((self.res))
        # difference of the mean value to the initial reference value
        self.v_mean_diff = np.zeros((self.res))
        # standard deviation of the values
        self.v_std = np.zeros((self.res))
        # quantile values (95% rage)
        self.q_max = np.zeros((self.res))
        self.q_min = np.zeros((self.res))
        # covariance matrix
        self.cov_matrix = np.zeros((self.res,self.res))
        # correlation matrix
        self.corr_matrix = np.zeros((self.res,self.res))
        # packed values for the statistical analysis
        self.values_packed = np.zeros((self.trials, self.res))
        # information from the evaluation
        self.spectrumResults = MCSpectrumResults()
        # database with the latest results
        self.MCTable = pd.DataFrame()

    def init_values(self):
        # Array containing instances of the class lx.SPD to store the results of the MC simulation
        self.values=np.ndarray((self.trials,),dtype=object)
        for i in range(self.trials):
            self.values[i] = McSpectrumX(self.spd)

    # save all data in a binary format using pickle (only usable with python)
    def save(self, filename = None):
        if filename is None:
            file = open(self.name+'.pkl','wb')
        else:
            file = open(filename+'.pkl', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    # load all data from binary pickle file to reconstruct the complete instance
    def load(self, filename = None):
        if filename is None:
            file = open(self.name+'.pkl','rb')
        else:
            file = open(filename+'.pkl', 'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)

    def __str__(self):
        return str(self.__dict__)

    # save/load csv:
    #   store only mean vector and covariance matrix
    #   load only mean vector and covariance matrix and generate the missing data
    def save_to_csv(self, filename = None):
        if filename is None:
            filename = self.name+'.csv'
        else:
            filename = filename+'.csv'

        help = np.vstack((self.v_wl_ref, self.v_wl, self.v_mean, self.v_std))

        with open(filename, 'w') as f:
            f.write(type(self).__name__ + '\n')
            f.write('V0.1\n')
            f.write( '%d\n' %(self.trials_current))
            f.write( '%d\n' %(self.res))
            f.write('v_wl_ref;v_wl;v_mean;v_std\n')
            np.savetxt(f, help.T, delimiter=';')
            f.write('covariance\n')
            np.savetxt(f, self.cov_matrix, delimiter=';')
            # let's store both matrices for easy further evaluations
            f.write('correlation\n')
            np.savetxt(f, self.corr_matrix, delimiter=';')
        return filename

    def load_from_csv(self, filename=None):
        if filename is None:
            filename = self.name+'.csv'
        else:
            filename = filename+'.csv'

        help = np.vstack((self.v_wl_ref, self.v_wl, self.v_mean, self.v_std))
        help[:,:]=0
        self.cov_matrix[:,:]=0

        with open(filename, 'r') as f:
            type_str = f.readline()
            version_str = f.readline()
            # oDo: Test type_str and version_str
            self.trials_current = int(f.readline())
            self.res = int(f.readline())
            # read the header line and ignore
            f.readline()
            help = np.loadtxt(f, delimiter=';', max_rows=self.res)
            self.v_wl_ref = np.zeros((self.res))
            self.v_wl_ref = help[:,0].T
            self.v_wl = np.zeros((self.res))
            self.v_wl = help[:,1].T
            self.v_mean = np.zeros((self.res))
            self.v_mean = help[:,2].T
            self.v_std = np.zeros((self.res))
            self.v_std = help[:,3].T

            self.cov_matrix = np.zeros((self.res, self.res))
            # read the header line and ignore
            f.readline()
            tmp_matrix = np.loadtxt(f, delimiter=';', max_rows=self.res)
            self.cov_matrix = tmp_matrix.copy()
            # read the header line and ignore
            f.readline()
            tmp_matrix = np.loadtxt(f, delimiter=';', max_rows=self.res)
            self.corr_matrix = tmp_matrix.copy()
        return filename

    # generate all information from v_wl, v_mean and cov_matrix
    # do not calculate new statistical evaluation here, only reconstruct the base data
    def reconstruct_information(self):
        self.trials = self.trials_current
        # Array containing instances of the class SPD to store the results of the MC simulation
        self.values=np.ndarray((self.trials,),dtype=object)
        spd_tmp = np.vstack((self.v_wl, self.v_mean))
        for i in range(self.trials):
            self.values[i] = McSpectrumX(spd_tmp)
        self.res = len(self.values[0].spd.wl)
        self.v_mean_diff = np.zeros((self.res))
        self.q_max = np.zeros((self.res))
        self.q_min = np.zeros((self.res))
        # generate new random data
        rand_data = np.random.default_rng().multivariate_normal(self.v_mean, self.cov_matrix, self.trials)
        for i in range(self.trials):
            self.values[i].spd.value = rand_data[i].copy()
        self.values_packed = np.zeros((self.trials, self.res))
        return

    # generate a packed array to use the numpy statistical functions later on
    def array2analyse(self, wavelength_stat=True, scale_to_ref=True):
        for i in range(self.trials_current):
            if scale_to_ref:
                if wavelength_stat:
                    self.values_packed[i] = self.values[i].spd.wl - self.spd[0]
                else:
                    self.values_packed[i] = self.values[i].spd.value - self.spd[1]
            else:
                if wavelength_stat:
                    self.values_packed[i] = self.values[i].spd.wl
                else:
                    self.values_packed[i] = self.values[i].spd.value

    def init_MCTable(self):
        self.MCTable = pd.DataFrame()
        return

    def run_MC(self, mc_enable_loc, wished_results, quantil=0.95, filename=None):
        for iMC in mc_enable_loc:
            if mc_enable_loc[iMC] == False: continue
            print( iMC)
            if iMC == 'file':
                self.load(filename=filename)
            else:
                if filename is not None and iMC == 'cov':
                    self.load_from_csv(filename=filename)
                    self.reconstruct_information()
                else:
                    print( 'generate_random_numbers')
                    d_mean, d_stddev, d_dist, d_nb = self.generate_random_numbers(iMC, mc_enable_loc, init=True)

            self.normalize_wl_scale()
            self.calc_cov_matrix()

            self.spectrumResults.clear_result_stat()
            self.spectrumResults.calculate(self, wished_results)
            self.spectrumResults.calc_result_stat(quantil=quantil)

            if iMC=='all' or iMC=='cov' or iMC == 'file':
                df = pd.DataFrame( { \
                     'Contributon': [iMC], \
                     'Mean':'',
                    'StdDev':'',
                    'NB':'',
                    'Distribution':''})
            else:
                df = pd.DataFrame( { \
                     'Contributon': [iMC], \
                     'Mean':d_mean,
                    'StdDev':d_stddev,
                    'NB':d_nb,
                    'Distribution':d_dist})

            for i, result_type in enumerate(_RESULT_TYPE): # store all in single nested dict
                if not math.isnan(self.spectrumResults.result[result_type] ['Mean']):
                    df[result_type]=self.spectrumResults.result[result_type] ['Mean']
                    df['s'+result_type]=self.spectrumResults.result[result_type] ['Std']
                    df['p95Min'+result_type]=self.spectrumResults.result[result_type] ['p95Min']
                    df['p95Max'+result_type]=self.spectrumResults.result[result_type] ['p95Max']

            self.MCTable = pd.concat( [self.MCTable,df])
        return

    def generate_random_numbers(self, iMC, mc_enable_loc, init=True):
        if init:
            self.init_values()

        # for all trials
        d_nb = 0
        d_mean = 0
        d_stddev = 0
        d_dist = 'none'
        for i in range(0, self.trials):
            if i == 0: continue
            self.trials_current=i+1
            if mc_enable_loc['wl_noise_nc'] and (iMC == 'wl_noise_nc' or iMC == 'all'):
                if i == 1: print('A')
                d_mean = 0
                d_stddev = 1
                d_dist = 'normal'
                self.values[i].add_wl_noise_nc( d_mean, d_stddev, distribution=d_dist)
            if mc_enable_loc['wl_noise_c'] and (iMC == 'wl_noise_c' or iMC == 'all'):
                if i == 1: print('B')
                d_mean = 0
                d_stddev = 1
                d_dist = 'normal'
                self.values[i].add_wl_noise_c( d_mean, d_stddev, distribution=d_dist)
            if mc_enable_loc['wl_fourier_noise'] and (iMC == 'wl_fourier_noise' or iMC == 'all'):
                if i == 1: print('C')
                d_mean = 0
                d_stddev = 1
                d_dist = 'F'
                d_nb = 4
                self.values[i].add_wl_fourier_noise( self.values[0], d_nb, stddev=1.)

            if mc_enable_loc['value_noise_nc'] and (iMC == 'value_noise_nc' or iMC == 'all'):
                if i == 1: print('D')
                d_mean = 0
                d_stddev = 0.01
                d_dist = 'normal'
                self.values[i].add_value_noise_nc( d_mean, d_stddev, distribution=d_dist)
            if mc_enable_loc['value_noise_c'] and (iMC == 'value_noise_c' or iMC == 'all'):
                if i == 1: print('E')
                d_mean = 0
                d_stddev = 0.01
                d_dist = 'normal'
                self.values[i].add_value_noise_c( d_mean, d_stddev, distribution=d_dist)
            if mc_enable_loc['value_fourier_noise'] and (iMC == 'value_fourier_noise' or iMC == 'all'):
                if i == 1: print('F')
                d_mean = 0
                d_stddev = 0.01
                d_dist = 'F'
                d_nb = 4
                self.values[i].add_value_fourier_noise( self.values[0], d_nb, stddev=d_stddev)
        if iMC=='cov' and self.cov_matrix[0,0] != 0:
            print('G')
            rand_data = np.random.default_rng().multivariate_normal(self.v_mean, self.cov_matrix, self.trials)
            for i in range(1, self.trials):
                self.values[i].add_value_noise(rand_data[i])
        return d_mean, d_stddev, d_dist, d_nb

    def calc_summary(self, wavelength_stat=True, scale_to_ref=True):
        self.array2analyse(wavelength_stat=wavelength_stat, scale_to_ref=False)
        [loc_result_sum_mcv, loc_interval] = sumMCV(self.values_packed, Coverage=0.95)
        self.v_wl = self.values[0].spd.wl.copy()
        self.v_wl_diff = self.v_wl - self.spd[0]
        self.v_mean = loc_result_sum_mcv[0].copy()
        self.v_mean_diff = self.v_mean - self.spd[1]
        self.v_std = loc_result_sum_mcv[1].copy()
        self.q_min = loc_interval[0].copy()
        self.q_max = loc_interval[1].copy()
        self.cov_matrix = np.cov(self.values_packed.T)
        self.calc_corr_matrix()
        return self.trials_current

    def get_result(self):
        return self.v_wl_ref, self.v_wl, self.v_wl_diff, self.v_mean, self.v_mean_diff, self.v_std, self.q_min, self.q_max, self.cov_matrix, self.corr_matrix, self.trials_current

    # calc a correlation matrix from covariance matrix
    # We need the covariance matrix first!
    # THX: https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    def calc_corr_matrix(self):
        v = np.sqrt(np.diag(self.cov_matrix))
        outer_v = np.outer(v, v)
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

    # interpolate all data in self.values to the nominal wavelength scale
    # Attention: No update of statistical information here
    def normalize_wl_scale(self):
        for i in range(0, self.trials_current):
            self.values[i].spd.cie_interp(self.v_wl_ref, kind = 'linear',negative_values_allowed=True)
        return


