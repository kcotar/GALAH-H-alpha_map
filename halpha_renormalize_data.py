import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un

from astropy.table import Table
from scipy.signal import savgol_filter, argrelextrema
from scipy.interpolate import lagrange

from lmfit import minimize, Parameters, report_fit, Minimizer
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, LinearModel, ConstantModel

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
galah_general = Table.read(galah_data_dir+'sobject_iraf_general_1.1.fits')

spectra_file_csv = 'galah_dr51_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe.csv'
# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file_csv)
ccd = csv_param.get_ccd()
wvl_start, wvl_end = csv_param.get_wvl_range()
wvl_step = csv_param.get_wvl_step()
wvl_values = wvl_start + np.float64(range(0, np.int32(np.ceil((wvl_end-wvl_start)/wvl_step)))) * wvl_step

# object selection criteria
# must be a giant - objects are further away than dwarfs
print 'Number of all objects: '+str(len(galah_param))
idx_use = galah_param['logg_cannon'] < 3.5
# must have a decent snr value
idx_use = np.logical_and(idx_use,
                         galah_general['snr_c3_guess'] > 180)
print 'Number of used objects: '+str(np.sum(idx_use))
# create a subset of input object tables
galah_param = galah_param[idx_use].filled()
galah_general = galah_general[idx_use].filled()
# convert ra dec to coordinates object
ra_dec = coord.ICRS(ra=galah_general['ra'].data*un.deg,
                    dec=galah_general['dec'].data*un.deg)

HALPHA_WVL = 6562.81
wvl_min = 6500
wvl_max = 6620
idx_read = np.where(np.logical_and(wvl_values > wvl_min,
                                   wvl_values < wvl_max))
wvl_read = wvl_values[idx_read]
# read appropriate subset of data
print 'Reading resampled GALAH spectra'
spectral_data = pd.read_csv(galah_data_dir + spectra_file_csv, sep=',', header=None, na_values='nan',
                            usecols=idx_read[0], skiprows=np.where(np.logical_not(idx_use))[0]).values

# change to output directory
out_dir = 'Halpha_region_test'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
os.chdir(out_dir)


# function to be minimized
def gaussian_fit(parameters, data, wvls, continuum, evaluate=True):
    n_keys = (len(parameters)) / 3
    # function_val = parameters['offset']*np.ones(len(wvls))
    function_val = np.array(continuum)
    for i_k in range(n_keys):
        function_val -= parameters['amp'+str(i_k)] * np.exp(-0.5 * (parameters['wvl'+str(i_k)] - wvls) ** 2 / parameters['std'+str(i_k)])
    if evaluate:
        # likelihood = np.nansum(np.power(data - function_val, 2))
        likelihood = np.power(data - function_val, 2)
        return likelihood
    else:
        return function_val

# plot some randomly selected spectra
idx_rand = np.arange(50)
for i_r in idx_rand:
    sobj_id = galah_general[i_r]['sobject_id']
    print sobj_id
    # create line models
    spectra = spectral_data[i_r]

    idx_fit = np.logical_and(wvl_read > 6550, wvl_read < 6580)
    profile = VoigtModel(prefix='abs_')
    # line = ConstantModel(prefix='const_')  # constant or
    line = LinearModel(prefix='line_')  # linear model
    params = Parameters()
    params.add('abs_center', value=HALPHA_WVL, min=HALPHA_WVL-0.1, max=HALPHA_WVL+0.1)  # fix center wavelength of the profile
    params.add('abs_sigma', value=0.5, min=0.0001, max=4.0)
    params.add('abs_gamma', value=0.5, min=0.0001, max=4.0)  # manipulate connection between sigma and gamma
    params.add('abs_amplitude', value=1., min=0.01, max=5.0)
    # params.add('const_c', value=1., min=0.92, max=1.08)  # continuum approximation
    params.add('line_intercept', value=1., min=0.8, max=1.2)
    params.add('line_slope', value=0., min=-0.1, max=0.1)
    # fit the data
    final_model = line - profile
    abs_line_fit = final_model.fit(spectra[idx_fit], params=params, x=wvl_read[idx_fit], method='brute')
    abs_line_fit.params.pretty_print()

    # # second fit option
    # fit_param = Parameters()
    # fit_param.add('wvl0', value=HALPHA_WVL, min=HALPHA_WVL-0.1, max=HALPHA_WVL+0.1)
    # fit_param.add('std0', value=0.01, min=0.0001, max=0.700)
    # fit_param.add('amp0', value=0.01, min=0.0001, max=1.00)
    # # minimize the model
    # cont_line = np.ones(len(wvl_read))
    # fit_res = minimize(gaussian_fit, fit_param, method='brute',
    #                    args=(spectra[idx_fit], wvl_read[idx_fit], cont_line[idx_fit]))
    # fit_res.params.pretty_print()
    # report_fit(fit_res)

    # plot results
    plt.plot(wvl_read, spectra, color='black')
    # plt.plot(wvl_read, gaussian_fit(fit_res.params, 0., wvl_read, cont_line, evaluate=False), color='red')
    plt.plot(wvl_read[idx_fit], abs_line_fit.best_fit, color='red')
    plt.ylim((0.1, 1.1))
    plt.xlim((6550, 6580))
    plt.savefig(str(sobj_id)+'.png', dpi=250)
    plt.close()