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
galah_data_dir = '/home/nandir/Desktop/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv')

spectra_file_ccd1 = 'galah_dr52_ccd1_5640_5880_interpolated_wvlstep_0.04_spline_restframe.csv'
spectra_file_ccd3 = 'galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe.csv'
# parse resampling settings from filename
csv_param_ccd1 = CollectionParameters(spectra_file_ccd1)
wvl_values_ccd1 = csv_param_ccd1.get_wvl_values()
csv_param_ccd3 = CollectionParameters(spectra_file_ccd3)
wvl_values_ccd3 = csv_param_ccd3.get_wvl_values()

# object selection criteria
# must be a giant - objects are further away than dwarfs
print 'Number of all objects: '+str(len(galah_param))
idx_object_use = galah_param['logg_guess'] < 3.5
# must have a decent snr value
idx_object_use = np.logical_and(idx_object_use,
                                galah_param['snr_c3_guess'] > 150)
print 'Number of used objects: '+str(np.sum(idx_object_use))
# create a subset of input object tables
galah_param = galah_param[idx_object_use].filled()  # handle masked values
# convert ra dec to coordinates object
ra_dec = coord.ICRS(ra=galah_param['ra'].data*un.deg,
                    dec=galah_param['dec'].data*un.deg)

wvl_read_range = 45

HBETA_WVL = 4861.36
wvl_min_beta = HBETA_WVL - wvl_read_range
wvl_max_beta = HBETA_WVL + wvl_read_range
idx_beta = np.where(np.logical_and(wvl_values_ccd1 > wvl_min_beta,
                                   wvl_values_ccd1 < wvl_max_beta))
wvl_read_ccd1 = wvl_values_ccd1[idx_beta]

HALPHA_WVL = 6562.81
wvl_min_alpha = HALPHA_WVL - wvl_read_range
wvl_max_alpha = HALPHA_WVL + wvl_read_range
idx_alpha = np.where(np.logical_and(wvl_values_ccd3 > wvl_min_alpha,
                                    wvl_values_ccd3 < wvl_max_alpha))
wvl_read_ccd3 = wvl_values_ccd3[idx_alpha]

# read appropriate subset of data
print 'Reading resampled GALAH spectra'
print ' cols for ccd1: '+str(len(wvl_read_ccd1))
ccd1_data = pd.read_csv(galah_data_dir + spectra_file_ccd1, sep=',', header=None, na_values='nan',
                        usecols=wvl_read_ccd1[0], skiprows=np.where(np.logical_not(idx_object_use))[0]).values
print ' cols for ccd3: '+str(len(wvl_read_ccd3))
ccd3_data = pd.read_csv(galah_data_dir + spectra_file_ccd3, sep=',', header=None, na_values='nan',
                        usecols=idx_alpha[0], skiprows=np.where(np.logical_not(idx_object_use))[0]).values

# change to output directory
out_dir = 'H_spectra_normalization'
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


def fit_h_profile(spectrum, wavelengths, wvl_center, verbose=False):
    profile = VoigtModel(prefix='abs_')
    profile2 = VoigtModel(prefix='abs2_')
    # profile2 = VoigtModel(prefix='abs3_')
    # line = ConstantModel(prefix='const_')  # constant or
    line = LinearModel(prefix='line_')  # linear model
    params = Parameters()
    params.add('abs_center', value=0, min=-0.1, max=+0.1)  # fix center wavelength of the profile
    params.add('abs_sigma', value=0.5, min=0.0001, max=5.0)
    params.add('abs_gamma', value=0.5, min=0.0001, max=5.0)  # manipulate connection between sigma and gamma
    params.add('abs_amplitude', value=1., min=0.01, max=10.0)
    params.add('abs2_center', value=0, min=-0.1, max=+0.1)  # fix center wavelength of the profile
    params.add('abs2_sigma', value=0.5, min=0.0001, max=20.0)
    params.add('abs2_gamma', value=0.5, min=0.0001, max=20.0)  # manipulate connection between sigma and gamma
    params.add('abs2_amplitude', value=1., min=0.01, max=10.0)
    # params.add('const_c', value=1., min=0.92, max=1.08)  # continuum approximation
    params.add('line_intercept', value=1., min=0.8, max=1.2)
    params.add('line_slope', value=0., min=-0.1, max=0.1)
    # fit the data
    final_model = line - profile - profile2
    # start fitting procedure
    abs_line_fit = final_model.fit(spectrum, params=params, x=wavelengths - wvl_center)  # , method='brute')
    if verbose:
        abs_line_fit.params.pretty_print()
    return abs_line_fit


def new_txt_file(filename):
    temp = open(filename, 'w')
    temp.close()


def append_line(filename, line_string, new_line=False):
    temp = open(filename, 'a')
    if new_line:
        temp.write(line_string+'\n')
    else:
        temp.write(line_string)
    temp.close()

print 'Writing initial outputs'
# determine csv outputs
txt_out_sobject = 'report_sobject_ids.csv'
txt_out_wvl1 = 'report_wavelengths_ccd1.csv'
txt_out_wvl3 = 'report_wavelengths_ccd3.csv'
txt_out_spectra1 = 'normalized_spectra_ccd1.csv'
txt_out_spectra3 = 'normalized_spectra_ccd3.csv'

# create outputs
out_files = [txt_out_sobject, txt_out_wvl1, txt_out_wvl3, txt_out_spectra1, txt_out_spectra3]
for out_file in out_files:
    new_txt_file(out_file)

# write outputs
append_line(txt_out_sobject, ','.join([str(s_id) for s_id in galah_param['sobject_id'].data]))
append_line(txt_out_wvl1, ','.join([str(wvl) for wvl in wvl_read_ccd1]))
append_line(txt_out_wvl3, ','.join([str(wvl) for wvl in wvl_read_ccd3]))

print 'Fitting procedure started'
# plot some randomly selected spectra
# idx_rand = np.arange(1400)
# idx_rand = np.where(galah_param['sobject_id'] == 151109003601023)[0]
idx_rand = range(len(idx_object_use))
for i_r in idx_rand:
    object_param = galah_param[i_r]
    sobj_id = object_param['sobject_id']
    print str(i_r+1)+':  '+str(sobj_id)

    # fit h-alpha profile
    spectra_ccd3 = ccd3_data[i_r]
    h_alpha_fit = fit_h_profile(spectra_ccd3, wvl_read_ccd3, HALPHA_WVL)
    # fit b-beta profile
    spectra_ccd1 = ccd1_data[i_r]
    h_beta_fit = fit_h_profile(spectra_ccd1, wvl_read_ccd1, HBETA_WVL)

    # spectra normalization
    spectra_ccd3_norm = spectra_ccd3 / h_alpha_fit.best_fit
    spectra_ccd3_sub = spectra_ccd3 - h_alpha_fit.best_fit
    spectra_ccd1_norm = spectra_ccd1 / h_beta_fit.best_fit
    spectra_ccd1_sub = spectra_ccd1 - h_beta_fit.best_fit

    # save renormalized results to csv file
    append_line(txt_out_spectra3, ','.join([str(flx) for flx in spectra_ccd3_sub]), new_line=True)
    append_line(txt_out_spectra1, ','.join([str(flx) for flx in spectra_ccd1_sub]), new_line=True)

    # plot results
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Guess   ->   teff:{:4.0f}   logg:{:1.1f}   feh:{:1.1f}'.format(object_param['teff_guess'],object_param['logg_guess'],object_param['feh_guess']))
    # h-alpha plots
    axs[0, 0].plot(wvl_read_ccd3, h_alpha_fit.best_fit, color='red', linewidth=0.8)
    axs[0, 0].plot(wvl_read_ccd3, spectra_ccd3, color='black', linewidth=0.4)
    axs[0, 0].set(xlim=(wvl_min_alpha, wvl_max_alpha), ylim=(0.1, 1.1), title='H-alpha', ylabel='Flux')
    axs[1, 0].plot(wvl_read_ccd3, spectra_ccd3_sub, color='black', linewidth=0.5)
    axs[1, 0].set(xlim=(wvl_min_alpha, wvl_max_alpha), ylim=(-0.2, 0.2), ylabel='Subtracted flux', xlabel='Wavelength')
    # h-beta plots
    axs[0, 1].plot(wvl_read_ccd1, h_beta_fit.best_fit, color='red', linewidth=0.75)
    axs[0, 1].plot(wvl_read_ccd1, spectra_ccd1, color='black', linewidth=0.4)
    axs[0, 1].set(xlim=(wvl_min_beta, wvl_max_beta), ylim=(0.1, 1.1), title='H-beta')
    axs[1, 1].plot(wvl_read_ccd1, spectra_ccd1_sub, color='black', linewidth=0.5)
    axs[1, 1].set(xlim=(wvl_min_beta, wvl_max_beta), ylim=(-0.2, 0.2), xlabel='Wavelength')
    # plt.tight_layout()
    plt.savefig(str(sobj_id)+'.png', dpi=350)
    plt.close()


# -------------------
# unused code
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