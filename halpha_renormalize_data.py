import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un
import time, datetime

from astropy.table import Table
from halpha_renormalize_data_functions import *

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv')

spectra_file_ccd1 = 'galah_dr52_ccd1_4710_4910_interpolated_wvlstep_0.04_spline_restframe.csv'
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
# idx_object_use = galah_param['sobject_id'] == 131216001101091
# must have a decent snr value
idx_object_use = np.logical_and(idx_object_use,
                                galah_param['snr_c3_guess'] > 80)
print 'Number of used objects: '+str(np.sum(idx_object_use))
# create a subset of input object tables
galah_param = galah_param[idx_object_use].filled()  # handle masked values
# convert ra dec to coordinates object
ra_dec = coord.ICRS(ra=galah_param['ra'].data*un.deg,
                    dec=galah_param['dec'].data*un.deg)

wvl_read_range = 60

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
                        usecols=idx_beta[0], skiprows=np.where(np.logical_not(idx_object_use))[0]).values
print ' cols for ccd3: '+str(len(wvl_read_ccd3))
ccd3_data = pd.read_csv(galah_data_dir + spectra_file_ccd3, sep=',', header=None, na_values='nan',
                        usecols=idx_alpha[0], skiprows=np.where(np.logical_not(idx_object_use))[0]).values

# change to output directory
out_dir = 'H_spectra_normalization_3'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
os.chdir(out_dir)


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
append_line(txt_out_wvl1, ','.join([str(wvl) for wvl in wvl_read_ccd1]))
append_line(txt_out_wvl3, ','.join([str(wvl) for wvl in wvl_read_ccd3]))

print 'Fitting procedure started'
# time keeping
i_t = 1
total_sec = 0.
# plot some randomly selected spectra
# idx_rand = np.arange(1400)
idx_rand = range(len(galah_param))

# write outputs
append_line(txt_out_sobject, ','.join([str(s_id) for s_id in galah_param['sobject_id'][idx_rand].data]))

for i_r in idx_rand:
    object_param = galah_param[i_r]
    sobj_id = object_param['sobject_id']
    print str(i_r+1)+':  '+str(sobj_id)

    time_start = time.time()
    # fit h-alpha profile
    spectra_ccd3 = ccd3_data[i_r]
    # spectra_ccd3_filtered = filter_spectra(spectra_ccd3, wvl_read_ccd3, wvl_center=HALPHA_WVL, center_width=10., median_width=15)
    h_alpha_fited_curve = fit_h_profile_with_clipping(spectra_ccd3, wvl_read_ccd3, wvl_center=HALPHA_WVL,
                                                      steps=4, std_lower=2.5, std_upper=2.5, std_step_change=0.1,
                                                      verbose=False, profile_kwargs={'voigt1':True, 'voigt2':True})
    # fit b-beta profile
    spectra_ccd1 = ccd1_data[i_r]
    # spectra_ccd1_filtered = filter_spectra(spectra_ccd1, wvl_read_ccd1, wvl_center=HBETA_WVL, center_width=10., median_width=15)
    h_beta_fited_curve = fit_h_profile_with_clipping(spectra_ccd1, wvl_read_ccd1, wvl_center=HBETA_WVL,
                                                     steps=1, std_lower=2.5, std_upper=2.5, std_step_change=0.1,
                                                     verbose=False, profile_kwargs={'voigt1':True, 'voigt2':True})
    time_end = time.time()

    # Create time estimation outputs
    time_delta = time_end-time_start
    print 'Fit time: '+str(datetime.timedelta(seconds=time_delta))
    total_sec += time_delta
    time_to_end = total_sec/i_t * (len(idx_rand)-i_t)
    print 'Estimated finished in: '+str(datetime.timedelta(seconds=time_to_end))
    i_t += 1

    # test plots for initial continuum fit
    # plt.plot(wvl_read_ccd1, spectra_ccd1, color='black')
    # plt.plot(wvl_read_ccd3, fit_h_profile_initial(spectra_ccd3, wvl_read_ccd3, HALPHA_WVL, method='sliding'), color='red')
    # plt.plot(wvl_read_ccd3, fit_h_profile_initial(spectra_ccd3, wvl_read_ccd3, HALPHA_WVL, method='savgol'), color='green')
    # plt.plot(wvl_read_ccd1, fit_h_profile_with_clipping(spectra_ccd1, wvl_read_ccd1, wvl_center=HBETA_WVL, steps=1), color='red')
    # plt.plot(wvl_read_ccd1, fit_h_profile_with_clipping(spectra_ccd1, wvl_read_ccd1, wvl_center=HBETA_WVL, steps=6), color='green')
    # plt.savefig(str(sobj_id) + '.png', dpi=250)
    # plt.close()
    # continue

    # spectra normalization
    # spectra_ccd3_norm = spectra_ccd3 / h_alpha_fited_curve
    spectra_ccd3_sub = spectra_ccd3 - h_alpha_fited_curve
    # spectra_ccd1_norm = spectra_ccd1 / h_beta_fited_curve
    spectra_ccd1_sub = spectra_ccd1 - h_beta_fited_curve

    # save renormalized results to csv file
    append_line(txt_out_spectra3, ','.join([str(flx) for flx in spectra_ccd3_sub]), new_line=True)
    append_line(txt_out_spectra1, ','.join([str(flx) for flx in spectra_ccd1_sub]), new_line=True)

    # plot results
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Guess   ->   teff:{:4.0f}   logg:{:1.1f}   feh:{:1.1f}'.format(object_param['teff_guess'],object_param['logg_guess'],object_param['feh_guess']))
    # h-alpha plots
    axs[0, 0].plot(wvl_read_ccd3, h_alpha_fited_curve, color='red', linewidth=0.8)
    axs[0, 0].plot(wvl_read_ccd3, spectra_ccd3, color='black', linewidth=0.4)
    # axs[0, 0].plot(wvl_read_ccd3, spectra_ccd3_filtered, color='blue', linewidth=0.4)
    axs[0, 0].set(xlim=(wvl_min_alpha, wvl_max_alpha), ylim=(0.1, 1.1), title='H-alpha', ylabel='Flux')
    axs[1, 0].plot(wvl_read_ccd3, spectra_ccd3_sub, color='black', linewidth=0.5)
    axs[1, 0].set(xlim=(wvl_min_alpha, wvl_max_alpha), ylim=(-0.2, 0.2), ylabel='Subtracted flux', xlabel='Wavelength')
    # h-beta plots
    axs[0, 1].plot(wvl_read_ccd1, h_beta_fited_curve, color='red', linewidth=0.75)
    axs[0, 1].plot(wvl_read_ccd1, spectra_ccd1, color='black', linewidth=0.4)
    # axs[0, 1].plot(wvl_read_ccd1, spectra_ccd1_filtered, color='blue', linewidth=0.4)
    axs[0, 1].set(xlim=(wvl_min_beta, wvl_max_beta), ylim=(0.1, 1.1), title='H-beta')
    axs[1, 1].plot(wvl_read_ccd1, spectra_ccd1_sub, color='black', linewidth=0.5)
    axs[1, 1].set(xlim=(wvl_min_beta, wvl_max_beta), ylim=(-0.2, 0.2), xlabel='Wavelength')
    # plt.tight_layout()
    plt.savefig(str(sobj_id)+'.png', dpi=250)
    plt.close()
    print ''


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


    # test profiles
    # test_curve_1, idx_bad1 = fit_h_profile_with_clipping(spectra_ccd3, wvl_read_ccd3, wvl_center=HALPHA_WVL,
    #                                            steps=4, std_lower=2.2, std_upper=2.2, verbose=False, diagnostics=True, profile_kwargs={'voigt1': True, 'voigt2': True}, return_ommitted=True)
    # test_curve_2 = fit_h_profile_with_clipping(spectra_ccd3_filtered, wvl_read_ccd3, wvl_center=HALPHA_WVL,
    #                                            steps=4, profile_kwargs={'voigt1': True})
    # test_curve_3, idx_bad = fit_h_profile_with_clipping(spectra_ccd3_filtered, wvl_read_ccd3, wvl_center=HALPHA_WVL,
    #                                            steps=4, profile_kwargs={'gauss1': True}, return_ommitted=True)
    # test_curve_4 = fit_h_profile_with_clipping(spectra_ccd3_filtered, wvl_read_ccd3, wvl_center=HALPHA_WVL,
    #                                            steps=4, profile_kwargs={'lorentz1': True})
    # test profiles plot
    # plt.plot(wvl_read_ccd3, spectra_ccd3, color='black', linewidth=0.6)
    # plt.plot(wvl_read_ccd3, test_curve_1, color='red', linewidth=0.3)
    # plt.plot(wvl_read_ccd3, test_curve_2, color='green', linewidth=0.3)
    # plt.plot(wvl_read_ccd3, test_curve_3, color='blue', linewidth=0.3)
    # plt.plot(wvl_read_ccd3, test_curve_4, color='purple', linewidth=0.3)
    # plt.scatter(wvl_read_ccd3[idx_bad1], spectra_ccd3[idx_bad1], lw=0, s=4, c='red')
    # plt.xlim((wvl_min_alpha, wvl_max_alpha))
    # plt.ylim((0.2, 1.2))
    # plt.savefig(str(sobj_id) + '_test.png', dpi=350)
    # plt.close()