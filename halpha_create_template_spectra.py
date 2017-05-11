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

# if processing should be resumed from the endpoint, TODO: wvl range should be read from output csv files
RESUME_PROCESSING = True

# determine csv outputs
txt_out_sobject = 'report_sobject_ids.csv'
txt_out_wvl1 = 'report_wavelengths_ccd1.csv'
txt_out_wvl3 = 'report_wavelengths_ccd3.csv'
txt_out_spectra1 = 'template_spectra_ccd1.csv'
txt_out_spectra3 = 'template_spectra_ccd3.csv'

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

# change to output directory
out_dir = 'H_spectra_templates'
ch_dir(out_dir)

# object selection criteria
# must be a giant - objects are further away than dwarfs
print 'Number of all objects: '+str(len(galah_param))
if not RESUME_PROCESSING:
    idx_object_use = galah_param['logg_guess'] < 3.5
    # idx_object_use = galah_param['sobject_id'] == 131216001101091
    # must have a decent snr value
    idx_object_use = np.logical_and(idx_object_use,
                                    galah_param['snr_c3_guess'] > 0)
else:
    sobjects_used = pd.read_csv(txt_out_sobject, sep=',', header=None).values[0]
    idx_object_use = np.in1d(galah_param['sobject_id'].data, sobjects_used)

print 'Number of used objects: '+str(np.sum(idx_object_use))
# create a subset of input object tables
galah_param = galah_param[idx_object_use].filled()  # handle masked values

# some program setting
TEFF_STEP = 300.
LOGG_STEP = 0.5
FEH_STEP = 0.25
plot_graphs = False
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

# create empty nan spectra
empty_ccd1 = np.ndarray(len(wvl_read_ccd1))
empty_ccd1.fill(np.nan)
empty_ccd3 = np.array(empty_ccd1)

if not RESUME_PROCESSING:
    print 'Writing initial outputs'
    # create outputs
    out_files = [txt_out_sobject, txt_out_wvl1, txt_out_wvl3, txt_out_spectra1, txt_out_spectra3]
    for out_file in out_files:
        new_txt_file(out_file)

    # write outputs
    append_line(txt_out_wvl1, ','.join([str(wvl) for wvl in wvl_read_ccd1]))
    append_line(txt_out_wvl3, ','.join([str(wvl) for wvl in wvl_read_ccd3]))

print 'Template creation procedure started'
# time keeping
i_t = 1
total_sec = 0.
# plot some randomly selected spectra
# idx_process = np.arange(1400)
idx_process = range(len(galah_param))

# write outputs
if not RESUME_PROCESSING:
    append_line(txt_out_sobject, ','.join([str(s_id) for s_id in galah_param['sobject_id'][idx_process].data]))

# determine the point where the process was terminated
if RESUME_PROCESSING:
    print 'Determining number of already processed objects'
    with open(txt_out_spectra3) as foo:
        n_out_lines = len(foo.readlines())
    idx_process = idx_process[n_out_lines:]

# select data with high SNR
idx_snr_ok_ccd1 = galah_param['snr_c1_guess'] > 20
idx_snr_ok_ccd3 = galah_param['snr_c3_guess'] > 100

for i_r in idx_process:
    object_param = galah_param[i_r]
    sobj_id = object_param['sobject_id']
    print str(i_r+1)+':  '+str(sobj_id)

    # get object spectra
    spectra_ccd1 = ccd1_data[i_r]
    spectra_ccd3 = ccd3_data[i_r]

    time_start = time.time()
    # determine objects parameters
    obj_teff = object_param['teff_guess']
    obj_logg = object_param['teff_guess']
    obj_feh = object_param['teff_guess']

    # get objects that have similar physical parameters to the observed object
    idx_select = np.logical_and(np.abs(galah_param['logg_guess'] - obj_logg) < (LOGG_STEP / 2.),
                                np.abs(galah_param['teff_guess'] - obj_teff) < (TEFF_STEP / 2.))
    idx_select = np.logical_and(idx_select,
                                np.abs(galah_param['feh_guess'] - obj_feh) < (FEH_STEP / 2.))
    # idx_select = np.logical_and(idx_select,
    #                             idx_snr_ok_ccd3)

    # TODO: some further object selection refinement based on the "likeness" to the observed spectra

    n_similar = np.sum(idx_select)
    print ' Number of similar objects: ' + str(n_similar)
    if n_similar < 3:
        print '  Skipping this object'
        # save empty spectra line to csv files
        append_line(txt_out_spectra1, ','.join([str(flx) for flx in empty_ccd1]), new_line=True)
        append_line(txt_out_spectra3, ','.join([str(flx) for flx in empty_ccd3]), new_line=True)
    else:
        spectra_ccd1_median = np.median(ccd1_data[idx_select], axis=0)
        spectra_ccd3_median = np.median(ccd3_data[idx_select], axis=0)
        # save median spectra line to csv files
        append_line(txt_out_spectra1, ','.join([str(flx) for flx in spectra_ccd1_median]), new_line=True)
        append_line(txt_out_spectra3, ','.join([str(flx) for flx in spectra_ccd3_median]), new_line=True)

        if plot_graphs:
            # plot results
            fig, axs = plt.subplots(2, 2)
            fig.suptitle('Guess   ->   teff:{:4.0f}   logg:{:1.1f}   feh:{:1.1f}'.format(obj_teff,obj_logg,obj_feh))
            # h-alpha plots
            axs[0, 0].plot(wvl_read_ccd3, spectra_ccd3_median, color='red', linewidth=0.8)
            axs[0, 0].plot(wvl_read_ccd3, spectra_ccd3, color='black', linewidth=0.6)
            axs[0, 0].set(xlim=(wvl_min_alpha, wvl_max_alpha), ylim=(0.1, 1.1), title='H-alpha', ylabel='Flux')
            axs[1, 0].plot(wvl_read_ccd3, spectra_ccd3-spectra_ccd3_median, color='black', linewidth=0.6)
            axs[1, 0].set(xlim=(wvl_min_alpha, wvl_max_alpha), ylim=(-0.2, 0.2), ylabel='Subtracted flux', xlabel='Wavelength')
            # h-beta plots
            axs[0, 1].plot(wvl_read_ccd1, spectra_ccd1_median, color='red', linewidth=0.8)
            axs[0, 1].plot(wvl_read_ccd1, spectra_ccd1, color='black', linewidth=0.6)
            axs[0, 1].set(xlim=(wvl_min_beta, wvl_max_beta), ylim=(0.1, 1.1), title='H-beta')
            axs[1, 1].plot(wvl_read_ccd1, spectra_ccd1-spectra_ccd1_median, color='black', linewidth=0.6)
            axs[1, 1].set(xlim=(wvl_min_beta, wvl_max_beta), ylim=(-0.2, 0.2), xlabel='Wavelength')
            # plt.tight_layout()
            plt.savefig(str(sobj_id) + '.png', dpi=200)
            plt.close()
            print ''

    time_end = time.time()
    # Create time estimation outputs
    time_delta = time_end - time_start
    print 'Fit time: ' + str(datetime.timedelta(seconds=time_delta))
    total_sec += time_delta
    time_to_end = total_sec / i_t * (len(idx_process) - i_t)
    print 'Estimated finished in: ' + str(datetime.timedelta(seconds=time_to_end))
    i_t += 1
