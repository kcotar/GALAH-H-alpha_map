import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un
import time, datetime

from astropy.table import Table
from halpha_renormalize_data_functions import ch_dir, append_line, new_txt_file, euclidean_distance_filter, refine_idx_selection, spectra_stat

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters

# if processing should be resumed from the endpoint
RESUME_PROCESSING = True
# some program setting
TEFF_SPAN = 250.
LOGG_SPAN = 0.5
FEH_SPAN = 0.25
limit_snr = True
snr_limits = [15, 25, 40, 45]  # TODO: better definition of those values
spectra_selection = False
n_spectra_selection_max = 150
median_correction = True
spectra_filtering = True
spectra_filtering_std = 2.5
plot_graphs = False
plot_include_all_spectra = False

print 'Reading data sets'
galah_data_input = '/home/klemen/GALAH_data/'
galah_data_output = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_input+'sobject_iraf_52_reduced.csv')

spectra_file = 'galah_dr52_ccd3_6475_6745_interpolated_wvlstep_0.06_spline_restframe.csv'

# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file)
wvl_values = csv_param.get_wvl_values()
wvl_limits = csv_param.get_wvl_range()
ccd_number = int(csv_param.get_ccd())

# determine csv outputs
suffix = '_teff_{:.0f}_logg_{:1.2f}_feh_{:1.2f}'.format(TEFF_SPAN, LOGG_SPAN, FEH_SPAN)
if snr_limits:
    suffix += '_snr_{:3.0f}'.format(snr_limits[ccd_number - 1])
if spectra_selection:
    suffix += '_best_{:3.0f}'.format(n_spectra_selection_max)
if median_correction:
    suffix += '_medianshift'
if spectra_filtering:
    suffix += '_std_{:1.1f}'.format(spectra_filtering_std)
# final output csv file
txt_out_spectra = spectra_file[:-4] + suffix + '.csv'

# change to output directory
ch_dir(galah_data_output)
ch_dir('Spectra_template_ccd{0}'.format(ccd_number))

galah_param = galah_param.filled()  # handle masked values

# read appropriate subset of data
print 'Reading resampled GALAH spectra'
print ' cols for ccd'+str(ccd_number)+': '+str(len(wvl_values))
spectral_data = pd.read_csv(galah_data_input + spectra_file, sep=',', header=None, na_values='nan').values

# create empty nan spectra
empty_spectra = np.ndarray(len(wvl_values))
empty_spectra.fill(np.nan)

if not RESUME_PROCESSING:
    print 'Writing initial outputs'
    # create outputs
    out_files = [txt_out_spectra]
    for out_file in out_files:
        new_txt_file(out_file)

print 'Template creation procedure started'
# time keeping
i_t = 1
total_sec = 0.
# plot some randomly selected spectra
idx_process = range(len(galah_param))

# determine the point where the process was terminated
if RESUME_PROCESSING:
    print 'Determining number of already processed objects'
    with open(txt_out_spectra) as foo:
        n_out_lines = len(foo.readlines())
    idx_process = idx_process[n_out_lines:]

# select data with high SNR
idx_snr_ok = galah_param['snr_c{0}_iraf'.format(ccd_number)] > snr_limits[ccd_number-1]

for i_r in idx_process:
    object_param = galah_param[i_r]
    sobj_id = object_param['sobject_id']
    print str(i_r+1)+':  '+str(sobj_id)

    # get object spectra
    spectra = spectral_data[i_r]

    time_start = time.time()
    # determine objects parameters
    obj_teff = object_param['teff_guess']
    obj_logg = object_param['logg_guess']
    obj_feh = object_param['feh_guess']

    # get objects that have similar physical parameters to the observed object
    idx_select = np.logical_and(np.abs(galah_param['logg_guess'] - obj_logg) < (LOGG_SPAN / 2.),
                                np.abs(galah_param['teff_guess'] - obj_teff) < (TEFF_SPAN / 2.))
    idx_select = np.logical_and(idx_select,
                                np.abs(galah_param['feh_guess'] - obj_feh) < (FEH_SPAN / 2.))

    n_similar = np.sum(idx_select)
    print ' Number of similar objects: ' + str(n_similar)
    n_used = n_similar
    if limit_snr:
        idx_select = np.logical_and(idx_select, idx_snr_ok)
        n_used = np.sum(idx_select)
        print ' After SNR limit {0}'.format(n_used)

    if n_similar < 5:
        print '  Skipping this object'
        # save empty spectra line to csv files
        append_line(txt_out_spectra, ','.join([str(flx) for flx in empty_spectra]), new_line=True)
    else:
        if spectra_selection:
            idx_select_use = euclidean_distance_filter(spectral_data[idx_select], spectra, n_spectra_selection_max)
            n_used = np.sum(idx_select_use)
            print ' After likens estimation {0}'.format(n_used)
        else:
            idx_select_use = None

        spectral_data_selected = spectral_data[refine_idx_selection(idx_select, idx_select_use)]

        if median_correction:
            # shift spectra in order to have the same median value as tho whole stack of spectra
            median_all = np.nanmedian(spectral_data_selected)
            for i_row in range(spectral_data_selected.shape[0]):
                median_row = np.nanmedian(spectral_data_selected[i_row])
                spectral_data_selected[i_row] += median_all-median_row

        if spectra_filtering:
            # remove outliers for every spectra pixel
            spectra_template = np.apply_along_axis(spectra_stat, 0, spectral_data_selected,
                                                   median=True, std=spectra_filtering_std)
        else:
            spectra_template = np.nanmedian(spectral_data_selected, axis=0)

        # save template spectra line to csv files
        if n_used >= 3:
            append_line(txt_out_spectra, ','.join([str(flx) for flx in spectra_template]), new_line=True)
        else:
            print ' Nan output spectra'
            append_line(txt_out_spectra, ','.join([str(flx) for flx in empty_spectra]), new_line=True)

        if plot_graphs:
            # plot results
            fig, axs = plt.subplots(2, 1)
            if plot_include_all_spectra:
                for i_row in range(spectral_data_selected.shape[0]):
                    axs[0].plot(wvl_values, spectral_data_selected[i_row], color='blue', linewidth=0.2, alpha=0.05)
            axs[0].plot(wvl_values, spectra_template, color='red', linewidth=0.8)
            axs[0].plot(wvl_values, spectra, color='black', linewidth=0.6)
            axs[0].set(xlim=wvl_limits, ylim=(0.1, 1.1), ylabel='Flux', title='Guess   ->   teff:{:4.0f}   logg:{:1.1f}   feh:{:1.1f}   spectra:{:.0f}'.format(obj_teff, obj_logg, obj_feh, n_used))
            axs[1].plot(wvl_values, spectra-spectra_template, color='black', linewidth=0.6)
            axs[1].set(xlim=wvl_limits, ylim=(-0.2, 0.2), ylabel='Subtracted flux', xlabel='Wavelength')
            plt.tight_layout()
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
