import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un

from astropy.table import Table
from halpha_renormalize_data_functions import *
from template_spectra_function import *

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import CollectionParameters


# Functions
def is_in_emission(spectra, wvl, wvl_center, span=0.5):
    idx_peak = np.abs(wvl - wvl_center) < span
    mean_peak = np.mean(np.abs(spectra[idx_peak]))
    return mean_peak > PEAK_THR


def turnover_around_center(spectra, wvl, wvl_center, span=0.5):
    # TODO
    return False


# if processing should be resumed from the endpoint
RESUME_PROCESSING = False
USE_TEMPLATE_GRID = True
# constants
PEAK_THR = 0.08
STD_MAX = 0.03

# determine csv outputs
txt_out_sobject = 'report_sobject_ids.csv'
txt_out_wvl1 = 'report_wavelengths_ccd1.csv'
txt_out_wvl3 = 'report_wavelengths_ccd3.csv'
txt_out_spectra1 = 'residuum_spectra_ccd1.csv'
txt_out_spectra3 = 'residuum_spectra_ccd3.csv'

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_template_dir = '/home/klemen/GALAH_data/Spectra_template/'
galah_grid_dir_ccd1 = '/home/klemen/GALAH_data/Spectra_template_grid/galah_dr52_ccd1_4710_4910_wvlstep_0.02_lin_RF_renorm/Teff_300_logg_0.50_feh_0.20_snr_15_medianshift_std_2.5_redflag/'
galah_grid_dir_ccd3 = '/home/klemen/GALAH_data/Spectra_template_grid/galah_dr52_ccd3_6475_6745_wvlstep_0.03_lin_RF_renorm/Teff_300_logg_0.50_feh_0.20_snr_40_medianshift_std_2.5_redflag/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv')

spectra_file_ccd1 = 'galah_dr52_ccd1_4710_4910_wvlstep_0.02_lin_RF_renorm.csv'
spectra_file_ccd3 = 'galah_dr52_ccd3_6475_6745_wvlstep_0.03_lin_RF_renorm.csv'
template_file_ccd3 = ''
# parse resampling settings from filename
csv_param_ccd1 = CollectionParameters(spectra_file_ccd1)
wvl_values_ccd1 = csv_param_ccd1.get_wvl_values()
csv_param_ccd3 = CollectionParameters(spectra_file_ccd3)
wvl_values_ccd3 = csv_param_ccd3.get_wvl_values()

# change to output directory
out_dir = 'H_flux_template_grid_alpha_beta_complete_renorm'
ch_dir(out_dir)

# object selection criteria
# must be a giant - objects are further away than dwarfs
print 'Number of all objects: '+str(len(galah_param))
if not RESUME_PROCESSING:
    # OLD definition of giant
    # idx_object_use = galah_param['logg_guess'] < 3.5
    # NEW definition of giants
    idx_object_use = (-2./1500.*galah_param['teff_guess'] + 10) > galah_param['logg_guess']

    idx_object_use = np.logical_and(idx_object_use,
                                    galah_param['snr_c3_guess'] > 50)
    # spectra reduction quality cuts
    idx_object_use = np.logical_and(idx_object_use,
                                    np.bitwise_and(galah_param['red_flag'], 1) == 0)  # remove bad ccd1 wvl reductions
    idx_object_use = np.logical_and(idx_object_use,
                                    np.bitwise_and(galah_param['red_flag'], 4) == 0)  # remove bad ccd3 wvl reductions
    idx_object_use = np.logical_and(idx_object_use,
                                    np.bitwise_and(galah_param['red_flag'], 16) == 0)  # remove failed molecfit in ccd3
    idx_object_use = np.logical_and(idx_object_use,
                                    np.bitwise_and(galah_param['red_flag'], 64) == 0)  # remove twilight flat
else:
    sobjects_used = pd.read_csv(txt_out_sobject, sep=',', header=None).values[0]
    idx_object_use = np.in1d(galah_param['sobject_id'].data, sobjects_used)

print 'Number of used objects: '+str(np.sum(idx_object_use))
# create a subset of input object tables
galah_param = galah_param[idx_object_use].filled()  # handle masked values
# convert ra dec to coordinates object
ra_dec = coord.ICRS(ra=galah_param['ra'].data*un.deg,
                    dec=galah_param['dec'].data*un.deg)

wvl_read_range = 250
wvl_plot_range = 40

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

if USE_TEMPLATE_GRID:
    template_grid_list_ccd1 = Table.read(galah_grid_dir_ccd1 + 'grid_list.csv', format='ascii.csv')
    template_grid_list_ccd3 = Table.read(galah_grid_dir_ccd3 + 'grid_list.csv', format='ascii.csv')
else:
    print 'Reading template GALAH spectra'
    template_ccd3_data = pd.read_csv(galah_template_dir + template_file_ccd3, sep=',', header=None, na_values='nan',
                                     usecols=idx_alpha[0], skiprows=np.where(np.logical_not(idx_object_use))[0]).values

if not RESUME_PROCESSING:
    print 'Writing initial outputs'
    # create outputs
    out_files = [txt_out_sobject, txt_out_wvl1, txt_out_wvl3, txt_out_spectra1, txt_out_spectra3]
    for out_file in out_files:
        new_txt_file(out_file)

    # write outputs
    append_line(txt_out_wvl1, ','.join([str(wvl) for wvl in wvl_read_ccd1]))
    append_line(txt_out_wvl3, ','.join([str(wvl) for wvl in wvl_read_ccd3]))

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

# gather data
for i_r in idx_process:
    object_param = galah_param[i_r]
    sobj_id = object_param['sobject_id']
    print str(i_r+1)+':  '+str(sobj_id)

    spectra_ccd1 = ccd1_data[i_r]
    spectra_ccd3 = ccd3_data[i_r]
    if USE_TEMPLATE_GRID:
        template_file_ccd1 = get_best_match(object_param['teff_guess'], object_param['logg_guess'],
                                            object_param['feh_guess'], template_grid_list_ccd1, midpoint=False) + '.csv'
        template_file_ccd3 = get_best_match(object_param['teff_guess'], object_param['logg_guess'],
                                            object_param['feh_guess'], template_grid_list_ccd3, midpoint=False) + '.csv'
        template_ccd1 = np.loadtxt(galah_grid_dir_ccd1 + template_file_ccd1, delimiter=',')[idx_beta]
        template_ccd3 = np.loadtxt(galah_grid_dir_ccd3 + template_file_ccd3, delimiter=',')[idx_alpha]
    else:
        template_ccd3 = template_ccd3_data[i_r]

    # subtract spectra
    residuum_spectra_ccd1 = spectra_ccd1 - template_ccd1
    residuum_spectra_ccd3 = spectra_ccd3 - template_ccd3

    # detection of strange fluxes compared to grid
    # mark and exclude them from the final resulting table of residuum fluxes
    std_residuum_ccd3 = np.nanstd(residuum_spectra_ccd3)
    large_std_ccd = std_residuum_ccd3 > STD_MAX
    print 'Std residuum ccd3: '+str(std_residuum_ccd3)
    # determine emission around
    emission_ccd3 = is_in_emission(residuum_spectra_ccd3, wvl_read_ccd3, HALPHA_WVL)

    if emission_ccd3 or large_std_ccd:
        possible_bad = True
        print ' BAD or STRANGE'
    else:
        possible_bad = False

    # save renormalized results to csv file
    if not possible_bad:
        append_line(txt_out_spectra3, ','.join([str(flx) for flx in residuum_spectra_ccd3]), new_line=True)
        append_line(txt_out_spectra1, ','.join([str(flx) for flx in residuum_spectra_ccd1]), new_line=True)
    else:
        append_line(txt_out_spectra1, ','.join([str(flx) for flx in np.full_like(residuum_spectra_ccd1, np.nan)]), new_line=True)
        append_line(txt_out_spectra3, ','.join([str(flx) for flx in np.full_like(residuum_spectra_ccd3, np.nan)]), new_line=True)

    # plot results
    if possible_bad:
        fig, axs = plt.subplots(2, 2)
        suptitle = 'Guess  ->  teff:{:4.0f}  logg:{:1.1f}  feh:{:1.1f}'.format(object_param['teff_guess'],
                                                                               object_param['logg_guess'],
                                                                               object_param['feh_guess'])
        # if USE_TEMPLATE_GRID:
        #     suptitle += ', template: '
        fig.suptitle(suptitle)
        # h-alpha plots
        axs[0, 0].plot(wvl_read_ccd3, template_ccd3, color='red', linewidth=0.5)
        axs[0, 0].plot(wvl_read_ccd3, spectra_ccd3, color='black', linewidth=0.4)
        # axs[0, 0].plot(wvl_read_ccd3, spectra_ccd3_filtered, color='blue', linewidth=0.4)
        axs[0, 0].set(xlim=(HALPHA_WVL-wvl_plot_range, HALPHA_WVL+wvl_plot_range), ylim=(0.1, 1.1), title='H-alpha t:'+template_file_ccd1[:-4], ylabel='Flux')
        axs[1, 0].plot(wvl_read_ccd3, residuum_spectra_ccd3, color='black', linewidth=0.5)
        axs[1, 0].set(xlim=(HALPHA_WVL-wvl_plot_range, HALPHA_WVL+wvl_plot_range), ylim=(-0.2, 0.2), ylabel='Subtracted flux', xlabel='Wavelength')

        # # temp h-alpha plots aka zoom plots
        # axs[0, 1].plot(wvl_read_ccd3, template_ccd3, color='red', linewidth=0.5)
        # axs[0, 1].plot(wvl_read_ccd3, spectra_ccd3, color='black', linewidth=0.4)
        # # axs[0, 0].plot(wvl_read_ccd3, spectra_ccd3_filtered, color='blue', linewidth=0.4)
        # axs[0, 1].set(xlim=(HALPHA_WVL-5, HALPHA_WVL+5), ylim=(0.1, 1.1), title='H-alpha')
        # axs[1, 1].plot(wvl_read_ccd3, residuum_spectra_ccd3, color='black', linewidth=0.5)
        # axs[1, 1].set(xlim=(HALPHA_WVL-5, HALPHA_WVL+5), ylim=(-0.2, 0.2), xlabel='Wavelength')

        # h-beta plots
        axs[0, 1].plot(wvl_read_ccd1, template_ccd1, color='red', linewidth=0.75)
        axs[0, 1].plot(wvl_read_ccd1, spectra_ccd1, color='black', linewidth=0.4)
        # axs[0, 1].plot(wvl_read_ccd1, spectra_ccd1_filtered, color='blue', linewidth=0.4)
        axs[0, 1].set(xlim=(HBETA_WVL-wvl_plot_range, HBETA_WVL+wvl_plot_range), ylim=(0.1, 1.1), title='H-beta t:'+template_file_ccd1[:-4])
        axs[1, 1].plot(wvl_read_ccd1, residuum_spectra_ccd1, color='black', linewidth=0.5)
        axs[1, 1].set(xlim=(HBETA_WVL-wvl_plot_range, HBETA_WVL+wvl_plot_range), ylim=(-0.2, 0.2), xlabel='Wavelength')
        plt.tight_layout()
        plt.savefig(str(sobj_id) + '.png', dpi=200)
        plt.close()
    print ''

# merge residuum data with some statistics
# USE: halpha_determine_strength.py
