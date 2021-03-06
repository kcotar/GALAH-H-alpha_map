import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un
import time, datetime

from astropy.table import Table, join
from halpha_renormalize_data_functions import ch_dir, append_line, new_txt_file, refine_idx_selection, spectra_stat

imp.load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import *

# if processing should be resumed from the endpoint
RESUME_PROCESSING = False
# some program setting
TEFF_SPAN = 100.  # span represents the range of parameters around the grid values
TEFF_STEP = 75.  # step between grid values
LOGG_SPAN = 0.2
LOGG_STEP = 0.1
FEH_SPAN = 0.2
FEH_STEP = 0.1
limit_snr = True
# TODO: better definition of SNR limits values
# snr_limits = [20, 30, 40, 45]  # for guess snr values (snr_c1_guess)
snr_limits = [10, 15, 25, 30]  # for iraf snr values (snr_c1_iraf)
spectra_selection = False  # should be always set to FALSE as it is
n_spectra_selection_max = 150  # not possible for this kind of template
median_correction = False
spectra_filtering = True
spectra_filtering_std = 3.
rv_filtering = True
rv_filtering_e = 3  # km/s
flags_filtering = True
plot_graphs = True
plot_include_all_spectra = False

print 'Reading data sets'
galah_data_input = '/home/klemen/data4_mount/'
galah_data_output = '/home/klemen/data4_mount/'
galah_param = Table.read(galah_data_input+'sobject_iraf_52_reduced_20171111.fits')
cannon_param = Table.read(galah_data_input+'sobject_iraf_iDR2_171103_cannon.fits')['sobject_id', 'Teff_cannon', 'Feh_cannon', 'Logg_cannon', 'flag_cannon']
cannon_param = cannon_param[cannon_param['flag_cannon'] == 0]

print len(galah_param), 'Before join'
galah_param = join(galah_param, cannon_param, keys='sobject_id', join_type='left')
print len(galah_param), 'After join'


spectra_file = 'galah_dr52_ccd3_6475_6745_wvlstep_0.030_ext0_renorm_20171111.pkl'

# parse resampling settings from filename
csv_param = CollectionParameters(spectra_file)
wvl_values = csv_param.get_wvl_values()
wvl_limits = csv_param.get_wvl_range()
ccd_number = int(csv_param.get_ccd())

# determine csv outputs
suffix = 'Teff_{:.0f}_logg_{:1.2f}_feh_{:1.2f}'.format(TEFF_SPAN, LOGG_SPAN, FEH_SPAN)
if snr_limits:
    suffix += '_snr_{:.0f}'.format(snr_limits[ccd_number - 1])
# if spectra_selection:
#     suffix += '_best_{:3.0f}'.format(n_spectra_selection_max)
if median_correction:
    suffix += '_medianshift'
if spectra_filtering:
    suffix += '_std_{:.1f}'.format(spectra_filtering_std)
if flags_filtering:
    suffix += '_redflag'
    # 0 for no flags
    # +1 for bad wavelength solution in ccd_1
    # +2 for bad wavelength solution in ccd_2
    # +4 for ccd_3
    # +8 for ccd_4
    # +16 for molecfit fail in ccd_3
    # +32 for molecfit fail in ccd_4. Molecfit is not used for ccd_1 and ccd_2
    # +64 if the object is actually a twilight flat
    # ccd_badwvl_flag = [1, 2, 4, 8]
    # mollecfit_fail = [128, 128, 16, 32]
    # idx_red_flag = np.bitwise_and(galah_param['red_flag'], ccd_badwvl_flag[ccd_number - 1]) == ccd_badwvl_flag[ccd_number - 1]
    # idx_red_flag = np.logical_or(idx_red_flag,
    #                              np.bitwise_and(galah_param['red_flag'], mollecfit_fail[ccd_number - 1]) == mollecfit_fail[ccd_number - 1])
    # idx_red_flag = np.logical_or(idx_red_flag,
    #                              np.bitwise_and(galah_param['red_flag'], 64) == 64)
    idx_red_flag = galah_param['red_flag'] != 0
    print 'Number of removed spectra by reduction flag: '+str(np.sum(idx_red_flag))

# select data with high SNR
idx_snr_ok = galah_param['snr_c{0}_iraf'.format(ccd_number)] > snr_limits[ccd_number-1]
if rv_filtering:
    idx_snr_ok = np.logical_and(idx_snr_ok, galah_param['e_rv_guess'] > rv_filtering_e)
print 'Number of object with bad snr or rv error value: '+str(np.sum(idx_snr_ok))
idx_bad = np.logical_or(idx_red_flag, idx_snr_ok)
idx_ok = np.logical_not(idx_bad)
print 'Final number of spectra that will be used: '+str(np.sum(idx_ok))

# change to output directory
ch_dir(galah_data_output)
# first subdirectory describing input dataset - parsed from input spectra file
ch_dir(spectra_file[:-4])
# second subdirectory describing
ch_dir(suffix)

galah_param = galah_param.filled()  # handle masked values

# read appropriate subset of data
print 'Reading resampled GALAH spectra'
print ' cols for ccd'+str(ccd_number)+': '+str(len(wvl_values))
if flags_filtering:
    # remove or skip from reading flagged data that should not be used
    spectral_data = read_pkl_spectra(galah_data_input+spectra_file, read_rows=np.where(idx_ok)[0], read_cols=None)
    print 'Removing flagged data'
    galah_param = galah_param[idx_ok]
else:
    spectral_data = read_pkl_spectra(galah_data_input+spectra_file, read_rows=None, read_cols=None)

print 'Template creation procedure started'
# time keeping
i_t = 1
total_sec = 0.

# define template spectra grid values
TEFF_GRID = np.arange(3000, 8200, TEFF_STEP)
LOGG_GRID = np.arange(0, 7, LOGG_STEP)
FEH_GRID = np.arange(-2.5, 0.5, FEH_STEP)

n_grid_points = len(TEFF_GRID) * len(LOGG_GRID) * len(FEH_GRID)
print 'Number of grid points: '+str(n_grid_points)

# create a list of output wavelengths
grid_list_csv = 'wvl_list.csv'
txt_file = open(grid_list_csv, 'w')
txt_file.write(','.join([str(f) for f in wvl_values]))
txt_file.close()

# file with template spectra list
grid_list_csv = 'grid_list.csv'
if not RESUME_PROCESSING:
    new_txt_file(grid_list_csv)
    append_line(grid_list_csv, 'teff,logg,feh,n_spectra', new_line=True)

for obj_teff in TEFF_GRID:
    for obj_logg in LOGG_GRID:
        for obj_feh in FEH_GRID:
            grid_id = 'T_{:.0f}_L_{:1.2f}_F_{:1.2f}'.format(obj_teff, obj_logg, obj_feh)
            print str(i_t)+':  '+grid_id

            # determine name of the output files
            out_csv = grid_id + '.csv'
            out_png = grid_id + '.png'

            # resume option
            if RESUME_PROCESSING:
                if os.path.isfile():
                    i_t += 1
                    continue

            time_start = time.time()

            # get objects that have similar physical parameters to the grid object
            idx_select = np.logical_and(np.abs(galah_param['Logg_cannon'] - obj_logg) < (LOGG_SPAN / 2.),
                                        np.abs(galah_param['Teff_cannon'] - obj_teff) < (TEFF_SPAN / 2.))
            idx_select = np.logical_and(idx_select,
                                        np.abs(galah_param['Feh_cannon'] - obj_feh) < (FEH_SPAN / 2.))

            n_similar = np.sum(idx_select)
            print ' Number of objects: ' + str(n_similar)
            n_used = n_similar
            # NOTE: The following lines were removed as this is already done in the data reading stage
            # if limit_snr:
            #     idx_select = np.logical_and(idx_select, idx_snr_ok)
            #     n_used = np.sum(idx_select)
            #     print ' After SNR limit {0}'.format(n_used)

            if n_similar < 10:
                print '  Skipping this object'
                # do not save any results
            else:
                if spectra_selection:
                    # Select only few spectra with the best match to some spectra
                    # idx_select_use = euclidean_distance_filter(spectral_data[idx_select], spectra, n_spectra_selection_max)
                    # n_used = np.sum(idx_select_use)
                    # print ' After likens estimation {0}'.format(n_used)
                    pass
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
                    # TODO: optional weights based on per-pixel noise information from original fits file, resampling problem
                    spectra_template = np.nanmedian(spectral_data_selected, axis=0)

                # save template spectra line to csv files
                if n_used >= 3:
                    # write outputs
                    append_line(out_csv, ','.join([str(flx) for flx in spectra_template]), new_line=False)
                    append_line(grid_list_csv, '{:.0f},{:1.2f},{:1.2f},{:.0f}'.format(obj_teff, obj_logg, obj_feh, n_used), new_line=True)
                    # add graphs
                    if plot_graphs:
                        # plot results
                        fig, axs = plt.subplots(2, 1)
                        if plot_include_all_spectra:
                            for i_row in range(spectral_data_selected.shape[0]):
                                axs[0].plot(wvl_values, spectral_data_selected[i_row], color='blue', linewidth=0.2,
                                            alpha=0.05)
                        axs[0].plot(wvl_values, spectra_template, color='red', linewidth=0.8)
                        axs[0].set(xlim=wvl_limits, ylim=(0.1, 1.1), ylabel='Flux',
                                   title='Guess   ->   teff:{:4.0f}   logg:{:1.1f}   feh:{:1.1f}   spectra:{:.0f}'.format(
                                       obj_teff, obj_logg, obj_feh, n_used))
                        plt.tight_layout()
                        plt.savefig(out_png, dpi=150)
                        plt.close()
                else:
                    # do not save any results
                    pass

            time_end = time.time()
            # Create time estimation outputs
            time_delta = time_end - time_start
            print 'Fit time: ' + str(datetime.timedelta(seconds=time_delta))
            total_sec += time_delta
            time_to_end = total_sec / i_t * (n_grid_points - i_t)
            print 'Estimated finished in: ' + str(datetime.timedelta(seconds=time_to_end))
            i_t += 1
            print ''
