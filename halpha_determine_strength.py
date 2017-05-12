import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un

from astropy.table import Table
from halpha_renormalize_data_functions import ch_dir, spectra_stat, mask_outliers_rows


print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv')

ch_dir('H_spectra_normalization_3')

txt_out_sobject = 'report_sobject_ids.csv'
txt_out_wvl1 = 'report_wavelengths_ccd1.csv'
txt_out_wvl3 = 'report_wavelengths_ccd3.csv'
txt_out_spectra1 = 'normalized_spectra_ccd1.csv'
txt_out_spectra3 = 'normalized_spectra_ccd3.csv'

sobjects_normalized = pd.read_csv(txt_out_sobject, sep=',', header=None).values[0]
wvl_ccd1 = np.loadtxt(txt_out_wvl1, delimiter=',')
hbeta_lim = (np.min(wvl_ccd1), np.max(wvl_ccd1))
wvl_ccd3 = np.loadtxt(txt_out_wvl3, delimiter=',')
halpha_lim = (np.min(wvl_ccd3), np.max(wvl_ccd3))
# normalized_data_ccd1 = pd.read_csv(txt_out_spectra1, header=None, sep=',', na_values='nan').values
normalized_data_ccd3 = pd.read_csv(txt_out_spectra3, header=None, sep=',', na_values='nan').values

n_lines = normalized_data_ccd3.shape[0]

# select normalized objects from the complete set
idx_use = np.in1d(galah_param['sobject_id'].data, sobjects_normalized[:n_lines])
galah_param_norm = galah_param[idx_use].filled()

ra_dec = coord.ICRS(ra=galah_param_norm['ra'].data*un.deg,
                    dec=galah_param_norm['dec'].data*un.deg)

all_mean_halpha = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3)
# all_mean_hbeta = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd1)
filtered = mask_outliers_rows(normalized_data_ccd3)
ch_dir('Stacked_plots')

# idx_plot = np.logical_and(all_mean_halpha <= 0.02,
#                           all_mean_halpha >= -0.02)
plt.plot(wvl_ccd3, all_mean_halpha, color='black', linewidth=0.75)
plt.xlim(halpha_lim)
plt.ylim((-0.05, 0.05))
plt.savefig('halpha_all.png', dpi=350)
plt.close()

# create a meshgrid of ra/dec coordinates
max_dist = 20 * un.deg
for ra_center in np.arange(10, 360, 30):
    for dec_center in np.arange(-80, 90, 20):
        print ra_center, dec_center
        ra_dec_center = coord.ICRS(ra=ra_center * un.deg,
                                   dec=dec_center * un.deg)
        dist_center = ra_dec.separation(ra_dec_center)
        idx_in_field = dist_center < max_dist

        # compute statistics
        n_in_filed = np.sum(idx_in_field)
        print n_in_filed
        if n_in_filed <= 0:
            continue
        field_mean_halpha = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :])
        # field_std_halpha = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'spread':True})
        # field_mean_hbeta = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd1[idx_in_field, :])
        # field_std_hbeta = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd1[idx_in_field, :], **{'spread':True})

        # plot 1
        plt.plot(wvl_ccd3, all_mean_halpha - field_mean_halpha, color='red', alpha=0.8)
        # idx_plot = np.logical_and(field_mean_halpha <= 0.02,
        #                           field_mean_halpha >= -0.02)
        # plt.fill_between(wvl_ccd3, np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'percentile':90}),
        #                            np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'percentile':10}),
        #                  color='0.8')
        # plt.fill_between(wvl_ccd3, np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'percentile':70}),
        #                            np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'percentile':30}),
        #                  color='0.4')
        plt.plot(wvl_ccd3, field_mean_halpha, color='black', linewidth=0.6)
        # plt.plot(wvl_ccd3, all_mean_halpha / field_mean_halpha, color='blue')
        # plt.errorbar(wvl_ccd3, field_mean_halpha, yerr=field_std_halpha, markersize=0, elinewidth=0.01, color='black', errorevery=10)
        plt.xlim(halpha_lim)
        plt.ylim((-0.05, 0.05))
        plt.savefig('halpha_ra{:03.1f}_dec{:03.1f}.png'.format(ra_center, dec_center), dpi=350)
        plt.close()

        # plot 2
        # idx_plot = np.logical_and(field_mean_hbeta <= 0.02,
        #                           field_mean_hbeta >= -0.02)
        # plt.plot(wvl_ccd1, field_mean_hbeta, color='black')
        # plt.plot(wvl_ccd1, all_mean_hbeta - field_mean_hbeta, color='blue')
        # plt.plot(wvl_ccd1, all_mean_hbeta / field_mean_hbeta, color='blue')
        # plt.errorbar(wvl_ccd1, field_mean_hbeta, yerr=field_std_hbeta, markersize=0)
        # plt.xlim(hbeta_lim)
        # plt.ylim((-0.05, 0.05))
        # plt.savefig('hbeta_ra{:03.1f}_dec{:03.1f}.png'.format(ra_center, dec_center), dpi=350)
        # plt.close()

