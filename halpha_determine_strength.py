import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un

from astropy.table import Table



def spectra_stat(spect_col):
    # remove outliers
    mean = np.nanmean(spect_col)
    std = np.nanstd(spect_col)
    idx_use = np.abs(spect_col - mean) < (std * 2.)
    return np.nanmean(spect_col[idx_use])

print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv')

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
# normalized_data_ccd3 = pd.read_csv(txt_out_spectra3, header=None, sep=',', na_values='nan').values

# select normalized objects from the complete set
idx_use = np.in1d(galah_param['sobject_id'].data, sobjects_normalized)
galah_param_norm = galah_param[idx_use].filled()

ra_dec = coord.ICRS(ra=galah_param['ra'].data*un.deg,
                    dec=galah_param['dec'].data*un.deg)

# create a meshgrid of ra/dec coordinates
max_dist = 30 * un.deg
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
        field_mean_hbeta = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd1[idx_in_field, :])
        # plot 1
        plt.plot(wvl_ccd3, field_mean_halpha)
        plt.xlim(halpha_lim)
        plt.ylim((-0.1, 0.1))
        plt.savefig('halpha_ra{:f03.1}_dec{:f03.1}.png'.format(ra_center, dec_center), dpi=350)
        plt.close()
        # plot 2
        plt.plot(wvl_ccd1, field_mean_hbeta)
        plt.xlim(halpha_lim)
        plt.ylim((-0.1, 0.1))
        plt.savefig('halpha_ra{:f03.1}_dec{:f03.1}.png'.format(ra_center, dec_center), dpi=350)
        plt.close()

