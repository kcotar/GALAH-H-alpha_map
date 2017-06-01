import os, imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as un
import healpy as hp

from astropy.table import Table
from halpha_renormalize_data_functions import ch_dir, spectra_stat, mask_outliers_rows, spectra_resample


print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_52_reduced.csv')
dibs = Table.read('dibs.csv', format='ascii.csv')

ch_dir('H_flux_template_grid_0.03_renorm')
# ch_dir('_test')


C_LIGHT = 299792458  # m/s
to_barycentric = True
txt_out_sobject = 'report_sobject_ids.csv'
txt_out_wvl1 = 'report_wavelengths_ccd1.csv'
txt_out_wvl3 = 'report_wavelengths_ccd3.csv'
txt_out_spectra1 = 'residuum_spectra_ccd1.csv'
txt_out_spectra3 = 'residuum_spectra_ccd3.csv'

sobjects_normalized = pd.read_csv(txt_out_sobject, sep=',', header=None).values[0]
# wvl_ccd1 = np.loadtxt(txt_out_wvl1, delimiter=',')
# hbeta_lim = (np.min(wvl_ccd1), np.max(wvl_ccd1))
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
l_b = ra_dec.transform_to(coord.Galactic)

if to_barycentric:
    # filter out dibs in the observed field
    dibs = dibs[np.logical_and(dibs['W_center']>=np.min(wvl_ccd3), dibs['W_center']<=np.max(wvl_ccd3))]
    # shift all data to barycentric system
    print 'Shifting all observed spectra to barycentric system'
    for i_r in range(len(galah_param_norm)):
        if i_r % 10000 == 0:
            print i_r
        galah_obj = galah_param_norm[i_r]
        velocity_shift = galah_obj['rv_guess_shift']
        wvl_shifted = wvl_ccd3 * (1 + velocity_shift * 1000. / C_LIGHT)
        spectra_obj = normalized_data_ccd3[i_r, :]
        spectra_obj_bary = spectra_resample(spectra_obj, wvl_shifted, wvl_ccd3)
        normalized_data_ccd3[i_r, :] = spectra_obj_bary
    print 'Finished'


all_mean_halpha = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3, std=3)
# all_mean_hbeta = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd1)
filtered = mask_outliers_rows(normalized_data_ccd3)

out_dir = 'Stacked_plots'
if to_barycentric:
    out_dir += '_barycentric'
ch_dir(out_dir)

# idx_plot = np.logical_and(all_mean_halpha <= 0.02,
#                           all_mean_halpha >= -0.02)
plt.plot(wvl_ccd3, all_mean_halpha, color='black', linewidth=0.75)
plt.xlim(halpha_lim)
if to_barycentric:
    for line_wvl in dibs['W_center']:
        plt.axvline(x=line_wvl, color='blue', alpha=0.8, linewidth=0.5)
plt.ylim((-0.05, 0.05))
plt.grid(True, linestyle='--')
plt.savefig('halpha_all.png', dpi=350)
plt.close()
# step = 20.
# max_dist = step/2. * un.deg
# for l_center in np.arange(0, 360, step):
#     for b_center in np.arange(-80, 90, step):
#         print l_center, b_center
#         # in field based on radius search around the center
#         l_b_center = coord.Galactic(l=l_center * un.deg,
#                                     b=b_center * un.deg)
#         dist_center = l_b.separation(l_b_center)
#         idx_in_field = dist_center < max_dist
#         # OR in field based on bined search
#         # idx_in_field = dist_center < max_dist


# create a meshgrid of ra/dec coordinates
n_side = 8
n_pix = hp.nside2npix(n_side)
pix_object = hp.ang2pix(n_side, np.deg2rad(np.abs(l_b.b.value - 90.)), np.deg2rad(l_b.l.value))
line_strength = np.ndarray(n_pix, dtype=np.float64)
line_strength.fill(-1.6375e+30)  # healpy unseen pixel value
idx_line = np.abs(wvl_ccd3 - 6613.9) <= 0.15
print np.sum(idx_line)

for pix in list([466]): # np.unique(pix_object):
    idx_in_field = pix_object == pix
    b_center, l_center = np.rad2deg(hp.pix2ang(n_side, pix))
    print l_center, b_center
    # compute statistics
    n_in_filed = np.sum(idx_in_field)
    print n_in_filed
    if n_in_filed <= 0:
        continue
    field_mean_halpha = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], std=3)
    # field_std_halpha = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'spread':True})
    # field_mean_hbeta = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd1[idx_in_field, :])
    # field_std_hbeta = np.apply_along_axis(spectra_stat, 0, normalized_data_ccd1[idx_in_field, :], **{'spread':True})

    # plot 1
    plt.plot(wvl_ccd3, all_mean_halpha - field_mean_halpha - 0.02, color='red', alpha=0.5)
    # idx_plot = np.logical_and(field_mean_halpha <= 0.02,
    #                           field_mean_halpha >= -0.02)
    # plt.fill_between(wvl_ccd3, np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'percentile':90}),
    #                            np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'percentile':10}),
    #                  color='0.8')
    plt.fill_between(wvl_ccd3, np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'percentile':70}),
                               np.apply_along_axis(spectra_stat, 0, normalized_data_ccd3[idx_in_field, :], **{'percentile':30}),
                     color='0.4')
    plt.plot(wvl_ccd3, field_mean_halpha, color='black', linewidth=0.5)
    # plt.plot(wvl_ccd3, all_mean_halpha / field_mean_halpha, color='blue')
    # plt.errorbar(wvl_ccd3, field_mean_halpha, yerr=field_std_halpha, markersize=0, elinewidth=0.01, color='black', errorevery=10)
    plt.xlim(halpha_lim)
    if to_barycentric:
        for line_wvl in dibs['W_center']:
            plt.axvline(x=line_wvl, color='blue', alpha=0.8, linewidth=0.5)
    plt.ylim((-0.05, 0.05))
    plt.grid(True, linestyle='--')
    plt.title('H-alpha  l:{:05.1f}  b:{:05.1f}  n:{:.0f}'.format(l_center, b_center, n_in_filed))
    plt.savefig('halpha_l{:05.1f}_b{:05.1f}_pix{:.0f}.png'.format(l_center, b_center, pix), dpi=350)
    plt.close()

    line_strength[pix] = np.nanmedian(field_mean_halpha[idx_line])

    i_p = 0
    for i_s in np.where(idx_in_field)[0]:
        plt.plot(wvl_ccd3, normalized_data_ccd3[i_s,:] + i_p*0.02, color='black', linewidth=0.5)
        i_p += 1
        if i_p >= 20:
            plt.show()
            plt.close()
            i_p = 0


hp.mollview(line_strength, min=-0.01, max=0.01)
plt.savefig('line_strength_map.png', dpi=350)
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

