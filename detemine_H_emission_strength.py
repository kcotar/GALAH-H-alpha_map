import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 14

from os import chdir
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, join
from scipy.signal import savgol_filter, argrelextrema, decimate, bspline
from scipy.stats import skew, kurtosis

from helper_functions import move_to_dir
from spectra_collection_functions import *

from multiprocessing import Pool
from joblib import Parallel, delayed
from functools import partial
from socket import gethostname
# PC hostname
pc_name = gethostname()


# --------------------------------------------------------
# ---------------- Read data -----------------------------
# --------------------------------------------------------
print 'Reading GALAH parameters'
date_string = '20190801'
snr_limit = 5.
remove_spikes = False

n_multi = 4
galah_data_dir = '/shared/ebla/cotar/'
out_dir = '/shared/camelot/data-camelot/'

general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')

spectra_ccd1 = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl'
median_spectra_ccd1 = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'_median_250_snr_15_teff_150_logg_0.20_feh_0.10.pkl'
spectra_ccd3 = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl'
median_spectra_ccd3 = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'_median_250_snr_15_teff_150_logg_0.20_feh_0.10.pkl'

# parse interpolation and averaging settings from filename
ccd1_wvl = CollectionParameters(spectra_ccd1).get_wvl_values()
ccd3_wvl = CollectionParameters(spectra_ccd3).get_wvl_values()


# determine wvls that will be read from the spectra
wvl_read_range = 50
wvl_plot_range = 40
wvl_int_range = 5
HBETA_WVL = 4861.36
HALPHA_WVL = 6562.81

idx_read_ccd1 = np.where(np.logical_and(ccd1_wvl >= HBETA_WVL - wvl_read_range,
                                        ccd1_wvl <= HBETA_WVL + wvl_read_range))[0]
idx_read_ccd3 = np.where(np.logical_and(ccd3_wvl >= HALPHA_WVL - wvl_read_range,
                                        ccd3_wvl <= HALPHA_WVL + wvl_read_range))[0]

wvl_val_ccd1 = ccd1_wvl[idx_read_ccd1]
wvl_val_ccd3 = ccd3_wvl[idx_read_ccd3]
dwvl_ccd1 = wvl_val_ccd1[1] - wvl_val_ccd1[0]
dwvl_ccd3 = wvl_val_ccd3[1] - wvl_val_ccd3[0]

# read limited number of columns instead of full spectral dataset
print 'Reading resampled/interpolated GALAH spectra'
spectra_ccd1 = read_pkl_spectra(galah_data_dir + spectra_ccd1, read_cols=idx_read_ccd1)
spectra_ccd3 = read_pkl_spectra(galah_data_dir + spectra_ccd3, read_cols=idx_read_ccd3)
print 'Reading merged median GALAH spectra'
spectra_median_ccd1 = read_pkl_spectra(galah_data_dir + median_spectra_ccd1, read_cols=idx_read_ccd1)
spectra_median_ccd3 = read_pkl_spectra(galah_data_dir + median_spectra_ccd3, read_cols=idx_read_ccd3)

# select initial data by parameters
idx_object_ok = general_data['sobject_id'] > 0  # can filter by date even later
idx_object_ok = np.logical_and(idx_object_ok, np.bitwise_and(general_data['red_flag'], 64) == 0)  # remove twilight flats

# --------------------------------------------------------
# ---------------- Determine objects to be observed ------
# --------------------------------------------------------

# determine object sobject_id numbers
sobject_ids = general_data[idx_object_ok]['sobject_id']

move_to_dir(out_dir+'H_band_strength_all_'+date_string)

# binary flag describing processing step where something went wrong:
# 1000 or 8 =
# 0100 or 4 =
# 0010 or 2 =
# 0001 or 1 = No reference spectrum in one of the bands
results = Table(names=('sobject_id', 'Ha_EW', 'Hb_EW', 'Ha_wvl', 'Hb_wvl', 'x1', 'x2', 'x3', 'x4', 'flag'),
                dtype=('int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int32'))
n_cols_out = len(results.columns)

# --------------------------------------------------------
# ---------------- Main analysis -------------------------
# --------------------------------------------------------
print 'Number of spectra that will be evaluated:', len(sobject_ids)


def integrate_ew_spectra(spectra_data, wvl_data, wvl_range=None, offset=0.):
    if wvl_range is not None:
        # integrate part of the spectrum representing selected band
        idx_spectra_integrate = np.logical_and(np.logical_and(wvl_data >= wvl_range[0],
                                                              wvl_data <= wvl_range[1]),
                                               np.isfinite(spectra_data))
    else:
        idx_spectra_integrate = np.isfinite(spectra_data)

    if np.isfinite(spectra_data[idx_spectra_integrate]).all():
        integral = np.trapz(spectra_data[idx_spectra_integrate] - offset,  # values corrected for the continuum/offset level
                            wvl_data[idx_spectra_integrate])
        return integral
    else:
        return np.nan


def process_selected_id(s_id):
    print 'Working on object '+str(s_id)
    # define flag parameter that will describe processing problem(s) in resulting table
    proc_flag = 0

    # get parameters of the observed object
    idx_object = np.where(general_data['sobject_id'] == s_id)[0]
    object_parameters = general_data[idx_object]

    # get both spectra of the object and it's reduced reference median comparison spectra
    spectra_object_c1 = spectra_ccd1[idx_object, :][0]
    spectra_object_c3 = spectra_ccd3[idx_object, :][0]
    spectra_median_c1 = spectra_median_ccd1[idx_object, :][0]
    spectra_median_c3 = spectra_median_ccd3[idx_object, :][0]

    # check validity of reference spectra
    if not np.isfinite(spectra_median_c1).any() or not np.isfinite(spectra_median_c3).any():
        proc_flag += 0b1000
        results.add_row(np.hstack([s_id, np.repeat(np.nan, n_cols_out-2), proc_flag]))
        txt_out = open(results_csv_out, 'a')
        txt_out.write(','.join([str(v) for v in [s_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, proc_flag]])+'\n')
        txt_out.close()
        print ' Reference spectra is not defined'
        return False

    # compute spectra difference and division
    # spectra_dif = spectra_object - spectra_median
    # spectra_dif = np.exp(-np.log(spectra_median) - (-np.log(spectra_object)))
    spectra_dif_c1 = -np.log(spectra_median_c1) - (-np.log(spectra_object_c1))
    spectra_dif_c3 = -np.log(spectra_median_c3) - (-np.log(spectra_object_c3))
    spectra_div_c1 = spectra_object_c1 / spectra_median_c1
    spectra_div_c3 = spectra_object_c3 / spectra_median_c3

    # find strange things in spectra comparision:
    # - bad fit to reference spectra
    # - large continuum difference
    # - emission like ccd spikes
    # - discontinuities in observed spectra

    # TODO: analysis

    # integrate part of the spectrum representing H bands
    Ha_EW = integrate_ew_spectra(spectra_div_c3, wvl_val_ccd3, offset=1., wvl_range=[HALPHA_WVL - wvl_int_range, HALPHA_WVL + wvl_int_range])
    Hb_EW = integrate_ew_spectra(spectra_div_c1, wvl_val_ccd1, offset=1., wvl_range=[HBETA_WVL - wvl_int_range, HBETA_WVL + wvl_int_range])

    # add to results
    results.add_row([s_id, Ha_EW, Hb_EW, 0, 0, 0, 0, 0, 0, proc_flag])

    txt_out = open(results_csv_out, 'a')
    txt_out.write(','.join([str(v) for v in [s_id, Ha_EW, Hb_EW, 0, 0, 0, 0, 0, 0, proc_flag]]) + '\n')
    txt_out.close()

    # --------------------------------------------------------
    # ---------------- Plot results --------------------------
    # --------------------------------------------------------
    suffix = ''  #
    print ' Plotting results'

    fig, axs = plt.subplots(2, 3, figsize=(7, 10))

    # h-alpha plots
    axs[0, 0].plot(wvl_val_ccd3, spectra_median_c3, color='red', linewidth=0.5, label='Median')
    axs[0, 0].plot(wvl_val_ccd3, spectra_object_c3, color='black', linewidth=0.4, label='Star')
    axs[0, 0].set(xlim=(HALPHA_WVL - wvl_plot_range, HALPHA_WVL + wvl_plot_range), ylim=(0.1, 1.1),
                  ylabel='Flux - H alpha')
    axs[0, 0].legend()
    axs[1, 0].plot(wvl_val_ccd3, spectra_dif_c3, color='black', linewidth=0.5)
    axs[1, 0].set(xlim=(HALPHA_WVL - wvl_plot_range, HALPHA_WVL + wvl_plot_range), ylim=(-0.5, 0.5),
                  ylabel='Difference log(flux)', xlabel='Wavelength')
    axs[2, 0].plot(wvl_val_ccd3, spectra_div_c3, color='black', linewidth=0.5)
    axs[2, 0].set(xlim=(HALPHA_WVL - wvl_plot_range, HALPHA_WVL + wvl_plot_range), ylim=(0.5, 1.5),
                  ylabel='Division flux', xlabel='Wavelength')
    axs[2, 0].axhline(1., c='C0', alpha=0.75)
    for ip in range(3):
        axs[ip, 0].axvline(HALPHA_WVL, c='black', alpha=0.75)
        axs[ip, 0].axvline(HALPHA_WVL - wvl_int_range, c='black', alpha=0.33, ls='--')
        axs[ip, 0].axvline(HALPHA_WVL + wvl_int_range, c='black', alpha=0.33, ls='--')

    # h-beta plots
    axs[0, 1].plot(wvl_val_ccd1, spectra_median_c1, color='red', linewidth=0.5, label='Median')
    axs[0, 1].plot(wvl_val_ccd1, spectra_object_c1, color='black', linewidth=0.4, label='Star')
    axs[0, 1].set(xlim=(HBETA_WVL - wvl_plot_range, HBETA_WVL + wvl_plot_range), ylim=(0.1, 1.1),
                  ylabel='Flux - H beta')
    axs[0, 1].legend()
    axs[1, 1].plot(wvl_val_ccd1, spectra_dif_c1, color='black', linewidth=0.5)
    axs[1, 1].set(xlim=(HBETA_WVL - wvl_plot_range, HBETA_WVL + wvl_plot_range), ylim=(-0.5, 0.5),
                  ylabel='Difference log(flux)', xlabel='Wavelength')
    axs[2, 1].plot(wvl_val_ccd1, spectra_div_c1, color='black', linewidth=0.5)
    axs[2, 1].set(xlim=(HBETA_WVL - wvl_plot_range, HBETA_WVL + wvl_plot_range), ylim=(0.5, 1.5),
                  ylabel='Division flux', xlabel='Wavelength')
    axs[2, 1].axhline(1., c='C0', alpha=0.75)
    for ip in range(3):
        axs[ip, 1].axvline(HBETA_WVL, c='black', alpha=0.75)
        axs[ip, 1].axvline(HBETA_WVL - wvl_int_range, color='black', alpha=0.5, ls='--')
        axs[ip, 1].axvline(HBETA_WVL + wvl_int_range, color='black', alpha=0.5, ls='--')

    for ip in range(3):
        for il in range(2):
            axs[ip, il].grid(color='black', alpha=0.2, ls='--')

    s_date = np.int32(s_id/10e10)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(str(s_date)+'/'+str(s_id)+suffix+'.png', dpi=200)
    plt.close()

    return True


# create all possible output subdirectories
sobject_dates = np.unique(np.int32(sobject_ids/10e10))
for s_date in sobject_dates:
    move_to_dir(str(s_date))
    chdir('..')

results_csv_out = 'results_H_lines.csv'
txt_out = open(results_csv_out,  'w')
txt_out.write('sobject_id,swan_integ,swan_fit_integ,amp,sig,offset,wvl,amp_lin,offset_lin,flag\n')
txt_out.close()

# # without any multiprocessing - for test purpose only
# for so_id in sobject_ids:
#     process_selected_id(so_id)

# multiprocessing
pool = Pool(processes=n_multi)
process_return = np.array(pool.map(process_selected_id, sobject_ids))
pool.close()

# save results
results.write('results_H_lines.fits')
