import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 14

from os import chdir
from imp import load_source
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, join
from scipy.signal import argrelextrema, correlate
from lmfit.models import LinearModel, LorentzianModel, GaussianModel
from astropy.modeling import models, fitting

load_source('helper_functions', '../Carbon-Spectra/helper_functions.py')
from helper_functions import move_to_dir, spectra_normalize
load_source('s_collection', '../Carbon-Spectra/spectra_collection_functions.py')
from s_collection import *

from multiprocessing import Pool
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

n_multi = 62
galah_data_dir = '/shared/ebla/cotar/'
out_dir = '/shared/data-camelot/cotar/'

general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')
params_data = Table.read(galah_data_dir + 'GALAH_iDR3_main_alpha_190529.fits')
general_data = join(general_data, params_data['sobject_id', 'teff', 'fe_h', 'logg', 'flag_sp'], join_type='left')

spectra_ccd1_pkl = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl'
median_spectra_ccd1_pkl = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'_median_350_snr_15_teff_150_logg_0.20_feh_0.20_vbroad_10_hbeta.pkl'
spectra_ccd3_pkl = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl'
median_spectra_ccd3_pkl = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'_median_350_snr_25_teff_150_logg_0.20_feh_0.20_vbroad_10_halpha.pkl'

# parse interpolation and averaging settings from filename
ccd1_wvl = CollectionParameters(spectra_ccd1_pkl).get_wvl_values()
ccd3_wvl = CollectionParameters(spectra_ccd3_pkl).get_wvl_values()

# determine wvls that will be read from the spectra
wvl_read_range = 60
wvl_plot_range_s = 24
wvl_plot_range_z = 5
wvl_int_range = 3
HBETA_WVL = 4861.36
HALPHA_WVL = 6562.81

# exact values of lines taken from http://newt.phys.unsw.edu.au/~jkw/alpha/useful_lines.pdf
SII_bands = [6716.47, 6730.85]
NII_bands = [6548.03, 6583.41]
SII_WVL = np.mean(SII_bands)
NII_WVL = np.mean(NII_bands)

idx_read_ccd1 = np.where(np.logical_and(ccd1_wvl >= HBETA_WVL - wvl_read_range,
                                        ccd1_wvl <= HBETA_WVL + wvl_read_range))[0]
idx_read_ccd3 = np.where(np.logical_and(ccd3_wvl >= HALPHA_WVL - wvl_read_range,
                                        ccd3_wvl <= HALPHA_WVL + wvl_read_range))[0]
idx_read_ccd3_SII = np.where(np.logical_and(ccd3_wvl >= SII_WVL - wvl_read_range,
                                            ccd3_wvl <= SII_WVL + wvl_read_range))[0]

wvl_val_ccd1 = ccd1_wvl[idx_read_ccd1]
wvl_val_ccd3 = ccd3_wvl[idx_read_ccd3]
wvl_val_ccd3_SII = ccd3_wvl[idx_read_ccd3_SII]
dwvl_ccd1 = wvl_val_ccd1[1] - wvl_val_ccd1[0]
dwvl_ccd3 = wvl_val_ccd3[1] - wvl_val_ccd3[0]

# read limited number of columns instead of full spectral dataset
print 'Reading resampled/interpolated GALAH spectra'
spectra_ccd1 = read_pkl_spectra(out_dir + spectra_ccd1_pkl, read_cols=idx_read_ccd1)
print ' --'
ccd3_all = read_pkl_spectra(out_dir + spectra_ccd3_pkl)
print ' --'
spectra_ccd3 = ccd3_all[:, idx_read_ccd3]
spectra_ccd3_SII = ccd3_all[:, idx_read_ccd3_SII]
ccd3_all = None
print 'Reading merged median GALAH spectra'
spectra_median_ccd1 = read_pkl_spectra(out_dir + median_spectra_ccd1_pkl, read_cols=idx_read_ccd1)
print ' --'
ccd3_median_all = read_pkl_spectra(out_dir + median_spectra_ccd3_pkl)
print ' --'
spectra_median_ccd3 = ccd3_median_all[:, idx_read_ccd3]
spectra_median_ccd3_SII = ccd3_median_all[:, idx_read_ccd3_SII]
ccd3_median_all = None
# remove full spectra arrays
print ' Done'

# # TEMP: save/store data pkl subsets to output dir
# print 'Saving subsets'
# save_pkl_spectra(spectra_ccd3, out_dir + 'Emissions_ccd3_1.pkl')
# save_pkl_spectra(spectra_ccd1, out_dir + 'Emissions_ccd1_1.pkl')
# save_pkl_spectra(spectra_ccd3_SII, out_dir + 'Emissions_ccd3_2.pkl')
# save_pkl_spectra(spectra_median_ccd3, out_dir + 'Emissions_ccd3_me_1.pkl')
# save_pkl_spectra(spectra_median_ccd1, out_dir + 'Emissions_ccd1_me_1.pkl')
# save_pkl_spectra(spectra_median_ccd3_SII, out_dir + 'Emissions_ccd3_me_2.pkl')
#
# print 'Reading saved subsets'
# spectra_ccd3 = read_pkl_spectra(out_dir + 'Emissions_ccd3_1.pkl')
# spectra_ccd1 = read_pkl_spectra(out_dir + 'Emissions_ccd1_1.pkl')
# spectra_ccd3_SII = read_pkl_spectra(out_dir + 'Emissions_ccd3_2.pkl')
# print ' --'
# spectra_median_ccd3 = read_pkl_spectra(out_dir + 'Emissions_ccd3_me_1.pkl')
# spectra_median_ccd1 = read_pkl_spectra(out_dir + 'Emissions_ccd1_me_1.pkl')
# spectra_median_ccd3_SII = read_pkl_spectra(out_dir + 'Emissions_ccd3_me_2.pkl')
# print ' --'

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
results = Table(names=('sobject_id', 'Ha_EW', 'Hb_EW', 'Ha_EW_abs', 'Hb_EW_abs', 'SB2_c3', 'SB2_c1', 'wvl_c3', 'wvl_c1', 'NII', 'SII', 'flag'),
                dtype=('int64', 'float64', 'float64', 'float64', 'float64', 'int16', 'int16', 'int16', 'int16', 'int16', 'int16', 'int16'))
n_cols_out = len(results.columns)

# --------------------------------------------------------
# ---------------- Main analysis -------------------------
# --------------------------------------------------------
print 'Number of spectra that will be evaluated:', len(sobject_ids)


def integrate_ew_spectra(spectra_data, wvl_data, wvl_range=None, offset=0., abs=False):
    if wvl_range is not None:
        # integrate part of the spectrum representing selected band
        idx_spectra_integrate = np.logical_and(np.logical_and(wvl_data >= wvl_range[0],
                                                              wvl_data <= wvl_range[1]),
                                               np.isfinite(spectra_data))
    else:
        idx_spectra_integrate = np.isfinite(spectra_data)

    if np.isfinite(spectra_data[idx_spectra_integrate]).all():
        integ_signal = spectra_data[idx_spectra_integrate] - offset
        if abs:
            integ_signal = np.abs(integ_signal)
        integral = np.trapz(integ_signal,  # values corrected for the continuum/offset level to absolute values
                            wvl_data[idx_spectra_integrate])
        return integral
    else:
        return np.nan


def process_selected_id(s_id):
    print '\nWorking on object '+str(s_id)
    # define flag parameter that will describe processing problem(s) in resulting table
    proc_flag = 0

    # get parameters of the observed object
    idx_object = np.where(general_data['sobject_id'] == s_id)[0]
    object_parameters = general_data[idx_object][0]

    # get both spectra of the object and it's reduced reference median comparison spectra
    spectra_object_c1 = spectra_ccd1[idx_object, :][0]
    spectra_object_c3 = spectra_ccd3[idx_object, :][0]
    spectra_object_c3_SII = spectra_ccd3_SII[idx_object, :][0]
    spectra_median_c1 = spectra_median_ccd1[idx_object, :][0]
    spectra_median_c3 = spectra_median_ccd3[idx_object, :][0]
    spectra_median_c3_SII = spectra_median_ccd3_SII[idx_object, :][0]

    # check validity of reference spectra
    no_ref_plot = False
    if not np.isfinite(spectra_median_c1).any() or not np.isfinite(spectra_median_c3).any():
        no_ref_plot = True
        proc_flag += 0b1000

    def renorm_by_ref(s_obj, s_ref, wvl):
        # renormalization with reference spectrum
        idx_ref_px = np.abs(s_ref - 1.) < 0.1
        if np.sum(idx_ref_px) < 10:
            return s_obj
        s_obj_norm_curve = spectra_normalize(wvl, s_obj / s_ref,
                                             steps=3, sigma_low=2., sigma_high=2., n_min_perc=5.,
                                             order=2, func='poly', fit_mask=idx_ref_px, return_fit=True)
        return s_obj / s_obj_norm_curve

    spectra_object_c1 = renorm_by_ref(spectra_object_c1, spectra_median_c1, wvl_val_ccd1)
    spectra_object_c3 = renorm_by_ref(spectra_object_c3, spectra_median_c3, wvl_val_ccd3)
    spectra_object_c3_SII = renorm_by_ref(spectra_object_c3_SII, spectra_median_c3_SII, wvl_val_ccd3_SII)

    def fill_nans(s_obj):
        # first replace possible nans
        idx_nan_o = ~np.isfinite(s_obj)
        if np.sum(idx_nan_o) > 0:
            s_obj[idx_nan_o] = 1.
        return s_obj
    # correct part of the spectrum that might not be filled with valid data
    spectra_object_c3 = fill_nans(spectra_object_c3)
    spectra_object_c3_SII = fill_nans(spectra_object_c3_SII)
    spectra_median_c3 = fill_nans(spectra_median_c3)
    spectra_median_c3_SII = fill_nans(spectra_median_c3_SII)
    spectra_object_c1 = fill_nans(spectra_object_c1)
    spectra_median_c1 = fill_nans(spectra_median_c1)

    # compute spectra difference and division
    # spectra_dif = spectra_object - spectra_median
    # spectra_dif = np.exp(-np.log(spectra_median) - (-np.log(spectra_object)))
    spectra_dif_c1 = -np.log(spectra_median_c1) - (-np.log(spectra_object_c1))
    spectra_dif_c3 = -np.log(spectra_median_c3) - (-np.log(spectra_object_c3))
    spectra_dif_c3_SII = -np.log(spectra_median_c3_SII) - (-np.log(spectra_object_c3_SII))
    spectra_div_c1 = spectra_object_c1 / spectra_median_c1
    spectra_div_c3 = spectra_object_c3 / spectra_median_c3
    spectra_div_c3_SII = spectra_object_c3_SII / spectra_median_c3_SII

    # find strange things in spectra comparision:
    # - bad fit to reference spectra
    # - large continuum difference
    # - emission like ccd spikes
    # - discontinuities in observed spectra

    # detect possible presence of SB2 spectrum
    def sb2_ccf(s_obj, s_ref,
                a_thr=10., s_thr_l=5., s_thr_u=20):

        # create array of ones, with the same length as investigated signal - will be used for normalization
        ones_y = np.full_like(s_obj, fill_value=1.)
        ccf_y = correlate(1. - s_obj, 1. - s_ref, mode='same', method='fft')  # subtract from 1 to remove continuum
        ccf_norm = correlate(ones_y, ones_y, mode='same', method='fft')
        ccf_x = np.arange(len(ccf_y))+1 - int(len(ccf_y)/2.)  # could be wrong to +/- 1
        # use only the central part of CCF that has some usable information
        w_ccf = 150.
        idx_keep = np.abs(ccf_x) <= w_ccf
        ccf_x = ccf_x[idx_keep]
        # normalize correlation by a number of investigated wvl pixels
        ccf_y = ccf_y[idx_keep] / ccf_norm[idx_keep]
        # better normalize values between min and max values of ccf - easier and better fir
        ccf_y -= np.median(ccf_y)  # set new zero-point
        ccf_y /= np.percentile(ccf_y, 98)  # normalise ccf values and roughly equalise both data axes

        # fit n_gauss Gaussian functions to the CCF
        n_gauss = 3
        # approximate uncorrelated levels with a polynomial of low degree (usually up to max 3rd degreee)
        # e_fit = models.Polynomial1D(2, c0=np.median(ccf_y), c1=0., c2=-1.*1e-8)
        e_fit = models.Polynomial1D(1, c0=np.median(ccf_y), c1=0.)
        # e_fit = models.Trapezoid1D(amplitude=np.percentile(ccf_y, 90), x_0=0., width=0., slope=1., fixed={'width': True})
        # e_fit = models.Const1D(amplitude=np.mean(ccf_y))
        for i_l in range(n_gauss):
            g_amp = 0.5
            if i_l == 1:
                g_amp = 1.
            e_fit += models.Gaussian1D(amplitude=g_amp, mean=(n_gauss-2)*45.-45.*i_l, stddev=8)
            e_fit.bounds['mean_' + str(i_l + 1)] = [-w_ccf+20, w_ccf-20]
            e_fit.bounds['stddev_' + str(i_l + 1)] = [0., 75.]
            e_fit.bounds['amplitude_' + str(i_l + 1)] = [0., 2.]
        # fit_t = fitting.SLSQPLSQFitter()
        fit_t = fitting.LevMarLSQFitter()
        fit_res = fit_t(e_fit, ccf_x, ccf_y, maxiter=300)

        # is there a sign of multiplicity in the spectrum?
        # define number of meaningful peaks in the ccf curve
        is_sb2 = 0
        n_peaks = 0
        ccf_peaks = []
        # g_amps = []
        # g_std = []
        for i_l in range(n_gauss):
            ccf_peaks.append(fit_res[i_l + 1].mean.value)
            # g_amps.append(fit_res[i_l + 1].amplitude.value)
            # g_std.append(fit_res[i_l + 1].stddev.value)
            # check if fitted peak is meaningful
            if fit_res[i_l + 1].amplitude.value > a_thr and s_thr_l < fit_res[i_l + 1].stddev.value < s_thr_u:
                n_peaks += 1
        if n_peaks > 1:
            is_sb2 = 1

        # is peak centered?
        is_cent = True
        ccf_peak = ccf_x[np.argmax(ccf_y)]
        if np.abs(ccf_peak) >= 3.:
            is_cent = False

        # print 'SB2 fit results'
        # print is_sb2, is_cent, ccf_peak
        # print 'C poly:', [fit_res[0].c0.value, fit_res[0].c1.value]#, fit_res[0].c2.value]
        # print 'Peaks:', ccf_peaks
        # print 'Amps:', g_amps
        # print 'Std:', g_std

        return is_sb2, is_cent, [ccf_x, ccf_y, fit_res(ccf_x), ccf_peak, ccf_peaks]

    sb2_c3, wvl_c3, ccf_res_c3 = sb2_ccf(spectra_object_c3, spectra_median_c3, a_thr=0.45, s_thr_l=3.5, s_thr_u=60.)
    sb2_c1, wvl_c1, ccf_res_c1 = sb2_ccf(spectra_object_c1, spectra_median_c1, a_thr=0.45, s_thr_l=3.5, s_thr_u=60.)

    # detect possible presence of [NII] and [SII] nebular emission lines
    def fit_emission_lines(s_obj, w_obj, e_lines,
                           thr_l=0.05, thr_u=1., s_thr_l=0.1, s_thr_u=0.5,
                           d_w=6., d_e_l=3.):
        # first try to detect strong emission peaks

        # create model to be fitted
        e_fit = models.Const1D(amplitude=np.median(s_obj))
        for i_l, e_l in enumerate(e_lines):
            # select the best suitable peak in the vicinity of selected emission line
            idx_ep = np.where(np.logical_and(w_obj >= e_l-d_e_l,
                                             w_obj <= e_l+d_e_l))[0]
            idx_max = idx_ep[np.argmax(s_obj[idx_ep])]
            e_mean = w_obj[idx_max]
            # add emission peak to the fitting procedure
            e_fit += models.Gaussian1D(amplitude=0.5, mean=e_mean, stddev=0.3,
                                       fixed={'mean': True})  # mean is fixed for simplicity
            e_fit.bounds['amplitude_' + str(i_l + 1)] = [0., 2.]
            e_fit.bounds['stddev' + str(i_l + 1)] = [0., 0.5]
        fit_t = fitting.LevMarLSQFitter()  # SLSQPLSQFitter
        fit_res = fit_t(e_fit, w_obj, s_obj, maxiter=300)
        fitted_curve = fit_res(w_obj)

        line_det = 0
        wvl_peaks = []
        g_amps = []
        g_std = []
        for i_l, e_l in enumerate(e_lines):
            wvl_peaks.append(fit_res[i_l+1].mean.value)
            g_amps.append(fit_res[i_l + 1].amplitude.value)
            g_std.append(fit_res[i_l + 1].stddev.value)
            if thr_l < fit_res[i_l+1].amplitude.value < thr_u and s_thr_l < fit_res[i_l+1].stddev.value < s_thr_u:
                line_det += 1

        # print 'Emission lines fit results, is detected? =', line_det
        # print 'Peaks:', wvl_peaks, np.array(wvl_peaks) - np.array(e_lines)
        # print 'Amps:', g_amps
        # print 'Std:', g_std

        # check if fitted curve even exists -> test if fit did converge
        # TODO: find possible better way to check this
        if ~np.isfinite(fitted_curve).all():
            # fit did not converge properly
            line_det = 0

        return line_det, wvl_peaks, fitted_curve

    nii_det, nii_peaks, nii_fit = fit_emission_lines(spectra_dif_c3, wvl_val_ccd3, NII_bands)
    sii_det, sii_peaks, sii_fit = fit_emission_lines(spectra_dif_c3_SII, wvl_val_ccd3_SII, SII_bands)

    # integrate part of the spectrum representing H bands
    Ha_EW = integrate_ew_spectra(spectra_div_c3, wvl_val_ccd3, offset=1., wvl_range=[HALPHA_WVL - wvl_int_range, HALPHA_WVL + wvl_int_range])
    Ha_EW_abs = integrate_ew_spectra(spectra_div_c3, wvl_val_ccd3, abs=True, offset=1., wvl_range=[HALPHA_WVL - wvl_int_range, HALPHA_WVL + wvl_int_range])
    Hb_EW = integrate_ew_spectra(spectra_div_c1, wvl_val_ccd1, offset=1., wvl_range=[HBETA_WVL - wvl_int_range, HBETA_WVL + wvl_int_range])
    Hb_EW_abs = integrate_ew_spectra(spectra_div_c1, wvl_val_ccd1, abs=True, offset=1., wvl_range=[HBETA_WVL - wvl_int_range, HBETA_WVL + wvl_int_range])

    # add to results
    output_array = [s_id, Ha_EW, Hb_EW, Ha_EW_abs, Hb_EW_abs, sb2_c3, sb2_c1, wvl_c3, wvl_c1, nii_det, sii_det, proc_flag]
    results.add_row(output_array)

    txt_out = open(results_csv_out, 'a')
    txt_out.write(','.join([str(v) for v in output_array]) + '\n')
    txt_out.close()

    # --------------------------------------------------------
    # ---------------- Plot results --------------------------
    # --------------------------------------------------------
    suffix = ''  #
    # print ' Plotting results'

    fig, axs = plt.subplots(4, 3, figsize=(16, 12))
    fig.suptitle('Red flag: ' + str(object_parameters['red_flag']) + ',    flag sp: ' + str(object_parameters['flag_sp']) + ',    proc flag: ' + str(int(proc_flag)))

    # h-alpha plots
    axs[0, 0].plot(wvl_val_ccd3, spectra_median_c3, color='red', linewidth=0.5, label='Median')
    axs[0, 0].plot(wvl_val_ccd3, spectra_object_c3, color='black', linewidth=0.5, label='Star')
    axs[0, 0].set(xlim=(HALPHA_WVL - wvl_plot_range_s, HALPHA_WVL + wvl_plot_range_s), ylim=(0.1, 1.3),
                  ylabel='Flux - H alpha, [NII]')
    axs[0, 0].legend()
    axs[0, 1].plot(wvl_val_ccd3, spectra_dif_c3, color='black', linewidth=0.5)
    axs[0, 1].set(xlim=(HALPHA_WVL - wvl_plot_range_s, HALPHA_WVL + wvl_plot_range_s), ylim=(-0.3, 0.8),
                  ylabel='Difference log(flux)', xlabel='Valid [NII] lines = {:.0f}'.format(nii_det))
    axs[0, 2].plot(wvl_val_ccd3, spectra_div_c3, color='black', linewidth=0.5)
    axs[0, 2].set(xlim=(HALPHA_WVL - wvl_plot_range_z, HALPHA_WVL + wvl_plot_range_z), ylim=(0.7, 4),
                  ylabel='Division flux')
    axs[0, 1].axhline(0., c='C0', alpha=0.75)
    axs[0, 2].axhline(1., c='C0', alpha=0.75)
    for ip in range(3):
        axs[0, ip].axvline(HALPHA_WVL - wvl_int_range, c='black', alpha=0.7, ls='--')
        axs[0, ip].axvline(HALPHA_WVL + wvl_int_range, c='black', alpha=0.7, ls='--')
        for nii_w in NII_bands:
            axs[0, ip].axvline(nii_w, color='C2', alpha=0.7, ls='--')
        if True:#nii_det:
            axs[0, 1].plot(wvl_val_ccd3, nii_fit, color='C3', linewidth=0.5)
            for nii_w in nii_peaks:
                axs[0, ip].axvline(nii_w, color='C3', alpha=0.8, ls='--')

    # [SII] plots
    axs[1, 0].plot(wvl_val_ccd3_SII, spectra_median_c3_SII, color='red', linewidth=0.5, label='Median')
    axs[1, 0].plot(wvl_val_ccd3_SII, spectra_object_c3_SII, color='black', linewidth=0.5, label='Star')
    axs[1, 0].set(xlim=(SII_WVL - wvl_plot_range_s, SII_WVL + wvl_plot_range_s), ylim=(0.1, 1.3),
                  ylabel='Flux - [SII]')
    axs[1, 0].legend()
    axs[1, 1].plot(wvl_val_ccd3_SII, spectra_dif_c3_SII, color='black', linewidth=0.5)
    axs[1, 1].set(xlim=(SII_WVL - 2.*wvl_plot_range_z, SII_WVL + 2.*wvl_plot_range_z), ylim=(-0.3, 0.8),
                  ylabel='Difference log(flux)', xlabel='Valid [SII] lines = {:.0f}'.format(sii_det))
    axs[1, 2].plot(wvl_val_ccd3_SII, spectra_div_c3_SII, color='black', linewidth=0.5)
    axs[1, 2].set(xlim=(SII_WVL - 2.*wvl_plot_range_z, SII_WVL + 2.*wvl_plot_range_z), ylim=(0.7, 2.),
                  ylabel='Division flux')
    axs[1, 1].axhline(0., c='C0', alpha=0.75)
    axs[1, 2].axhline(1., c='C0', alpha=0.75)
    for ip in range(3):
        for sii_w in SII_bands:
            axs[1, ip].axvline(sii_w, color='C2', alpha=0.7, ls='--')
        if True:#sii_det:
            axs[1, 1].plot(wvl_val_ccd3_SII, sii_fit, color='C3', linewidth=0.5)
            for sii_w in sii_peaks:
                axs[1, ip].axvline(sii_w, color='C3', alpha=0.8, ls='--')
                axs[1, ip].axvline(sii_w, color='C3', alpha=0.8, ls='--')

    # h-beta plots
    axs[2, 0].plot(wvl_val_ccd1, spectra_median_c1, color='red', linewidth=0.5, label='Median')
    axs[2, 0].plot(wvl_val_ccd1, spectra_object_c1, color='black', linewidth=0.5, label='Star')
    axs[2, 0].set(xlim=(HBETA_WVL - wvl_plot_range_s, HBETA_WVL + wvl_plot_range_s), ylim=(0.1, 1.3),
                  ylabel='Flux - H beta', xlabel='Wavelength')
    axs[2, 0].legend()
    axs[2, 1].plot(wvl_val_ccd1, spectra_dif_c1, color='black', linewidth=0.5)
    axs[2, 1].set(xlim=(HBETA_WVL - wvl_plot_range_s, HBETA_WVL + wvl_plot_range_s), ylim=(-0.3, 2),
                  ylabel='Difference log(flux)', xlabel='Wavelength')
    axs[2, 2].plot(wvl_val_ccd1, spectra_div_c1, color='black', linewidth=0.5)
    axs[2, 2].set(xlim=(HBETA_WVL - wvl_plot_range_z, HBETA_WVL + wvl_plot_range_z), ylim=(0.7, 4),
                  ylabel='Division flux', xlabel='Wavelength')
    axs[2, 1].axhline(0., c='C0', alpha=0.75)
    axs[2, 2].axhline(1., c='C0', alpha=0.75)
    for ip in range(3):
        axs[2, ip].axvline(HBETA_WVL - wvl_int_range, color='black', alpha=0.7, ls='--')
        axs[2, ip].axvline(HBETA_WVL + wvl_int_range, color='black', alpha=0.7, ls='--')

    # analysis plots - CCF, fits, flags ...
    # [ccf_x, ccf_y, , ccf_peak, ccf_peaks]
    axs[3, 0].plot(ccf_res_c3[0], ccf_res_c3[1],
                   color='black', linewidth=0.6, label='CCF H alpha')
    axs[3, 0].plot(ccf_res_c3[0], ccf_res_c3[2],
                   color='C3', linewidth=0.6, label='CCF fit')
    axs[3, 0].legend()
    axs[3, 1].plot(ccf_res_c1[0], ccf_res_c1[1],
                   color='black', linewidth=0.6, label='CCF H beta')
    axs[3, 1].plot(ccf_res_c1[0], ccf_res_c1[2],
                   color='C3', linewidth=0.6, label='CCF fit')
    axs[3, 1].legend()
    # add peaks
    if wvl_c3:
        axs[3, 0].axvline(ccf_res_c3[3], color='C2', alpha=0.8, ls='-')
    else:
        axs[3, 0].axvline(ccf_res_c3[3], color='C3', alpha=0.8, ls='-')
    if wvl_c1:
        axs[3, 1].axvline(ccf_res_c1[3], color='C2', alpha=0.8, ls='-')
    else:
        axs[3, 1].axvline(ccf_res_c1[3], color='C3', alpha=0.8, ls='-')
    # add gaussian peaks
    for c_p in ccf_res_c3[4]:
        axs[3, 0].axvline(c_p, color='black', alpha=0.8, ls='--')
    for c_p in ccf_res_c1[4]:
        axs[3, 1].axvline(c_p, color='black', alpha=0.8, ls='--')

    plot_w = 125
    idx_m = np.logical_and(ccf_res_c3[0] >= -plot_w, ccf_res_c3[0] <= plot_w)
    if np.isfinite(ccf_res_c3[1]).all():
        axs[3, 0].set(xlim=(-plot_w, plot_w), xlabel='Pixel shifts - is SB2? = '+str(int(sb2_c3)),
                      ylim=(np.min(ccf_res_c3[1][idx_m]), np.max(ccf_res_c3[1]) * 1.05))
    if np.isfinite(ccf_res_c1[1]).all():
        axs[3, 1].set(xlim=(-plot_w, plot_w), xlabel='Pixel shifts - is SB2? = '+str(int(sb2_c1)),
                      ylim=(np.min(ccf_res_c1[1][idx_m]), np.max(ccf_res_c1[1]) * 1.05))

    # add grid lines to every plot available
    for ip in range(3):
        for il in range(4):
            axs[il, ip].grid(color='black', alpha=0.2, ls='--')

    s_date = np.int32(s_id/10e10)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig(str(s_date)+'/'+str(s_id)+suffix+'.png', dpi=200)
    # plt.savefig('tests/'+str(s_id)+suffix+'.png', dpi=200)
    plt.close()

    return True


# create all possible output subdirectories
print 'Creating sub-folders in advance'
sobject_dates = np.unique(np.int32(sobject_ids/10e10))
for s_date in sobject_dates:
    move_to_dir(str(s_date))
    chdir('..')
for s_date in ['no_ref', 'strongest', 'strongest_absolute', 'tests', 'SB2']:
    move_to_dir(str(s_date))
    chdir('..')

results_csv_out = 'results_H_lines.csv'
txt_out = open(results_csv_out,  'w')
txt_out.write(','.join([str(sc) for sc in results.colnames]) + '\n')
txt_out.close()

# # without any multiprocessing - for test purpose only
# sobject_ids = [140808001101103, 150106002201377, 150405001501216, 150405002001246, 150428000101176, 150607003601066,
#                151009005101283, 160129004701296, 160326000101077, 160420003801237, 170109002801349, 170217002201245,
#                170408003501326, 170509002201287, 170604002601330, 170828001601228, 170911002601194, 140112002301230,
#                140112002301339, 140608003101196, 140609002601035, 140609003601247, 140711000601112, 140713001901218,
#                170603004601078, 170604002101031, 180131002701292, 140311005601066, 140311005601088, 140312001701004,
#                140608003101355, 141102003201190, 150210002701176, 150411004601220, 150428002601115, 150428003101113,
#                150428003101134, 151110002101261, 160325002101011, 160401004401107, 160423003801226, 170511000601249,
#                140112002301103, 140112002301176, 140118002501376, 140609002601375, 150504003001213, 150607003601372,
#                151225002101232, 151228003301129, 160415004601132, 170530004101189, 170603003101013, 170603004101373]
# for so_id in np.sort(sobject_ids):
#     process_selected_id(so_id)

# multiprocessing
pool = Pool(processes=n_multi)
process_return = np.array(pool.map(process_selected_id, sobject_ids))
pool.close()
pool = None

# save results
results.write('results_H_lines.fits', overwrite=True)
