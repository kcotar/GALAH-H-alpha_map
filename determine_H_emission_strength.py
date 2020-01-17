import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 14

from os import chdir
from importlib.machinery import SourceFileLoader
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const

from astropy.table import Table, join
from scipy.signal import correlate, medfilt, argrelextrema
from astropy.modeling import models, fitting
from sys import argv
from getopt import getopt
from copy import deepcopy
from time import time

SourceFileLoader('helper_functions', '../Carbon-Spectra/helper_functions.py').load_module()
from helper_functions import move_to_dir, spectra_normalize
SourceFileLoader('s_collection', '../Carbon-Spectra/spectra_collection_functions.py').load_module()
from s_collection import CollectionParameters, read_pkl_spectra, save_pkl_spectra

from multiprocessing import Pool

VERBOSE = False
PLOT_FIG = False
TEST_RUN = False
if len(argv) > 1:
    # parse input options
    opts, args = getopt(argv[1:], '', ['test=', 'verbose=', 'plot='])
    # set parameters, depending on user inputs
    print(opts)
    for o, a in opts:
        if o == '--test':
            TEST_RUN = int(a)
        if o == '--verbose':
            VERBOSE = int(a)
        if o == '--plot':
            PLOT_FIG = int(a)


def make_mask(values, ranges,
              drange=0., invert=False):
    idx_mask = np.full_like(values, fill_value=False)
    for vr in ranges:
        idx_mask[np.logical_and(values >= (vr[0]-drange), values <= (vr[1]+drange))] = True

    if invert:
        return np.logical_not(idx_mask)
    return idx_mask


# --------------------------------------------------------
# ---------------- Read data -----------------------------
# --------------------------------------------------------
print('Reading GALAH parameters')
date_string = '20190801'
remove_spikes = False

n_multi = 45
galah_data_dir = '/shared/ebla/cotar/'
out_dir = '/shared/data-camelot/cotar/'

# additional data and products about observed spectra
general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')['sobject_id', 'red_flag', 'rv_guess_shift', 'v_bary']
params_data = Table.read(galah_data_dir + 'GALAH_iDR3_main_alpha_190529.fits')
general_data = join(general_data, params_data['sobject_id', 'teff', 'fe_h', 'logg', 'flag_sp'], join_type='left')
del params_data
# auxiliary data-sets
sky_lineslist = Table.read(galah_data_dir + 'sky_emission_linelist.csv', format='ascii.csv')

spectra_ccd1_pkl = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'.pkl'
# median_spectra_ccd1_pkl = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'_median_350_snr_15_teff_150_logg_0.20_feh_0.20_vbroad_10_hbeta.pkl'
# median_spectra_ccd1_pkl = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'_median_500_snr_15_teff_300_logg_0.30_feh_0.30_vbroad_10_hbeta.pkl'
median_spectra_ccd1_pkl = 'galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_'+date_string+'_ann_median.pkl'
spectra_ccd3_pkl = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'.pkl'
# median_spectra_ccd3_pkl = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'_median_350_snr_25_teff_150_logg_0.20_feh_0.20_vbroad_10_halpha.pkl'
# median_spectra_ccd3_pkl = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'_median_500_snr_25_teff_300_logg_0.30_feh_0.30_vbroad_10_halpha.pkl'
median_spectra_ccd3_pkl = 'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_'+date_string+'_ann_median.pkl'

# parse interpolation and averaging settings from filename
ccd1_wvl = CollectionParameters(spectra_ccd1_pkl).get_wvl_values()
ccd3_wvl = CollectionParameters(spectra_ccd3_pkl).get_wvl_values()

# determine wvls that will be read from the spectra
wvl_read_range = 60
wvl_plot_range_s = 24
wvl_plot_range_z = 5
wvl_peak_range = 2
wvl_int_range = 3.5
HBETA_WVL = 4861.3615
HALPHA_WVL = 6562.8518

# remove weak and out-of-bounds sky emission lines
sky_lineslist = sky_lineslist[sky_lineslist['Flux'] >= 0.9]
sky_lineslist = sky_lineslist[np.abs(sky_lineslist['Ang'] - HALPHA_WVL) <= wvl_read_range]
# sort for further refinement of lines
sky_lineslist = sky_lineslist[np.argsort(sky_lineslist['Ang'])]

# merge close nearby emission lines that are separated for less than the GALAH resolution (~a few wavelength pixels)
min_sky_wvl_sep = 0.2
remove_sky_rows = []
for i_sl, sl in enumerate(sky_lineslist):
    if i_sl >= len(sky_lineslist) - 1:
        continue
    if sky_lineslist['Ang'][i_sl+1] - sky_lineslist['Ang'][i_sl] < min_sky_wvl_sep:
        sky_lineslist['Ang'][i_sl] = np.mean(sky_lineslist['Ang'][i_sl:i_sl+2])
        remove_sky_rows.append(i_sl+1)
if len(remove_sky_rows) > 0:
    sky_lineslist.remove_rows(remove_sky_rows)

print('Remaining set of sky lines:', len(sky_lineslist))
print(sky_lineslist)

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

# ccf masks that exclude halpha and hbeta
idx_ccf_ccd1 = make_mask(wvl_val_ccd1, [[HBETA_WVL-2.*wvl_plot_range_z, HBETA_WVL+2.*wvl_plot_range_z]],
                         drange=0., invert=True)
idx_ccf_ccd3 = make_mask(wvl_val_ccd3, [[HALPHA_WVL-2.*wvl_plot_range_z, HALPHA_WVL+2.*wvl_plot_range_z]],
                         drange=0., invert=True)
print('Non masked pixels used in CCF:', np.sum(idx_ccf_ccd1), np.sum(idx_ccf_ccd3))

# read limited number of columns instead of full spectral dataset
print('Reading resampled/interpolated GALAH spectra')
spectra_ccd1 = read_pkl_spectra(out_dir + spectra_ccd1_pkl, read_cols=idx_read_ccd1)
print(' --')
ccd3_all = read_pkl_spectra(out_dir + spectra_ccd3_pkl)
print(' --')
spectra_ccd3 = ccd3_all[:, idx_read_ccd3]
spectra_ccd3_SII = ccd3_all[:, idx_read_ccd3_SII]
ccd3_all = None
print('Reading merged median GALAH spectra')
spectra_median_ccd1 = read_pkl_spectra(out_dir + median_spectra_ccd1_pkl, read_cols=idx_read_ccd1)
print(' --')
ccd3_median_all = read_pkl_spectra(out_dir + median_spectra_ccd3_pkl)
print(' --')
spectra_median_ccd3 = ccd3_median_all[:, idx_read_ccd3]
spectra_median_ccd3_SII = ccd3_median_all[:, idx_read_ccd3_SII]
ccd3_median_all = None
# remove full spectra arrays
print(' Done')

# # TEMP: save/store data pkl subsets to output dir
# print('Saving subsets'
# save_pkl_spectra(spectra_ccd3, out_dir + 'Emissions_ccd3_1.pkl')
# save_pkl_spectra(spectra_ccd1, out_dir + 'Emissions_ccd1_1.pkl')
# save_pkl_spectra(spectra_ccd3_SII, out_dir + 'Emissions_ccd3_2.pkl')
# save_pkl_spectra(spectra_median_ccd3, out_dir + 'Emissions_ccd3_me_1.pkl')
# save_pkl_spectra(spectra_median_ccd1, out_dir + 'Emissions_ccd1_me_1.pkl')
# save_pkl_spectra(spectra_median_ccd3_SII, out_dir + 'Emissions_ccd3_me_2.pkl')
#
# print('Reading saved subsets'
# spectra_ccd3 = read_pkl_spectra(out_dir + 'Emissions_ccd3_1.pkl')
# spectra_ccd1 = read_pkl_spectra(out_dir + 'Emissions_ccd1_1.pkl')
# spectra_ccd3_SII = read_pkl_spectra(out_dir + 'Emissions_ccd3_2.pkl')
# print(' --'
# spectra_median_ccd3 = read_pkl_spectra(out_dir + 'Emissions_ccd3_me_1.pkl')
# spectra_median_ccd1 = read_pkl_spectra(out_dir + 'Emissions_ccd1_me_1.pkl')
# spectra_median_ccd3_SII = read_pkl_spectra(out_dir + 'Emissions_ccd3_me_2.pkl')
# print(' --'

# select initial data by parameters
idx_object_ok = general_data['sobject_id'] > 0  # can filter by date even later
idx_object_ok = np.logical_and(idx_object_ok, np.bitwise_and(general_data['red_flag'], 64) == 0)  # remove twilight flats

# --------------------------------------------------------
# ---------------- Determine objects to be observed ------
# --------------------------------------------------------

# determine object sobject_id numbers
sobject_ids = general_data[idx_object_ok]['sobject_id']

# move_to_dir(out_dir+'H_band_strength_all_'+date_string)
# move_to_dir(out_dir+'H_band_strength_complete_'+date_string+'_BroadRange_191005')
move_to_dir(out_dir+'H_band_strength_complete_'+date_string+'_ANN-medians_newthrs')

# binary flag describing processing step where something went wrong or data are missing:
# 10000000 or 128 = Missing median spectrum in ccd3 (red spectral range around H-alpha)
# 01000000 or  64 = Missing median spectrum in ccd1 (blue spectral range around H-beta)
# 00100000 or  32 = Large difference between median and observed spectrum in ccd3 - median squared error
# 00010000 or  16 = Large difference between median and observed spectrum in ccd1
# 00001000 or   8 = Spectrum most likely shows duplicated spectral absorption lines
# 00000100 or   4 = Strong contamination with sky emission features in ccd3 (one of them falls inside Ha region)
#                   Flag indicates to strong or to weak background correction.
# 00000010 or   2 = Wavelength solution (or RV) might be wrong in ccd3 - CCF is not centered at RV = 0
# 00000001 or   1 = Wavelength solution (or RV) might be wrong in ccd1
results = Table(names=('sobject_id', 'Ha_EW', 'Hb_EW', 'Ha_EW_abs', 'Hb_EW_abs', 'rv_Ha_peak', 'rv_Hb_peak',
                       'Ha_W10', 'Ha_EW_asym', 'Hb_EW_asym', 'SB2_c3', 'SB2_c1',
                       'NII', 'SII', 'NII_EW', 'SII_EW', 'rv_NII', 'rv_SII', 'flag'),
                dtype=('int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
                       'float64', 'float64', 'float64', 'bool', 'bool',
                       'int16', 'int16', 'float64', 'float64', 'float64', 'float64', 'int16'))
n_cols_out = len(results.columns)

# --------------------------------------------------------
# ---------------- Functions -----------------------------
# --------------------------------------------------------
print('Number of spectra that will be evaluated:', len(sobject_ids))


def fill_nans(s_obj):
    # first replace possible nans
    idx_nan_o = ~np.isfinite(s_obj)
    if np.sum(idx_nan_o) > 0:
        s_obj[idx_nan_o] = 1.
    return s_obj


def renorm_by_ref(s_obj, s_ref, wvl):
    # renormalization with reference spectrum
    min_renorm_px = 200
    idx_ref_px = np.abs(s_ref - 1.) < 0.1
    if np.sum(idx_ref_px) < min_renorm_px:
        idx_ref_px = np.abs(s_ref - 1.) < 0.15
        if np.sum(idx_ref_px) < min_renorm_px:
            return s_obj
    try:
        s_obj_norm_curve = spectra_normalize(wvl, s_obj / s_ref,
                                             steps=5, sigma_low=2., sigma_high=2., n_min_perc=5.,
                                             order=3, func='poly', fit_mask=idx_ref_px, return_fit=True)
        return s_obj / s_obj_norm_curve
    except:
        print('  Renormalization problem')
        return s_obj


def integrate_ew_spectra(spectra_data_orig, wvl_data_orig,
                         res_scale=11.,
                         wvl_range=None, offset=0., absolute=False,
                         reutrn_maxpeak_rv=False, wvl_rv_ref=None):

    # first resample input data to a (much) finer resolution
    if res_scale > 1.:
        wvl_data = np.linspace(wvl_data_orig[0], wvl_data_orig[-1], int(len(wvl_data_orig) * res_scale))
        spectra_data = np.interp(wvl_data, wvl_data_orig, spectra_data_orig)
    else:
        wvl_data = deepcopy(wvl_data_orig)
        spectra_data = deepcopy(spectra_data_orig)

    if wvl_range is not None:
        # integrate part of the spectrum representing selected band
        idx_spectra_integrate = np.logical_and(np.logical_and(wvl_data >= wvl_range[0],
                                                              wvl_data <= wvl_range[1]),
                                               np.isfinite(spectra_data))
        # integration of left part of the given spectral range
        idx_spectra_integrate_L = np.logical_and(np.logical_and(wvl_data >= wvl_range[0],
                                                                wvl_data <= np.mean(wvl_range)),
                                                 np.isfinite(spectra_data))
        # integration of right part of the given spectral range
        idx_spectra_integrate_R = np.logical_and(np.logical_and(wvl_data > np.mean(wvl_range),
                                                                wvl_data <= wvl_range[1]),
                                                 np.isfinite(spectra_data))
    else:
        idx_spectra_integrate = np.isfinite(spectra_data)

    if np.isfinite(spectra_data[idx_spectra_integrate]).all():
        integ_signal = spectra_data - offset
        if absolute:
            integ_signal = np.abs(integ_signal)

        # integrate flux values corrected for the continuum/offset level to absolute values
        # left and right part of the range
        integral_L = np.trapz(integ_signal[idx_spectra_integrate_L],
                              wvl_data[idx_spectra_integrate_L])
        integral_R = np.trapz(integ_signal[idx_spectra_integrate_R],
                              wvl_data[idx_spectra_integrate_R])
        # sum both parts
        integral = integral_L + integral_R

        # derive asymmetry index of the integrated range - not to be confused with actual asymmetry of the actual line
        ew_asym = (integral_R - integral_L) / integral

        if reutrn_maxpeak_rv:
            c_vel = const.c.value  # in units of m/s
            idx_max_inrange = np.argmax(medfilt(spectra_data[idx_spectra_integrate], kernel_size=int(3. * res_scale)))
            wvl_peak = wvl_data[idx_spectra_integrate][idx_max_inrange]
            rv_peak = (wvl_peak - wvl_rv_ref) / wvl_rv_ref * c_vel
            return integral, ew_asym, rv_peak/1000.  # convert to km/s
        return integral, ew_asym

    else:
        if reutrn_maxpeak_rv:
            return np.nan, np.nan, np.nan
        return np.nan, np.nan


# detect possible presence of [NII] and [SII] nebular emission lines
def fit_emission_lines(s_obj, w_obj, e_lines,
                       thr_l=0.05, thr_u=1., s_thr_l=0.1, s_thr_u=0.5,
                       d_w=6., d_e_l=3.,
                       verbose=False, fixed_cont=True):

    # first try to detect strong emission peaks and try to correct them if possible
    # TODO: is this even necessary with RV constrained fit?

    c_vel = const.c.value  # in units of m/s
    amp_init = 0.5
    std_init = 0.3
    # create model to be fitted
    if not fixed_cont:
        e_fit = models.Const1D(amplitude=np.median(s_obj))
    else:
        e_fit = models.Const1D(amplitude=0., fixed={'amplitude': True})
    rvs_peak = []
    for i_l, e_l in enumerate(e_lines):
        # select the best suitable peak in the vicinity of selected emission line
        idx_ep = np.where(np.logical_and(w_obj >= e_l-d_e_l,
                                         w_obj <= e_l+d_e_l))[0]
        idx_max = idx_ep[np.argmax(s_obj[idx_ep])]
        e_mean = w_obj[idx_max]
        # set means to init rv value as mean rv of both strongest peaks
        rvs_peak.append((e_mean - e_l) / e_l * c_vel)
    rv_init = np.mean(rvs_peak)

    for i_l, e_l in enumerate(e_lines):
        # # Version1 - add emission peak to the fitting procedure
        # e_fit += models.Gaussian1D(amplitude=amp_init, mean=rvs_peak[i_l] / c_vel * e_lines[i_l] + e_lines[i_l],
        #                            stddev=std_init, fixed={'mean': True})  # mean is fixed for simplicity

        # # Version2 - add emission peak to the fitting procedure - start fitting at laboratory peak values
        # e_fit += models.Gaussian1D(amplitude=amp_init, mean=e_l, stddev=std_init)

        # Version3 - add emission peak to the fitting procedure - start at mean rv of the strongest nearby peaks
        e_fit += models.Gaussian1D(amplitude=amp_init, mean=rv_init / c_vel * e_lines[i_l] + e_lines[i_l],
                                   stddev=std_init)

        e_fit.bounds['amplitude_' + str(i_l + 1)] = [0., 2.]
        e_fit.bounds['stddev' + str(i_l + 1)] = [0., 0.5]

    # tie RV value (or means of gaussians) of both emission lines
    def tie_RVs(fit_model, ref_wvl=None):
        c_vel = const.c.value  # in units of m/s
        rv2 = (fit_model.mean_2 - ref_wvl[1]) / ref_wvl[1] * c_vel
        wvl1 = rv2 / c_vel * ref_wvl[0] + ref_wvl[0]
        return wvl1
    e_fit.tied['mean_1'] = partial(tie_RVs, ref_wvl=e_lines)

    fit_t = fitting.LevMarLSQFitter()  # SLSQPLSQFitter
    fit_res = fit_t(e_fit, w_obj, s_obj, maxiter=400)
    fitted_curve = fit_res(w_obj)

    line_det = 0
    converged = True
    wvl_peaks = []
    g_amps = []
    g_std = []
    for i_l, e_l in enumerate(e_lines):
        wvl_peaks.append(fit_res[i_l+1].mean.value)
        g_amps.append(fit_res[i_l + 1].amplitude.value)
        g_std.append(fit_res[i_l + 1].stddev.value)
        if thr_l < fit_res[i_l+1].amplitude.value < thr_u and s_thr_l < fit_res[i_l+1].stddev.value < s_thr_u:
            line_det += 1
    # compute fitted rv value
    rv_fitted = np.mean((wvl_peaks - np.array(e_lines)) / np.array(e_lines) * c_vel)

    if verbose:
        print('Emission lines fit results, is detected? =', line_det)
        print('RV init peaks:', rvs_peak)
        print('Peaks:', wvl_peaks, np.array(wvl_peaks) - np.array(e_lines))
        print('Amps:', g_amps)
        print('Std:', g_std)

    # check if fitted curve even exists -> test if fit did converge
    # TODO: find possible better way to check this
    if ~np.isfinite(fitted_curve).all() or (np.all(np.array(g_amps) == amp_init) and np.all(np.array(g_std) == std_init)):
        # fit did not converge properly
        line_det = 0
        converged = False

    e_fit = None
    fit_res = None
    del e_fit
    del fit_res

    # return RV converted to km/s
    return line_det, wvl_peaks, rv_fitted/1000., fitted_curve, converged


# detect possible presence of SB2 spectrum
def sb2_ccf(s_obj, s_ref,
            a_thr=10., s_thr_l=5., s_thr_u=20., min_peak_sep=5, max_peak_offset=75,
            verbose=False):

    # create array of ones, with the same length as investigated signal - will be used for normalization
    ones_y = np.full_like(s_obj, fill_value=1.)
    # remove strong outliers in observed spectrum before correlating with reference spectrum
    # strong spikes could produce spurious signal in derived CCF signal
    ccf_y = deepcopy(s_obj)
    ccf_y[ccf_y >= 1.2] = 1.2
    ccf_y[ccf_y <= 0.] = 0.
    # perform a small amount of noise filtering in the observed signal
    ccf_y = medfilt(ccf_y, kernel_size=3)
    # perform signal correlation
    ccf_y = correlate(1. - ccf_y, 1. - s_ref, mode='same', method='fft')  # subtract from 1 to remove continuum
    ccf_norm = correlate(ones_y, ones_y, mode='same', method='fft')
    ccf_x = np.arange(len(ccf_y)) - int(len(ccf_y)/2.)  # could be wrong to +/- 1 for even lengths
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

    # detect strongest peaks
    idx_peaks = argrelextrema(ccf_y, np.greater, order=2, mode='clip')[0]
    # limit peaks to a reasonable values inside expected CCF RV values
    idx_peaks = idx_peaks[np.abs(ccf_y[idx_peaks]) < 120]
    if len(idx_peaks) >= n_gauss:
        # select n_gauss strongest peaks
        idx_pekas_best = idx_peaks[np.argsort(ccf_y[idx_peaks])[-n_gauss:]]
        ccf_extrema = ccf_x[idx_pekas_best]
    else:
        ccf_extrema = [-40., 0., 40.]

    # approximate uncorrelated levels with a polynomial of low degree (usually up to max 3rd degreee)
    # e_fit = models.Polynomial1D(2, c0=np.median(ccf_y), c1=0., c2=-1.*1e-8)
    e_fit = models.Polynomial1D(1, c0=np.median(ccf_y), c1=0.)
    # e_fit = models.Trapezoid1D(amplitude=np.percentile(ccf_y, 90), x_0=0., width=0., slope=1., fixed={'width': True})
    # e_fit = models.Const1D(amplitude=np.mean(ccf_y))
    for i_l in range(n_gauss):
        g_amp = 0.8
        if i_l == 1:
            g_amp = 1.2
        e_fit += models.Gaussian1D(amplitude=g_amp, stddev=8,
                                   # mean=(n_gauss - 2)*40. - 40.*i_l,
                                   mean=ccf_extrema[i_l]
                                   )
        e_fit.bounds['mean_' + str(i_l + 1)] = [-w_ccf+2, w_ccf-2]
        e_fit.bounds['stddev_' + str(i_l + 1)] = [0., 70.]
        e_fit.bounds['amplitude_' + str(i_l + 1)] = [0., 2.]
    # fit_t = fitting.SLSQPLSQFitter()
    fit_t = fitting.LevMarLSQFitter()
    fit_res = fit_t(e_fit, ccf_x, ccf_y, maxiter=400)

    # is there a sign of multiplicity in the spectrum?
    # define number of meaningful peaks in the ccf curve
    is_sb2 = False
    n_peaks = 0
    ccf_peaks = []
    ccf_peaks_valid = []
    g_amps = []
    g_amps_valid = []
    g_std = []
    for i_l in range(n_gauss):
        ccf_peaks.append(fit_res[i_l + 1].mean.value)
        g_amps.append(fit_res[i_l + 1].amplitude.value)
        g_std.append(fit_res[i_l + 1].stddev.value)
        # check if fitted peak is meaningful - has a correct shape and amplitude, doesn't have too large RV offest
        if fit_res[i_l + 1].amplitude.value > a_thr and \
                s_thr_l < fit_res[i_l + 1].stddev.value < s_thr_u and \
                np.abs(fit_res[i_l + 1].mean.value) < max_peak_offset:
            n_peaks += 1
            ccf_peaks_valid.append(fit_res[i_l + 1].mean.value)
            g_amps_valid.append(fit_res[i_l + 1].amplitude.value)
    # did we detect more than one peak in the ccf function
    if n_peaks > 1:
        # check if the strongest valid peaks are sufficiently separated
        im1, im2 = np.argsort(g_amps_valid)[::-1][:2]
        if np.abs(ccf_peaks_valid[im1] - ccf_peaks_valid[im2]) > min_peak_sep:
            is_sb2 = True

    # is peak centered?
    is_cent = True
    ccf_peak = ccf_x[np.argmax(ccf_y)]
    if np.abs(ccf_peak) >= 5.:
        is_cent = False

    if verbose:
        print('SB2 fit results')
        print(is_sb2, is_cent, ccf_peak)
        print('CCF extrema:', ccf_extrema)
        print('C poly:', [fit_res[0].c0.value, fit_res[0].c1.value])#, fit_res[0].c2.value]
        print('Peaks:', ccf_peaks)
        print('Amps:', g_amps)
        print('Std:', g_std)

    fit_res_y = fit_res(ccf_x)
    e_fit = None
    fit_res = None
    del e_fit
    del fit_res

    return is_sb2, is_cent, [ccf_x, ccf_y, fit_res_y, ccf_peak, ccf_peaks]


def determine_sky_emission_strength(s_obj, w_obj, linelist,
                                    rv=0, rv_bary=0.,
                                    e_thr=0.15, d_wvl=0.2):
    # shift emission lines from the observed to the stellar reference frame
    c_vel = const.c.value / 1000.  # value of light speed in km/s
    rv_shift = rv - rv_bary
    linelist_star_frame = linelist * (1 - rv_shift / c_vel)

    # check lines one by one - simple peak thresholding, nothing fancy
    linelist_present = []
    linelist_present_neg = []
    for sky_e in linelist_star_frame:
        # specify wlv neighbourhood of the observed sky emission line
        idx_w = np.abs(w_obj - sky_e) <= d_wvl
        if np.sum(idx_w) <= 0:
            # skip this emission line as it is located outside the investigated range
            continue
        # determine too low atmospheric/background correction
        max_amp = np.nanmax(s_obj[idx_w])
        if max_amp > 1.*e_thr:
            linelist_present.append(sky_e)
        # determine too strong atmospheric/background correction
        min_amp = np.nanmin(s_obj[idx_w])
        if min_amp < -1.*e_thr:
            linelist_present_neg.append(sky_e)

    # return lines that are possiblly present in the spectrum difference
    return linelist_star_frame, linelist_present, linelist_present_neg


def emis_peak_width(s_obj, w_obj, peak_loc, w_peak,
                    at_perc=10.):
    # investigate emission line peak and return its width at selected peak depth
    idx_range = np.where(np.abs(w_obj - peak_loc) <= w_peak)[0]
    # determine highest pixel location at the selected location and range
    idx_peak = idx_range[np.argmax(s_obj[idx_range])]
    # peak strength and its limit at selected peak percentage
    p_offset = np.nanmedian(s_obj)
    p_max = s_obj[idx_peak] - p_offset
    p_thr = p_max * at_perc/100. + p_offset

    # determine peak limits at selected threshold p_thr
    def _peak_lim(w_s):
        # w_s == step width
        idx_cur = deepcopy(idx_peak)
        # find boundary bellow requested threshold
        while s_obj[idx_cur] >= p_thr:
            idx_cur += w_s

        # determine a spectrum point bellow and above the thresholding line
        y1 = w_obj[idx_cur]
        y2 = w_obj[idx_cur-1]
        x1 = s_obj[idx_cur]
        x2 = s_obj[idx_cur-1]
        # interpolate to find width
        return np.interp(p_thr, [x1, x2], [y1, y2])

    # compute peak width using both left and right peak wings
    width_l = _peak_lim(-1)
    width_r = _peak_lim(1)
    p_width = width_r - width_l
    # return emission line width as velocity - km/s
    return p_width / peak_loc * const.c.value/1000.


# --------------------------------------------------------
# ---------------- Main analysis procedure ---------------
# --------------------------------------------------------
def process_selected_id(s_id):
    print('\nWorking on object '+str(s_id))
    t_s = time()
    # define flag parameter that will describe processing problem(s) in resulting table
    proc_flag = 0

    # get parameters of the observed object
    idx_object = np.where(general_data['sobject_id'] == s_id)[0]
    if len(idx_object) == 0:
        print('Sobject', s_id, 'not in the dataset.')
        return []

    object_parameters = general_data[idx_object][0]

    # get both spectra of the object and it's reduced reference median comparison spectra
    spectra_object_c1 = spectra_ccd1[idx_object, :][0]
    spectra_object_c3 = spectra_ccd3[idx_object, :][0]
    spectra_object_c3_SII = spectra_ccd3_SII[idx_object, :][0]
    spectra_median_c1 = spectra_median_ccd1[idx_object, :][0]
    spectra_median_c3 = spectra_median_ccd3[idx_object, :][0]
    spectra_median_c3_SII = spectra_median_ccd3_SII[idx_object, :][0]

    # check validity of reference spectra
    if not np.isfinite(spectra_median_c3).any():
        proc_flag += 0b10000000
    if not np.isfinite(spectra_median_c1).any():
        proc_flag += 0b01000000

    spectra_object_c1 = renorm_by_ref(spectra_object_c1, spectra_median_c1, wvl_val_ccd1)
    spectra_object_c3 = renorm_by_ref(spectra_object_c3, spectra_median_c3, wvl_val_ccd3)
    spectra_object_c3_SII = renorm_by_ref(spectra_object_c3_SII, spectra_median_c3_SII, wvl_val_ccd3_SII)

    # compute MSE between median and observed spectrum
    rms_c1 = (np.nanmedian((spectra_median_c1 - spectra_object_c1)**2))
    rms_c3 = (np.nanmedian((spectra_median_c3 - spectra_object_c3)**2))
    rms_c3_SII = (np.nanmedian((spectra_median_c3_SII - spectra_object_c3_SII)**2))

    # flag badly correlated spectra and median spectra
    if rms_c3 >= 0.002:
        proc_flag += 0b00100000
    if rms_c1 >= 0.008:
        proc_flag += 0b00010000

    # correct part of the spectrum that might not be filled with valid data
    spectra_object_c3 = fill_nans(spectra_object_c3)
    spectra_object_c3_SII = fill_nans(spectra_object_c3_SII)
    spectra_median_c3 = fill_nans(spectra_median_c3)
    spectra_median_c3_SII = fill_nans(spectra_median_c3_SII)
    spectra_object_c1 = fill_nans(spectra_object_c1)
    spectra_median_c1 = fill_nans(spectra_median_c1)

    # compute spectra difference and division
    spectra_dif_c1 = spectra_object_c1 - spectra_median_c1
    spectra_dif_c3 = spectra_object_c3 - spectra_median_c3
    spectra_dif_c3_SII = spectra_object_c3_SII - spectra_median_c3_SII
    spectra_div_c1 = spectra_object_c1 / spectra_median_c1
    spectra_div_c3 = spectra_object_c3 / spectra_median_c3
    spectra_div_c3_SII = spectra_object_c3_SII / spectra_median_c3_SII

    # find strange things in spectra comparision:
    # - bad fit to reference spectra
    # - large continuum difference
    # - emission like ccd spikes
    # - discontinuities in observed spectra

    e_sky_thr = 0.1
    sky_all, sky_present, sky_present_neg = determine_sky_emission_strength(spectra_dif_c3, wvl_val_ccd3, sky_lineslist['Ang'],
                                                           rv=object_parameters['rv_guess_shift'],
                                                           rv_bary=object_parameters['v_bary'], e_thr=e_sky_thr)
    # set sky emission warning quality flag if many sky line were detected to be present in the spectrum difference
    if len(sky_present) >= 4 or len(sky_present_neg) >= 4:
        proc_flag += 0b00000100

    # compute cross-correlation function for both bands to identify possible SB2 object and wrong wavelength calibration
    sb2_c3, wvl_c3, ccf_res_c3 = sb2_ccf(spectra_object_c3[idx_ccf_ccd3], spectra_median_c3[idx_ccf_ccd3],
                                         a_thr=0.45, s_thr_l=2.5, s_thr_u=20., max_peak_offset=115,
                                         verbose=VERBOSE)
    sb2_c1, wvl_c1, ccf_res_c1 = sb2_ccf(spectra_object_c1[idx_ccf_ccd1], spectra_median_c1[idx_ccf_ccd1],
                                         a_thr=0.45, s_thr_l=2.5, s_thr_u=20., max_peak_offset=125,
                                         verbose=VERBOSE)

    # very high chance of being a SB2 binary star -> raise processing quality flag
    if sb2_c1 and sb2_c3:
        proc_flag += 0b00001000

    # add validity of wavelength solution to the processing flag
    if not wvl_c3:
        proc_flag += 0b00000010
    if not wvl_c1:
        proc_flag += 0b00000001

    nii_det, nii_peaks, nii_rv, nii_fit, nii_con = fit_emission_lines(spectra_dif_c3, wvl_val_ccd3, NII_bands, verbose=VERBOSE)
    sii_det, sii_peaks, sii_rv, sii_fit, sii_con = fit_emission_lines(spectra_dif_c3_SII, wvl_val_ccd3_SII, SII_bands, verbose=VERBOSE)
    nii_EW = np.trapz(nii_fit, wvl_val_ccd3)
    sii_EW = np.trapz(sii_fit, wvl_val_ccd3_SII)

    # integrate part of the spectrum representing H bands
    Ha_EW, _, Ha_rv = integrate_ew_spectra(spectra_dif_c3, wvl_val_ccd3,
                                    offset=0., reutrn_maxpeak_rv=True, wvl_rv_ref=HALPHA_WVL,
                                    wvl_range=[HALPHA_WVL - wvl_int_range, HALPHA_WVL + wvl_int_range])
    Ha_EW_abs, Ha_EW_asym = integrate_ew_spectra(spectra_dif_c3, wvl_val_ccd3,
                                    absolute=True, offset=0.,
                                    wvl_range=[HALPHA_WVL - wvl_int_range, HALPHA_WVL + wvl_int_range])
    Hb_EW, _, Hb_rv = integrate_ew_spectra(spectra_dif_c1, wvl_val_ccd1,
                                    offset=0., reutrn_maxpeak_rv=True, wvl_rv_ref=HBETA_WVL,
                                    wvl_range=[HBETA_WVL - wvl_int_range, HBETA_WVL + wvl_int_range])
    Hb_EW_abs, Hb_EW_asym = integrate_ew_spectra(spectra_dif_c1, wvl_val_ccd1,
                                    absolute=True, offset=0.,
                                    wvl_range=[HBETA_WVL - wvl_int_range, HBETA_WVL + wvl_int_range])

    # determine peak width in velocity units
    Ha_W10 = emis_peak_width(spectra_dif_c3, wvl_val_ccd3, HALPHA_WVL, wvl_peak_range, at_perc=10.)

    # add to results
    output_array = [s_id, Ha_EW, Hb_EW, Ha_EW_abs, Hb_EW_abs, Ha_rv, Hb_rv,
                    Ha_W10, Ha_EW_asym, Hb_EW_asym, sb2_c3, sb2_c1,
                    nii_det, sii_det, nii_EW, sii_EW, nii_rv, sii_rv, proc_flag]

    # write_csv_string = ','.join([str(v) for v in output_array]) + '\n'
    # with open(results_csv_out, 'a') as txt_csv_out:
    #     txt_csv_out.write(write_csv_string)

    # --------------------------------------------------------
    # ---------------- Plot results --------------------------
    # --------------------------------------------------------
    suffix = ''  #
    # print(' Plotting results'

    if PLOT_FIG:
        fig, axs = plt.subplots(4, 3, figsize=(16, 12))
        fig.suptitle('Red flag: ' + str(object_parameters['red_flag']) + ',    flag sp: ' + str(object_parameters['flag_sp']) + ',    proc flag: ' + str(int(proc_flag)))

        # h-alpha plots
        axs[0, 0].plot(wvl_val_ccd3, spectra_median_c3, color='red', linewidth=0.5, label='Median')
        axs[0, 0].plot(wvl_val_ccd3, spectra_object_c3, color='black', linewidth=0.5, label='Star')
        axs[0, 0].set(xlim=(HALPHA_WVL - wvl_plot_range_s, HALPHA_WVL + wvl_plot_range_s), ylim=(0.1, 1.3),
                      ylabel='Flux - H alpha, [NII]', xlabel='MSE = {:.4f}'.format(rms_c3))
        axs[0, 0].legend()
        axs[0, 1].plot(wvl_val_ccd3, spectra_dif_c3, color='black', linewidth=0.5)
        axs[0, 1].set(xlim=(HALPHA_WVL - wvl_plot_range_s, HALPHA_WVL + wvl_plot_range_s), ylim=(-0.3, 0.8),
                      ylabel='Difference lux', xlabel='Valid [NII] lines = {:.0f}'.format(nii_det))
        axs[0, 2].plot(wvl_val_ccd3, spectra_div_c3, color='black', linewidth=0.5)
        ymlim_max = np.max(spectra_div_c3[np.abs(wvl_val_ccd3 - HALPHA_WVL) <= wvl_int_range]) * 1.05  # determine ylim
        axs[0, 2].set(xlim=(HALPHA_WVL - wvl_plot_range_z, HALPHA_WVL + wvl_plot_range_z), ylim=(0.7, ymlim_max),
                      ylabel='Division flux')
        axs[0, 1].axhline(0., c='C0', alpha=0.75)
        axs[0, 2].axhline(1., c='C0', alpha=0.75)
        for ip in range(3):
            axs[0, ip].axvline(HALPHA_WVL, c='black', alpha=0.45, ls='--')
            axs[0, ip].axvline(HALPHA_WVL - wvl_int_range, c='black', alpha=0.6, ls='--')
            axs[0, ip].axvline(HALPHA_WVL + wvl_int_range, c='black', alpha=0.6, ls='--')
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
                      ylabel='Flux - [SII]', xlabel='MSE = {:.4f}'.format(rms_c3_SII))
        axs[1, 0].legend()
        axs[1, 1].plot(wvl_val_ccd3_SII, spectra_dif_c3_SII, color='black', linewidth=0.5)
        axs[1, 1].set(xlim=(SII_WVL - 2.*wvl_plot_range_z, SII_WVL + 2.*wvl_plot_range_z), ylim=(-0.3, 0.8),
                      ylabel='Difference flux', xlabel='Valid [SII] lines = {:.0f}'.format(sii_det))
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
                      ylabel='Flux - H beta', xlabel='Wavelength, MSE = {:.4f}'.format(rms_c1))
        axs[2, 0].legend()
        axs[2, 1].plot(wvl_val_ccd1, spectra_dif_c1, color='black', linewidth=0.5)
        axs[2, 1].set(xlim=(HBETA_WVL - wvl_plot_range_s, HBETA_WVL + wvl_plot_range_s), ylim=(-0.3, 2),
                      ylabel='Difference flux', xlabel='Wavelength')
        axs[2, 2].plot(wvl_val_ccd1, spectra_div_c1, color='black', linewidth=0.5)
        ymlim_max = np.max(spectra_div_c1[np.abs(wvl_val_ccd1 - HBETA_WVL) <= wvl_int_range]) * 1.05  # determine ylim
        axs[2, 2].set(xlim=(HBETA_WVL - wvl_plot_range_z, HBETA_WVL + wvl_plot_range_z), ylim=(0.7, ymlim_max),
                      ylabel='Division flux', xlabel='Wavelength')
        axs[2, 1].axhline(0., c='C0', alpha=0.75)
        axs[2, 2].axhline(1., c='C0', alpha=0.75)
        for ip in range(3):
            axs[2, ip].axvline(HBETA_WVL, color='black', alpha=0.45, ls='--')
            axs[2, ip].axvline(HBETA_WVL - wvl_int_range, color='black', alpha=0.6, ls='--')
            axs[2, ip].axvline(HBETA_WVL + wvl_int_range, color='black', alpha=0.6, ls='--')

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
        if np.isfinite(ccf_res_c3[1]).all():
            axs[3, 0].set(xlim=(-plot_w, plot_w), xlabel='Pixel shifts - is SB2? = '+str(int(sb2_c3)),
                          ylim=(-0.2, 1.6))
        if np.isfinite(ccf_res_c1[1]).all():
            axs[3, 1].set(xlim=(-plot_w, plot_w), xlabel='Pixel shifts - is SB2? = '+str(int(sb2_c1)),
                          ylim=(-0.2, 1.6))

        # visualize position and detected sky lines
        axs[3, 2].plot(wvl_val_ccd3, spectra_dif_c3, color='black', linewidth=0.5)
        axs[3, 2].set(xlim=(HALPHA_WVL - wvl_plot_range_s, HALPHA_WVL + wvl_plot_range_s), ylim=(-0.2, 0.2),
                      ylabel='Difference flux', xlabel='Strongest sky emissions detected = {:.0f}, {:.0f}'.format(len(sky_present), len(sky_present_neg)))
        # add thresholding value for detection of emission peaks
        axs[3, 2].axhline(e_sky_thr, color='black', alpha=0.6, ls='--')
        axs[3, 2].axhline(-1. * e_sky_thr, color='black', alpha=0.6, ls='--')
        for e_sky in sky_all:  # visualize all sky lines and their transformed wvl position
            axs[3, 2].axvline(e_sky, color='C2', alpha=0.75, ls='--')
        for e_sky in sky_present:  # highlight only detected strong pozitive sky lines
            axs[3, 2].axvline(e_sky + 1.5 * dwvl_ccd3, color='C3', alpha=0.75, ls='--')
        for e_sky in sky_present_neg:  # highlight only detected strong negative sky lines
            axs[3, 2].axvline(e_sky - 1.5 * dwvl_ccd3, color='C1', alpha=0.75, ls='--')

        # add grid lines to every plot available
        for ip in range(3):
            for il in range(4):
                axs[il, ip].grid(color='black', alpha=0.2, ls='--')

        s_date = np.int32(s_id/10e10)
        plt.tight_layout()
        # plt.subplots_adjust(wspace=0., hspace=0.)
        if not TEST_RUN:
            plt.savefig(str(s_date)+'/'+str(s_id)+suffix+'.png', dpi=200)
        else:
            plt.savefig('tests/'+str(s_id)+suffix+'.png', dpi=200)
        plt.close()

        fig = None
        axs = None
        del fig
        del axs

    t_e = time()
    print(' Elapsed time {:.1f}s'.format(t_e - t_s))

    return output_array


# create all possible output subdirectories
print('Creating sub-folders in advance')
sobject_dates = np.unique(np.int32(sobject_ids/10e10))
for s_date in sobject_dates:
    move_to_dir(str(s_date))
    chdir('..')
for s_date in ['strongest', 'strongest_absolute', 'tests', 'SB2', 'nebular']:
    move_to_dir(str(s_date))
    chdir('..')

out_suffix = ''
if TEST_RUN:
    out_suffix += '_tests'
    VERBOSE = True
    PLOT_FIG = True

results_csv_out = 'results_H_lines' + out_suffix + '.csv'
txt_out = open(results_csv_out,  'w')
txt_out.write(','.join([str(sc) for sc in results.colnames]) + '\n')
txt_out.close()

# without any multiprocessing - for test purpose only
if TEST_RUN:
    sobject_ids = [140808001101103, 150106002201377, 150405001501216, 150405002001246, 150428000101176, 150607003601066,
                   151009005101283, 160129004701296, 160326000101077, 160420003801237, 170109002801349, 170217002201245,
                   170408003501326, 170509002201287, 170604002601330, 170828001601228, 170911002601194, 140112002301230,
                   140112002301339, 140608003101196, 140609002601035, 140609003601247, 140711000601112, 140713001901218,
                   170603004601078, 170604002101031, 180131002701292, 140311005601066, 140311005601088, 140312001701004,
                   140608003101355, 141102003201190, 150210002701176, 150411004601220, 150428002601115, 150428003101113,
                   150428003101134, 151110002101261, 160325002101011, 160401004401107, 160423003801226, 170511000601249,
                   140112002301103, 140112002301176, 140118002501376, 140609002601375, 150504003001213, 150607003601372,
                   151225002101232, 151228003301129, 160415004601132, 170530004101189, 170603003101013, 170603004101373,
                   150409007601105, 150409007601271, 150410002801043, 150411005101227, 150427000801008, 150427002301128,
                   150427002301134, 150427004801127, 150427002301365, 150428002101107, 150607003601060, 150703003101333,
                   150705002901303, 170113001601235, 170119003101214, 170127001601294, 170312001601249, 170413004601298,
                   131216001601159, 140209002201249, 140305002601079, 140309004701291, 140310004301119, 140311006101192,
                   140414005101274, 151219003101351, 151229004001146, 151230003201255, 160325002101026, 160328004701205,
                   160331005301092, 160421003601079, 160424002601337, 160812003601067, 161012002101178, 131121001901282,
                   131123004101054, 131216001601054, 140118003001080, 140307001101317, 140309002101085, 160523003501122,
                   160525003601045, 160529005401023, 160530001601177, 160530004501223, 160530005501177, 160608002501389,
                   171206002601361, 160421005101358, 160426004001249, 150703004101143, 160401002601397, 170725004601301,
                   140314003601104, 150831005001212, 170601003101248, 170202000701343, 140811002101039, 150103003501177,
                   150829004301276, 160524006601134, 140413003201189, 160112001601130, 160527002101081, 170416005801142,
                   170105003101389, 170106002601378, 170106004101089, 170106004601001, 170106004601083, 170107002601006,
                   170711003501086, 180126002601034, 171208002601103, 180131004101338, 180103001601060, 171003002601045,
                   170905002101249, 170805004601106, 170725005101394, 170603005601057, 170404003101254, 160111001601202,
                   140112002301115, 140309003601373, 150607004101179, 140600921011178, 140609003101388, 150428001601398,
                   140309002101378, 140305001301198, 140113002901294, 140309003601367, 140308001401205, 140309002601352,
                   140308003801011, 140308003801056, 140309004101056, 140301004701056, 140308003801394, 140309003601014,
                   140808002101004, 140809001601034, 171102001601041, 170513003501193, 170109002101011, 190225003701061,
                   150606002401302, 140812003801021, 160419002101393, 140812003801202, 140307002601272, 170515004101070,
                   140824003501176, 141231004001024, 150607003601015, 170108002201038, 170806005801117]
    '''
    sobject_ids = sobject_ids[sobject_ids > 190000000000000]
    '''
    for so_id in np.unique(sobject_ids):
        so_id_results = process_selected_id(so_id)
        if len(so_id_results) > 0:
            results.add_row(so_id_results)
    results.write(results_csv_out[:-3] + 'fits', overwrite=True)
else:

    # # first half of the dataset
    # sobject_ids = sobject_ids[:330000]
    # out_fits_file = 'results_H_lines' + out_suffix + '_1.fits'

    # second half of the dataset
    sobject_ids = sobject_ids[330000:]
    out_fits_file = 'results_H_lines' + out_suffix + '_2.fits'

    # multiprocessing - run in multiple subsets as this might reduce memory usage and clear saved astropy.fitting models
    n_subsets = 50
    sobj_ranges = np.int32(np.linspace(0, len(sobject_ids), n_subsets+1))
    for i_subset in range(n_subsets):
        pool = Pool(processes=n_multi)
        subset_sobject_ids = sobject_ids[sobj_ranges[i_subset]: sobj_ranges[i_subset+1]]
        process_return = pool.map(process_selected_id, subset_sobject_ids)
        pool.close()
        del pool
        pool = None

        # insert individual object results into resulting table
        for so_id_results in process_return:
            if len(so_id_results) > 0:
                results.add_row(so_id_results)

        # save partial results at the end of every data subset
        print('Saving partial results to a table')
        results.write(out_fits_file, overwrite=True)
