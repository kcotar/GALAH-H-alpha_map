import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, argrelextrema, medfilt
from scipy.interpolate import lagrange, splrep, splev

from lmfit import minimize, Parameters, report_fit, Minimizer
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, LinearModel, ConstantModel, PolynomialModel


# function to be minimized
def gaussian_fit(parameters, data, wvls, continuum, evaluate=True):
    n_keys = (len(parameters)) / 3
    # function_val = parameters['offset']*np.ones(len(wvls))
    function_val = np.array(continuum)
    for i_k in range(n_keys):
        function_val -= parameters['amp'+str(i_k)] * np.exp(-0.5 * (parameters['wvl'+str(i_k)] - wvls) ** 2 / parameters['std'+str(i_k)])
    if evaluate:
        # likelihood = np.nansum(np.power(data - function_val, 2))
        likelihood = np.power(data - function_val, 2)
        return likelihood
    else:
        return function_val


def fit_h_profile_initial(spectrum, wavelengths, wvl_center=0, verbose=False, method='savgol',
                          window=45, savgol_order=2, poly_order=20):
    # median/mean method with sliding windows
    if method == 'sliding':
        n_obs_mean = np.convolve(np.ones(len(spectrum)), np.ones(window), mode='same')
        y_fit = np.convolve(spectrum, np.ones(window), mode='same')/n_obs_mean
    # savgol filtering
    elif method == 'savgol':
        y_fit = savgol_filter(spectrum, window, savgol_order)
    # polynominal method
    elif method == 'poly':
        chb_coef = np.polynomial.chebyshev.chebfit(wavelengths-wvl_center, spectrum, poly_order)
        y_fit = np.polynomial.chebyshev.chebval(wavelengths-wvl_center, chb_coef)
    elif method == 'spline':
        bspline = splrep(wavelengths - wvl_center, spectrum)  # k and der must be odd
        y_fit = splev(wavelengths - wvl_center, bspline)  # der <= k
    # return fitted function
    return y_fit


def sigma_clip(y, y_ref, idx_y_use=None,
               std_lower=2., std_upper=2., wvl=None, wvl_center=None, wvl_center_range=1.5, diagnostics=False):
    delta = y_ref - y
    if idx_y_use is None:
        sigma = np.nanstd(delta)
        mean = np.nanmean(delta)
    else:
        sigma = np.nanstd(delta[idx_y_use])
        mean = np.nanmean(delta[idx_y_use])
    if diagnostics:
        plt.plot(wvl, y, color='black')
        plt.plot(wvl, y_ref, color='red')
        delta_offest = 0.7
        plt.plot(wvl, delta-mean+delta_offest, color='blue')
        plt.axhline(y=delta_offest)
        plt.axhline(y=delta_offest - sigma * std_lower)
        plt.axhline(y=delta_offest + sigma * std_upper)
        plt.show()
    idx_bad = np.logical_or((delta - mean) < -1. * sigma * std_lower,
                            (delta - mean) > sigma * std_upper)
    # region around center of absorption line should always be included in the data
    if wvl is not None and wvl_center is not None:
        idx_line_center = np.logical_and(wvl >= (wvl_center - wvl_center_range),
                                         wvl <= (wvl_center + wvl_center_range))
        idx_bad[idx_line_center] = False
    return np.logical_not(idx_bad)


def fit_h_profile_with_clipping(spectrum, wavelengths, wvl_center=0., verbose=False, diagnostics=False,
                                steps=3, std_lower=2., std_upper=2., std_step_change=0.05,
                                return_ommitted=False, profile_kwargs=None):
    idx_use = np.isfinite(spectrum)
    for n_s in range(steps):
        len_use = np.sum(idx_use)
        if verbose:
            print len_use
        fit_model, fit_param = fit_h_profile(spectrum[idx_use], wavelengths[idx_use], wvl_center=wvl_center,
                                             verbose=False, **profile_kwargs)
        spectrum_fitted = fit_model.eval(fit_param, x=wavelengths - wvl_center)
        # end
        idx_use = np.logical_and(idx_use,
                                 sigma_clip(spectrum, spectrum_fitted, idx_y_use=idx_use,
                                            std_lower=std_lower - n_s*std_step_change,
                                            std_upper=std_upper - n_s*std_step_change,
                                            wvl=wavelengths, wvl_center=wvl_center, diagnostics=diagnostics))
        # steps is maximum number of steps, break if no change in number of excluded points
        if np.sum(idx_use) >= len_use:
            break
    if verbose:
        fit_param.pretty_print()

    # return selected results
    if return_ommitted:
        return spectrum_fitted, np.logical_not(idx_use)
    else:
        return spectrum_fitted


def fit_h_profile_eval(params, model, X_ref, Y_ref):
    Y_model = model.eval(params, x=X_ref)
    return np.power(Y_ref - Y_model, 2)


def fit_h_profile(spectrum, wavelengths, wvl_center=0., verbose=False,
                  gauss1=False, lorentz1=False, voigt1=False, voigt2=False):
    # parameters container
    params = Parameters()

    # basic linear continuum
    line = LinearModel(prefix='line_')  # linear model
    params.add('line_intercept', value=1., min=0.8, max=1.2)
    params.add('line_slope', value=0., min=-0.1, max=0.1)
    # other possible continuum profiles
    # line = ConstantModel(prefix='const_')  # constant or
    # params.add('const_c', value=1., min=0.92, max=1.08)
    # poly_degree = 5
    # line = PolynomialModel(prefix='poly_', degree=poly_degree)
    # for i_p in range(poly_degree+1):
    #     if i_p == 0:
    #         params.add('poly_c'+str(i_p), value=1., min=0.8, max=1.2)
    #     else:
    #         params.add('poly_c'+str(i_p), value=0., min=-0.1, max=0.1)
    final_model = line

    if voigt1:
        profile_v1 = VoigtModel(prefix='voigt1_')
        final_model -= profile_v1
        params.add('voigt1_center', value=0, min=-0.2, max=+0.2)  # fix center wavelength of the profile
        params.add('voigt1_sigma', value=0.5, min=0.0, max=5.0)
        params.add('voigt1_gamma', value=0.5, min=0.0, max=5.0)  # manipulate connection between sigma and gamma
        params.add('voigt1_amplitude', value=1., min=0.0, max=10.0)

    if voigt2:
        profile_v2 = VoigtModel(prefix='voigt2_')
        final_model -= profile_v2
        params.add('voigt2_center', value=0, min=-0.2, max=+0.2)  # fix center wavelength of the profile
        params.add('voigt2_sigma', value=0.5, min=0.0, max=20.0)
        params.add('voigt2_gamma', value=0.5, min=0.0, max=20.0)  # manipulate connection between sigma and gamma
        params.add('voigt2_amplitude', value=1., min=0.0, max=10.0)

    if gauss1:
        profile_g1 = GaussianModel(prefix='gauss1_')
        final_model -= profile_g1
        params.add('gauss1_center', value=0, min=-0.2, max=+0.2)
        params.add('gauss1_sigma', value=0.5, min=0.0, max=5.0)
        params.add('gauss1_amplitude', value=1., min=0.0, max=10.0)

    if lorentz1:
        profile_g1 = LorentzianModel(prefix='lorentz1_')
        final_model -= profile_g1
        params.add('lorentz1_center', value=0, min=-0.2, max=+0.2)
        params.add('lorentz1_sigma', value=0.5, min=0.0, max=5.0)
        params.add('lorentz1_amplitude', value=1., min=0.0, max=10.0)

    # start fitting procedure
    # begin - FIT version 1
    abs_line_fit = final_model.fit(spectrum, params=params, x=wavelengths - wvl_center, method='least_square')
    # end
    # begin - FIT version 2
    # abs_line_fit = minimize(fit_h_profile_eval, params, method='least_square', args=(final_model, wavelengths - wvl_center, spectrum))
    # mini = Minimizer(fit_h_profile_eval, params, fcn_args=(final_model, wavelengths - wvl_center, spectrum))
    # abs_line_fit = mini.emcee(burn=10, steps=2000, params=params)
    # end
    if verbose:
        abs_line_fit.params.pretty_print()
    # return results
    return final_model, abs_line_fit.params


def filter_spectra(spectr, wvl, wvl_center, center_width=10., median_width=15):
    idx_retain = np.logical_and(wvl >= wvl_center - center_width,
                                wvl <= wvl_center + center_width)
    spectra_out = medfilt(spectr, median_width)
    spectra_out[idx_retain] = spectr[idx_retain]
    return spectra_out


def new_txt_file(filename):
    temp = open(filename, 'w')
    temp.close()


def append_line(filename, line_string, new_line=False):
    temp = open(filename, 'a')
    if new_line:
        temp.write(line_string+'\n')
    else:
        temp.write(line_string)
    temp.close()