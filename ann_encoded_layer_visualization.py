import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 15
import joblib
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

# --------------------------------------------------------
# ---------------- Read data -----------------------------
# --------------------------------------------------------
print('Reading GALAH parameters')
galah_data_dir = '/shared/ebla/cotar/'
out_dir = '/shared/data-camelot/cotar/'

# additional data and products about observed spectra
date_string = '20190801'
general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')['sobject_id', 'red_flag', 'rv_guess_shift', 'v_bary']
params_data = Table.read(galah_data_dir + 'GALAH_iDR3_main_191213.fits')
general_data = join(general_data, params_data['sobject_id', 'teff', 'fe_h', 'logg', 'flag_sp'], join_type='left')

# directory with encoded data
ccd = 3
fname = f'/shared/data-camelot/cotar/Autoencoder_dense_test_complex_ccd{ccd}_prelu_5D_350epoch_4layer_adam_mae/encoded_spectra_ccd{ccd}_nf5.pkl'
fname_dir = fname[0: fname.rfind('/')+1]

print('Reading encoded data')
encoded_data = joblib.load(fname)
print('Done')

n_features = encoded_data.shape[1]

idx_valid = general_data['flag_sp'] == 0
for param_comb in [['teff', 'logg'], ['teff', 'fe_h'], ['fe_h','logg']]:
    fig, ax = plt.subplots(n_features, n_features,
                           figsize=(13,11),
                           sharey='row', sharex='col')

    for i1 in range(n_features):
        for i2 in range(n_features):
            print(i1, i2)

            # upper triangle
            if i1 > i2:
                sc1 = ax[i2, i1].scatter(encoded_data[idx_valid, i1], encoded_data[idx_valid, i2],
                                   c=general_data[param_comb[1]][idx_valid],
                                   cmap='viridis',
                                   vmin=np.nanpercentile(general_data[param_comb[1]][idx_valid], 1),
                                   vmax=np.nanpercentile(general_data[param_comb[1]][idx_valid], 99),
                                   lw=0, s=0.5)

            # middle linear layer
            if i1 == i2:
                ax[i2, i1].scatter(encoded_data[idx_valid, i1], encoded_data[idx_valid, i2],
                                   color='black', alpha=0.2,
                                   lw=0, s=0.5)

            # lower triangle
            if i1 < i2:
                sc2 = ax[i2, i1].scatter(encoded_data[idx_valid, i1], encoded_data[idx_valid, i2],
                                   c=general_data[param_comb[0]][idx_valid],
                                   cmap='viridis',
                                   vmin=np.nanpercentile(general_data[param_comb[0]][idx_valid], 1),
                                   vmax=np.nanpercentile(general_data[param_comb[0]][idx_valid], 99),
                                   lw=0, s=0.5)

            ax[i2, i1].set(xlim=np.nanpercentile(encoded_data[idx_valid, i1], [0.02, 99.98]),
                           ylim=np.nanpercentile(encoded_data[idx_valid, i2], [0.02, 99.98]))

            # add axis labels
            if i2 == n_features-1:
                ax[i2, i1].set(xlabel=f'Feature {(i1 + 1):d}')
            if i1 == 0:
                ax[i2, i1].set(ylabel=f'Feature {(i2 + 1):d}')

    # align subplots labels
    fig.align_xlabels()
    fig.align_ylabels()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0, right=1, bottom=0.0)

    # add colorbars for both parameters
    cbar2 = fig.colorbar(sc2, ax=ax[:, :],
                         shrink=0.85, location='bottom', pad=0.09, fraction=0.1,
                         anchor=(0.0, 1.0), panchor=(0.0, 1.0))
    cbar1 = fig.colorbar(sc1, ax=ax[:, :],
                         shrink=1.0, location='right', pad=0.05, fraction=0.1)

    # save image
    plt.savefig(fname_dir + f'encoded_features_scatter_{param_comb[0]}_{param_comb[1]}_ccd{ccd}.png',
                dpi=125)
    plt.close(fig)
