from os import chdir
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from astropy.table import Table, join
from importlib.machinery import SourceFileLoader
SourceFileLoader('helper_functions', '../Carbon-Spectra/helper_functions.py').load_module()
from helper_functions import move_to_dir
SourceFileLoader('s_collection', '../Carbon-Spectra/spectra_collection_functions.py').load_module()
from s_collection import CollectionParameters, read_pkl_spectra, save_pkl_spectra

move_to_dir('paper_plots')
plt.rcParams['font.size'] = 15

galah_data_dir = '/shared/ebla/cotar/'
out_dir = '/shared/data-camelot/cotar/'

date_string = '20190801'
general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')

PLOT_LOSS = False
PLOT_SPECTRA = True

HBETA_WVL = 4861.3615
HALPHA_WVL = 6562.8518

# loss function plot
if PLOT_LOSS:
    ann_fit_loss1 = np.loadtxt(out_dir + 'Autoencoder_dense_test_complex_ccd1_prelu_5D_350epoch_4layer_adam_mae/ann_network_loss.txt')
    ann_fit_loss3 = np.loadtxt(out_dir + 'Autoencoder_dense_test_complex_ccd3_prelu_5D_350epoch_4layer_adam_mae/ann_network_loss.txt')

    x_loss = np.arange(350) + 1
    plt.plot(x_loss, ann_fit_loss1[:, 0], label='Training (blue arm)', lw=0.75, c='C0')
    plt.plot(x_loss, ann_fit_loss1[:, 1], label='Validation (blue arm)',lw=0.75, ls='--', c='C0')
    plt.plot(x_loss, ann_fit_loss3[:, 0], label='Training (red arm)', lw=0.75, c='C3')
    plt.plot(x_loss, ann_fit_loss3[:, 1], label='Validation (red arm)', lw=0.75, ls='--', c='C3')
    plt.xlabel('Training epoch')
    plt.ylabel('Mean absolute prediction error ')
    plt.xlim(-1, 352)
    # if ccd == '1':
    #     plt.ylim(1.75e-2, 3e-2)
    # elif ccd == '3':
    plt.ylim(1.1e-2, 3e-2)
    plt.gca().yaxis.set_major_locator(MultipleLocator(5.0e-3))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(2.5e-3))
    plt.gca().yaxis.get_major_formatter().set_powerlimits((-1, 1))
    plt.gca().get_yaxis().get_offset_text().offset_text_position = "top"
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.grid(ls='--', color='black', lw=0.5, alpha=0.2, which='both')
    plt.savefig('ann_network_loss_ccd13.pdf')
    plt.close()

# plot median ANN and observed spectra
if PLOT_SPECTRA:
    print('plotting selected spectra')
    ccd = 3
    spec_pkl = ['galah_dr53_ccd1_4710_4910_wvlstep_0.040_ext4_20190801.pkl',
                '',
                'galah_dr53_ccd3_6475_6745_wvlstep_0.060_ext4_20190801.pkl',
                ''][ccd - 1]
    ann_pkl = spec_pkl[:-4] + '_ann_median.pkl'
    plot_wvl = [HBETA_WVL, 0, HALPHA_WVL, 0][ccd - 1]
    wvl_plot_span = [20, 25, 30, 35][ccd - 1]

    print(' read 1')
    obs_spec = read_pkl_spectra(out_dir + spec_pkl)
    print(' read 2')
    ann_spec = read_pkl_spectra(out_dir + ann_pkl)
    print(' read done')
    wvl_spec = CollectionParameters(spec_pkl).get_wvl_values()

    # plotting repeats spectra
    plot_s_ids = [[140308000101142, 161213002601381, 161219002101381, 171206002101381, 171207002101381],
                  [161209001801307, 170904002101301, 171206002601307],
                  [150427002801012, 150427004801012, 150606003901012],
                  [151230002201218, 171206004101218, 171208003601218],
                  [140823000401145, 140823000901144, 140823001401145, 140823001801144, 140824002501145, 140824003001144, 140824003501145, 140824003901144]
                  ]

    fig, ax = plt.subplots(1, len(plot_s_ids), figsize=(14., 5), sharey=True)
    for i_ax, s_ids in enumerate(plot_s_ids):
        for s_id in s_ids:
            s_id = np.where(general_data['sobject_id'] == s_id)[0][0]
            ax[i_ax].plot(wvl_spec, obs_spec[s_id, :], lw=0.8, alpha=0.75, label='Observed')
            ax[i_ax].set(xlim=(plot_wvl - wvl_plot_span/3, plot_wvl + wvl_plot_span/3))
            # if np.floor(len(plot_ids)/2.) - 1 == i_ax:
            #     ax[i_ax].set(xlabel=u'Wavelength [$\AA$]')
        ax[i_ax].grid(ls='--', lw=0.4, alpha=0.2, color='black')
    ax[0].set(ylim=(0.4, 1.2), ylabel='Normalised flux')
    fig.text(0.5, 0.02, u'Wavelength [$\AA$]', ha='center')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0.12)
    fig.savefig('repetas_spectra_spectra_ccd' + str(ccd) + '.pdf')
    plt.close(fig)

    # which spectra are to be plotted
    plot_ids = [248872, 50588, 67497, 141778]

    fig, ax = plt.subplots(1, len(plot_ids), figsize=(14., 5), sharey=True)
    for i_ax, s_id in enumerate(plot_ids):
        ax[i_ax].plot(wvl_spec, obs_spec[s_id, :], lw=0.8, alpha=0.9, color='black', label='Observed')
        ax[i_ax].plot(wvl_spec, ann_spec[s_id, :], lw=0.8, alpha=0.9, color='C0', label='Reference')
        ax[i_ax].set(xlim=(plot_wvl - wvl_plot_span, plot_wvl + wvl_plot_span))
        # if np.floor(len(plot_ids)/2.) - 1 == i_ax:
        #     ax[i_ax].set(xlabel=u'Wavelength [$\AA$]')
        ax[i_ax].grid(ls='--', lw=0.4, alpha=0.2, color='black')
    ax[0].set(ylim=(0.3, 1.2), ylabel='Normalised flux')
    ax[-1].legend()
    fig.text(0.5, 0.02, u'Wavelength [$\AA$]', ha='center')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0.12)
    fig.savefig('sample_spectra_spectra_ccd'+str(ccd)+'.pdf')
    plt.close(fig)
