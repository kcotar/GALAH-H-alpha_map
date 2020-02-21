from os import system, chdir, path, listdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from astropy.table import Table, unique, join, vstack
import astropy.coordinates as coord
import astropy.units as un

plt.rcParams['font.size'] = 15
ANALYSE_REPEATS = False
MAKE_PLOTS = True
MOVE_ORDER_RESULT_PLOTS = False

data_dir = '/shared/ebla/cotar/'
# results_data_dir = '/shared/data-camelot/cotar/H_band_strength_all_20190801/'
results_data_dir = '/shared/data-camelot/cotar/H_band_strength_complete_20190801_ANN-medians_newthrs/'
out_dir = results_data_dir

print('Reading results')
res_hdet = vstack((Table.read(results_data_dir + 'results_H_lines_1.fits'),
                   Table.read(results_data_dir + 'results_H_lines_2.fits')))
res_hdet = res_hdet[np.argsort(res_hdet['sobject_id'])]
res_hdet.write(results_data_dir + 'results_H_lines_complete.fits', overwrite=True)
# res_hdet = Table.read(results_data_dir + 'results_H_lines_complete.fits')

res_hdet['sobject_id', 'SB2_c1', 'SB2_c3'].write(results_data_dir + 'results_H_lines_complete_Gregor.fits', overwrite=True)
# print(res_hdet)
print('Results so far:', len(res_hdet))
# s_u, c_u = np.unique(res_all['sobject_id'], return_counts=True)
# for get_s_u in s_u[c_u >= 2][:10]:
#     print(res_all[res_all['sobject_id'] == get_s_u]
res_hdet = unique(res_hdet, keys='sobject_id', keep='first')
print('Unique results:', len(res_hdet))

# read and add additional GALAH data tables
binary_candidates_gregor = Table.read(results_data_dir + 'galah_binary_all_candidates_radec.csv', format='ascii.csv')
galah_all = Table.read(data_dir + 'sobject_iraf_53_reduced_20190801.fits')
sme_all = Table.read(data_dir + 'GALAH_iDR3_main_191213.fits')
gaiadr2_xmatch = unique(Table.read(data_dir + 'sobject_iraf_53_reduced_20190801_po_gaiadr2.csv'), keep='first', keys='sobject_id')
oc_all = Table.read(data_dir + 'clusters/members_open_gaia_r2.fits')
res_hdet = join(res_hdet, galah_all['sobject_id', 'ra', 'dec', 'rv_guess_shift'], keys='sobject_id', join_type='left')
res_hdet = join(res_hdet, sme_all['sobject_id', 'teff', 'logg', 'fe_h', 'flag_sp', 'rv_guess','parallax', 'pmra', 'pmdec'], keys='sobject_id', join_type='left')
res_hdet = join(res_hdet, gaiadr2_xmatch['sobject_id', 'source_id'], keys='sobject_id', join_type='left')
drew_tracks = Table.read(results_data_dir + 'Drew_2015_MS_tracks.csv', format='ascii.csv')

# read IPHAS DR2 and VPHAS DR2 photometry values
iphas_cols = ['sobject_id', 'ha', 'haErr', 'r', 'rErr', 'i', 'iErr']
vphas_cols = ['sobject_id', 'Hamag', 'e_Hamag', 'rmag', 'e_rmag', 'imag', 'e_imag']
iphas = unique(Table.read(data_dir + 'photometry/' + 'IPHAS_dr2_Barentsen_20190801_clean.csv', format='ascii.csv'), keys='sobject_id', keep='first')[iphas_cols]
vphas = unique(Table.read(data_dir + 'photometry/' + 'VPHAS_dr2_Drew_20190801.csv', format='ascii.csv'), keys='sobject_id', keep='first')[vphas_cols]

res_hdet = join(res_hdet, iphas,
                keys='sobject_id', join_type='left')
res_hdet = join(res_hdet, vphas,
                keys='sobject_id', join_type='left')

iphas_vphas_stack = vstack((iphas, vphas))
for i_c in range(1, len(iphas_cols)):
    iphas[iphas_cols[i_c]].name = vphas_cols[i_c]
print('IPHAS stars: ', len(iphas))
print('VPHAS stars: ', len(vphas))
print('(I+V)PHAS stars: ', len(np.unique(iphas_vphas_stack['sobject_id'])))
# # add photometry tot he final table
# res_hdet = join(res_hdet, iphas_vphas_stack,
#                 keys='sobject_id', join_type='left')

print(' Flag stats:')
vf, nf = np.unique(res_hdet['flag'], return_counts=True)
for ff in range(len(nf)):
    print('  {:.0f} => {:.0f}'.format(vf[ff], nf[ff]))

print(' Flag stats 2:')
vf, nf = np.unique(res_hdet['flag'], return_counts=True)
for ff in range(10):
    print('  {:.0f} => {:.0f}'.format(2**ff, np.sum(np.bitwise_and(res_hdet['flag'], 2**ff) == 2**ff)))

res_all = res_hdet[res_hdet['flag'] == 0]
print('Results unflagged:', len(res_all))

d_rv_nii = 15

# use flags to determine subsets
idx_unf = res_hdet['flag'] == 0
idx_bin = np.logical_or(res_hdet['SB2_c1'] >= 1, res_hdet['SB2_c3'] >= 1)
idx_bin_conf = np.logical_and(res_hdet['SB2_c1'] >= 1, res_hdet['SB2_c3'] >= 1)
idx_neb = (res_hdet['NII'] + res_hdet['SII']) >= 3
idx_emi = np.logical_and(res_hdet['Ha_EW'] > 0.25, res_hdet['Ha_EW_abs'] > 0.)  # EW thresholds for weak emissions, TODO
idx_emi2 = np.logical_and(res_hdet['Ha_EW'] > 0.2, res_hdet['Ha_EW_abs'] > 0.)
idx_emi3 = np.logical_and(res_hdet['Ha_EW'] > 0.3, res_hdet['Ha_EW_abs'] > 0.)
idx_emi4 = np.logical_and(res_hdet['Ha_EW'] > 0.4, res_hdet['Ha_EW_abs'] > 0.)
idx_emi5 = np.logical_and(res_hdet['Ha_EW'] > 0.5, res_hdet['Ha_EW_abs'] > 0.)
idx_emi6 = np.logical_and(res_hdet['Ha_EW'] > 0.6, res_hdet['Ha_EW_abs'] > 0.)
idx_mark_niirv = np.abs(res_hdet['rv_NII'] - res_hdet['rv_SII']) > d_rv_nii
idx_emi_strong = np.logical_and(res_hdet['Ha_EW'] > 0.8, res_hdet['Ha_EW_abs'] > 0.5)  # EW thresholds for the strongest
idx_asym_h = np.sqrt(res_hdet['Ha_EW_asym']**2 + res_hdet['Hb_EW_asym']**2) > 0.3
idx_sme_valid = res_hdet['flag_sp'] == 0

res_hdet['emiss'] = 0
res_hdet['nebular'] = 0
res_hdet['emiss'][idx_emi * idx_unf * ~idx_bin] = 1
res_hdet['nebular'][idx_neb * ~idx_mark_niirv] = 1

print('Final emission stars:', np.sum(res_hdet['emiss']))
print('Final nebular stars:', np.sum(res_hdet['nebular']))
print('Final emission nebular union:', np.sum(res_hdet['emiss'] * res_hdet['nebular']))
print('Final emission stars 0.2:', np.sum(idx_emi2 * idx_unf))
print('Final emission stars 0.3:', np.sum(idx_emi3 * idx_unf))
print('Final emission stars 0.4:', np.sum(idx_emi4 * idx_unf))
print('Final emission stars 0.5:', np.sum(idx_emi5 * idx_unf))
print('Final emission stars 0.6:', np.sum(idx_emi6 * idx_unf))
print('Final nebular stars noflag:', np.sum(idx_neb))

# ----------------------------------------
# ------------ FUNCTIONS -----------------
# ----------------------------------------
def copy_ids_to_curr_map(sel_ids, cp_dir, suffix='',
                         prefixes=None, suffixes=None):
    system('mkdir ' + cp_dir)
    chdir(cp_dir)
    for i_s, star_id in enumerate(sel_ids):
        s_date = np.int32(star_id / 10e10)
        sobj = str(star_id)
        # print(sobj)

        source = results_data_dir + str(s_date) + '/' + str(sobj) + suffix + '.png'
        prefix = ''
        suffix2 = ''
        if prefixes is not None:
            prefix += prefixes[i_s] + '_'
        if suffixes is not None:
            suffix2 += '_' + suffixes[i_s]
        target = cp_dir + '/' + prefix + str(sobj) + suffix + suffix2 + '.png'

        if path.isfile(target):
            continue
        # execute copy of the strongest emitters
        system('cp ' + source + ' ' + target)
    chdir('..')


# ----------------------------------------
# Compare our binaries with possible binaries detected and analysed by Traven+ 2020
# ----------------------------------------
# binary candidates statistics
print("Binary statistics - comparison with Gregor's paper:")
print('Stats: ', len(binary_candidates_gregor), np.sum(idx_bin_conf))
print('Union:', np.sum(np.in1d(binary_candidates_gregor['sobject_id'], res_hdet['sobject_id'][idx_bin_conf])))
print('Unique:', np.sum(np.in1d(binary_candidates_gregor['sobject_id'], res_hdet['sobject_id'][idx_bin_conf], invert=True)))
idx_show_ids = np.in1d(res_hdet['sobject_id'][idx_bin_conf], binary_candidates_gregor['sobject_id'], invert=True)
print('Unique:', np.sum(idx_show_ids))
# print(res_hdet['sobject_id'][idx_bin_conf][idx_show_ids])

# # copy objects that were not detected by the mention paper into a separate folder where they can be investigated
# copy_ids_to_curr_map(res_hdet['sobject_id'][idx_bin_conf][idx_show_ids],
#                      out_dir + 'SB2_notin_Gregor/',
#                      suffix='', prefixes=None)

# ----------------------------------------
# Get repeated observations that were at least once detected as in emission
# - repeats could be from different programs than the GALAH -> (TESS, K2, clusters, pilot, Orion ...), therefore
#   repeats detection is based on logged coordinates of the spectrum
# ----------------------------------------
rep_fits = 'results_H_lines_complete_with-rep.fits'
if ANALYSE_REPEATS:
    ra_dec_all = coord.ICRS(ra=res_hdet['ra']*un.deg,
                            dec=res_hdet['dec']*un.deg)
    # use only the strongest detected emissions when dealing with repeated observations
    idx_detected = idx_emi5 * idx_unf * ~idx_bin
    res_hdet['id_rep'] = np.int32(0)
    id_rep = 1

    rep_dir = out_dir + 'repeats/'
    system('mkdir ' + rep_dir)
    chdir(rep_dir)

    for star in res_hdet:
        ra_dec_star = coord.ICRS(ra=star['ra']*un.deg,
                                 dec=star['dec']*un.deg)

        if star['id_rep'] > 0:
            # was already found to be a repeated observation of a star, skip that
            continue

        idx_close = ra_dec_all.separation(ra_dec_star) < 0.5 * un.arcsec
        n_close = np.sum(idx_close)
        if n_close > 1 and np.sum(res_hdet['id_rep'][idx_close]) == 0:
            # we have a repeated observation of a star that was not yet detected and analysed
            # add a repeate id to observations of the same star
            res_hdet['id_rep'][idx_close] = id_rep
            id_rep += 1

            # must pass minimum number of repeats to be visually checked and analysed
            min_rep = 3
            if n_close < min_rep:
                continue

            idx_detected_close = idx_close * idx_detected
            if np.sum(idx_detected_close) >= 1:
                print('Repeated observations id', id_rep)
                # at least one of the repeats was detected as emission object
                rep_subdir = rep_dir + '{:05.0f}'.format(id_rep) + '/'
                copy_ids_to_curr_map(res_hdet['sobject_id'][idx_close], rep_subdir, suffix='', prefixes=None)

    chdir('..')
    # save results
    res_hdet.write(results_data_dir + rep_fits, overwrite=True)

if path.isfile(out_dir + rep_fits):
    rep_data = Table.read(out_dir + rep_fits)

    def rep_stat_print(uniq_id, uniq_c):
        for n_rep in np.unique(uniq_c):
            print(' rep', n_rep, 'count:', np.sum(uniq_c == n_rep))
        return True

    # repeats stats
    print('Repeats all data:')
    u_id1, u_c1 = np.unique(rep_data['id_rep'], return_counts=True)
    rep_stat_print(u_id1, u_c1)

    idx_detected = idx_emi5 * idx_unf * ~idx_bin
    print('Repeats with at least one detection:')
    u_id2, u_c2 = np.unique(rep_data['id_rep'][idx_detected], return_counts=True)
    idx_print = np.in1d(u_id1, u_id2)
    rep_stat_print(u_id1[idx_print], u_c1[idx_print])

    print('Variability modes')
    all_rep_dirs = listdir(out_dir + 'repeats/')
    rep_mode = [ff.split('_')[1] for ff in all_rep_dirs]
    u_id3, u_c3 = np.unique(rep_mode, return_counts=True)
    for iu in range(len(u_id3)):
        print(' rep', u_id3[iu], 'count', u_c3[iu], 'perc', 100.*u_c3[iu]/np.sum(u_c3))

if MAKE_PLOTS:
    rep_dir = out_dir + 'pretty_plots/'
    system('mkdir ' + rep_dir)
    chdir(rep_dir)

    # ----------------------------------------
    # Make nice analysis plots that will potentially appear in the final published paper
    # ----------------------------------------

    # determine data rows that will be used for production of plots
    idx_plot = idx_neb
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.plot((-200, 200), (-200, 200), lw=2, c='C2', alpha=0.8, ls='-')
    ax.plot((-200, 200), (-200-d_rv_nii, 200-d_rv_nii), lw=2, c='C2', alpha=0.5, ls='--')
    ax.plot((-200, 200), (-200+d_rv_nii, 200+d_rv_nii), lw=2, c='C2', alpha=0.5, ls='--')
    ax.scatter(res_hdet['rv_NII'][idx_plot*~idx_mark_niirv], res_hdet['rv_SII'][idx_plot*~idx_mark_niirv],
               lw=0, s=6, c='black', alpha=0.5)
    ax.scatter(res_hdet['rv_NII'][idx_plot*idx_mark_niirv], res_hdet['rv_SII'][idx_plot*idx_mark_niirv],
               lw=0, s=6, c='grey', alpha=0.5)
    ax.set(xlim=(-120, 120), ylim=(-120, 120),
           xlabel='Measured radial velocity of the [NII] doublet',
           ylabel='Measured radial velocity of the [SII] doublet')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('sii_nii_rv.png', dpi=250)
    plt.close(fig)

    # export stars with confident contribution of nebular emission present in the spectrum
    res_hdet['sobject_id', 'ra', 'dec'][idx_plot].write(results_data_dir + 'results_H_lines_nebular-pos-only.fits', overwrite=True)

    # determine data rows that will be used for production of plots
    idx_plot = idx_emi * idx_unf * ~idx_bin * idx_sme_valid * ~idx_neb
    xlims = [(3300, 8000), (0, 5), (-2.5, 1.0)]
    for i_c, p_col in enumerate(['teff', 'logg', 'fe_h']):
        # make a plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
        ax.plot((-1, 1), (-1, 1), lw=2, c='C2', alpha=0.75, ls='--')
        ax.scatter(res_hdet[p_col][idx_plot], res_hdet['Ha_EW'][idx_plot], lw=0, s=6, c='black', alpha=0.5)
        ax.set(xlim=xlims[i_c], ylim=(0.2, 2.8),
               xlabel=r'T$_{eff}$ [K]',
               ylabel=r'Equivalent width of the H$\alpha$ emission')
        ax.grid(ls='--', alpha=0.25, c='black')
        fig.tight_layout()
        fig.savefig('haew_'+p_col+'_corr_without_neb.png', dpi=250)
        plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_emi * idx_unf * ~idx_bin
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    # TODO: accretion separation line
    ax.plot((0.15, 0.85), (0.2, 0.55), lw=2, c='C0', alpha=0.75, ls='-.', label='Threshold')
    ax.plot(drew_tracks['ri0'], drew_tracks['rha0'],
            lw=2, c='C2', alpha=0.75, ls='--', label='Main-sequence track')
    ax.scatter((res_hdet['rmag'] - res_hdet['imag'])[idx_plot],
               (res_hdet['rmag'] - res_hdet['Hamag'])[idx_plot],
               lw=0, s=22, c='black', alpha=1., marker="v", label="VPHAS")
    ax.scatter((res_hdet['r'] - res_hdet['i'])[idx_plot],
               (res_hdet['r'] - res_hdet['ha'])[idx_plot],
               lw=0, s=28, c='black', alpha=1., marker="X", label="IPHAS")
    ax.set(xlim=(0.15, 1.2), ylim=(0.05, 0.55),
           xlabel=r'Color index r - i',
           ylabel=r'Color index r - H$\alpha$')
    ax.grid(ls='--', alpha=0.25, c='black')
    ax.legend()
    fig.tight_layout()
    fig.savefig('mag_iphas_vphas_sep.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_neb * ~idx_mark_niirv
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.plot((-1, 1), (-1, 1), lw=2, c='C2', alpha=0.75, ls='--')
    ax.scatter(res_hdet['NII_EW'][idx_plot], res_hdet['SII_EW'][idx_plot], lw=0, s=6, c='black', alpha=0.5)
    ax.set(xlim=(-0.01, 0.85), ylim=(-0.01, 0.7),
           xlabel='Equivalent width of the fitted [NII] lines',
           ylabel='Equivalent width of the fitted [SII] lines')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('sii_nii_ew_corr.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_neb * ~idx_mark_niirv
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.plot((-1, 1), (-1, 1), lw=2, c='C2', alpha=0.75, ls='--')
    ax.plot((-1, 8), (-1/8, 1), lw=2, c='C2', alpha=0.75, ls='--')
    ax.scatter(res_hdet['Ha_EW'][idx_plot], res_hdet['SII_EW'][idx_plot], lw=0, s=6, c='black', alpha=0.5)
    ax.set(xlim=(-0.05, 3.5), ylim=(-0.01, 0.7),
           xlabel=r'Equivalent width of H$\alpha$ emission',
           ylabel='Equivalent width of the fitted [SII] lines')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('ha_nii_ew_corr.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_neb * ~idx_mark_niirv
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.plot((-1, 1), (-1, 1), lw=2, c='C2', alpha=0.75, ls='--')
    ax.plot((-1, 8), (-1/8, 1), lw=2, c='C2', alpha=0.75, ls='--')
    ax.scatter(res_hdet['Ha_EW'][idx_plot], res_hdet['NII_EW'][idx_plot], lw=0, s=6, c='black', alpha=0.5)
    ax.set(xlim=(-0.05, 3.5), ylim=(-0.01, 0.85),
           xlabel=r'Equivalent width of H$\alpha$ emission',
           ylabel='Equivalent width of the fitted [NII] lines')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('ha_sii_ew_corr.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_emi * idx_unf * ~idx_bin
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.plot((-1, 5), (-1, 5), lw=2, c='C2', alpha=0.75, ls='--')
    f = 2.
    ax.plot((-1, 5), (-1/f, 5/f), lw=2, c='C1', alpha=0.75, ls='-.')
    ax.scatter(res_hdet['Ha_EW'][idx_plot], res_hdet['Hb_EW'][idx_plot], lw=0, s=4, c='black', alpha=0.33)
    ax.set(xlim=(-0.05, 3.5), ylim=(-0.7, 3),
           xlabel=r'Equivalent width of the H$\alpha$ emission',
           ylabel=r'Equivalent width of the H$\beta$ emission')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('H_emission_EW_dist.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot1 = idx_emi * idx_unf * ~idx_bin
    idx_plot2 = ~idx_emi * idx_unf * ~idx_bin
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.plot((-1, 5), (-1, 5), lw=2, c='C2', alpha=0.75, ls='--')
    f = 2.
    ax.plot((-1, 5), (-1/f, 5/f), lw=2, c='C1', alpha=0.75, ls='-.')
    ax.scatter(res_hdet['Ha_EW'][idx_plot1], res_hdet['Hb_EW'][idx_plot1], lw=0, s=4, c='black', alpha=0.33)
    ax.scatter(res_hdet['Ha_EW'][idx_plot2], res_hdet['Hb_EW'][idx_plot2], lw=0, s=4, c='grey', alpha=0.33)
    ax.set(xlim=(-0.05, 3.5), ylim=(-0.7, 3),
           xlabel=r'Equivalent width of the H$\alpha$ emission',
           ylabel=r'Equivalent width of the H$\beta$ emission')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('H_emission_EW_dist_all.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_emi * idx_unf * ~idx_bin
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.plot((-1, 5), (-1, 5), lw=2, c='C2', alpha=0.75, ls='--')
    ax.scatter(res_hdet['Ha_EW'][idx_plot], res_hdet['Ha_W10'][idx_plot], lw=0, s=4, c='black', alpha=0.33)
    ax.set(xlim=(-0.05, 4), ylim=(-0., 400),
           xlabel=r'Equivalent width of the H$\alpha$ emission',
           ylabel=r'H$\alpha$ 10%')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('H_emission_H10_dist.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_emi * idx_unf * ~idx_bin
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6.5))

    ax.scatter(res_hdet['Ha_EW_asym'][idx_plot], res_hdet['Hb_EW_asym'][idx_plot], lw=0, s=5, c='black', alpha=0.5)
    ax.set(xlim=(-1, 1), ylim=(-1, 1),
           xlabel=r'Asymmetry index of H$\alpha$ emission',
           ylabel=r'Asymmetry index of H$\beta$ emission')
    ax.grid(ls='--', alpha=0.25, c='black')
    circle = mpatches.Circle((0.0, 0.0), radius=0.25, alpha=1., fill=False,
                             lw=2.5, ls='--', color='C2')
    ax.add_patch(circle)
    fig.tight_layout()
    fig.savefig('H_emission_asymmetry.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_emi * idx_unf * ~idx_bin
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.scatter(res_hdet['Ha_EW_asym'][idx_plot], res_hdet['Ha_EW'][idx_plot], lw=0, s=3, c='black', alpha=0.33)
    ax.set(xlim=(-1, 1), ylim=(-0.05, 3.5),
           xlabel=r'Asymmetry index of H$\alpha$ emission',
           ylabel=r'Equivalent width of H$\alpha$ emission')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('Ha_emission_asymmetry_strength.png', dpi=250)
    plt.close(fig)

    # determine data rows that will be used for production of plots
    idx_plot = idx_emi * idx_unf * ~idx_bin
    # make a plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.scatter(res_hdet['Hb_EW_asym'][idx_plot], res_hdet['Hb_EW'][idx_plot], lw=0, s=3, c='black', alpha=0.33)
    ax.set(xlim=(-1, 1), ylim=(-0.7, 3),
           xlabel=r'Asymmetry index of H$\beta$ emission',
           ylabel=r'Equivalent width of H$\beta$ emission')
    ax.grid(ls='--', alpha=0.25, c='black')
    fig.tight_layout()
    fig.savefig('Hb_emission_asymmetry_strength.png', dpi=250)
    plt.close(fig)

### Final published electronic and latex tables
res_hdet['rv_NII'] -= res_hdet['rv_guess_shift']
res_hdet['rv_SII'] -= res_hdet['rv_guess_shift']
# export all data with relevant columns
final_published_res = res_hdet['source_id', 'sobject_id', 'ra', 'dec', 'Ha_EW', 'Hb_EW', 'Ha_EW_abs', 'Hb_EW_abs', 'Ha_W10', 'SB2_c3', 'SB2_c1', 'NII', 'SII', 'NII_EW', 'rv_NII', 'rv_SII', 'nebular', 'emiss', 'flag']
final_published_res.write(results_data_dir + 'results_emission_lines_final.fits', overwrite=True, format='fits')
final_published_res.write(results_data_dir + 'results_emission_lines_final.csv', overwrite=True, format='ascii.csv')
# export detection subsets
res_hdet[idx_emi * idx_unf * ~idx_bin]['sobject_id', 'ra', 'dec', 'Ha_EW', 'rv_guess','parallax', 'pmra', 'pmdec'].write(results_data_dir + 'results_H_lines_emmis.fits', overwrite=True)
res_hdet[idx_neb * idx_unf * ~idx_bin_conf]['sobject_id', 'ra', 'dec', 'rv_NII', 'rv_SII', 'rv_guess','parallax', 'pmra', 'pmdec'].write(results_data_dir + 'results_H_lines_nebul.fits', overwrite=True)

# LaTeX data export
res_latex = res_hdet[idx_emi * ~idx_bin * idx_unf]
out_cols = ['source_id', 'Ha_EW', 'Ha_EW_abs', 'Ha_W10', 'Ha_EW_asym', 'NII', 'SII', 'NII_EW', 'rv_NII', 'rv_SII', 'flag']
# out_formats = ['%.2f' for ic in range(len(out_cols))]
out_formats = ['%.0f', '%.2f', '%.2f', '%.2f', '%.2f', '%.0f', '%.0f', '%.2f', '%.2f', '%.2f', '%.0f']
res_latex[np.argsort(res_latex['Ha_EW'])[::-1]][:30][out_cols].write(out_dir + 'results_H_lines.tex',
                                                                     format='ascii.latex', overwrite=True,
                                                                     formats=dict(zip(out_cols, out_formats)))

if not MOVE_ORDER_RESULT_PLOTS:
    raise SystemExit

n_random_det = 450
# ----------------------------------------
# Spectra with possible crosstalk signal present in the red spectral arm
# ----------------------------------------
res = res_hdet[np.logical_and(res_hdet['flag'] == 0,
                              res_hdet['SB2_c3'] >= 1)]
res = res[res['SB2_c1'] == 0]
print('N possible crosstalk:', len(res))
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'possible_crosstalk/')

# ----------------------------------------
# Totally random subset of results - easy way to browse for strange results
# ----------------------------------------
print('N for random selection:', len(res_hdet))
idx_sel = np.int64(np.random.uniform(0, len(res_hdet), n_random_det))
copy_ids_to_curr_map(res_hdet['sobject_id'][idx_sel], out_dir + 'random/',
                     prefixes=['{:.3f}'.format(p_val) for p_val in res_hdet['Ha_EW'][idx_sel]],
                     suffixes=['{:.3f}'.format(p_val) for p_val in res_hdet['Ha_EW_abs'][idx_sel]])

# # ----------------------------------------
# # sample of missing spectrum flagged objects - should not exist in the final version of the detection pipeline
# # ----------------------------------------
# res = res_hdet[np.logical_or(np.bitwise_and(res_hdet['flag'], 128),
#                              np.bitwise_and(res_hdet['flag'], 64))]
# print('N no reference:', len(res))
# idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
# copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'flag_noref/')

# ----------------------------------------
# sample of spectra with large difference towards reference
# ----------------------------------------
res = res_hdet[np.logical_or(np.bitwise_and(res_hdet['flag'], 32),
                             np.bitwise_and(res_hdet['flag'], 16))]
print('N bad reference:', len(res))
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'flag_badref/')

# ----------------------------------------
# sample of spectra with possible erroneous wvl reduction or determined RV velocity
# ----------------------------------------
res = res_hdet[np.logical_not(np.logical_or(np.bitwise_and(res_hdet['flag'], 128),
                                            np.bitwise_and(res_hdet['flag'], 64)))]
res = res[np.logical_or(np.bitwise_and(res['flag'], 2),
                        np.bitwise_and(res['flag'], 1))]
print('N bad wvl reduction:', len(res))
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'flag_wvlred/')

# ----------------------------------------
# sample of nebular emissions
# ----------------------------------------
res = res_hdet[(res_hdet['NII'] + res_hdet['SII']) >= 3]
res = res[res['rv_NII'] - res['rv_SII'] <= d_rv_nii]
print('N nebulous:', len(res))
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'nebular/')

# ----------------------------------------
# sample of sb2 stars
# ----------------------------------------
# res = res_all[np.logical_or(res_all['SB2_c1'] >= 1,
#                             res_all['SB2_c3'] >= 1)]
res = res_hdet[idx_bin]
print('N SB2:', len(res))
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'SB2/')

n_random_det = 600
# ----------------------------------------
# sample of stars with asymmetric emission line - based on computed EW
# ----------------------------------------
idx_asmy = np.logical_and(idx_emi, idx_unf)
idx_asmy = np.logical_and(idx_asmy, idx_asym_h)
res = res_hdet[idx_asmy]
print('N asymmetric:', len(res))
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'asym/',
                     prefixes=['{:.3f}'.format(p_val) for p_val in res['Ha_EW'][idx_sel]],
                     suffixes=['{:.3f}'.format(p_val) for p_val in res['Ha_EW_abs'][idx_sel]])

n_best = 1250
# ----------------------------------------,
# sort by strongest H-alpha emission spectrum
# ----------------------------------------
sort_by = 'Ha_EW'
print('N '+sort_by+':', n_best)
res = res_all[np.isfinite(res_all[sort_by])]
idx_strong = np.argsort(res[sort_by])[::-1]
copy_ids_to_curr_map(res['sobject_id'][idx_strong[:n_best]], out_dir + 'strongest/',
                     prefixes=['{:.3f}'.format(p_val) for p_val in res[sort_by][idx_strong[:n_best]]])

plot_stars = galah_all[np.in1d(galah_all['sobject_id'],
                               res[idx_strong[:n_best]]['sobject_id'])]
plt.scatter(galah_all['ra'], galah_all['dec'], lw=0, s=0.5, c='0.8',)
plt.scatter(plot_stars['ra'], plot_stars['dec'], lw=0, s=0.5, c='black')
plt.xlim(0., 360.)
plt.ylim(-90., 90.)
plt.grid(ls='--', alpha=0.2, color='black')
plt.tight_layout()
# plt.show()
plt.savefig('ra_dec_plot.png', dpi=450)
plt.close()
plt.scatter(galah_all['ra'], galah_all['dec'], lw=0, s=0.5, c='0.8')
# plt.scatter(plot_stars['ra'], plot_stars['dec'], lw=0, s=0.5, c='black')
idx_inoc = np.in1d(galah_all['sobject_id'], oc_all['sobject_id'])
plt.scatter(galah_all['ra'][idx_inoc], galah_all['dec'][idx_inoc], lw=0, s=0.5, c='red')
plt.xlim(0., 360.)
plt.ylim(-90., 90.)
plt.grid(ls='--', alpha=0.2, color='black')
plt.tight_layout()
plt.savefig('ra_dec_plot_clusters.png', dpi=450)
plt.close()
res = None

# ----------------------------------------
# sort by strongest absolute H-alpha emission spectrum
# ----------------------------------------
sort_by = 'Ha_EW_abs'
print('N '+sort_by+':', n_best)
res = res_all[np.logical_and(np.isfinite(res_all[sort_by]),
                             res_all['Ha_EW'] > -2)]  # as we want to discover strange objects with large Ha deviation
idx_strong = np.argsort(res[sort_by])[::-1]
copy_ids_to_curr_map(res['sobject_id'][idx_strong[:n_best]], out_dir + 'strongest_absolute/',
                     prefixes=['{:.3f}'.format(p_val) for p_val in res[sort_by][idx_strong[:n_best]]])
