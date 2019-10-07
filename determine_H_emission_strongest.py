from os import system, chdir, path
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, unique, join
import astropy.coordinates as coord
import astropy.units as un

data_dir = '/shared/ebla/cotar/'
results_data_dir = '/shared/data-camelot/cotar/H_band_strength_all_20190801/'
out_dir = results_data_dir

print 'Reading results'
# # TEMP FIX:
# csv_temp = open(results_data_dir + 'results_H_lines_temp.csv', 'w')
# with open(results_data_dir + 'results_H_lines.csv', 'r') as csv_orig:
#     print 'Skipping bad csv lines'
#     for i_l, csv_line in enumerate(csv_orig):
#         if 110 < len(csv_line) < 300:
#             line_split = csv_line.split(',')
#             if (len([0]) == 15 or i_l <= 0) and len(line_split) == 16:
#                 csv_temp.write(csv_line)
# csv_temp.close()
# Read corrected csv with less lines than optimal
# res_hdet = Table.read(results_data_dir + 'results_H_lines.csv', format='ascii.csv')
res_hdet = Table.read(results_data_dir + 'results_H_lines.fits')
res_hdet = res_hdet[np.argsort(res_hdet['sobject_id'])]
print res_hdet
print 'Results so far:', len(res_hdet)
# s_u, c_u = np.unique(res_all['sobject_id'], return_counts=True)
# for get_s_u in s_u[c_u >= 2][:10]:
#     print res_all[res_all['sobject_id'] == get_s_u]
res_hdet = unique(res_hdet, keys='sobject_id', keep='first')
print 'Unique results:', len(res_hdet)

galah_all = Table.read(data_dir + 'sobject_iraf_53_reduced_20190801.fits')
oc_all = Table.read(data_dir + 'clusters/members_open_gaia_r2.fits')
res_hdet = join(res_hdet, galah_all['sobject_id', 'ra', 'dec'], keys='sobject_id', join_type='left')


print ' Flag stats:'
vf, nf = np.unique(res_hdet['flag'], return_counts=True)
for ff in range(len(nf)):
    print '  {:.0f} => {:.0f}'.format(vf[ff], nf[ff])

res_all = res_hdet[res_hdet['flag'] == 0]
print 'Results unflagged:', len(res_hdet)

# use flags to determine subsets
idx_unf = res_hdet['flag'] == 0
idx_bin = np.logical_or(res_hdet['SB2_c1'] >= 1, res_hdet['SB2_c3'] >= 1)
idx_neb = (res_hdet['NII'] + res_hdet['SII']) >= 3
idx_emi = np.logical_and(res_hdet['Ha_EW'] > 0.1, res_hdet['Ha_EW_abs'] > 0.3)  # TODO: better definition of EW thresholds


def copy_ids_to_curr_map(sel_ids, cp_dir, suffix='', prefixes=None):
    system('mkdir ' + cp_dir)
    chdir(cp_dir)
    for i_s, star_id in enumerate(sel_ids):
        s_date = np.int32(star_id / 10e10)
        sobj = str(star_id)
        # print sobj

        source = results_data_dir + str(s_date) + '/' + str(sobj) + suffix + '.png'
        prefix = ''
        if prefixes is not None:
            prefix += prefixes[i_s] + '_'
        target = cp_dir + '/' + prefix + str(sobj) + suffix + '.png'

        if path.isfile(target):
            continue
        # execute copy of the strongest emitters
        system('cp ' + source + ' ' + target)
    chdir('..')


# ----------------------------------------
# Get repeated observations that were at least once detected as in emission
# - repeats could be from different programs than the GALAH -> (TESS, K2, clusters, pilot, Orion ...)
# ----------------------------------------
ra_dec_all = coord.ICRS(ra=res_hdet['ra']*un.deg,
                        dec=res_hdet['dec']*un.deg)
idx_detected = idx_emi * idx_unf * np.logical_not(idx_bin)
res_hdet['repeated'] = 0
id_rep = 1

rep_dir = out_dir + 'repeats/'
system('mkdir ' + rep_dir)
chdir(rep_dir)

for star in res_hdet:
    ra_dec_star = coord.ICRS(ra=star['ra']*un.deg,
                             dec=star['dec']*un.deg)
    idx_close = ra_dec_all.separation(ra_dec_star) < 0.5 * un.arcsec
    if np.sum(idx_close) > 1 and np.sum(res_hdet['repeated'][idx_close]) == 0:
        # we have a repeated observation of a star that wan not yet detected and analysed
        res_hdet['repeated'][idx_close] = id_rep
        id_rep += 1

        idx_detected_close = idx_close * idx_detected
        if np.sum(idx_detected_close) >= 1:
            print 'Repeated observations id', id_rep
            # at least one of the repeats was detected as emission object
            rep_subdir = rep_dir + '{:05.0f}'.format(id_rep) + '/'
            copy_ids_to_curr_map(res_hdet['sobject_id'][idx_close], rep_subdir, suffix='', prefixes=None)

chdir('..')

raise SystemError

n_random_det = 350
# ----------------------------------------
# Totally random subset of results - easy way to browse for strange results
# ----------------------------------------
print 'N for random selection:', len(res_hdet)
idx_sel = np.int64(np.random.uniform(0, len(res_hdet), n_random_det))
copy_ids_to_curr_map(res_hdet['sobject_id'][idx_sel], out_dir + 'random/')

# ----------------------------------------
# sample of missing spectrum flagged objects
# ----------------------------------------
res = res_hdet[np.logical_or(np.bitwise_and(res_hdet['flag'], 32),
                             np.bitwise_and(res_hdet['flag'], 16))]
print 'N no reference:', len(res)
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'flag_noref/')

# ----------------------------------------
# sample of spectra with large difference towards reference
# ----------------------------------------
res = res_hdet[np.logical_or(np.bitwise_and(res_hdet['flag'], 8),
                             np.bitwise_and(res_hdet['flag'], 4))]
print 'N bad reference:', len(res)
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'flag_badref/')

# ----------------------------------------
# sample of spectra with possible erroneous wvl reduction or determined RV velocity
# ----------------------------------------
res = res_hdet[np.logical_not(np.logical_or(np.bitwise_and(res_hdet['flag'], 32),
                                            np.bitwise_and(res_hdet['flag'], 16)))]
res = res[np.logical_or(np.bitwise_and(res['flag'], 2),
                        np.bitwise_and(res['flag'], 1))]
print 'N bad wvl reduction:', len(res)
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'flag_wvlred/')

# ----------------------------------------
# sample of nebular emissions
# ----------------------------------------
res = res_all[(res_all['NII'] + res_all['SII']) >= 3]
print 'N nebulous:', len(res)
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'nebular/')

plot_stars = galah_all[np.in1d(galah_all['sobject_id'], res['sobject_id'])]
plt.scatter(galah_all['ra'], galah_all['dec'], lw=0, s=0.5, c='0.8',)
plt.scatter(plot_stars['ra'], plot_stars['dec'], lw=0, s=0.5, c='black')
plt.xlim(0., 360.)
plt.ylim(-90., 90.)
plt.grid(ls='--', alpha=0.2, color='black')
plt.tight_layout()
# plt.show()
plt.savefig('ra_dec_plot_nebular.png', dpi=450)
plt.close()
res = None

# ----------------------------------------
# sample of sb2 stars
# ----------------------------------------
if res_all['SB2_c1'].dtype == np.dtype('S5'):
    res = res_all[np.logical_or(res_all['SB2_c1'] == 'True',
                                res_all['SB2_c3'] == 'True')]
else:
    res = res_all[np.logical_or(res_all['SB2_c1'] >= 1,
                                res_all['SB2_c3'] >= 1)]
print 'N SB2:', len(res)
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'SB2/')

# ----------------------------------------
# sample of stars with asymmetric emission line - based on computed EW
# ----------------------------------------
idx_asmy = np.logical_and(res_all['Ha_EW_abs'] > 0.4, res_all['Ha_EW'] > 0.1)
idx_asmy = np.logical_and(idx_asmy, np.sqrt(res_all['Ha_EW_asym']**2 + res_all['Hb_EW_asym']**2) > 0.2)
res = res_all[idx_asmy]
print 'N asymmetric:', len(res)
idx_sel = np.int64(np.random.uniform(0, len(res), n_random_det))
copy_ids_to_curr_map(res['sobject_id'][idx_sel], out_dir + 'asym/')

n_best = 250
# ----------------------------------------
# sort by strongest H-alpha emission spectrum
# ----------------------------------------
sort_by = 'Ha_EW'
print 'N '+sort_by+':', n_best
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
idx_inoc = np.in1d(galah_all['sobject_id'], oc_all['sobject_id'])
plt.scatter(galah_all['ra'], galah_all['dec'], lw=0, s=0.5, c='0.8')
# plt.scatter(plot_stars['ra'], plot_stars['dec'], lw=0, s=0.5, c='black')
plt.scatter(plot_stars['ra'][idx_inoc], plot_stars['dec'][idx_inoc], lw=0, s=0.5, c='red')
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
print 'N '+sort_by+':', n_best
res = res_all[np.logical_and(np.isfinite(res_all[sort_by]),
                             res_all['Ha_EW'] > 0.)]
idx_strong = np.argsort(res[sort_by])[::-1]
copy_ids_to_curr_map(res['sobject_id'][idx_strong[:n_best]], out_dir + 'strongest_absolute/',
                     prefixes=['{:.3f}'.format(p_val) for p_val in res[sort_by][idx_strong[:n_best]]])
