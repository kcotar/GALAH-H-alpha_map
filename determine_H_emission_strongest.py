from os import system, chdir, path
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

data_dir = '/shared/ebla/cotar/'
results_data_dir = '/shared/data-camelot/cotar/H_band_strength_all_20190801/'
out_dir = results_data_dir

print 'Reading results'
res_all = Table.read(results_data_dir + 'results_H_lines.csv', format='ascii.csv')
galah_all = Table.read(data_dir + 'sobject_iraf_53_reduced_20190801.fits')
oc_all = Table.read(data_dir + 'clusters/members_open_gaia_r2.fits')
print 'Results so far:', len(res_all)

# sample of sb2 stars
res = res_all[np.logical_and(res_all['SB2_c1'] > 0,
                             res_all['SB2_c3'] > 0)]
print 'N SB2:', len(res)
idx_sel = np.int64(np.random.uniform(0, len(res), 250))
sb2_dir = out_dir + 'SB2/'
system('mkdir '+sb2_dir)
chdir(sb2_dir)

suffix = ''
for star in res[idx_sel]:
    s_date = np.int32(star['sobject_id'] / 10e10)
    sobj = str(star['sobject_id'])
    print sobj

    source = results_data_dir + str(s_date) + '/' + str(sobj) + suffix + '.png'
    target = sb2_dir + '/' + str(sobj) + suffix + '.png'
    if path.isfile(target):
        continue
    # execute copy of the strongest emitters
    system('cp ' + source + ' ' + target)
chdir('..')

# sort by strongest emission
sort_by = 'Ha_EW'
res = res_all[np.isfinite(res_all[sort_by])]
idx_strong = np.argsort(res[sort_by])[::-1]

strong_dir = out_dir + 'strongest/'
system('mkdir '+strong_dir)
chdir(strong_dir)

suffix = ''
n_best = 5500
for star in res[idx_strong[:n_best]]:
    s_date = np.int32(star['sobject_id'] / 10e10)
    sobj = str(star['sobject_id'])
    print sobj

    source = results_data_dir + str(s_date) + '/' + str(sobj) + suffix + '.png'
    target = strong_dir + '{:06.2f}_'.format(star[sort_by]) + str(sobj) + suffix + '.png'
    if path.isfile(target):
        continue
    # execute copy of the strongest emitters
    system('cp ' + source + ' ' + target)

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
idx_inoc = np.in1d(plot_stars['sobject_id'], oc_all['sobject_id'])
plt.scatter(galah_all['ra'], galah_all['dec'], lw=0, s=0.5, c='0.8')
plt.scatter(plot_stars['ra'], plot_stars['dec'], lw=0, s=0.5, c='black')
plt.scatter(plot_stars['ra'][idx_inoc], plot_stars['dec'][idx_inoc], lw=0, s=0.5, c='red')
plt.xlim(0., 360.)
plt.ylim(-90., 90.)
plt.grid(ls='--', alpha=0.2, color='black')
plt.tight_layout()
plt.savefig('ra_dec_plot_clusters.png', dpi=450)
plt.close()
chdir('..')
res = None

# sort by strongest emission
sort_by = 'Ha_EW_abs'
res = res_all[np.logical_and(np.isfinite(res_all[sort_by]),
                             res_all['Ha_EW'] > 0.)]
idx_strong = np.argsort(res[sort_by])[::-1]

strong_dir = out_dir + 'strongest_absolute/'
system('mkdir '+strong_dir)
chdir(strong_dir)

suffix = ''
for star in res[idx_strong[:n_best]]:
    s_date = np.int32(star['sobject_id'] / 10e10)
    sobj = str(star['sobject_id'])
    print sobj

    source = results_data_dir + str(s_date) + '/' + str(sobj) + suffix + '.png'
    target = strong_dir + '{:06.2f}_'.format(star[sort_by]) + str(sobj) + suffix + '.png'
    if path.isfile(target):
        continue
    # execute copy of the strongest emitters
    system('cp ' + source + ' ' + target)
chdir('..')
