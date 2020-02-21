import stilism
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as un
import astropy.coordinates as coord
from astropy.table import Table
from multiprocessing import Pool
from os import system, path
from astropy.wcs import WCS
plt.rcParams['font.size'] = 15

date_string = '20190801'
galah_data_dir = '/shared/ebla/cotar/'
out_dir = '/shared/data-camelot/cotar/'
results_data_dir = out_dir + 'H_band_strength_complete_20190801_ANN-medians_newthrs/'
general_data = Table.read(galah_data_dir + 'sobject_iraf_53_reduced_'+date_string+'.fits')['sobject_id','ra','dec']
emis_results = Table.read(results_data_dir + 'results_H_lines_complete.fits')
const_data = Table.read(galah_data_dir + 'Constellations/VI_49_bound_18.dat.fits')

ra_dec_stars = coord.ICRS(ra=general_data['ra']*un.deg, dec=general_data['dec']*un.deg)
l_b_stars = ra_dec_stars.transform_to(coord.Galactic)


# reddening = list([])
# for ll_s, bb_s in zip(ll_v, bb_v):
def get_reddening(i):
    ll_s = ll_v[i]
    bb_s = bb_v[i]
    print(ll_s, bb_s)
    d_vals, r_vals, _, _, _ = stilism.reddening(ll_s, un.deg, bb_s, un.deg, frame='galactic', step_pc=5)
    idx_dist = np.argmin(np.abs(d_vals - dist))
    # reddening.append(r_vals[idx_dist])
    return r_vals[idx_dist]


def add_const_lines(in_ax, const_data):
    const_list = np.unique(const_data['cst'])

    for const in const_list:
        vert = const_data[const_data['cst'] == const]
        ra_dec_vert = coord.ICRS(ra=vert['RAhr'], dec=vert['DEdeg'])
        l_b_vert = ra_dec_vert.transform_to(coord.Galactic)
        for i_v in range(len(vert) -1):
            # get vertex coordinates and sort them by increasing l
            l = l_b_vert.l.value[[i_v, i_v + 1]]
            idx_sort = np.argsort(l)
            l = l[idx_sort]
            b = l_b_vert.b.value[[i_v, i_v+1]][idx_sort]

            if l[1] - l[0] > 180:
                in_ax.plot([l[0], l[1] - 360], b, c='black', lw=0.1)
                in_ax.plot([360 + l[0], l[1]], b, c='black', lw=0.1)
            else:
                in_ax.plot(l, b, c='black', lw=0.1)

        if np.abs(np.nanmedian(l_b_vert.b.value)) < 75:
            in_ax.text(np.nanmedian(l_b_vert.l.value),
                       np.nanmedian(l_b_vert.b.value),
                       const,
                       horizontalalignment='center',
                       verticalalignment='center', fontsize=7)


d_deg = 0.5
n_deg = int(round(360. / d_deg))
ll = np.linspace(0., 360., n_deg)
bb = np.linspace(-90., 90., int(round(n_deg / 2.)))
dist = 2000
file_out = 'red_values_ll_bb_d{:04d}_n{:d}.csv'.format(dist, n_deg)

if path.isfile(file_out):
    print('Reading old table')
    all_data = np.loadtxt(file_out)
    reddening = all_data[:, 2]
    ll_v = all_data[:, 0]
    bb_v = all_data[:, 1]
    ll = np.unique(ll_v)
    bb = np.unique(bb_v)

else:
    ll_v, bb_v = np.meshgrid(ll, bb)
    ll_v = ll_v.flatten()
    bb_v = bb_v.flatten()

    print('Mulitprocessing with stilism reddening code')
    pool = Pool(processes=10)
    reddening = pool.map(get_reddening, range(len(ll_v)))
    pool.close()

    print('Saving')
    all_data = np.vstack((ll_v, bb_v, reddening)).T
    np.savetxt(file_out, all_data)

print(ll[0], ll[-1])
print(bb[0], bb[-1])

print('Plotting and contouring')
red_img = np.reshape(reddening, (len(bb), len(ll)))

plt.imshow(red_img, interpolation='none', origin='lower', cmap='viridis',
           extent=[0, 360, -90, 90],
           vmin=np.percentile(red_img, 0),
           vmax=np.percentile(red_img, 100),
           alpha=1.)
plt.xlabel(u'Galactic longitude $l$ [$^{\circ}$]')
plt.ylabel(u'Galactic latitude $b$ [$^{\circ}$]')
plt.colorbar()
plt.tight_layout()
plt.savefig('reddening_d{:04d}_n{:d}_map.png'.format(dist, n_deg), dpi=175)
plt.close()

idx_unf = emis_results['flag'] == 0
idx_bin = np.logical_or(emis_results['SB2_c1'] >= 1, emis_results['SB2_c3'] >= 1)
idx_bin_conf = np.logical_and(emis_results['SB2_c1'] >= 1, emis_results['SB2_c3'] >= 1)
idx_neb = (emis_results['NII'] + emis_results['SII']) >= 3
idx_neb_final = idx_unf * idx_neb * ~idx_bin
sobj_nebular = emis_results[idx_neb_final]['sobject_id']
idx_mark = np.in1d(general_data['sobject_id'], sobj_nebular)
plt.figure(figsize=(14., 7.8))
plt.scatter(l_b_stars.l.value, l_b_stars.b.value, s=0.25, c='darkgrey', lw=0)
plt.scatter(l_b_stars.l.value[idx_mark], l_b_stars.b.value[idx_mark], s=0.6, c='black', lw=0)
plt.contour(ll, bb, red_img,
            np.arange(0, 1.1, 0.1),
            colors=['darkgreen'], linewidths=[0.25])
add_const_lines(plt.gca(), const_data)
plt.xlabel(u'Galactic longitude $l$ [$^{\circ}$]')
plt.ylabel(u'Galactic latitude $b$ [$^{\circ}$]')
plt.xlim(0, 360)
plt.ylim(-75, 75)
plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
plt.yticks([-75, -50, -25, 0, 25, 50, 75])
plt.grid(ls='--', color='black', alpha=0.2, lw=0.5)
plt.tight_layout()
plt.savefig('reddening_d{:04d}_n{:d}_cont_wnebular.png'.format(dist, n_deg), dpi=175)
plt.close()

idx_emi = np.logical_and(emis_results['Ha_EW'] > 0.25, emis_results['Ha_EW_abs'] > 0.)
idx_emiss_final = idx_emi * idx_unf * ~idx_bin_conf
sobj_nebular = emis_results[idx_emiss_final]['sobject_id']
idx_mark = np.in1d(general_data['sobject_id'], sobj_nebular)
plt.figure(figsize=(14., 7.8))
plt.scatter(l_b_stars.l.value, l_b_stars.b.value, s=0.25, c='darkgrey', lw=0)
plt.scatter(l_b_stars.l.value[idx_mark], l_b_stars.b.value[idx_mark], s=0.6, c='black', lw=0)
plt.contour(ll, bb, red_img,
            np.arange(0, 1.1, 0.1),
            colors=['darkgreen'], linewidths=[0.25])
add_const_lines(plt.gca(), const_data)
plt.xlabel(u'Galactic longitude $l$ [$^{\circ}$]')
plt.ylabel(u'Galactic latitude $b$ [$^{\circ}$]')
plt.xlim(0, 360)
plt.ylim(-75, 75)
plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
plt.yticks([-75, -50, -25, 0, 25, 50, 75])
plt.grid(ls='--', color='black', alpha=0.2, lw=0.5)
plt.tight_layout()
plt.savefig('reddening_d{:04d}_n{:d}_cont_wemisions.png'.format(dist, n_deg), dpi=175)
# plt.show()
plt.close()
