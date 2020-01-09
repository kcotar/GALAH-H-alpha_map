import stilism
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as un
import astropy.coordinates as coord
from multiprocessing import Pool
from os import system, path
plt.rcParams['font.size'] = 10


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


d_deg = 0.5
n_deg = int(round(360. / d_deg))
ll = np.linspace(0., 360., n_deg)
bb = np.linspace(-90., 90., int(round(n_deg / 2.)))
dist = 2500
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
plt.savefig('reddening_d{:04d}_n{:d}_map.png'.format(dist, n_deg), dpi=250)

plt.figure(figsize=(7., 4.))
plt.contour(ll, bb, red_img,
            np.arange(0, 1.1, 0.1),
            colors=['darkgreen'], linewidths=[0.25])
plt.xlabel(u'Galactic longitude $l$ [$^{\circ}$]')
plt.ylabel(u'Galactic latitude $b$ [$^{\circ}$]')
plt.xlim(0, 360)
plt.ylim(-90, 90)
plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
plt.yticks([-75, -50, -25, 0, 25, 50, 75])
plt.grid(ls='--', color='black', alpha=0.2, lw=0.5)
plt.tight_layout()
plt.savefig('reddening_d{:04d}_n{:d}_cont.png'.format(dist, n_deg), dpi=250)
