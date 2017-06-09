from astropy.table import Table
from halpha_renormalize_data_functions import spectra_resample

import imp
import numpy as np

imp.load_source('helper', '../Carbon-Spectra/helper_functions.py')
from helper import move_to_dir, get_spectra_dr51, get_spectra_dr52

spectra_dir = '/media/storage/HERMES_REDUCED/dr5.1/'
galah_data_dir = '/home/klemen/GALAH_data/'
data_target_dir = '/home/klemen/GALAH_data/'

renormalize_data = True
plot_individual = False

flats_data = Table.read(galah_data_dir+'flat_objects.csv', format='ascii.csv')

for ccd in list([1,2,3,4]):  # 1-4
    print 'Working on ccd {0}'.format(ccd)
    min_wvl = list([4710, 5655, 6475, 7700])[ccd-1]
    max_wvl = list([4910, 5855, 6745, 7875])[ccd-1]
    step_wvl = list([0.02, 0.02, 0.03, 0.03])[ccd-1]

    target_wvl = min_wvl + np.float64(range(0, np.int32(np.ceil((max_wvl - min_wvl) / step_wvl)))) * step_wvl
    # target_wvl *= (1. + (-20.)/3e5)
    spectra_out = np.full_like(target_wvl, 0.)
    spectra_n = np.full_like(target_wvl, 0.)

    out_txt_file = 'galah_dr52_ccd{:1.0f}_{:4.0f}_{:4.0f}_wvlstep_{:01.2f}_lin_flat.csv'.format(ccd, min_wvl, max_wvl, step_wvl)

    for i_d in flats_data['sobject_id']:
        # print i_d
        object_id = str(i_d)
        spectrum, wavelengths = get_spectra_dr51(object_id, root=spectra_dir, bands=[ccd], read_sigma=False)
        spectra_res = spectra_resample(spectrum[0], wavelengths[0], target_wvl)
        idx_ok = np.isfinite(spectra_res)
        spectra_out[idx_ok] += spectra_res[idx_ok]
        spectra_n[idx_ok] += 1

    # save result
    txt_out = open(out_txt_file, 'w')
    txt_out.write(','.join([str(f) for f in spectra_out/spectra_n]))
    txt_out.close()
