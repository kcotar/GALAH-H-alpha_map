from os import system, chdir, path
import numpy as np
from astropy.table import Table

results_data_dir = '/shared/data-camelot/cotar/H_band_strength_all_20190801/'
out_dir = results_data_dir

print 'Reading results'
res = Table.read(results_data_dir + 'results_H_lines.csv', format='ascii.csv')
print 'Results so far:', len(res)

# sort by strongest emission
sort_by = 'Ha_EW'
res = res[np.isfinite(res[sort_by])]
idx_strong = np.argsort(res[sort_by])[::-1]

strong_dir = out_dir + 'strongest/'
system('mkdir '+strong_dir)
chdir(strong_dir)

suffix = ''
n_best = 200
for star in res[idx_strong[:n_best]]:
    s_date = np.int32(star['sobject_id'] / 10e10)
    sobj = str(star['sobject_id'])
    print sobj

    source = results_data_dir + str(s_date) + '/' + str(sobj) + suffix + '.png'
    target = strong_dir + '{:06.2f}_'.format(star[sort_by]) + str(sobj) + suffix + '.png'
    # execute copy of the strongest emitors
    system('cp ' + source + ' ' + target)
