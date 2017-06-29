import numpy as np


def get_best_match(teff, logg, feh, grid_list, midpoint=False):
    teff_values = np.unique(grid_list['teff'])
    teff_best = teff_values[np.argsort(np.abs(teff_values-teff))]

    feh_values = np.unique(grid_list[grid_list['teff'] == teff_best[0]]['feh'])
    feh_best = feh_values[np.argsort(np.abs(feh_values - feh))]

    logg_values = np.unique(grid_list[np.logical_and(grid_list['teff'] == teff_best[0], grid_list['feh'] == feh_best[0])]['logg'])
    logg_best = logg_values[np.argsort(np.abs(logg_values - logg))]

    str_best = 'T_{:.0f}_L_{:1.2f}_F_{:1.2f}'.format(teff_best[0], logg_best[0], feh_best[0])
    if midpoint:
        str_second = 'T_{:.0f}_L_{:1.2f}_F_{:1.2f}'.format(teff_best[1], logg_best[1], feh_best[1])
        return str_best, str_second
    else:
        return str_best