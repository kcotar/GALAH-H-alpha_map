#!/usr/bin/env python3
# coding: utf-8

"""Methods used for calculations on STILISM cube

Authors: N. Leclerc, G. Plum, S. Ferron
"""

# get current dir
import os
here = os.path.dirname(__file__)
# use matplotlib without X server
import matplotlib
matplotlib.use('Agg')

import numpy as np
import scipy.interpolate as spi
import h5py
from astropy import units as u
from astropy.coordinates import SkyCoord

# Module global vars
hdf5file = '/shared/ebla/cotar/stilism_cube.h5'
headers = None
cube = None
cubeXErr = None
cubeYErrMin = None
cubeYErrMax = None
axes = None
min_axes = None
max_axes = None

# For map, used by S. Ferron
step = None
hw = None
points = None
s = None

# Load HDF5 file
def init(hdf5file):
    """Load hdf5, calculate axes values corresponding to data.

    Args:
        hdf5file (str): full path for STILISM HDF5 file.

    Returns:
        dict: headers contains in HDF5 file.
        :func:`np.array`: 3D array which contains E(B-V).
        tuple: (x, y, z) where x,y,z contains array of axes
            corresponding to cube values.
        array: value min for x, y, z axes.
        array: value max for x, y, z axes.

    """
    # read hdf5 file
    with h5py.File(hdf5file, 'r') as hf:
        cube = hf['stilism/cube_datas'][:]
        cubeXerr = hf['stilism/cube_err_distance'][:]
        cubeYerrMin = hf['stilism/cube_err_magnitudemin'][:]
        cubeYerrMax = hf['stilism/cube_err_magnitudemax'][:]
        dc = hf['stilism/cube_datas']

        # Less method call are done with this version:
        headers = {k: v for k, v in dc.attrs.items()}

    sun_position = headers["sun_position"]
    gridstep_values = headers["gridstep_values"]

    # Calculate axes for cube value, with sun at position (0, 0, 0)
    min_axes = -1 * sun_position * gridstep_values
    max_axes = np.abs(min_axes)
    axes = (
        np.linspace(min_axes[0], max_axes[0], cube.shape[0]),
        np.linspace(min_axes[1], max_axes[1], cube.shape[1]),
        np.linspace(min_axes[2], max_axes[2], cube.shape[2])
    )

    # S. Ferron variable for map
    step = np.array(headers["gridstep_values"])
    hw = (np.copy(cube.shape) - 1) / 2.
    points = (
        np.arange(0, cube.shape[0]),
        np.arange(0, cube.shape[1]),
        np.arange(0, cube.shape[2])
    )
    s = hw * step

    return (headers, cube, cubeXerr, cubeYerrMin, cubeYerrMax,
        axes, min_axes, max_axes,
        step, hw, points, s)


# Initialisation
headers, cube, cubeXErr, cubeYErrMin, cubeYErrMax, axes, min_axes, max_axes, step, hw, points, s = init(hdf5file)


def reddening(vlong, ulong, vlat, ulat, frame, step_pc=5):
    """Calculate Reddening versus distance from Sun.

    Args:
        vlong (str or double): Longitude value.
        ulong (str): Longitude unit used in :class:`SkyCoord`.
        vlat (str or double): Latitude value.
        ulat (str): Latitude unit used in :class:`SkyCoord`.
        frame (str): Galactic, icrs ... values supported by :class:`SkyCoord`.

    Kwargs:
        step_pc (int): Incremental distance in parsec

    Returns:
        array: Parsec values.
        array: E(B-V) value obtain with integral of linear extrapolation.

    """
    # Calculate the position for 1pc
    sc = SkyCoord(
        vlong,
        vlat,
        distance=1 * u.pc,
        unit=(ulong, ulat),
        frame=frame
    )
    coords_xyz = sc.transform_to('galactic').represent_as('cartesian').get_xyz().value

    # Find the number of parsec I can calculate before go out the cube
    # (exclude divide by 0)
    not0 = np.where(coords_xyz != 0)
    max_pc = np.amin(
        np.abs(
            np.take(max_axes, not0) / np.take(coords_xyz, not0)
        )
    )

    # Calculate all coordinates to interpolate (use step_pc)
    distances = np.arange(0, max_pc, step_pc)
    sc = SkyCoord(
        vlong,
        vlat,
        distance=distances,
        unit=(ulong, ulat, 'pc'),
        frame=frame
    )
    sc = sc.transform_to('galactic').represent_as('cartesian')
    coords_xyz = np.array([coord.get_xyz().value for coord in sc])

    # linear interpolation with coordinates
    interpolation = spi.interpn(
        axes,
        cube,
        coords_xyz,
        method='linear'
    )
    xvalues = np.arange(0, len(interpolation) * step_pc, step_pc)
    yvalues = np.cumsum(interpolation) * step_pc

    # errors
    xerrors = spi.interpn(
        axes,
        cubeXErr,
        coords_xyz,
        method='linear'
    )
    yerrorsMin = spi.interpn(
        axes,
        cubeYErrMin,
        coords_xyz,
        method='linear'
    )
    yerrorsMax = spi.interpn(
        axes,
        cubeYErrMax,
        coords_xyz,
        method='linear'
    )

    return (
        xvalues,
        np.around(yvalues, decimals=3),
        np.around(xerrors, decimals=0),
        np.around(yerrorsMin, decimals=3),
        np.around(yerrorsMax, decimals=3)
    )


def cube_cut(vlong, ulong, vlat, ulat, frame, vdist, udist, vnlong, unlong, vnlat, unlat):
    """Calculate map cut of cube.

    Args:
        vlong (str or double): Longitude value.
        ulong (str): Longitude unit used in :class:`SkyCoord`.
        vlat (str or double): Latitude value.
        ulat (str): Latitude unit used in :class:`SkyCoord`.
        frame (str): Galactic, icrs, etc
            (values supported by :class:`SkyCoord`).
        vdist (str or double): Distance.
        udist (str): Distance unit used in Skycoord.
        vnlong (str or double): Normal longitude value.
        unlong (str or double): Longitude unit used in :class:`SkyCoord`
            for normal.
        vnlat (str or double): Normal latitude value.
        unlat (str): Latitude unit used in :class:`SkyCoord` for normal.

    Returns:
        img (str):plot of map

    """
    #  Transforming the reference position point into cartesian coordinates:
    sc = SkyCoord(
        vlong,
        vlat,
        distance=vdist,
        unit=(ulong, ulat, udist),
        frame=frame
    )
    r = sc.represent_as('cartesian').get_xyz().value

    #  Getting the normal to the plane in cartesian coordinates:
    wp = SkyCoord(
        vnlong,
        vnlat,
        unit=(unlong, unlat),
        frame=frame
    )
    w = wp.represent_as('cartesian').get_xyz().value

    #  Creating a direct base (u, v, w) using the normal vector:
    lon = np.radians(wp.l.degree)
    u = [np.sin(lon), -np.cos(lon), 0]
    u_latitude = 0 # arcsin(0) = 0 because u[2] = 0
    u_longitude = np.degrees(np.arctan2(u[1], u[0]))

    #  The last vector is just the results of the vector product of w and u
    #  (the order is important to keep the referential oriented as expected):
    v = np.cross(w, u)
    v_latitude = np.degrees(np.arcsin(v[2]))
    v_longitude = np.degrees(np.arctan2(v[1], v[0]))

    # maximum extension maximal of the slice
    #  Taking the two norm of the vector giving the sun position
    #  (if I have understand well the meaning of hw and step):
    f = np.linalg.norm(2 * hw * step, 2)
    #  Then we construct an array giving the cube border:
    c = np.arange(-f, f, step.min())

    #  Transforming this array into a mesh grid (it consist essentially of a
    #  coordinate matrix):
    cu, cv = np.meshgrid(c, c)

    #  The outer product of two vector u_1 and u_2 is equivalent to do the
    #  matrix multiplication u_1 u_2^T (T:: transpose). Here, the
    #  :func:`numpy.outer` function will flatten the matrix of size
    #  (amin(step), amin(step)) to have a vector of dimension $amin(step)^2$.
    #  Then it will multiple, following the matrix multiplication rule all
    #  coordinates of the argument to get in return a matrix of size
    #  (amin(step)^2, len(u)) (same for v and $r+s$ which should be of the same
    #  length).
    #  You can also search for the definition of the kroenecker product as the
    #  outer product is supposed to be a special case of it.
    #  This should give us a matrix to transform the cube coordinates into the
    #  plane coordinate, if I have understand everything correctly:
    cz = np.outer(cu, u) + np.outer(cv, v) + np.outer(np.ones_like(cu), r + s)

    z = np.ones([c.size * c.size])
    z[:] = None

    ix = np.floor((cz[:, 0]) / step[0])
    iy = np.floor((cz[:, 1]) / step[1])
    iz = np.floor((cz[:, 2]) / step[2])
    w = np.where(
        (ix >= 0) & (ix <= cube.shape[0] - 1) &
        (iy >= 0) & (iy <= cube.shape[1] - 1) &
        (iz >= 0) & (iz <= cube.shape[2] - 1)
    )

    z[w] = spi.interpn(
        points,
        cube,
        (ix[w], iy[w], iz[w]),
        method='linear',
        fill_value=1
    )
    z = np.reshape(z, [c.size, c.size])

    f = np.take(
        np.reshape(cu, [c.size * c.size]),
        w
    )
    wx = np.squeeze(
        np.where(
            (c >= np.amin(f)) & (c <= np.amax(f))
        )
    )
    f = np.take(
        np.reshape(cv, [c.size * c.size]),
        w
    )
    wy = np.squeeze(
        np.where(
            (c >= np.amin(f)) & (c <= np.amax(f))
        )
    )

    smap = z[np.ix_(wy, wx)]
    
    if v_latitude < 0:
        v_latitude = -v_latitude
        v_longitude = (v_longitude + 180)%360
        smap = smap[::-1]

    result = {}
    
    # x,y values for axes, log(z) values for a better representation of contour map
    addforLog0 = 1e-7
    result["addforlog0"] = addforLog0
    result["xJsTab"] = np.array2string(c[wx][::2], max_line_width=80, separator=',', precision=2, threshold=np.nan).replace("nan", "NaN")
    result["xTitle"] = "'Left to right towards ⇒ (l=%.1f°,b=%.1f°)'"%(u_longitude, u_latitude)
    result["yJsTab"] = np.array2string(c[wy][::2], max_line_width=80, separator=',', precision=2, threshold=np.nan).replace("nan", "NaN")
    result["yTitle"] = "'Bottom to top towards ⇒ (l=%.1f°,b=%.1f°)'"%(v_longitude, v_latitude)
    logsmap = np.log(smap[::2,::2] + addforLog0)
    result["zJsTab"] = np.array2string(logsmap, max_line_width=80, separator=',', precision=2, threshold=np.nan).replace("nan", "NaN")
    
    # Calculate color bar values (5 values between [min, max])
    # Use log because log(z), and corresponding true value
    logScaletmp = np.linspace(np.nanmin(logsmap), np.nanmax(logsmap), 5)
    result["logScale"] = np.array2string(logScaletmp, separator=",")
    scaletmp = [format(np.exp(v) - addforLog0, '.2e').replace("e-0", "e-") for v in logScaletmp]
    result["scale"] = "[" + ", ".join(["'%s'"%v for v in scaletmp]) + "]"
    
    result["title"] = "'Origin (l=%.1f°,b=%.1f°), distance=%.1fpc<br />Normal to the plane (l=%.1f°,b=%.1f°)'"%(
            sc.l.degree, sc.b.degree,  sc.distance.value,
            wp.l.degree, wp.b.degree)
    
    return result
