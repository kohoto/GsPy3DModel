# Some GeostatsPy Functions - by Michael Pyrcz, maintained at https://git.io/fNgR7.
# A set of functions to provide access to GSLIB in Python.
# GSLIB executables: nscore.exe, declus.exe, gam.exe, gamv.exe, vmodel.exe, kb2d.exe & sgsim.exe must be in the working directory
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math
import scipy.signal as signal
import decimal

image_type = 'png';
dpi = 600


# utility to convert 1D or 2D numpy ndarray to a GSLIB Geo-EAS file for use with GSLIB methods
def ndarray2GSLIB(array, data_file, col_name):
    file_out = open(data_file, "w")
    file_out.write(data_file + '\n')
    file_out.write('1 \n')
    file_out.write(col_name + '\n')
    if array.ndim == 2:
        ny = (array.shape[0])
        nx = (array.shape[1])
        for iy in range(0, ny):
            for ix in range(0, nx):
                file_out.write(str(array[ny - 1 - iy, ix]) + '\n')
    elif array.ndim == 1:
        nx = len(array)
        for ix in range(0, nx):
            file_out.write(str(array[ix]) + '\n')
    else:
        print("Error: must use a 2D array")
        file_out.close()
        return
    file_out.close()


# utility to convert GSLIB Geo-EAS files to a 1D or 2D numpy ndarray for use with Python methods
def GSLIB2ndarray(data_file, kcol, nx, ny):
    if ny > 1:
        array = np.ndarray(shape=(ny, nx), dtype=float, order='F')
    else:
        array = np.zeros(nx)
    with open(data_file) as myfile:  # read first two lines
        head = [next(myfile) for x in range(2)]
        line2 = head[1].split()
        ncol = int(line2[0])  # get the number of columns
        for icol in range(0, ncol):  # read over the column names
            head = [next(myfile) for x in range(1)]
            if icol == kcol:
                col_name = head[0].split()[0]
        if ny > 1:
            for iy in range(0, ny):
                for ix in range(0, nx):
                    head = [next(myfile) for x in range(1)]
                    array[ny - 1 - iy][ix] = head[0].split()[kcol]
        else:
            for ix in range(0, nx):
                head = [next(myfile) for x in range(1)]
                array[ix] = head[0].split()[kcol]
    return array, col_name


# utility to convert pandas DataFrame to a GSLIB Geo-EAS file for use with GSLIB methods
def Dataframe2GSLIB(data_file, df):
    ncol = len(df.columns)
    nrow = len(df.index)
    file_out = open(data_file, "w")
    file_out.write(data_file + '\n')
    file_out.write(str(ncol) + '\n')
    for icol in range(0, ncol):
        file_out.write(df.columns[icol] + '\n')
    for irow in range(0, nrow):
        for icol in range(0, ncol):
            file_out.write(str(df.iloc[irow, icol]) + ' ')
        file_out.write('\n')
    file_out.close()


# utility to convert GSLIB Geo-EAS files to a pandas DataFrame for use with Python methods
def GSLIB2Dataframe(data_file):
    colArray = []
    with open(data_file) as myfile:  # read first two lines
        head = [next(myfile) for x in range(2)]
        line2 = head[1].split()
        ncol = int(line2[0])
        for icol in range(0, ncol):
            head = [next(myfile) for x in range(1)]
            colArray.append(head[0].split()[0])
        data = np.loadtxt(myfile, skiprows=0)
        df = pd.DataFrame(data)
        df.columns = colArray
        return df


# histogram, reimplemented in Python of GSLIB hist with MatPlotLib methods, displayed and as image file
def hist(array, xmin, xmax, log, cumul, bins, weights, xlabel, title, fig_name):
    plt.figure(figsize=(8, 6))
    cs = plt.hist(array, alpha=0.2, color='red', edgecolor='black', bins=bins, range=[xmin, xmax], weights=weights,
                  log=log, cumulative=cumul)
    plt.title(title)
    plt.xlabel(xlabel);
    plt.ylabel('Frequency')
    plt.savefig(fig_name + '.' + image_type, dpi=dpi)
    plt.show()
    return


# histogram, reimplemented in Python of GSLIB hist with MatPlotLib methods (version for subplots)
def hist_st(array, xmin, xmax, log, cumul, bins, weights, xlabel, title):
    cs = plt.hist(array, alpha=0.2, color='red', edgecolor='black', bins=bins, range=[xmin, xmax], weights=weights,
                  log=log, cumulative=cumul)
    plt.title(title)
    plt.xlabel(xlabel);
    plt.ylabel('Frequency')
    return


# location map, reimplemention in Python of GSLIB locmap with MatPlotLib methods
def locmap(df, xcol, ycol, vcol, xmin, xmax, ymin, ymax, vmin, vmax, title, xlabel, ylabel, vlabel, cmap, fig_name):
    plt.figure(figsize=(8, 6))
    im = plt.scatter(df[xcol], df[ycol], s=None, c=df[vcol], marker=None, cmap=cmap, norm=None, vmin=vmin, vmax=vmax,
                     alpha=0.8, linewidths=0.8, verts=None, edgecolors="black")
    plt.title(title)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(im, orientation='vertical', ticks=np.linspace(vmin, vmax, 10))
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    plt.savefig(fig_name + '.' + image_type, dpi=dpi)
    plt.show()
    return im


# location map, reimplemention in Python of GSLIB locmap with MatPlotLib methods (version for subplots)
def locmap_st(df, xcol, ycol, vcol, xmin, xmax, ymin, ymax, vmin, vmax, title, xlabel, ylabel, vlabel, cmap):
    im = plt.scatter(df[xcol], df[ycol], s=None, c=df[vcol], marker=None, cmap=cmap, norm=None, vmin=vmin, vmax=vmax,
                     alpha=0.8, linewidths=0.8, verts=None, edgecolors="black")
    plt.title(title)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(im, orientation='vertical', ticks=np.linspace(vmin, vmax, 10))
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return im


# pixel plot, reimplemention in Python of GSLIB pixelplt with MatPlotLib methods
def pixelplt(array, xmin, xmax, ymin, ymax, step, vmin, vmax, title, xlabel, ylabel, vlabel, cmap, fig_name):
    print(str(step))
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step))
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.contourf(xx, yy, array, cmap=cmap, vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, 100))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.gca().set_aspect("equal")
    # setting for color bar
    decimal.getcontext().prec = 1
    imratio = array.shape[0] / array.shape[1]
    cbar = plt.colorbar(im, ticks=np.linspace(float(decimal.Decimal(vmin)), float(decimal.Decimal(vmax)), 5),
                        fraction=0.046 * imratio, pad=0.04)
    cbar.set_label(vlabel, rotation=270)
    fig.savefig(fig_name + '.' + image_type, dpi=dpi, transparent=True)
    fig.tight_layout()
    plt.show()
    return im


# pixel plot, reimplemention in Python of GSLIB pixelplt with MatPlotLib methods(version for subplots)
def pixelplt_st(array, xmin, xmax, ymin, ymax, step, vmin, vmax, title, xlabel, ylabel, vlabel, cmap):
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step))
    x = [];
    y = [];
    v = []  # use dummy since scatter plot controls legend min and max appropriately and contour does not!
    cs = plt.contourf(xx, yy, array, cmap=cmap, vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, 100))
    im = plt.scatter(x, y, s=None, c=v, marker=None, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, linewidths=0.8,
                     verts=None, edgecolors="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.clim(vmin, vmax)
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs


# pixel plot, reimplemention in Python of GSLIB pixelplt with MatPlotLib methods (version for subplots, log scale)
def pixelplt_log_st(array, xmin, xmax, ymin, ymax, step, vmin, vmax, title, xlabel, ylabel, vlabel, cmap):
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step))
    x = [];
    y = [];
    v = []  # use dummy since scatter plot controls legend min and max appropriately and contour does not!
    color_int = np.r_[np.log(vmin):np.log(vmax):0.5]
    color_int = np.exp(color_int)
    cs = plt.contourf(xx, yy, array, cmap=cmap, vmin=vmin, vmax=vmax, levels=color_int,
                      norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    im = plt.scatter(x, y, s=None, c=v, marker=None, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, linewidths=0.8,
                     verts=None, edgecolors="black", norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs


# pixel plot and location map, reimplementation in Python of a GSLIB MOD with MatPlotLib methods
def locpix(array, xmin, xmax, ymin, ymax, step, vmin, vmax, df, xcol, ycol, vcol, title, xlabel, ylabel, vlabel, cmap,
           fig_name):
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step))
    plt.figure(figsize=(8, 6))
    cs = plt.contourf(xx, yy, array, cmap=cmap, vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, 100))
    im = plt.scatter(df[xcol], df[ycol], s=None, c=df[vcol], marker=None, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8,
                     linewidths=0.8, verts=None, edgecolors="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar(orientation='vertical')
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    plt.savefig(fig_name + '.' + image_type, dpi=dpi)
    plt.show()
    return cs


# pixel plot and location map, reimplementation in Python of a GSLIB MOD with MatPlotLib methods(version for subplots)
def locpix_st(array, xmin, xmax, ymin, ymax, step, vmin, vmax, df, xcol, ycol, vcol, title, xlabel, ylabel, vlabel,
              cmap):
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step))
    cs = plt.contourf(xx, yy, array, cmap=cmap, vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, 100))
    im = plt.scatter(df[xcol], df[ycol], s=None, c=df[vcol], marker=None, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8,
                     linewidths=0.8, verts=None, edgecolors="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar(orientation='vertical')
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs


def locpix_log_st(array, xmin, xmax, ymin, ymax, step, vmin, vmax, df, xcol, ycol, vcol, title, xlabel, ylabel, vlabel,
                  cmap, fig_name):
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step))
    color_int = np.r_[np.log(vmin):np.log(vmax):0.5]
    color_int = np.exp(color_int)
    cs = plt.contourf(xx, yy, array, cmap=cmap, vmin=vmin, vmax=vmax, levels=color_int,
                      norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    im = plt.scatter(df[xcol], df[ycol], s=None, c=df[vcol], marker=None, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8,
                     linewidths=0.8, verts=None, edgecolors="black",
                     norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cbar = plt.colorbar(orientation='vertical')
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    return cs


# affine distribution correction reimplemented in Python with numpy methods
def affine(array, tmean, tstdev):
    mean = np.average(array)
    stdev = np.std(array)
    array = (tstdev / stdev) * (array - mean) + tmean
    return (array)


# normal score transform, wrapper for nscore from GSLIB (.exe must be in working directory)(not used in this demo)
def nscore(x):
    import numpy as np
    initial_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    nx = np.ma.size(x)
    ny = 1
    ndarray2GSLIB(x, "nscore.dat", "value")
    nscore_lines = ["                  Parameters for NSCORE                                    \n",
                    "                  *********************                                    \n",
                    "                                                                           \n",
                    "START OF PARAMETERS:                                                       \n",
                    "nscore.dat           -file with data                                       \n",
                    "1   0                    -  columns for variable and weight                \n",
                    "-1.0e21   1.0e21         -  trimming limits                                \n",
                    "0                        -1=transform according to specified ref. dist.    \n",
                    "../histsmth/histsmth.out -  file with reference dist.                      \n",
                    "1   2                    -  columns for variable and weight                \n",
                    "nscore.out               -file for output                                  \n",
                    "nscore.trn               -file for output transformation table             \n"]
    open("nscore.par", 'w').write(''.join(nscore_lines))

    os.system('nscore.exe nscore.par')
    y, name = GSLIB2ndarray('nscore.out', 1, nx, ny)
    os.chdir(initial_dir)
    return (y)


def make_variogram(nug, nst, it1, cc1, azi1, hmaj1, hmin1, it2=1, cc2=0, azi2=0, hmaj2=0, hmin2=0):
    if cc2 == 0:
        nst = 1
    var = dict(
        [('nug', nug), ('nst', nst), ('it1', it1), ('cc1', cc1), ('azi1', azi1), ('hmaj1', hmaj1), ('hmin1', hmin1),
         ('it2', it2), ('cc2', cc2), ('azi2', azi2), ('hmaj2', hmaj2), ('hmin2', hmin2)])
    if nug + cc1 + cc2 != 1:
        print('\x1b[0;30;41m make_variogram Warning: sill does not sum to 1.0, do not use in simulation \x1b[0m')
    if cc1 < 0 or cc2 < 0 or nug < 0 or hmaj1 < 0 or hmaj2 < 0 or hmin1 < 0 or hmin2 < 0:
        print('\x1b[0;30;41m make_variogram Warning: contributions and ranges must be all positive \x1b[0m')
    if hmaj1 < hmin1 or hmaj2 < hmin2:
        print('\x1b[0;30;41m make_variogram Warning: major range should be greater than minor range \x1b[0m')
    return var


# irregularly sampled variogram, 2D wrapper for gam from GSLIB (.exe must be in working directory)
def gamv_2d(df, xcol, ycol, vcol, nlag, lagdist, azi, atol, bstand):
    initial_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    lag = [];
    gamma = [];
    npair = []
    df_ext = pd.DataFrame({'X': df[xcol], 'Y': df[ycol], 'Z': df[vcol]})
    Dataframe2GSLIB("gamv_out.dat", df_ext)

    gamv_lines = ["                  Parameters for GAMV                                      \n",
                  "                  *******************                                      \n",
                  "                                                                           \n",
                  "START OF PARAMETERS:                                                       \n",
                  "gamv_out.dat                    -file with data                            \n",
                  "1   2   0                         -   columns for X, Y, Z coordinates      \n",
                  "1   3   0                         -   number of variables,col numbers      \n",
                  "-1.0e21     1.0e21                -   trimming limits                      \n",
                  "gamv.out                          -file for variogram output               \n",
                  str(nlag) + "                      -number of lags                          \n",
                  str(lagdist) + "                       -lag separation distance                 \n",
                  str(lagdist * 0.5) + "                   -lag tolerance                           \n",
                  "1                                 -number of directions                    \n",
                  str(azi) + " " + str(atol) + " 99999.9 0.0  90.0  50.0  -azm,atol,bandh,dip,dtol,bandv \n",
                  str(bstand) + "                    -standardize sills? (0=no, 1=yes)        \n",
                  "1                                 -number of variograms                    \n",
                  "1   1   1                         -tail var., head var., variogram type    \n"]
    open("gamv.par", 'w').write(''.join(gamv_lines))

    os.system("gamv.exe gamv.par")
    reading = True
    with open("gamv.out", 'r') as myfile:
        head = [next(myfile) for x in range(1)]  # skip the first line
        iline = 0
        while reading:
            try:
                head = [next(myfile) for x in range(1)]
                lag.append(float(head[0].split()[1]))
                gamma.append(float(head[0].split()[2]))
                npair.append(float(head[0].split()[3]))
                iline = iline + 1
            except StopIteration:
                reading = False
    os.chdir(initial_dir)
    return (lag, gamma, npair)


# irregular spaced data, 2D wrapper for varmap from GSLIB (.exe must be in working directory)
def varmapv_2d(df, xcol, ycol, vcol, nx, ny, lagdist, minpairs, vmax, bstand, title, vlabel):
    import numpy as np

    df_ext = pd.DataFrame({'X': df[xcol], 'Y': df[ycol], 'Z': df[vcol]})
    Dataframe2GSLIB("varmap_out.dat", df_ext)

    varmap_lines = ["              Parameters for VARMAP                                        \n",
                    "              *********************                                        \n",
                    "                                                                           \n",
                    "START OF PARAMETERS:                                                       \n",
                    "varmap_out.dat          -file with data                                    \n",
                    "1   3                        -   number of variables: column numbers       \n",
                    "-1.0e21     1.0e21           -   trimming limits                           \n",
                    "0                            -1=regular grid, 0=scattered values           \n",
                    " 50   50    1                -if =1: nx,     ny,   nz                      \n",
                    "1.0  1.0  1.0                -       xsiz, ysiz, zsiz                      \n",
                    "1   2   0                    -if =0: columns for x,y, z coordinates        \n",
                    "varmap.out                   -file for variogram output                    \n",
                    str(nx) + " " + str(ny) + " 0 " + "-nxlag, nylag, nzlag                     \n",
                    str(lagdist) + " " + str(lagdist) + " 1.0              -dxlag, dylag, dzlag \n",
                    str(minpairs) + "             -minimum number of pairs                      \n",
                    str(bstand) + "               -standardize sill? (0=no, 1=yes)              \n",
                    "1                            -number of variograms                         \n",
                    "1   1   1                    -tail, head, variogram type                   \n"]
    open("varmap.par", 'w').write(''.join(varmap_lines))

    os.system('varmap.exe varmap.par')
    nnx = nx * 2 + 1;
    nny = ny * 2 + 1
    varmap, name = GSLIB2ndarray("varmap.out", 0, nnx, nny)

    xmax = ((float(nx) + 0.5) * lagdist);
    xmin = -1 * xmax;
    ymax = ((float(ny) + 0.5) * lagdist);
    ymin = -1 * ymax;
    pixelplt(varmap, xmin, xmax, ymin, ymax, lagdist, 0, vmax, title, 'X', 'Y', vlabel, cmap)
    return (varmap)


# regular spaced data, 2D wrapper for varmap from GSLIB (.exe must be in working directory)
def varmap_2d(array, nx, ny, hsiz, nlagx, nlagy, minpairs, vmax, bstand, title, vlabel):

    ndarray2GSLIB(array, "varmap_out.dat", "gam.dat")

    varmap_lines = ["              Parameters for VARMAP                                        \n",
                    "              *********************                                        \n",
                    "                                                                           \n",
                    "START OF PARAMETERS:                                                       \n",
                    "varmap_out.dat          -file with data                                    \n",
                    "1   1                        -   number of variables: column numbers       \n",
                    "-1.0e21     1.0e21           -   trimming limits                           \n",
                    "1                            -1=regular grid, 0=scattered values           \n",
                    str(nx) + " " + str(ny) + " 1  -if =1: nx,     ny,   nz                     \n",
                    str(hsiz) + " " + str(hsiz) + " 1.0  - xsiz, ysiz, zsiz                     \n",
                    "1   2   0                    -if =0: columns for x,y, z coordinates        \n",
                    "varmap.out                   -file for variogram output                    \n",
                    str(nlagx) + " " + str(nlagy) + " 0 " + "-nxlag, nylag, nzlag               \n",
                    str(hsiz) + " " + str(hsiz) + " 1.0              -dxlag, dylag, dzlag       \n",
                    str(minpairs) + "             -minimum number of pairs                      \n",
                    str(bstand) + "               -standardize sill? (0=no, 1=yes)              \n",
                    "1                            -number of variograms                         \n",
                    "1   1   1                    -tail, head, variogram type                   \n"]
    open("varmap.par", 'w').write(''.join(varmap_lines))

    os.system('varmap.exe varmap.par')
    nnx = nlagx * 2 + 1;
    nny = nlagy * 2 + 1
    varmap, name = GSLIB2ndarray("varmap.out", 0, nnx, nny)

    xmax = ((float(nlagx) + 0.5) * hsiz);
    xmin = -1 * xmax;
    ymax = ((float(nlagy) + 0.5) * hsiz);
    ymin = -1 * ymax;
    pixelplt(varmap, xmin, xmax, ymin, ymax, hsiz, 0, vmax, title, 'X', 'Y', vlabel, cmap)
    return (varmap)


# variogram model, 2D wrapper for vmodel from GSLIB (.exe must be in working directory)
def vmodel_2d(nlag, step, azi, nug, nst, tstr1, c1, azi1, rmaj1, rmin1, tstr2=1, c2=0, azi2=0, rmaj2=0, rmin2=0):
    lag = [];
    gamma = []

    vmodel_lines = ["                                                                           \n",
                    "                  Parameters for VMODEL                                    \n",
                    "                  *********************                                    \n",
                    "                                                                           \n",
                    "START OF PARAMETERS:                                                       \n",
                    "vmodel.var                   -file for variogram output                    \n",
                    "1 " + str(nlag) + "          -number of directions and lags                \n",
                    str(azi) + " 0.0 " + str(step) + " -azm, dip, lag distance                  \n",
                    str(nst) + " " + str(nug) + " -nst, nugget effect                           \n",
                    str(tstr1) + " " + str(c1) + " " + str(azi1) + " 0.0   0.0   0.0 -it,cc,ang1,ang2,ang3 \n",
                    str(rmaj1) + " " + str(rmin1) + " 0.0 -a_hmax, a_hmin, a_vert               \n",
                    str(tstr2) + " " + str(c2) + " " + str(azi2) + " 0.0   0.0   0.0 -it,cc,ang1,ang2,ang3 \n",
                    str(rmaj2) + " " + str(rmin2) + " 0.0 -a_hmax, a_hmin, a_vert               \n"]
    open("vmodel.par", 'w').write(''.join(vmodel_lines))

    os.system('vmodel.exe vmodel.par')
    reading = True
    with open("vmodel.var") as myfile:
        head = [next(myfile) for x in range(1)]  # skip the first line
        iline = 0
        while reading:
            try:
                head = [next(myfile) for x in range(1)]
                lag.append(float(head[0].split()[1]))
                gamma.append(float(head[0].split()[2]))
                iline = iline + 1
            except StopIteration:
                reading = False

    return (lag, gamma)


# cell-based declustering, 2D wrapper for declus from GSLIB (.exe must be in working directory)
def declus(df, xcol, ycol, vcol, cmin, cmax, cnum, bmin):
    nrow = len(df)
    weights = []
    file = 'declus_out.dat'
    file_out = open(file, "w")
    file_out.write('declus_out.dat' + '\n')
    file_out.write('3' + '\n')
    file_out.write('x' + '\n')
    file_out.write('y' + '\n')
    file_out.write('value' + '\n')
    for irow in range(0, nrow):
        file_out.write(
            str(df.iloc[irow][xcol]) + ' ' + str(df.iloc[irow][ycol]) + ' ' + str(df.iloc[irow][vcol]) + ' \n')
    file_out.close()

    declus_lines = ["                  Parameters for DECLUS                                    \n",
                    "                  *********************                                    \n",
                    "                                                                           \n",
                    "START OF PARAMETERS:                                                       \n",
                    "declus_out.dat           -file with data                                   \n",
                    "1   2   0   3               -  columns for X, Y, Z, and variable           \n",
                    "-1.0e21     1.0e21          -  trimming limits                             \n",
                    "declus.sum                  -file for summary output                       \n",
                    "declus.out                  -file for output with data & weights           \n",
                    "1.0   1.0                   -Y and Z cell anisotropy (Ysize=size*Yanis)    \n",
                    str(bmin) + "                -0=look for minimum declustered mean (1=max)   \n",
                    str(cnum) + " " + str(cmin) + " " + str(
                        cmax) + " -number of cell sizes, min size, max size      \n",
                    "5                           -number of origin offsets                      \n"]
    open("declus.par", 'w').write(''.join(declus_lines))

    os.system('declus.exe declus.par')
    df = GSLIB2Dataframe("declus.out")
    for irow in range(0, nrow):
        weights.append(df.iloc[irow, 3])

    return (weights)


# sequential Gaussian simulation, 2D unconditional wrapper for sgsim from GSLIB (.exe must be in working directory)
def GSLIB_sgsim_2d_uncond(nreal, nx, ny, hsiz, seed, var, output_file):
    import numpy as np
    cd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    nug = var['nug']
    nst = var['nst'];
    it1 = var['it1'];
    cc1 = var['cc1'];
    azi1 = var['azi1'];
    hmaj1 = var['hmaj1'];
    hmin1 = var['hmin1']
    it2 = var['it2'];
    cc2 = var['cc2'];
    azi2 = var['azi2'];
    hmaj2 = var['hmaj2'];
    hmin2 = var['hmin2']
    max_range = max(hmaj1, hmaj2)
    hmn = hsiz * 0.5
    hctab = int(max_range / hsiz) * 2 + 1


    sgsim_lines = ["              Parameters for SGSIM                                         \n",
                   "              ********************                                         \n",
                   "                                                                           \n",
                   "START OF PARAMETER:                                                        \n",
                   "none                          -file with data                              \n",
                   "1  2  0  3  5  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n",
                   "-1.0e21 1.0e21                -  trimming limits                           \n",
                   "0                             -transform the data (0=no, 1=yes)            \n",
                   "none.trn                      -  file for output trans table               \n",
                   "1                             -  consider ref. dist (0=no, 1=yes)          \n",
                   "none.dat                      -  file with ref. dist distribution          \n",
                   "1  0                          -  columns for vr and wt                     \n",
                   "-4.0    4.0                   -  zmin,zmax(tail extrapolation)             \n",
                   "1      -4.0                   -  lower tail option, parameter              \n",
                   "1       4.0                   -  upper tail option, parameter              \n",
                   "0                             -debugging level: 0,1,2,3                    \n",
                   "nonw.dbg                      -file for debugging output                   \n",
                   str(output_file) + "           -file for simulation output                  \n",
                   str(nreal) + "                 -number of realizations to generate          \n",
                   str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n",
                   str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n",
                   "1 0.0 1.0                     - nz zmn zsiz                                \n",
                   str(seed) + "                  -random number seed                          \n",
                   "0     8                       -min and max original data for sim           \n",
                   "12                            -number of simulated nodes to use            \n",
                   "0                             -assign data to nodes (0=no, 1=yes)          \n",
                   "1     3                       -multiple grid search (0=no, 1=yes),num      \n",
                   "0                             -maximum data per octant (0=not used)        \n",
                   str(max_range) + " " + str(max_range) + " 1.0 -maximum search  (hmax,hmin,vert) \n",
                   str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n",
                   str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n",
                   "0     0.60   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n",
                   "none.dat                      -  file with LVM, EXDR, or COLC variable     \n",
                   "4                             -  column for secondary variable             \n",
                   str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n",
                   str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n",
                   " " + str(hmaj1) + " " + str(hmin1) + " 1.0 - a_hmax, a_hmin, a_vert        \n",
                   str(it2) + " " + str(cc2) + " " + str(azi2) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n",
                   " " + str(hmaj2) + " " + str(hmin2) + " 1.0 - a_hmax, a_hmin, a_vert        \n"]
    open("sgsim.par", 'w').write(''.join(sgsim_lines))
    print(os.getcwd())
    os.system('"sgsim.exe sgsim.par"')
    sim_array = GSLIB2ndarray(output_file, 0, nx, ny)

    os.chdir(cd)
    return (sim_array[0])


# kriging estimation, 2D wrapper for kb2d from GSLIB (.exe must be in working directory)
def GSLIB_kb2d_2d(df, xcol, ycol, vcol, nx, ny, hsiz, var, output_file):
    import numpy as np

    X = df[xcol];
    Y = df[ycol];
    V = df[vcol]
    df_temp = pd.DataFrame({'X': X, 'Y': Y, 'Var': V})
    Dataframe2GSLIB('data_temp.dat', df_temp)

    nug = var['nug']
    nst = var['nst'];
    it1 = var['it1'];
    cc1 = var['cc1'];
    azi1 = var['azi1'];
    hmaj1 = var['hmaj1'];
    hmin1 = var['hmin1']
    it2 = var['it2'];
    cc2 = var['cc2'];
    azi2 = var['azi2'];
    hmaj2 = var['hmaj2'];
    hmin2 = var['hmin2']
    max_range = max(hmaj1, hmaj2)
    hmn = hsiz * 0.5

    kb2d_lines = ["              Parameters for KB2D                                          \n",
                  "              ********************                                         \n",
                  "                                                                           \n",
                  "START OF PARAMETER:                                                        \n",
                  "data_temp.dat                         -file with data                      \n",
                  "1  2  3                               -  columns for X,Y,vr                \n",
                  "-1.0e21   1.0e21                      -   trimming limits                  \n",
                  "0                                     -debugging level: 0,1,2,3            \n",
                  "none.dbg                              -file for debugging output           \n",
                  str(output_file) + "                   -file for kriged output              \n",
                  str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n",
                  str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n",
                  "1    1                                -x and y block discretization        \n",
                  "1    30                               -min and max data for kriging        \n",
                  str(max_range) + "                     -maximum search radius               \n",
                  "1    -9999.9                          -0=SK, 1=OK,  (mean if SK)           \n",
                  str(nst) + " " + str(nug) + "          -nst, nugget effect                  \n",
                  str(it1) + " " + str(cc1) + " " + str(azi1) + " " + str(hmaj1) + " " + str(
                      hmin1) + " -it, c ,azm ,a_max ,a_min \n",
                  str(it2) + " " + str(cc2) + " " + str(azi2) + " " + str(hmaj2) + " " + str(
                      hmin2) + " -it, c ,azm ,a_max ,a_min \n"]
    open("kb2d.par", 'w').write(''.join(kb2d_lines))

    os.system('"kb2d.exe kb2d.par"')
    est_array = GSLIB2ndarray(output_file, 0, nx, ny)
    var_array = GSLIB2ndarray(output_file, 1, nx, ny)
    return (est_array[0], var_array[0])


# sequential Gaussian simulation, 2D wrapper for sgsim from GSLIB (.exe must be in working directory)
def GSLIB_sgsim_2d(nreal, df, xcol, ycol, vcol, nx, ny, hsiz, seed, var, output_file):
    import numpy as np

    X = df[xcol];
    Y = df[ycol];
    V = df[vcol]
    var_min = V.values.min();
    var_max = V.values.max()
    df_temp = pd.DataFrame({'X': X, 'Y': Y, 'Var': V})
    Dataframe2GSLIB('data_temp.dat', df_temp)

    nug = var['nug']
    nst = var['nst'];
    it1 = var['it1'];
    cc1 = var['cc1'];
    azi1 = var['azi1'];
    hmaj1 = var['hmaj1'];
    hmin1 = var['hmin1']
    it2 = var['it2'];
    cc2 = var['cc2'];
    azi2 = var['azi2'];
    hmaj2 = var['hmaj2'];
    hmin2 = var['hmin2']
    max_range = max(hmaj1, hmaj2)
    hmn = hsiz * 0.5
    hctab = int(max_range / hsiz) * 2 + 1

    sim_array = np.random.rand(nx, ny)

    file = open("../GsPy3DModel/sgsim.par", "w")
    sgsim_lines = ["              Parameters for SGSIM                                         \n",
                   "              ********************                                         \n",
                   "                                                                           \n",
                   "START OF PARAMETER:                                                        \n",
                   "data_temp.dat                 -file with data                              \n",
                   "1  2  0  3  0  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n",
                   "-1.0e21 1.0e21                -  trimming limits                           \n",
                   "1                             -transform the data (0=no, 1=yes)            \n",
                   "none.trn                      -  file for output trans table               \n",
                   "0                             -  consider ref. dist (0=no, 1=yes)          \n",
                   "none.dat                      -  file with ref. dist distribution          \n",
                   "1  0                          -  columns for vr and wt                     \n",
                   str(var_min) + " " + str(var_max) + "   zmin,zmax(tail extrapolation)       \n",
                   "1   " + str(var_min) + "      -  lower tail option, parameter              \n",
                   "1   " + str(var_max) + "      -  upper tail option, parameter              \n",
                   "0                             -debugging level: 0,1,2,3                    \n",
                   "nonw.dbg                      -file for debugging output                   \n",
                   str(output_file) + "           -file for simulation output                  \n",
                   str(nreal) + "                 -number of realizations to generate          \n",
                   str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n",
                   str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n",
                   "1 0.0 1.0                     - nz zmn zsiz                                \n",
                   str(seed) + "                  -random number seed                          \n",
                   "0     8                       -min and max original data for sim           \n",
                   "12                            -number of simulated nodes to use            \n",
                   "0                             -assign data to nodes (0=no, 1=yes)          \n",
                   "1     3                       -multiple grid search (0=no, 1=yes),num      \n",
                   "0                             -maximum data per octant (0=not used)        \n",
                   str(max_range) + " " + str(max_range) + " 1.0 -maximum search  (hmax,hmin,vert) \n",
                   str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n",
                   str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n",
                   "0     0.60   1.0              -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n",
                   "none.dat                      -  file with LVM, EXDR, or COLC variable     \n",
                   "4                             -  column for secondary variable             \n",
                   str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n",
                   str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n",
                   " " + str(hmaj1) + " " + str(hmin1) + " 1.0 - a_hmax, a_hmin, a_vert        \n",
                   str(it2) + " " + str(cc2) + " " + str(azi2) + " 0.0 0.0 -it,cc,ang1,ang2,ang3\n",
                   " " + str(hmaj2) + " " + str(hmin2) + " 1.0 - a_hmax, a_hmin, a_vert        \n"]
    open("sgsim.par", 'w').write(''.join(sgsim_lines))

    os.system('"sgsim.exe sgsim.par"')
    sim_array = GSLIB2ndarray(output_file, 0, nx, ny)
    return (sim_array[0])


# sequential Gaussian simulation, 2D unconditional wrapper for sgsim from GSLIB (.exe must be in working directory)
def GSLIB_cosgsim_2d_uncond(nreal, nx, ny, hsiz, seed, var, sec, correl, output_file):
    import numpy as np
    nug = var['nug']
    nst = var['nst'];
    it1 = var['it1'];
    cc1 = var['cc1'];
    azi1 = var['azi1'];
    hmaj1 = var['hmaj1'];
    hmin1 = var['hmin1']
    it2 = var['it2'];
    cc2 = var['cc2'];
    azi2 = var['azi2'];
    hmaj2 = var['hmaj2'];
    hmin2 = var['hmin2']

    max_range = max(hmaj1, hmaj2)
    hmn = hsiz * 0.5
    hctab = int(max_range / hsiz) * 2 + 1
    sim_array = np.random.rand(nx, ny)

    ndarray2GSLIB(sec, "sec.dat", 'sec_dat')

    sgsim_lines = ["              Parameters for SGSIM                                         \n",
                   "              ********************                                         \n",
                   "                                                                           \n",
                   "START OF PARAMETER:                                                        \n",
                   "none                          -file with data                              \n",
                   "1  2  0  3  5  0              -  columns for X,Y,Z,vr,wt,sec.var.          \n",
                   "-1.0e21 1.0e21                -  trimming limits                           \n",
                   "0                             -transform the data (0=no, 1=yes)            \n",
                   "none.trn                      -  file for output trans table               \n",
                   "0                             -  consider ref. dist (0=no, 1=yes)          \n",
                   "none.dat                      -  file with ref. dist distribution          \n",
                   "1  0                          -  columns for vr and wt                     \n",
                   "-4.0    4.0                   -  zmin,zmax(tail extrapolation)             \n",
                   "1      -4.0                   -  lower tail option, parameter              \n",
                   "1       4.0                   -  upper tail option, parameter              \n",
                   "0                             -debugging level: 0,1,2,3                    \n",
                   "nonw.dbg                      -file for debugging output                   \n",
                   str(output_file) + "           -file for simulation output                  \n",
                   str(nreal) + "                 -number of realizations to generate          \n",
                   str(nx) + " " + str(hmn) + " " + str(hsiz) + "                              \n",
                   str(ny) + " " + str(hmn) + " " + str(hsiz) + "                              \n",
                   "1 0.0 1.0                     - nz zmn zsiz                                \n",
                   str(seed) + "                  -random number seed                          \n",
                   "0     8                       -min and max original data for sim           \n",
                   "12                            -number of simulated nodes to use            \n",
                   "0                             -assign data to nodes (0=no, 1=yes)          \n",
                   "1     3                       -multiple grid search (0=no, 1=yes),num      \n",
                   "0                             -maximum data per octant (0=not used)        \n",
                   str(max_range) + " " + str(max_range) + " 1.0 -maximum search  (hmax,hmin,vert) \n",
                   str(azi1) + "   0.0   0.0       -angles for search ellipsoid                 \n",
                   str(hctab) + " " + str(hctab) + " 1 -size of covariance lookup table        \n",
                   "4 " + str(correl) + " 1.0     -ktype: 0=SK,1=OK,2=LVM,3=EXDR,4=COLC        \n",
                   "sec.dat                       -  file with LVM, EXDR, or COLC variable     \n",
                   "1                             -  column for secondary variable             \n",
                   str(nst) + " " + str(nug) + "  -nst, nugget effect                          \n",
                   str(it1) + " " + str(cc1) + " " + str(azi1) + " 0.0 0.0 -it,cc,ang1,ang2,ang3 \n",
                   " " + str(hmaj1) + " " + str(hmin1) + " 1.0 - a_hmax, a_hmin, a_vert        \n",
                   str(it2) + " " + str(cc2) + " " + str(azi2) + " 0.0 0.0 -it,cc,ang1,ang2,ang3 \n",
                   " " + str(hmaj2) + " " + str(hmin2) + " 1.0 - a_hmax, a_hmin, a_vert        \n"]
    open("../GsPy3DModel/sgsim.par", 'w').write(''.join(sgsim_lines))

    os.system('"sgsim.exe sgsim.par"')
    sim_array = GSLIB2ndarray(output_file, 0, nx, ny)
    return (sim_array[0])


# sample with provided X and Y and append to DataFrame
def sample(array, xmin, xmax, ymin, ymax, step, name, df, xcol, ycol):
    if array.ndim == 2:
        ny = (array.shape[0])
        nx = (array.shape[1])
    else:
        print('Array must be 2D')
    v = []
    nsamp = len(df)
    for isamp in range(0, nsamp):
        x = df.iloc[isamp][xcol]
        y = df.iloc[isamp][ycol]
        iy = min(ny - int((y - ymin) / step) - 1, ny - 1)
        ix = min(int((x - xmin) / step), nx - 1)
        v.append(array[iy, ix])
    df[name] = v
    return (df)


def gkern(kernlen=21, std=3):  # Stack overflow solution from Teddy Hartano
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


# extract regular spaced samples from a model
def regular_sample(array, xmin, xmax, ymin, ymax, step, mx, my, name):
    x = [];
    y = [];
    v = []
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax, ymin, -1 * step))
    iiy = 0
    for iy in range(0, ny):
        if iiy >= my:
            iix = 0
            for ix in range(0, nx):
                if iix >= mx:
                    x.append(xx[ix, iy]);
                    y.append(yy[ix, iy]);
                    v.append(array[ix, iy])
                    iix = 0;
                    iiy = 0
                iix = iix + 1
        iiy = iiy + 1
    df = pd.DataFrame(np.c_[x, y, v], columns=['X', 'Y', name])
    return (df)


def random_sample(array, xmin, xmax, ymin, ymax, step, nsamp, name):
    import random as rand
    x = [];
    y = [];
    v = []
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax - 1, ymin - 1, -1 * step))
    ny = xx.shape[0]
    nx = xx.shape[1]
    sample_index = rand.sample(range((nx) * (ny)), nsamp)
    for isamp in range(0, nsamp):
        iy = int(sample_index[isamp] / ny)
        ix = sample_index[isamp] - iy * nx
        x.append(xx[iy, ix])
        y.append(yy[iy, ix])
        v.append(array[iy, ix])
    df = pd.DataFrame(np.c_[x, y, v], columns=['X', 'Y', name])
    return (df)


# Take spatial data from a DataFrame and make a sparse ndarray (NaN where no data in cell)
def DataFrame2ndarray(df, xcol, ycol, vcol, xmin, xmax, ymin, ymax, step):
    xx, yy = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymax - 1, ymin - 1, -1 * step))
    ny = xx.shape[0]
    nx = xx.shape[1]
    array = np.full((ny, nx), np.nan)
    nsamp = len(df)
    sample_index = rand.sample(range((nx) * (ny)), nsamp)
    for isamp in range(0, nsamp):
        iy = min(ny - 1, ny - int((df.iloc[isamp][ycol] - ymin) / step) - 1)
        ix = min(nx - 1, int((df.iloc[isamp][xcol] - xmin) / step))
        array[iy, ix] = df.iloc[isamp][vcol]
    return (array)


# Calculate weighted average and standard deviation
def weighted_avg_and_std(values, weights):  # from Eric O Lebigot, stack overflow
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))
