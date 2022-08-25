import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import GsPy3DPrint.geostatspy as gsp
import GsPy3DPrint.model_3D as p3d


# specify the directory to create new dir for stl files
save_parent_dir = 'output/'
image_type = 'tif'
file_type = 'stl'
dpi = 600


def main():
    # read input file here!!

    # input for ridge
    ridge = 0
    ridge_height = 0.1  # inch
    ridge_margin = 0.0  # inch
    n_cut = [1, 1]  # Divide into how many files in each direction? (y, x)!!!
    # Modeling parameters, the model grid, the random seed, the target distribution, and color bar
    # Lx = 48;
    # Ly = 23.5;
    Lx = 7;
    Ly = 1.7;
    cell_size = 0.01
    # cell_size = 0.02   # Using for coarse main
    # cell_size = 0.01 # Using this for side frac, too precise for main
    # corr_length_x = 0.04; corr_length_z = 0.5;
    # corr_length_x = 0.5;    corr_length_z = 0.5;
    height = 2.0  # [inch]
    nx = int(Lx / cell_size) + 1
    ny = int(Ly / cell_size) + 1
    # hmaj1 = 11.75
    # hmin1 = 1.175
    hmaj1 = 1.0
    hmin1 = 1.0


    # TODO: throw error if Lx, Ly, height are not the multiple of cell_size.

    # grid number of cells and cell size
    xmin = 0.0
    ymin = 0.0  # grid origin
    xmax = Lx + cell_size
    ymax = Ly + cell_size  # calculate the extent of model
    seed = rand.randint(7000, 8000)  # random number seed  for stochastic simulation
    cmap = plt.cm.plasma  # color min and max and using the plasma color map
    # Width statistical parameters
    # cmin and cmax: min and max of colorbar
    mean = 1.0
    stdev = 0.025


    # name of the dir
    dir_name = 'dim(' + str(Lx) + ',' + str(Ly) + ')_dimcorr(' + str(hmaj1) + ',' + str(hmin1) + ')_mean=' + str(mean) + '_stdev=' + str(stdev)

    # create folder if not exist
    path = save_parent_dir + dir_name
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)

    # Make a truth model / unconditional simulation
    # mean = 0.18; stdev = 0.05; cmin = 0.0; cmax = 0.3
    var = gsp.make_variogram(nug=0.0, nst=1, it1=1, cc1=1.0, azi1=90.0, hmaj1=hmaj1, hmin1=hmin1)
    width = gsp.GSLIB_sgsim_2d_uncond(1, nx, ny, cell_size, seed + 3, var, "input/Porosity")
    width = gsp.affine(width, mean, stdev)

# copy from here

    # Add height for 3d sample
    width += height

    # Show distribution
    cmin = width.min()
    cmax = width.max()
    gsp.pixelplt_st(width, xmin, xmax, ymin, ymax, cell_size, cmin, cmax, "Artificial Surface Dist.", "X [in]",
                "Y [in]", "Width [in]", cmap)

    # Create mirrored surface

    width_mirror = width.max() + height - width
    # Get the difference of the minimum value of two surfaces
    dmin = width_mirror.min() - width.min()
    width_mirror -= dmin
    #width_mirror *= -1 # This is to make mirrored rough surface matches to the original surface

    # Show histogram of width
    #plt.hist(width.reshape(-1))
    #plt.show()
    cut_x = math.floor(width.shape[1] / n_cut[1])
    cut_y = math.floor(width.shape[0] / n_cut[0])

    widths = []
    widths_mirror = []
    start_x = 0; end_x = cut_x
    for ix in range(n_cut[1]):
        start_y = 0
        end_y = cut_y
        for iy in range(n_cut[0]):
            widths.append(width[start_y:end_y, start_x:end_x])
            widths_mirror.append(width_mirror[start_y:end_y, start_x:end_x])
            start_y = end_y - 1
            if iy == n_cut[0]-2:
                end_y = width.shape[0]
            else:
                end_y = start_y + cut_y

        start_x = end_x
        if ix == n_cut[1]-2:
            end_x = width.shape[1]
        else:
            end_x = start_x + cut_x - 1
    # divide into pieces
    # determine the position to cut

    # iterate for each small pieces
    for width, width_mirror, file_number in zip(widths, widths_mirror, range(n_cut[0]*n_cut[1])):
        filename = save_parent_dir + dir_name + '/surf_' + str(file_number)
        p3d.open_STL(filename)
        p3d.open_STL(filename + '_mirror')

        if ridge:
            # Create ridge in the middle
            # ridge location is going to be 3" heigher than the middle
            ridge_loc_y = 0.5 * ymax + 1.0 # has to be 1 -> 3 in in the real sample
            # top of ridge
            p3d.generateSTL_ridge(width, xmin, xmax, ymin, ymax, cell_size, ridge_loc_y, ridge_height, ridge_margin, height, filename)
            # top of ridge mirrored
            p3d.generateSTL_ridge(width_mirror, xmin, xmax, ymin, ymax, cell_size, ridge_loc_y, -ridge_height, ridge_margin, height, filename + '_mirror')

        else:

            xmin = 0
            xmax = width.shape[1] * cell_size
            ymin = 0
            ymax = width.shape[0] * cell_size
            if file_type == 'stl':
                # Generate STL file for 3D modeling
                p3d.generateSTL(width, xmin, xmax, ymin, ymax, cell_size, height, filename)
                # Generate STL file for matching surface
                p3d.generateSTL(width_mirror, xmin, xmax, ymin, ymax, cell_size, height, filename + '_mirror')
            elif file_type == 'ply':
                # Generate XYZ data file (.ply file)
                p3d.generatePLY(width, xmin, xmax, ymin, ymax, cell_size, height, 'surface')
                # Generate PLY for matching surface
                p3d.generatePLY(width_mirror, xmin, xmax, ymin, ymax, cell_size, height, 'surface_mirror')
            else:
                print("Error: file_type is not defined. [main]")

        p3d.close_STL(filename)
        p3d.close_STL(filename + '_mirror')

        # generate profilometer file
        p3d.create_profilometer_file(width, xmin, xmax, ymin, ymax, cell_size, height, filename)


if __name__ == "__main__":
    main()


# settings
# both surfaces?
# ridge?
# split in several plates?
# stl or other file format?
# investigate profilometer data


## Functions used directly in my code
# make_variogram
# GSLIB_sgsim_2d_uncond
# affine
# pixelplt_st

## Functions listed in Testing
# - Some of them have differnet names, but all of them are in library and can be called.
# gsp.ndarray2GSLIB()
# gsp.GSLIB2ndarray()
# gsp.Dataframe2GSLIB()
# gsp.GSLIB2Dataframe()
# gsp.hist()
# gsp.hist_st()
# gsp.GSLIB.locmap()
# gsp.GSLIB.locmap_st()
# gsp.GSLIB.pixelplt
# gsp.pixelplt_st
# gsp.pixelplt_log_st
# gsp.locpix
# gsp.locpix_st
# gsp.locpix_log_st
# gsp.affine()
# gsp.nscore()
# gsp.make_variogram()
# gsp.gamv_2d -> This may be called just gamv now
# gsp.varmapv_2d  -> This may be called just varmapv now
# gsp.varmap_2d -> This may be called just varmap now
# gsp.vmodel_2d -> vmodel
# gsp.declus()
# gsp.GSLIB_sgsim_2d_uncond
# gsp.GSLIB_kb2d_2d
# gsp.GSLIB_sgsim_2d
# gsp.GSLIB_cosgsim_2d_uncond
# gsp.sample()
# gsp.gkern()
# gsp.regular_sample()
# gsp.random_sample()
# gsp.DataFrame2ndarray()

## My functions
# generateSTL
# generatePLY
# create_ridge
