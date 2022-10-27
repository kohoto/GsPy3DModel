import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import GsPy3DModel.geostatspy as gsp
import GsPy3DModel.model_3D as m3d


def main():
    import shutil

    input_file_path = '../GsPy3DModel/input/inp'
    save_parent_dir = 'output/'

    [image_type, file_type, dpi, Lx, Ly, cell_size, height, mean, stdev, hmaj1, hmin1, n_cut, ridge, ridge_height, ridge_margin] = m3d.read_input(input_file_path)

    """ calc some parameters from inputs """
    # number of grids
    nx = int(Lx / cell_size) + 1
    ny = int(Ly / cell_size) + 1
    # cell size
    xmin = 0.0
    ymin = 0.0  # grid origin
    xmax = Lx + cell_size
    ymax = Ly + cell_size  # calculate the extent of model
    seed = rand.randint(7000, 8000)  # random number seed  for stochastic simulation
    cmap = plt.cm.plasma  # color min and max and using the plasma color map
    # cmin and cmax: min and max of colorbar
    #TODO: check how cmap is used. make the color closer to profilometer data.


    # name of the dir
    dir_name = 'dim(' + str(Lx) + ',' + str(Ly) + ')_dimcorr(' + str(hmaj1) + ',' + str(hmin1) + ')_mean=' + str(mean) + '_stdev=' + str(stdev)

    # create folder if not exist
    path = save_parent_dir + dir_name
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)

    # copy input file to the output dir
    shutil.copyfile(input_file_path, path + '/inp')

    # Make a truth model / unconditional simulation
    # mean = 0.18; stdev = 0.05; cmin = 0.0; cmax = 0.3
    var = gsp.make_variogram(nug=0.0, nst=1, it1=1, cc1=1.0, azi1=90.0, hmaj1=hmaj1, hmin1=hmin1)
    width = gsp.GSLIB_sgsim_2d_uncond(1, nx, ny, cell_size, seed + 3, var, "input/width_distribution")
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
    # transpose to match the surface
    width_mirror = np.flipud(width_mirror)
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
        m3d.open_STL(filename)
        m3d.open_STL(filename + '_mirror')

        if ridge:
            # Create ridge in the middle
            # ridge location is going to be 3" heigher than the middle
            ridge_loc_y = 0.5 * ymax + 1.0 # has to be 1 -> 3 in in the real sample
            # top of ridge
            m3d.generateSTL_ridge(width, xmin, xmax, ymin, ymax, cell_size, ridge_loc_y, ridge_height, ridge_margin, height, filename)
            # top of ridge mirrored
            m3d.generateSTL_ridge(width_mirror, xmin, xmax, ymin, ymax, cell_size, ridge_loc_y, -ridge_height, ridge_margin, height, filename + '_mirror')

        else:

            xmin = 0
            xmax = width.shape[1] * cell_size
            ymin = 0
            ymax = width.shape[0] * cell_size
            if file_type == 'stl':
                # Generate STL file for 3D modeling
                m3d.generateSTL(width, xmin, xmax, ymin, ymax, cell_size, height, filename)
                # Generate STL file for matching surface
                m3d.generateSTL(width_mirror, xmin, xmax, ymin, ymax, cell_size, height, filename + '_mirror')
            elif file_type == 'ply':
                # Generate XYZ data file (.ply file)
                m3d.generatePLY(width, xmin, xmax, ymin, ymax, cell_size, height, 'surface')
                # Generate PLY for matching surface
                m3d.generatePLY(width_mirror, xmin, xmax, ymin, ymax, cell_size, height, 'surface_mirror')
            else:
                print("Error: file_type is not defined. [main]")

        m3d.close_STL(filename)
        m3d.close_STL(filename + '_mirror')

        # generate profilometer file
        m3d.create_profilometer_file(width, xmin, xmax, ymin, ymax, cell_size, height, filename)

        apature = 0.078  # [inch]
        filename = save_parent_dir + dir_name + '/apature'
        m3d.open_STL(filename)
        m3d.create_frac_apature(width, xmin, xmax, ymin, ymax, cell_size, apature, filename)
        m3d.close_STL(filename)


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

