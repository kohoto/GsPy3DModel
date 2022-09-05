import numpy as np
import math
import random as rand
import matplotlib.pyplot as plt

def read_input(fpath):
    header = True
    inp = []
    for line in open(fpath):
        if header:
            header = False
        else:
            splited_line = line.split()
            inp.append(splited_line)
            header = True

    # specify the directory to create new dir for stl files
    image_type = inp[0][0]
    file_type = inp[0][1]
    # change data type. Rest is all numbers
    dpi = float(inp[0][2])

    # 2nd row
    Lx = float(inp[1][0])
    Ly = float(inp[1][1])
    cell_size = float(inp[1][2])
    height = float(inp[1][3])  # [inch]
    mean = float(inp[1][4])
    stdev = float(inp[1][5])


    # 3rd row
    hmaj1 = float(inp[2][0])
    hmin1 = float(inp[2][1])

    # 4th row
    n_cut = [int(inp[3][1]), int(inp[3][0])]  # Divide into how many files in each direction? (y, x)!!!

    # 5th row - input for ridge
    ridge = int(inp[4][0])
    ridge_height = float(inp[4][1]) # inch
    ridge_margin = float(inp[4][2])  # inch

    # TODO: throw error if Lx, Ly, height are not the multiple of cell_size.

    return image_type, file_type, dpi, Lx, Ly, cell_size, height, mean, stdev, hmaj1, hmin1, n_cut, ridge, ridge_height, ridge_margin


def generateSTL(array, xmin, xmax, ymin, ymax, step, height, filename):
    filename = filename + '.stl'
    # array += height  # inch

    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    nx = len(x)
    ny = len(y)
    x = np.tile(x, (ny, 1))
    y = np.transpose(np.tile(y, (nx, 1)))

    # Change from inch to mm
    # x = 25.4 * x
    # y = 25.4 * y
    # array = 25.4 * array

    # This only works for the rectangular shape.
    coord1 = np.empty((3,1,1))
    coord1[:] = np.nan
    coord2 = coord1
    coord3 = coord1
    # top
    coord1, coord2, coord3 = create_top(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3)
    # front
    coord1, coord2, coord3 = create_front(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3)
    # back
    coord1, coord2, coord3 = create_back(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3)
    #left
    coord1, coord2, coord3 = create_left(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3)
    # right
    coord1, coord2, coord3 = create_right(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3)

    # transpose
    coord1 = np.transpose(coord1, (1, 0, 2))
    coord2 = np.transpose(coord2, (1, 0, 2))
    coord3 = np.transpose(coord3, (1, 0, 2))

    # normal vectors
    normal = (coord2[:, 1] - coord1[:, 1]) * (coord3[:, 2] - coord1[:, 2]) - (coord3[:, 1] - coord1[:, 1]) * (
            coord2[:, 2] - coord1[:, 2])
    normal = np.concatenate((normal, (coord2[:, 2] - coord1[:, 2]) * (coord3[:, 0] - coord1[:, 0])
                             - (coord3[:, 2] - coord1[:, 2]) * (coord2[:, 0] - coord1[:, 0])), axis=1)
    normal = np.concatenate((normal, (coord2[:, 0] - coord1[:, 0]) * (coord3[:, 1] - coord1[:, 1])
                             - (coord3[:, 0] - coord1[:, 0]) * (coord2[:, 1] - coord1[:, 1])), axis=1)

    #  Start printing out triangles
    with open(filename, 'a') as file:
        for cd1, cd2, cd3, nml in zip(coord1, coord2, coord3, normal):
            x, y, z = nml[0], nml[1], nml[2]
            file.write(f'facet normal {x:.5f} {y:.5f} {z:.5f}\n')
            file.write(f'outer loop\n')
            x, y, z = cd1[0][0], cd1[1][0], cd1[2][0]
            file.write(f'vertex {x:.5f} {y:.5f} {z:.5f}\n')
            x, y, z = cd2[0][0], cd2[1][0], cd2[2][0]
            file.write(f'vertex {x:.5f} {y:.5f} {z:.5f}\n')
            x, y, z = cd3[0][0], cd3[1][0], cd3[2][0]
            file.write(f'vertex {x:.5f} {y:.5f} {z:.5f}\n')
            file.write(f'endloop\n endfacet\n')


def generateSTL_ridge(array, xmin, xmax, ymin, ymax, step, ridge_loc_y, ridge_height, ridge_margin, height, filename):
    filename = filename + '.stl'

    ## create ridge here
    array, array_top, array_btm = create_ridge_half(array, ymin, ymax, step, ridge_loc_y, ridge_height, ridge_margin)

    # This only works for the ridged shape.
    # top
    # -- top
    coord1 = np.empty((3,1,1))
    coord1[:] = np.nan
    coord2 = coord1
    coord3 = coord1
    coord1, coord2, coord3 = create_top(array_top, xmin, xmax, ridge_loc_y, ymax, step, height, coord1, coord2, coord3)
    # -- btm
    coord1, coord2, coord3 = create_top(array_btm, xmin, xmax, ymin, ridge_loc_y + step, step, height, coord1, coord2, coord3)
    # front
    # -- btm only
    coord1, coord2, coord3 = create_front(array_btm, xmin, xmax, ymin, ridge_loc_y + step, step, height, coord1, coord2, coord3)
    # back
    # -- top only
    coord1, coord2, coord3 = create_back(array_top, xmin, xmax, ridge_loc_y, ymax, step, height, coord1, coord2, coord3)
    # left
    coord1, coord2, coord3 = create_left(array_top, xmin, xmax, ridge_loc_y, ymax, step, height, coord1, coord2, coord3)
    coord1, coord2, coord3 = create_left(array_btm, xmin, xmax, ymin, ridge_loc_y + step, step, height, coord1, coord2, coord3)
    # right
    coord1, coord2, coord3 = create_right(array_top, xmin, xmax, ridge_loc_y, ymax, step, height, coord1, coord2, coord3)
    coord1, coord2, coord3 = create_right(array_btm, xmin, xmax, ymin, ridge_loc_y + step, step, height, coord1, coord2, coord3)
    # bottom
    coord1, coord2, coord3 = create_bottom(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3)
    # ridge
    top = array_top[[-1], :]
    btm = array_btm[[0], :]
    coord1, coord2, coord3 = create_ridge_surface(top, btm, xmin, xmax, ridge_loc_y, step, height, coord1, coord2, coord3)
    # transpose
    coord1 = np.transpose(coord1, (1, 0, 2))
    coord2 = np.transpose(coord2, (1, 0, 2))
    coord3 = np.transpose(coord3, (1, 0, 2))

    # delete the first row

    # normal vectors
    normal = (coord2[:, 1] - coord1[:, 1]) * (coord3[:, 2] - coord1[:, 2]) - (coord3[:, 1] - coord1[:, 1]) * (
            coord2[:, 2] - coord1[:, 2])
    normal = np.concatenate((normal, (coord2[:, 2] - coord1[:, 2]) * (coord3[:, 0] - coord1[:, 0])
                             - (coord3[:, 2] - coord1[:, 2]) * (coord2[:, 0] - coord1[:, 0])), axis=1)
    normal = np.concatenate((normal, (coord2[:, 0] - coord1[:, 0]) * (coord3[:, 1] - coord1[:, 1])
                             - (coord3[:, 0] - coord1[:, 0]) * (coord2[:, 1] - coord1[:, 1])), axis=1)

    #  Start printing out triangles
    with open(filename, 'a') as file:
        for cd1, cd2, cd3, nml in zip(coord1, coord2, coord3, normal):
            x, y, z = nml[0], nml[1], nml[2]
            file.write(f'facet normal {x:.5f} {y:.5f} {z:.5f}\n')
            file.write(f'outer loop\n')
            x, y, z = cd1[0][0], cd1[1][0], cd1[2][0]
            file.write(f'vertex {x:.5f} {y:.5f} {z:.5f}\n')
            x, y, z = cd2[0][0], cd2[1][0], cd2[2][0]
            file.write(f'vertex {x:.5f} {y:.5f} {z:.5f}\n')
            x, y, z = cd3[0][0], cd3[1][0], cd3[2][0]
            file.write(f'vertex {x:.5f} {y:.5f} {z:.5f}\n')
            file.write(f'endloop\n endfacet\n')


def create_top(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3):
    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    nx = len(x)
    ny = len(y)
    x = np.tile(x, (ny, 1))
    y = np.transpose(np.tile(y, (nx, 1)))

    # Change from inch to mm
    x = 25.4 * x
    y = 25.4 * y
    array = 25.4 * array

    # top surface - lower triangle
    cd1 = [x[:-1, :-1].reshape((nx - 1) * (ny - 1), 1), y[:-1, :-1].reshape((nx - 1) * (ny - 1), 1),
              array[:-1, :-1].reshape((nx - 1) * (ny - 1), 1)]
    cd2 = [x[:-1, 1:].reshape((nx - 1) * (ny - 1), 1), y[:-1, 1:].reshape((nx - 1) * (ny - 1), 1),
              array[:-1, 1:].reshape((nx - 1) * (ny - 1), 1)]
    cd3 = [x[1:, :-1].reshape((nx - 1) * (ny - 1), 1), y[1:, :-1].reshape((nx - 1) * (ny - 1), 1),
              array[1:, :-1].reshape((nx - 1) * (ny - 1), 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)

    print(np.isnan(coord1[:,0,:]).all and np.isnan(coord2[:,0,:]).all and np.isnan(coord3[:,0,:]).all)
    if np.isnan(coord1[:,0,:]).all and np.isnan(coord2[:,0,:]).all and np.isnan(coord3[:,0,:]).all:  # if all empty
        coord1 = np.delete(coord1, 0, axis=1)
        coord2 = np.delete(coord2, 0, axis=1)
        coord3 = np.delete(coord3, 0, axis=1)

    # top surface - upper triangle
    cd1 = cd2
    cd2 = [x[1:, 1:].reshape((nx - 1) * (ny - 1), 1), y[1:, 1:].reshape((nx - 1) * (ny - 1), 1),
           array[1:, 1:].reshape((nx - 1) * (ny - 1), 1)]
    cd3 = cd3
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    return coord1, coord2, coord3


def create_front(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3):
    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    nx = len(x)
    ny = len(y)
    x = np.tile(x, (ny, 1))
    y = np.transpose(np.tile(y, (nx, 1)))

    # Change from inch to mm
    x = 25.4 * x
    y = 25.4 * y
    array = 25.4 * array

    # front side - lower triangle
    cd1 = [x[[0], :-1].reshape((nx - 1), 1), y[[0], :-1].reshape((nx - 1), 1), np.zeros(((nx - 1), 1))]
    cd2 = [x[[0], 1:].reshape((nx - 1), 1), y[[0], 1:].reshape((nx - 1), 1), np.zeros(((nx - 1), 1))]
    cd3 = [x[[0], :-1].reshape((nx - 1), 1), y[[0], :-1].reshape((nx - 1), 1), array[[0], :-1].reshape((nx - 1), 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    # front side - upper triangle
    cd1 = cd2
    cd2 = [x[[0], 1:].reshape((nx - 1), 1), y[[0], 1:].reshape((nx - 1), 1), array[[0], 1:].reshape((nx - 1), 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    return coord1, coord2, coord3

def create_back(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3):
    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    nx = len(x)
    ny = len(y)
    x = np.tile(x, (ny, 1))
    y = np.transpose(np.tile(y, (nx, 1)))

    # Change from inch to mm
    x = 25.4 * x
    y = 25.4 * y
    array = 25.4 * array

    # back side - lower triangle
    cd1 = [x[[-1], :-1].reshape((nx - 1), 1), y[[-1], :-1].reshape((nx - 1), 1), np.zeros(((nx - 1), 1))]
    cd2 = [x[[-1], :-1].reshape((nx - 1), 1), y[[-1], :-1].reshape((nx - 1), 1), array[[-1], :-1].reshape((nx - 1), 1)]
    cd3 = [x[[-1], 1:].reshape((nx - 1), 1), y[[-1], 1:].reshape((nx - 1), 1), np.zeros(((nx - 1), 1))]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    # back side - upper triangle
    cd1 = cd2
    cd2 = [x[[-1], 1:].reshape((nx - 1), 1), y[[-1], 1:].reshape((nx - 1), 1), array[[-1], 1:].reshape((nx - 1), 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    return coord1, coord2, coord3

def create_left(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3):
    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    nx = len(x)
    ny = len(y)
    x = np.tile(x, (ny, 1))
    y = np.transpose(np.tile(y, (nx, 1)))

    # Change from inch to mm
    x = 25.4 * x
    y = 25.4 * y
    array = 25.4 * array

    # left side - lower triangle
    cd1 = [x[:-1, [0]].reshape((ny - 1), 1), y[:-1, [0]].reshape((ny - 1), 1), np.zeros(((ny - 1), 1))]
    cd2 = [x[:-1, [0]].reshape((ny - 1), 1), y[:-1, [0]].reshape((ny - 1), 1), array[:-1, [0]].reshape((ny - 1), 1)]
    cd3 = [x[1:, [0]].reshape((ny - 1), 1), y[1:, [0]].reshape((ny - 1), 1), np.zeros(((ny - 1), 1))]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    # left side - upper triangle
    cd1 = cd2
    cd2 = [x[1:, [0]].reshape((ny - 1), 1), y[1:, [0]].reshape((ny - 1), 1), array[1:, [0]].reshape((ny - 1), 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    return coord1, coord2, coord3

def create_right(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3):
    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    nx = len(x)
    ny = len(y)
    x = np.tile(x, (ny, 1))
    y = np.transpose(np.tile(y, (nx, 1)))

    # Change from inch to mm
    x = 25.4 * x
    y = 25.4 * y
    array = 25.4 * array

    # right side - lower triangle
    cd1 = [x[:-1, [-1]].reshape((ny - 1), 1), y[:-1, [-1]].reshape((ny - 1), 1), np.zeros(((ny - 1), 1))]
    cd2 = [x[1:, [-1]].reshape((ny - 1), 1), y[1:, [-1]].reshape((ny - 1), 1), np.zeros(((ny - 1), 1))]
    cd3 = [x[:-1, [-1]].reshape((ny - 1), 1), y[:-1, [-1]].reshape((ny - 1), 1),
           array[:-1, [-1]].reshape((ny - 1), 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    # right side - upper triangle
    cd1 = cd2
    cd2 = [x[1:, [-1]].reshape((ny - 1), 1), y[1:, [-1]].reshape((ny - 1), 1),
           array[1:, [-1]].reshape((ny - 1), 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    return coord1, coord2, coord3

def create_bottom(array, xmin, xmax, ymin, ymax, step, height, coord1, coord2, coord3):
    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    nx = len(x)
    ny = len(y)
    x = np.tile(x, (ny, 1))
    y = np.transpose(np.tile(y, (nx, 1)))

    # Change from inch to mm
    x = 25.4 * x
    y = 25.4 * y
    array = 25.4 * array

    # bottom surface - lower triangle
    # cd1 = [x[:-1, :-1].reshape((nx - 1) * (ny - 1), 1), y[:-1, :-1].reshape((nx - 1) * (ny - 1), 1),
    #        np.zeros(((nx - 1) * (ny - 1), 1))]
    # cd2 = [x[1:, :-1].reshape((nx - 1) * (ny - 1), 1), y[1:, :-1].reshape((nx - 1) * (ny - 1), 1),
    #        np.zeros(((nx - 1) * (ny - 1), 1))]
    # cd3 = [x[:-1, 1:].reshape((nx - 1) * (ny - 1), 1), y[:-1, 1:].reshape((nx - 1) * (ny - 1), 1),
    #        np.zeros(((nx - 1) * (ny - 1), 1))]
    # find the midpoint
    mid_x = math.floor(0.5 * x.shape[0])
    cd1 = [x[:-1:mid_x, 0].reshape(2, 1), y[:-1:mid_x, 0].reshape(2, 1), np.zeros((2, 1))]
    cd2 = [x[mid_x::x.shape[0] - 1 - mid_x, 0].reshape(2, 1), y[mid_x::x.shape[0] - 1 - mid_x, 0].reshape(2, 1),
           np.zeros((2, 1))]
    cd3 = [x[:-1:mid_x, -1].reshape(2, 1), y[:-1:mid_x, -1].reshape(2, 1), np.zeros((2, 1))]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    # bottom surface - upper triangle
    cd1 = cd2
    cd2 = [x[mid_x::mid_x - 1, -1].reshape(2, 1), y[mid_x::mid_x - 1, -1].reshape(2, 1), np.zeros((2, 1))]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    return coord1, coord2, coord3


def create_ridge_surface(top, btm, xmin, xmax, ridge_loc_y, step, height, coord1, coord2, coord3):
    x = np.array([i for i in np.arange(xmin, xmax, step)])
    nx = len(x)

    # Change from inch to mm
    x = 25.4 * x
    y = 25.4 * ridge_loc_y * np.ones(nx)
    top = 25.4 * np.transpose(top)
    btm = 25.4 * np.squeeze(btm)

    # lower triangle
    cd1 = [x[:-1].reshape(nx-1, 1), y[:-1].reshape(nx-1, 1), btm[:-1].reshape(nx-1, 1)]
    cd2 = [x[:-1].reshape(nx-1, 1), y[:-1].reshape(nx-1, 1), top[:-1].reshape(nx-1, 1)]
    cd3 = [x[1:].reshape(nx-1, 1), y[1:].reshape(nx-1, 1), btm[1:].reshape(nx-1, 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    # left side - upper triangle
    cd1 = cd2
    cd2 = [x[1:].reshape(nx-1, 1), y[1:].reshape(nx-1, 1), top[1:].reshape(nx-1, 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)
    coord2 = np.concatenate((coord2, cd2), axis=1)
    coord3 = np.concatenate((coord3, cd3), axis=1)
    return coord1, coord2, coord3


def generatePLY(array, xmin, xmax, ymin, ymax, step, height, filename):
    filename = filename + '.ply'
    array = np.transpose(array)  # inch
    # array = np.transpose(array) + height  # inch
    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    z = [i for i in np.arange(0, height, step)]
    nx = len(x)
    ny = len(y)
    nz = len(z) + 1  # +1 is for the point at the top surface.

    # top surface
    x_top = np.tile(x, (ny, 1))
    y_top = np.transpose(np.tile(y, (nx, 1)))
    coord1 = [x_top.reshape(nx * ny, 1), y_top.reshape(nx * ny, 1), array.reshape(nx * ny, 1)]

    # bottom surface
    cd1 = [x_top.reshape(nx * ny, 1), y_top.reshape(nx * ny, 1), np.zeros((nx * ny, 1))]
    coord1 = np.concatenate((coord1, cd1), axis=1)

    # front side
    x_front = np.tile(x, (nz, 1)).transpose()
    z_front = np.concatenate((np.tile(z, (nx, 1)), array[:, :1]), axis=1)
    cd1 = [x_front.reshape(nx * nz, 1), ymin * np.ones((nx * nz, 1)), z_front.reshape(nx * nz, 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)

    # back side
    z_front = np.concatenate((np.tile(z, (nx, 1)), array[:, -1:]), axis=1)
    cd1 = [x_front.reshape(nx * nz, 1), ymax * np.ones((nx * nz, 1)), z_front.reshape(nx * nz, 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)

    # left side
    y_left = np.tile(y, (nz, 1)).transpose()
    z_left = np.concatenate((np.tile(z, (ny, 1)), array[:1, :].transpose()), axis=1)
    cd1 = [xmin * np.ones((ny * nz, 1)), y_left.reshape(ny * nz, 1), z_left.reshape(ny * nz, 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)

    # right side
    z_left = np.concatenate((np.tile(z, (ny, 1)), array[-1:, :].transpose()), axis=1)
    cd1 = [xmax * np.ones((ny * nz, 1)), y_left.reshape(ny * nz, 1), z_left.reshape(ny * nz, 1)]
    coord1 = np.concatenate((coord1, cd1), axis=1)

    # fill the margin between height to top surface
    # front, back, left, right
    x_coords = np.concatenate([x, x, xmin * np.ones(ny), xmax * np.ones(ny)])
    y_coords = np.concatenate([ymin * np.ones(nx), ymax * np.ones(nx), y, y])
    z_coords = np.concatenate([array[:, :1], array[:, -1:], array[:1, :].transpose(), array[-1:, :].transpose()])

    coord1 = np.squeeze(coord1)
    for x_coord, y_coord, z_coord in zip(x_coords, y_coords, z_coords):
        zs = np.arange(height + step, z_coord, step)  # Note that np.arange doesn't include the last point
        cd1 = [x_coord * np.ones(zs.size), y_coord * np.ones(zs.size), zs]
        coord1 = np.concatenate((coord1, cd1), axis=1)

    #  Start printing out xyz coordinate
    with open(filename, 'w') as file:
        # Write header
        file.write(f'ply\nformat ascii 1.0\ncomment Kinect v1 generated\n')
        file.write(f'element vertex {coord1.shape[1]}\n')
        file.write(f'property double x\nproperty double y\nproperty double z\nend_header\n')
        for coord in coord1.transpose():
            file.write(f'{coord[0]:.5f} {coord[1]:.5f} {coord[2]:.5f}\n')


def create_ridge(array, array_mirror, ymin, ymax, step, ridge_y, ridge_height, ridge_margin):
    '''
    ridge_y: location of ridge
    ridge_height: ridge height from the original surface
    '''
    y = [i for i in np.arange(ymin, ymax, step)]
    ridge_ny = math.ceil(ridge_y / step)
    ridge_ny_mirror = math.ceil((ridge_y + ridge_margin) / step)
    array_top = array[ridge_ny:, :] + ridge_height
    array_mirror_top = array_mirror[ridge_ny_mirror:, :] - ridge_height

    array_btm = array[:ridge_ny+1, :]
    array_mirror_btm = array_mirror[:ridge_ny_mirror+1, :]
    return array_top, array_mirror_top, array_btm, array_mirror_btm


def create_ridge_half(array, ymin, ymax, step, ridge_y, ridge_height, ridge_margin):
    '''
    ridge_y: location of ridge
    ridge_height: ridge height from the original surface
    '''
    y = [i for i in np.arange(ymin, ymax, step)]
    ridge_ny = math.ceil(ridge_y / step)
    ridge_ny_mirror = math.ceil((ridge_y + ridge_margin) / step)
    array_top = array[ridge_ny:, :] + ridge_height
    array_btm = array[:ridge_ny+1, :]
    array[ridge_ny:, :] = array_top

    return array, array_top, array_btm


def open_STL(filename):
    filename = filename + '.stl'
    #  Start printing out triangles
    with open(filename, 'w') as file:
        file.write('solid object\n')


def close_STL(filename):
    filename = filename + '.stl'
    with open(filename, 'a') as file:
        file.write('endsolid object\n')

def create_profilometer_file(array, xmin, xmax, ymin, ymax, step, height, filename):
    # we need points at xmax and ymax for profiometer file! # of points in each direction is +1 than normal gridding
    x = [i for i in np.arange(xmin, xmax, step)]
    y = [i for i in np.arange(ymin, ymax, step)]
    nx = len(x)
    ny = len(y)
    x = np.tile(x, (ny, 1))
    y = np.transpose(np.tile(y, (nx, 1)))

    # write in file
    with open(filename, 'w') as file:
        # Write header
        file.write('Name:' + filename + '\n Experiment number: 1\n Sample Lenght: ' + str(xmax - step) + '\nSample Width: ' + str(ymax - step) + '\n Measurement Interval: ' + str(step) + '\n\n\n\n')
        file.write('X Position	Y Position	Z Displacement\n')
        for x, y, z in zip(x.reshape(-1), y.reshape(-1), array.reshape(-1)):
            file.write(str(x) + '   ' + str(y) + '  ' + str(z) + '  \n')
