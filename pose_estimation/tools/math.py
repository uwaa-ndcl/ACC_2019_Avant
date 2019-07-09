import math
import numpy as np

def R_x(theta):
    '''
    rotate about x-axis
    '''
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])


def R_y(theta):
    '''
    rotate about y-axis
    '''
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def R_z(theta):
    '''
    rotate about z-axis
    '''
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def normalize_array(arr):
    '''
    normalize a 1D array of values so the largest value is 1
    '''

    arr_nrm = (1/np.amax(arr))*arr

    return arr_nrm


def print_matrix_as_latex(mat, n_digs=3):
    '''
    print a matrix in latex form
    inputs:
        mat: matrix, n_digs: number of digits in largest value
    '''
    n_rows, n_cols = mat.shape
    mat_max = np.amax(mat) # max value in matrix
    max_pow = int(np.floor(np.log10(mat_max))) # max value power of 10
    # power of 10 to divide matrix by so largest element is in range [100,999]
    scl_pow = max_pow - (n_digs - 1)
    scl = 10**scl_pow # value to divide matrix by
    mat_scl = mat/scl

    # print in latex form
    print('10^{' + str(scl_pow) + '} \\times')
    print(r'\begin{bmatrix}')
    for i in range(n_cols):
        for j in range(n_rows):
            if j < n_rows - 1:
                print('%.0f' %  mat_scl[i, j], '& ', end='')
            else: # last row
                if i < n_rows - 1:
                    print('%.0f' %  mat_scl[i, j], '\\\ ')
                else: # last column and last row
                    print('%.0f' %  mat_scl[i, j])
    print(r'\end{bmatrix}')
