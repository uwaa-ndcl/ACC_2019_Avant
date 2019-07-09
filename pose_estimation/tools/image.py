import os
import cv2
import glob
import imageio
import warnings
import subprocess
import numpy as np


def load_im_np(filename):
    '''
    load 1 image to a numpy array
    resulting array will have entries on the interval [0, 1] of type float32
    '''
        
    a = imageio.imread(filename) # will load integers on interval [0,255]
    b = np.asarray(a)
    c = np.float32(b)
    d = c/255.0 # put elements on interval [0,1]
            
    return d


def write_im_np(filename, im):
    '''
    take an array with entries on the interval [0, 1] and save it as an image
    '''
    warnings.simplefilter('always')
    if np.any(im < 0) or np.any(im > 1):
        warnings.warn('image has values outside of the interval [0,1], ' \
                       'saturating...')
        im = np.clip(im, 0, 1)

    im = np.uint8(im * 255.0)
    imageio.imwrite(filename, im)


def overlay(im_overlay, im_background, mode='0to1'):
    '''
    overlay an RGBA image onto an RGB background image

    INPUTS
    im_overlay: RGBA image with values on interval [0, 255]
    im_background: RGB image with values on interval [0, 255]
    mode: '0to1' (image elements are from 0 to 1),
          '0to255' (image elements are from 0 to 255)
    '''

    alph = im_overlay[:,:,3]

    if mode == '0to255':
        alph = alph/255.0 # now alpha values are in interval [0,1]

    im_overlay_scaled = np.stack((alph*im_overlay[:,:,0],
                                  alph*im_overlay[:,:,1],
                                  alph*im_overlay[:,:,2]), 2)
    im_background_scaled = np.stack(((1 - alph)*im_background[:,:,0],
                                     (1 - alph)*im_background[:,:,1],
                                     (1 - alph)*im_background[:,:,2]), 2)
    im_out = im_overlay_scaled + im_background_scaled

    return im_out
