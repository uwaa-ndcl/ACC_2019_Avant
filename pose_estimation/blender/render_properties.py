import math
import numpy as np
import transforms3d as t3d

class RenderProperties:

    def __init__(self):
        # name of .blend file
        self.model_name = None

        # list of names of images, if none then images will be named
        # 000000.png, 000001.png, ...
        self.image_names = None

        # directory to save output images, etc.
        self.save_dir = None # this will be filled later

        # object
        self.ob = None # this will be set inside of Blender
        self.n_renders = 1
        self.xyz = np.array([[0],[0],[0]]) # size (3, n_renders)
        self.quat = np.array([[1],[0],[0],[0]]) # size (4, n_renders)

        # list of images (of length n_renders) for the render to be
        # superimposed upon, None if it is not superimposed
        self.bkgd_image_list = None

        # world lighting, size (3, n_renders)
        self.world_RGB = None  

        # transparent background?
        self.alpha = True 

        # gramian
        self.compute_gramian = False
        self.eps = 1e-2

        # perturbations, if None, use initial perturbation
        # order of perturbations: -1, +1, -2, +2, ...
        self.pert_xyz = None # size (3, 12, n_renders)
        self.pert_quat = None # size (4, 12, n_renders)

        # camera
        # default based on properties I think a Canon Powershot A2500 has
        self.cam_ob = None # this will be set inside of Blender
        self.cam_xyz = [0, 0, 0]
        self.cam_quat = t3d.euler.euler2quat(math.pi/2, 0, 0, axes='sxyz')
        self.pix_width = 300
        #self.pix_width = 30
        self.pix_height = 300
        #self.pix_height = 30
        self.lens = 9
        self.sensor_width = 6.2
        self.sensor_height = 4.6
