import os.path
import math
import pickle
import shutil
import numpy as np
import transforms3d as t3d

import pose_estimation.directories as dirs
import pose_estimation.blender.render as br
import pose_estimation.tools.math as tm
import pose_estimation.gramian.functions as gf
from pose_estimation.blender.render_properties import RenderProperties

# model info
name = 'chair'
#name = 'car'

# save directory
save_dir = dirs.best_views_dir

# sample azimuthal and elevation angles, and calculate the Gramian for each
# sample
n_ang_azi = 20
n_ang_ele = int(n_ang_azi/2)
n_renders = n_ang_azi*n_ang_ele
ang_azi = np.linspace(0, 2*math.pi, n_ang_azi, endpoint=False)
ang_ele = np.linspace(0, math.pi/2, n_ang_ele, endpoint=True)

# radius of camera orbit
if name == 'chair':
    rad = 2.0
elif name == 'car':
    rad = 14

# position and oreintation of object and camera
xyz = np.array([[0], [0], [0]]) 
quat = np.array([[1], [0], [0], [0]]) 
cam_xyz_dft = rad*np.array([[0], [-1], [0]]) # camera xyz default
cam_R_dft = t3d.euler.euler2mat(math.pi/2, 0, 0, 'sxyz')

# loop over azimuthal and elevation angles and generate quaternions
gram = np.full((6,6,n_renders), np.nan) # to be filled
for i, ang_azi_i in enumerate(ang_azi):
    R_z_i = tm.R_z(ang_azi_i)

    for j, ang_ele_j in enumerate(ang_ele):
        ij = i*n_ang_ele + j
        R_x_j = tm.R_x(-ang_ele_j)
        R_ij = R_z_i @ R_x_j
        cam_xyz = R_ij @ cam_xyz_dft
        cam_quat = t3d.quaternions.mat2quat(R_ij @ cam_R_dft)

        # render
        to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
        render_props = RenderProperties()
        render_props.model_name = name
        render_props.image_names = ['%06d.png' % ij]
        render_props.n_renders = 1
        render_props.xyz = xyz
        render_props.quat = quat
        render_props.cam_xyz = cam_xyz
        render_props.cam_quat = cam_quat
        render_props.compute_gramian = True
        render_props.alpha = False
        with open(to_render_pkl, 'wb') as output:
            pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
        br.blender_render(save_dir)

        # load gramian
        gram_npz = os.path.join(save_dir, 'gramian.npz')
        gram_data = np.load(gram_npz)
        gram_ij = gram_data['gram']
        gram[:,:,ij] = np.squeeze(gram_ij)

# measure the gramian
grm = gf.gramian_measures(gram)
# get max and min, put each into a dictionary
min_max_dict = {'index of min det': np.argmin(grm['det']),
                'index of max det': np.argmax(grm['det']),
                'index of min trace': np.argmin(grm['trace']),
                'index of max trace': np.argmax(grm['trace']),
                'index of min mineval': np.argmin(grm['min_eval']),
                'index of max mineval': np.argmax(grm['min_eval']),
                'index of min condition num': np.argmin(grm['cond_num']),
                'index of max condition num': np.argmax(grm['cond_num'])} 

# copy renders to files with descriptive names
for key, value in min_max_dict.items():
    print(key, ': ', value)
    file_old = os.path.join(save_dir, '%06d.png' % value)
    # remove 'index of ' and make spaces hyphens
    name_new = key[9:].replace(' ', '_') + '.png' 
    file_new = os.path.join(save_dir, name_new)
    shutil.copyfile(file_old, file_new)
