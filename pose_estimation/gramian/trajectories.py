import os
import math
import pickle
import numpy as np
import transforms3d as t3d

import pose_estimation.directories as dirs
import pose_estimation.tools.math as tm
import pose_estimation.blender.render as br
import pose_estimation.gramian.functions as gf
from pose_estimation.blender.render_properties import RenderProperties

# object and camera
#model_name = 'cone'
model_name = 'car2'

# where to put object
xyz = np.array([0, 0, 0])
xyz_col = np.expand_dims(xyz, 1)

# center and default rotation of object
xyz_cent = np.array([0, 0, 0])
quat = np.array([1, 0, 0, 0])
xyz_cent_col = np.expand_dims(xyz_cent, 1)
quat_col = np.expand_dims(quat, 1)

# angles
if model_name == 'cone':
    rad = 5 # radius of semicircle
elif model_name == 'car2':
    rad = 8

n_pts = 10 # points along the trajectory for taking images
n_ang_x = 10
n_ang_z = 10
n_ang = n_ang_x * n_ang_z
ang_x_vals = np.linspace(-math.pi/2, math.pi/2, n_ang_x)
ang_z_vals = np.linspace(0, math.pi, n_ang_z)
ang_x, ang_z = np.meshgrid(ang_x_vals, ang_z_vals)
ang_x = np.reshape(ang_x, n_ang)
ang_z = np.reshape(ang_z, n_ang)
ang_xz = np.stack((ang_x, ang_z), 1)

# camera properties
lens = 32
sensor_width = 36
sensor_height = 36

# keys, colors, and plot names
kys = ['ang_xz_det_min', 'ang_xz_det_max',
       'ang_xz_trace_min', 'ang_xz_trace_max',
       'ang_xz_min_eval_min', 'ang_xz_min_eval_max',
       'ang_xz_cond_num_min', 'ang_xz_cond_num_max']
#            red      blue     yellow     white
clrs_4 = [[.8,0,0], [0,0,.8], [.8,.8,0], [1,1,1]]
clrs = [val for pair in zip(clrs_4, clrs_4) for val in pair]

tex_names = [
    r'det($\widehat{\mathbf{W}}$)',
    r'det($\widehat{\mathbf{W}}$)',
    r'tr($\widehat{\mathbf{W}}$)',
    r'tr($\widehat{\mathbf{W}}$)',
    r'$\lambda_{\text{min}}(\widehat{\mathbf{W}}$)',
    r'$\lambda_{\text{min}}(\widehat{\mathbf{W}}$)',
    r'$\frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}' \
        '(\widehat{\mathbf{W}}$)',
    r'$\frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}' \
        '(\widehat{\mathbf{W}}$)']


def semicircle(rad, n_coord, ang_x, ang_z, xyz_offset):
    '''
    create a semi-circular curve
    '''

    # coordinates of the points which define the curve
    theta = np.linspace(0, math.pi, n_coord)
    coord = np.full((3, n_coord), np.nan)
 
    # camera
    # rotation matrix for camera when ang_x = ang_z = 0
    # blender frame to default frame (y forward, x right, z up)
    R_def = tm.R_x(math.pi/2)
    R_semi = tm.R_z(-math.pi/2) # default frame to frame of 1st semicircle
    cam_quat = np.full((4, n_coord), np.nan) # to be filled

    # calcuate each coordinate
    for i in range(n_coord):
        # coordinates of semicircle, negative-to-positive in x-direction
        coord[:,i] = [-rad*math.cos(theta[i]), 0, rad*math.sin(theta[i])]

        # rotation matrix from camera frame to default Blender camera
        R_ij = tm.R_z(ang_z) @ tm.R_x(ang_x) @ tm.R_y(theta[i]) @ R_semi \
               @ R_def

        cam_quat[:,i] = t3d.quaternions.mat2quat(R_ij)

    # rotate all coordinates
    coord = tm.R_z(ang_z) @ tm.R_x(ang_x) @ coord
    coord += np.tile(xyz_offset, n_coord)

    return coord, cam_quat


def evaluate_all_trajectories():
    '''
    evaluate all candidate trajectories, calculate the Gramian for each, and
    calculate the optimal Gramian over all trajectories
    '''

    # rendering
    save_dir = dirs.trajectories_dir
    # integrated gramian for all trajectories
    gram_all = np.full((6, 6, n_ang), np.nan)
    # loop over semicircles
    for i in range(n_ang):
        gram = np.full((6, 6), 0.0)
        coord, cam_quat = semicircle(rad, n_pts, ang_x[i], ang_z[i],
                                     xyz_cent_col)

        # loop over points along semicircle
        for j in range(n_pts):

            # render
            cam_pos_ij = np.expand_dims(coord[:,j], 1)
            cam_quat_ij = np.expand_dims(cam_quat[:,j], 1)

            # render properties object
            to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
            render_props = RenderProperties()
            render_props.model_name = model_name
            render_props.xyz = xyz_col
            render_props.quat = quat_col
            render_props.cam_xyz = cam_pos_ij
            render_props.cam_quat = cam_quat_ij
            render_props.lens = lens
            render_props.sensor_width = sensor_width
            render_props.sensor_height = sensor_height
            render_props.compute_gramian = True
            render_props.alpha = False
            render_props.image_names=['%03d' % j]

            with open(to_render_pkl, 'wb') as output:
                pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
            br.blender_render(save_dir)
            
            # load gramian
            gram_npz = os.path.join(save_dir, 'gramian.npz')
            gram_data = np.load(gram_npz)
            gram_ij = gram_data['gram']
            gram += gram_ij[:,:,0]

        # save integrated gramian
        gram_all[:,:,i] = gram
        
    # calculate measures of all integrated gramians
    grm = gf.gramian_measures(gram_all)
    det_min_ind = np.argmin(grm['det'])
    det_max_ind = np.argmax(grm['det'])
    trace_min_ind = np.argmin(grm['trace'])
    trace_max_ind = np.argmax(grm['trace'])
    min_eval_min_ind = np.argmin(grm['min_eval'])
    min_eval_max_ind = np.argmax(grm['min_eval'])
    cond_num_min_ind = np.argmin(grm['cond_num'])
    cond_num_max_ind = np.argmax(grm['cond_num'])

    opt_ang = {
        kys[0]: ang_xz[det_min_ind],
        kys[1]: ang_xz[det_max_ind],
        kys[2]: ang_xz[trace_min_ind],
        kys[3]: ang_xz[trace_max_ind],
        kys[4]: ang_xz[min_eval_min_ind],
        kys[5]: ang_xz[min_eval_max_ind],
        kys[6]: ang_xz[cond_num_min_ind],
        kys[7]: ang_xz[cond_num_max_ind]}
    opt_ang_npz = os.path.join(save_dir, 'opt_ang.npz')
    np.savez(opt_ang_npz, xyz=xyz, ang_x=ang_x, ang_z=ang_z, opt_ang=opt_ang)


if __name__ == '__main__':
    evaluate_all_trajectories()
