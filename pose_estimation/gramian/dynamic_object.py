import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as mp
import transforms3d as t3d
from scipy import integrate

import pose_estimation.directories as dirs
import pose_estimation.tools.math as tm
import pose_estimation.tools.image as ti
import pose_estimation.blender.render as br
import pose_estimation.gramian.functions as gf
import pose_estimation.gramian.rigid_body as gr
from pose_estimation.blender.render_properties import RenderProperties

# model info
name = 'cube'

# initial conditions
xyz_0 = np.array([-1.5, 8, -1.5]) # initial state (right, forward, up)
R_0 = t3d.euler.euler2mat(.1, .3, .1, 'sxyz')
q_0 = t3d.quaternions.mat2quat(R_0)
xyz_q_0 = np.concatenate((xyz_0, q_0))
v_0 = np.array([3, 5, 9])   # initial translational velocity (inertial frame)
om_0 = np.array([3, 2, 1.5])  # initial angular velocity (body frame) 
v_om_0 = np.concatenate((v_0, om_0))

# integrate rigid body dynamics, Newton-Euler
# since we do not change velocities for the Gramian, these solutions work for
# the perturbations too 
t0 = 0
tf = 2.0    # seconds
n_t = 3001  # number of time points for integration
inds = np.arange(n_t)
t = np.linspace(t0, tf, n_t)
sol = integrate.solve_ivp(gr.newton_euler, [t0,tf], v_om_0, method='RK45',
                          t_eval=t)
t_sol = sol.t
v_om = sol.y

# integrate kinematics of nominal trajectory
xyz_q, q_dot = gr.integrate_kinematics(t, v_om, xyz_q_0)
xyz = xyz_q[:3,:]
q = xyz_q[3:,:]

# integrate kinematics of perturbation trajectories
xyz_pert_0, q_pert_0 = gf.standard_pert(xyz_0, q_0)
xyz_q_pert_0 = np.concatenate((xyz_pert_0, q_pert_0))
xyz_q_pert = np.full((7, 12, n_t), np.nan)
for i in range(12):
    xyz_q_pert[:,i,:], q_dot_pert = gr.integrate_kinematics(t, v_om,
                                                            xyz_q_pert_0[:,i])
xyz_pert = xyz_q_pert[:3,:,:]
q_pert = xyz_q_pert[3:,:,:]

# render
n_frame = 60   # number of frames (30 fps)
step_frame = int(np.floor(n_t/n_frame))
inds_frame = inds[::step_frame] # gives n_frame + 1 indices
inds_frame = inds_frame[:-1] # remove last index
delta_t_frame = t[inds_frame[1]] - t[inds_frame[0]]
save_dir = dirs.dynamic_dir
to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
render_props = RenderProperties()
render_props.n_renders = n_frame
render_props.model_name = name
render_props.xyz = xyz[:,inds_frame]
render_props.quat = q[:,inds_frame]
render_props.alpha = False
render_props.compute_gramian = True
#render_props.compute_gramian = False
render_props.pert_xyz = xyz_pert[:,:,inds_frame]
render_props.pert_quat = q_pert[:,:,inds_frame]
with open(to_render_pkl, 'wb') as output:
    pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
br.blender_render(save_dir)

# gramian
data = np.load(os.path.join(save_dir, 'gramian.npz'))
gram = data['gram']
gram_sum = np.sum(gram, axis=2)
gram_sum = delta_t_frame*gram_sum

# render snapshots
n_snapshot = 14 # number of snapshots for figure
step_snapshot = int(np.floor(n_t/n_snapshot))
inds_snapshot = inds[::step_snapshot]
inds_snapshot = inds_snapshot[:-1]
png_name_snapshot = 'snapshot_%06d'
for i in range(n_snapshot):
    ind_i = inds_snapshot[i]
    render_props = RenderProperties()
    render_props.model_name = name
    render_props.image_names = [png_name_snapshot % ind_i]
    render_props.xyz = xyz[:,[ind_i]]
    render_props.quat = q[:,[ind_i]]

    # only make last snapshot have a background
    if i < n_snapshot-1:
        render_props.alpha = True
    else:
        render_props.alpha = False

    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    br.blender_render(save_dir)

# overlay snapshots
im_file_0 = os.path.join(save_dir,
                         png_name_snapshot % inds_snapshot[-1] + '.png')
im_snapshot = ti.load_im_np(im_file_0)
for i in reversed(inds_snapshot[:-1]):
    im_file_i = os.path.join(save_dir, png_name_snapshot % i + '.png')
    im_overlay = ti.load_im_np(im_file_i)
    im_snapshot = ti.overlay(im_overlay, im_snapshot)
ti.write_im_np(os.path.join(save_dir, 'snapshots.png'), im_snapshot)
print(gram_sum)
tm.print_matrix_as_latex(gram_sum, n_digs=2)
