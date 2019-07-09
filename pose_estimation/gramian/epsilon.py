'''
calculate empirical observability Gramian with different epsilon values
'''
import os
import math
import pickle
import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt

import pose_estimation.directories as dirs
import pose_estimation.tools.image as ti
import pose_estimation.blender.render as br
from pose_estimation.blender.render_properties import RenderProperties

# files
save_dir = dirs.gram_eps_dir
gram_npz = os.path.join(save_dir, 'gramian.npz')
gram_eps_npz = os.path.join(save_dir, 'gram_eps.npz')

# model info
name = 'cone'
#name = 'sphere'

# object position and orientations
xyz = np.array([0, 5, -0.5]) # right, forward, up
xyz = xyz[:, np.newaxis] # make it a column vector 
euler = [0, 0, 0]
quat = t3d.euler.euler2quat(*euler)
quat = quat[:, np.newaxis] # make it a column vector

# epsilons
n_eps = 30
eps = np.logspace(-7, 1, num=n_eps, base=10)

# iterate over epsilons and calculate Gramian
gram = np.full((6, 6, n_eps), np.nan)
for i in range(n_eps):

    # render
    to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
    render_props = RenderProperties()
    render_props.model_name = name
    render_props.xyz = xyz
    render_props.quat = quat
    render_props.compute_gramian = True
    render_props.eps = eps[i]
    render_props.alpha = False

    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)

    br.blender_render(save_dir)

    # load gramian
    gram_data = np.load(gram_npz)
    gram_i = gram_data['gram']
    gram_i = gram_i[:,:,0]
    gram[:,:,i] = gram_i

# for the record, save all gramians
np.savez(gram_eps_npz, gram=gram)

# load saved data
data = np.load(gram_eps_npz)
gram = data['gram']

# plot
plt.rc('text', usetex=True)
image_file = os.path.join(save_dir, '000000.png')
fig, ax = plt.subplots(1, 1, dpi=300)
l0, = ax.plot(eps, gram[0,0,:], label='1,1')
l1, = ax.plot(eps, gram[1,1,:], label='2,2')
l2, = ax.plot(eps, gram[2,2,:], label='3,3')
l3, = ax.plot(eps, gram[3,3,:], label='4,4')
l4, = ax.plot(eps, gram[4,4,:], label='5,5')
l5, = ax.plot(eps, gram[5,5,:], label='6,6')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\epsilon$', fontsize=24)
ax.tick_params(axis='x', which='minor', bottom=False)
ax.tick_params(axis='both', which='major', labelsize=13)
fig.subplots_adjust(bottom=0.15, top=0.95)

# legend
leg = ax.legend(handles=[l0, l1, l2, l3, l4, l5], loc='upper right',
                edgecolor='k', title='Gramian\nmatrix\nelement',
                fontsize=12, labelspacing=.2)
leg.get_title().set_fontsize('12')

# plot image
im = ti.load_im_np(image_file)
im_axes = [.22, .17, .3, .3] # upper left, [left, bottom, width, height]
ax_im = fig.add_axes(im_axes, frame_on=False, xticks=[], xticklabels=[],
                     yticks=[], yticklabels=[])
ax_im.set_title(r'nominal pose', fontsize=16)
ax_im.imshow(im, interpolation='lanczos')
save_png = os.path.join(save_dir, 'figure.png')
fig.savefig(save_png)
#plt.show()
