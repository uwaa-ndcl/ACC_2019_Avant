import os
import sys
import math
import pickle
import numpy as np
import transforms3d as t3d

import pose_estimation.directories as dirs
import pose_estimation.tools.math as tm
import pose_estimation.blender.render as br
from pose_estimation.blender.render_properties import RenderProperties

# model info
#name = 'sphere'
#name = 'lamp'
name = 'cone'

# object properties
if name == 'sphere':
    xyz = np.array([0.0, 5.0, 0.0])
elif name == 'lamp':
    xyz = np.array([0.0, 1.6, 0.0])
elif name == 'cone':
    xyz = np.array([0.0, 5.0, -0.5])
xyz = xyz[:, np.newaxis] # make it a column vector
euler = [0, 0, 0]
quat = t3d.euler.euler2quat(*euler)
quat = quat[:, np.newaxis] # make it a column vector

# save render info to file
save_dir = dirs.gramian_example_dir
to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
render_props = RenderProperties()
render_props.model_name = name
render_props.xyz = xyz
render_props.quat = quat
render_props.compute_gramian = True
render_props.alpha = False
render_props.eps = 1e-2

with open(to_render_pkl, 'wb') as output:
    pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
br.blender_render(save_dir)

# load gramian
data = np.load(os.path.join(save_dir, 'gramian.npz'))
gram = data['gram']
gram = gram[:,:,0]

# printing
print(gram)
tm.print_matrix_as_latex(gram)
