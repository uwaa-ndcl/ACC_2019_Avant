'''
this file is to be run as a blender function
blender --background --python /path/to/this/file.py
'''
import os
import math
import numpy as np
import transforms3d as t3d
import bpy

import pose_estimation.directories as dirs
import pose_estimation.tools.math as tm
import pose_estimation.blender.functions as bf
import pose_estimation.gramian.trajectories as gt

# see: https://blender.stackexchange.com/questions/6750/poly-bezier-curve-
# from-a-list-of-coordinates

# model info
blender_models_dir = dirs.blender_models_dir
blend_file = os.path.join(blender_models_dir, gt.model_name+'.blend')
bpy.ops.wm.open_mainfile(filepath=blend_file)
ob_name = 'all_parts'
ob = bpy.data.objects[ob_name]
ob_pos = gt.xyz
ob_quat = gt.quat

# set up camera
camera_name = 'cam0'
cam = bpy.data.cameras.new(camera_name)  # create a new camera

# define camera
cam_ob = bpy.data.objects.new(camera_name, cam)  # create a new camera object
bpy.context.scene.camera = cam_ob  # set the active camera
cam_dist = 9.9
cam_pos = [cam_dist, -cam_dist, cam_dist] + gt.xyz_cent
cam_euler = [0.96, 0, .78]
cam_quat = t3d.euler.euler2quat(*cam_euler, axes='sxyz')
bpy.data.objects[camera_name].location = cam_pos
bpy.data.objects[camera_name].rotation_euler = cam_euler

# camer lens: based on Canon Powershot A2500
bpy.data.cameras[camera_name].lens = gt.lens
bpy.data.cameras[camera_name].sensor_width = gt.sensor_width
bpy.data.cameras[camera_name].sensor_height = gt.sensor_height

# scene properties
bpy.data.scenes['Scene'].render.resolution_x = 1000
bpy.data.scenes['Scene'].render.resolution_y = 1000
bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGBA'
bpy.data.scenes['Scene'].render.image_settings.file_format = 'PNG'
bpy.data.scenes['Scene'].cycles.film_transparent = True
bpy.data.scenes['Scene'].render.engine = 'CYCLES'
bpy.data.scenes['Scene'].cycles.device = 'GPU'

# load optimal trajectory data
opt_ang_npz = os.path.join(dirs.trajectories_dir, 'opt_ang.npz')
dat = np.load(opt_ang_npz, allow_pickle=True)
xyz = dat['xyz']
ang_x = dat['ang_x']
ang_z = dat['ang_z']
opt_ang = dat['opt_ang'].item() # convert from numpy object to dictionary


def make_material(clr, trans=.0):
    '''
    create a solid-color material
    clr: a tuple of three numbers (R,G,B)
    trans: transparency value (0 to 1)
    '''

    # create the material and get its nodes
    mat = bpy.data.materials.new(name='my_material')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # if nodes have been created by default, remove them
    if nodes.get('Diffuse BSDF', None) is not None:
        nodes.remove(nodes['Diffuse BSDF'])
    if nodes.get('Material Output', None) is not None:
        nodes.remove(nodes['Material Output'])

    dif = nodes.new("ShaderNodeBsdfDiffuse")
    dif.inputs[0].default_value = clr + [1,]
    dif.location = (0,100)

    trn = nodes.new("ShaderNodeBsdfTransparent")
    trn.inputs[0].default_value = [1,1,1,1]
    trn.location = (0,-100)

    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = trans # this sets the transparency level!
    mix.location = (200,0)

    out = nodes.new(type='ShaderNodeOutputMaterial')
    out.location = (400,0)

    # link nodes
    links = mat.node_tree.links
    link0 = links.new(dif.outputs[0], mix.inputs[1])
    link1 = links.new(trn.outputs[0], mix.inputs[2])
    link2 = links.new(mix.outputs[0], out.inputs[0])

    return mat


def create_curve(coord, mat):
    '''
    create a semi-circular curve with material mat
    '''

    # create curve
    curve_data = bpy.data.curves.new('myCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = .1
    curve_data.bevel_resolution = 50
    curve_data.fill_mode = 'FULL'

    # create polyline from coords
    n_coord = coord.shape[1]
    # POLY, BEZIER, BSPLINE, CARDINAL, NURBS
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(n_coord-1)
    for i in range(n_coord):
        polyline.points[i].co = (coord[0,i], coord[1,i], coord[2,i], 1)

    # link scene to object
    curve_ob = bpy.data.objects.new('myCurve', curve_data)
    scn = bpy.context.scene
    scn.objects.link(curve_ob)
    scn.objects.active = curve_ob
    curve_ob.select = True
    curve_ob.color = [1,0,0,1]

    curve_ob.data.materials.append(mat)


def create_curve_striped(coord, n_stripes, clr_0, clr_1):
    '''
    create a striped curve
    '''

    n_pts = coord.shape[1]
    coord_split = np.array_split(coord, n_stripes, axis=1)
    mat_0 = make_material(clr_0)
    mat_1 = make_material(clr_1)

    for i, coord_i in enumerate(coord_split):

        # so there are no spaces in between the stripes, add the beginning
        # coordinate of the next stripe
        if i < n_stripes-1:
            coord_i = np.append(coord_i, coord_split[i+1][:,[0]], axis=1)

        if i % 2 == 0:
            create_curve(coord_i, mat_0)
        else:
            create_curve(coord_i, mat_1)


def draw_all_curves():
    '''
    draw all semicircle curves defined by ang_x and ang_z
    '''

    # loop over curves
    for i in range(gt.n_ang):
        coord, _ = gt.semicircle(gt.rad, 100, gt.ang_x[i], gt.ang_z[i],
                                 gt.xyz_cent_col)
        create_curve_striped(coord, n_stripes=15, clr_0=[0,0,0], clr_1=[1,0,0])

    # render a picture
    image_file = os.path.join(dirs.trajectories_dir, 'all_curves.png')
    bf.render_image(cam_ob, cam_pos, cam_quat, ob, ob_pos, ob_quat, image_file,
                    alpha=False, world_RGB=None)


def draw_opt_curves(min_or_max):
    '''
    draw the optimal curves
    min_or_max: 'min' or 'max', draw the min or max measures of the gramian
    '''

    if min_or_max == 'min':
        kys_opt = gt.kys[0:5:2]
    elif min_or_max == 'max':
        kys_opt = gt.kys[1:6:2]

    for ky_i in kys_opt:
        i = gt.kys.index(ky_i)
        xz_i = opt_ang[ky_i] # i'th angles

        if i == 0:
            shft_col = np.array([[0],[0],[0]]) # offset of center of curve
        elif i == 1:
            shft_col = np.array([[0],[0],[0]])
        elif i == 2:
            shft_col = np.array([[0],[0],[0]])
        elif i == 3:
            shft_col = np.array([[0],[0],[0]])
        elif i == 4:
            shft_col = np.array([[0],[0],[0]])
        elif i == 5:
            shft_col = np.array([[0],[0],[0]])
        elif i == 6:
            shft_col = np.array([[0],[0],[0]])
        elif i == 7:
            shft_col = np.array([[0],[0],[0]])
        coord, _ = gt.semicircle(gt.rad, 100, xz_i[0], xz_i[1],
                                 gt.xyz_cent_col + shft_col)

        mat = make_material(gt.clrs[i])
        create_curve(coord, mat)

    # render a picture
    output_image_name = min_or_max + '.png'
    output_image_file = os.path.join(dirs.trajectories_dir, output_image_name)
    bf.render_image(cam_ob, cam_pos, cam_quat, ob, ob_pos, ob_quat,
                    output_image_file, alpha=False, world_RGB=None)

# draw
#draw_all_curves()
draw_opt_curves('min')
#draw_opt_curves('max')
