import os
import bpy
import time
import math
import numpy as np
import imageio
import transforms3d as t3d

import pose_estimation.directories as dirs
import pose_estimation.tools.image as ti
import pose_estimation.gramian.functions as gf


def render_image(
        cam_ob, cam_pos, cam_quat, ob, ob_pos, ob_quat, image_file, alpha=True,
        world_RGB=None):
    '''
    set the camera and object to a position and orientation, take, and save an
    image
    '''

    cam_ob.location = cam_pos
    cam_ob.rotation_mode = 'QUATERNION'
    cam_ob.rotation_quaternion = cam_quat

    ob.location = ob_pos
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = ob_quat
    bpy.data.scenes['Scene'].render.filepath = image_file

    if world_RGB is not None:
        A = np.array([1.0]) # alpha for world RGBA lighting
        RGBA = np.concatenate((world_RGB, A))
        bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0]. \
            default_value = RGBA

    if not alpha:
        bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGB'
        bpy.data.scenes['Scene'].cycles.film_transparent = False

    else:
        bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGBA'
        bpy.data.scenes['Scene'].cycles.film_transparent = True

    bpy.ops.render.render(write_still=True)


def compute_gramian_object(render_props):
    '''
    compute the empirical observability Gramian for each render
    '''

    # scalars, vectors, and arrays
    n_states = 6 # x, y, z, x-rot, y-rot, z-rot
    n_el = 3*render_props.pix_width*render_props.pix_height
    mat = np.full((n_el, n_states), np.nan) # matrix of y^+ - y^- vectors
    # gramian for all renders
    gram = np.full((n_states, n_states, render_props.n_renders), np.nan)

    # check if background images are to be used load background image
    if render_props.bkgd_image_list is None:
        use_bkgd_image = False
    else:
        use_bkgd_image = True

    # generic save files
    temp_file_neg = os.path.join(dirs.gramian_image_save_dir,
                                 'temp_%d_neg.png')
    temp_file_pos = os.path.join(dirs.gramian_image_save_dir,
                                 'temp_%d_pos.png')

    # loop through renders
    for i in range(0, render_props.n_renders):

        # i'th position, quaternion, and perturbations
        xyz_i = render_props.xyz[:,i]
        quat_i = render_props.quat[:,i]
        if render_props.pert_xyz is None and render_props.pert_quat is None:
            pert_xyz_i, pert_quat_i = gf.standard_pert(xyz_i, quat_i,
                                                       render_props.eps)
        else:
            pert_xyz_i = render_props.pert_xyz[:,:,i]
            pert_quat_i = render_props.pert_quat[:,:,i]

        # if we are using a background image, get it
        if use_bkgd_image:
            bkgd_image = ti.load_im_np(render_props.bkgd_image_list[i])

        # set world_RGB for i'th render
        if render_props.world_RGB is None:
            world_RGB_i = None
        else:
            world_RGB_i = render_props.world_RGB[:,i]

        # loop through states
        for j in range(0, n_states):

            # perturbations in the negative direction
            temp_file_neg_j = temp_file_neg % j 
            temp_file_pos_j = temp_file_pos % j 
            pert_xyz_minus_ij = pert_xyz_i[:,2*j]
            pert_quat_minus_ij = pert_quat_i[:,2*j]
            pert_xyz_plus_ij = pert_xyz_i[:,2*j+1]
            pert_quat_plus_ij = pert_quat_i[:,2*j+1]

            # render positive & negative images
            render_image(
                cam_ob=render_props.cam_ob,
                cam_pos=render_props.cam_xyz,
                cam_quat=render_props.cam_quat,
                ob=render_props.ob,
                ob_pos=pert_xyz_minus_ij,
                ob_quat=pert_quat_minus_ij,
                image_file=temp_file_neg_j,
                world_RGB=world_RGB_i,
                alpha=render_props.alpha)

            render_image(
                cam_ob=render_props.cam_ob,
                cam_pos=render_props.cam_xyz,
                cam_quat=render_props.cam_quat,
                ob=render_props.ob,
                ob_pos=pert_xyz_plus_ij,
                ob_quat=pert_quat_plus_ij,
                image_file=temp_file_pos_j,
                world_RGB=world_RGB_i,
                alpha=render_props.alpha)

            # overlay render on background image?
            if use_bkgd_image:
                # negative
                # entries from 0 to 1
                overlay_image = ti.load_im_np(temp_file_neg_j)
                bkgd_image = ti.load_im_np(render_props.bkgd_image_list[i])
                y_minus = ti.overlay(overlay_image, bkgd_image)

                # positive
                # entries from 0 to 1
                overlay_image = ti.load_im_np(temp_file_pos_j)
                bkgd_image = ti.load_im_np(render_props.bkgd_image_list[i])
                y_plus = ti.overlay(overlay_image, bkgd_image)

            else:
                # negative
                y_minus = ti.load_im_np(temp_file_neg_j)[:,:,:3] # no alpha

                # positive
                y_plus = ti.load_im_np(temp_file_pos_j)[:,:,:3] # no alpha
            
            # compare positive to negative perturbations
            y_diff = y_plus - y_minus
            y_diff_vec = np.reshape(y_diff, n_el)
            mat[:,j] = y_diff_vec

        # compute gramian
        gram[:, :, i] = (1/(4*render_props.eps**2))*mat.T @ mat
    return gram


def render_pose(render_props):
    '''
    render the object at different x, y, and z locations and orientations (as
    quaternions)
    '''

    # preliminary things
    if render_props.image_names is None:
        image_numerical_name = '%06d.png' # generic name for each image

    # loop through poses to generate images
    for i in range(render_props.n_renders):

        # different world color?
        if render_props.world_RGB is not None:
            world_RGB_i = render_props.world_RGB[:, i]
        else:
            world_RGB_i = None
       
        # give the image a name
        if render_props.image_names is None:
            image_file_name_i = image_numerical_name % i
        else:
            image_file_name_i = render_props.image_names[i]
        image_file_i = os.path.join(render_props.save_dir, image_file_name_i)
        
        # render image i
        render_image(
            cam_ob=render_props.cam_ob,
            cam_pos=render_props.cam_xyz,
            cam_quat=render_props.cam_quat,
            ob=render_props.ob,
            ob_pos=render_props.xyz[:,i],
            ob_quat=render_props.quat[:,i],
            image_file=image_file_i,
            alpha=render_props.alpha,
            world_RGB=world_RGB_i)

        # if we have a list of background images, overlay render onto
        # background
        if render_props.bkgd_image_list is not None:
            im_i = ti.load_im_np(image_file_i)
            im_bkgd_i = ti.load_im_np(render_props.bkgd_image_list[i])
            im_overlay = ti.overlay(im_i, im_bkgd_i)
            ti.write_im_np(image_file_i, im_overlay)

    # compute gramian for all renders, then save data
    if render_props.compute_gramian:
        gram_file = os.path.join(render_props.save_dir, 'gramian.npz')
        gram = compute_gramian_object(render_props)
        np.savez(gram_file, gram=gram)
