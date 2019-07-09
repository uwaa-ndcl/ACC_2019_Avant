import numpy as np
import transforms3d as t3d
import pose_estimation.tools.math as tm

def standard_pert(xyz, quat, eps=1e-2):
    '''
    standard perturbation
    
    inputs:
        xyz: np array of size (3,1)
        quat: np array of size (4,1)
        eps: epsilon for perturbations

    outputs:
        pert_xyz: np array of xyz for 12 perturbations, size, (3, 12)
                  order of perturbations: -1, +1, -2, +2, ...
        pert_quat: np array of quat for 12 perturbations, size, (4, 12)
                  order of perturbations: -1, +1, -2, +2, ...
    '''

    # nominal rotation matrix
    R = t3d.quaternions.quat2mat(quat)

    # basis vectors
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    # translational perturbations in the negative direction
    xyz_minus_1 = xyz - eps*e1
    xyz_minus_2 = xyz - eps*e2
    xyz_minus_3 = xyz - eps*e3

    # translational perturbations in the positive direction
    xyz_plus_1 = xyz + eps*e1
    xyz_plus_2 = xyz + eps*e2
    xyz_plus_3 = xyz + eps*e3

    # perturbations in negative direction
    R_minus_1 = t3d.euler.euler2mat(-eps,0,0,axes='sxyz') # rotation about x
    R_minus_2 = t3d.euler.euler2mat(0,-eps,0,axes='sxyz') # rotation about y
    R_minus_3 = t3d.euler.euler2mat(0,0,-eps,axes='sxyz') # rotation about y
    quat_minus_1 = t3d.quaternions.mat2quat(R @ R_minus_1) 
    quat_minus_2 = t3d.quaternions.mat2quat(R @ R_minus_2) 
    quat_minus_3 = t3d.quaternions.mat2quat(R @ R_minus_3) 

    # perturbations in positive direction
    R_plus_1 = t3d.euler.euler2mat(eps,0,0,axes='sxyz') # rotation about x
    R_plus_2 = t3d.euler.euler2mat(0,eps,0,axes='sxyz') # rotation about y
    R_plus_3 = t3d.euler.euler2mat(0,0,eps,axes='sxyz') # rotation about z
    quat_plus_1 = t3d.quaternions.mat2quat(R @ R_plus_1) 
    quat_plus_2 = t3d.quaternions.mat2quat(R @ R_plus_2) 
    quat_plus_3 = t3d.quaternions.mat2quat(R @ R_plus_3) 

    # xyz with perturbations
    pert_xyz = np.stack((
        xyz_minus_1, xyz_plus_1,
        xyz_minus_2, xyz_plus_2,
        xyz_minus_3, xyz_plus_3,
        xyz, xyz, xyz, xyz, xyz, xyz),
        axis=1)

    # quat with perturbations
    pert_quat = np.stack((
        quat, quat, quat, quat, quat, quat,
        quat_minus_1, quat_plus_1,
        quat_minus_2, quat_plus_2,
        quat_minus_3, quat_plus_3),
        axis=1)

    return pert_xyz, pert_quat


def gramian_measures(gram):
    '''
    compute various measures of the gramian

    input:
    gram: an array of Gramians, of dimension (n_rows, n_cols, n_gramians)

    output:
    a dictionary containing the trace, determinant, minimum eigenvalue of each
    gramian, as well as the normalized values and normalized inverses of all of
    these measures
    '''

    # arrays to be filled
    n_gram = gram.shape[2]
    trace = np.full(n_gram, np.nan)
    det = np.full(n_gram, np.nan) # determinant
    min_eval = np.full(n_gram, np.nan) # minimum eigenvalue
    cond_num = np.full(n_gram, np.nan) # condition number
    trace_inv = np.full(n_gram, np.nan)
    det_inv = np.full(n_gram, np.nan) # determinant
    min_eval_inv = np.full(n_gram, np.nan) # minimum eigenvalue
    cond_num_inv = np.full(n_gram, np.nan) # condition number

    # loop over all gramians
    for i in range(n_gram):
        gram_i = gram[:, :, i]
        gram_inv_i = np.linalg.inv(gram_i)

        # measures
        trace[i] = np.trace(gram_i) 
        det[i] = np.linalg.det(gram_i)
        val_i, vec_i = np.linalg.eig(gram_i)
        min_eval[i] = np.amin(val_i)
        cond_num[i] = np.amax(val_i)/np.amin(val_i)
        
        # measures of inverse (i.e. unobservability indices)
        trace_inv[i] = np.trace(gram_inv_i) 
        det_inv[i] = np.linalg.det(gram_inv_i)
        val_i, vec_i = np.linalg.eig(gram_inv_i)
        min_eval_inv[i] = np.amin(val_i)
        cond_num_inv[i] = np.amax(val_i)/np.amin(val_i)

    # normalized measures
    det_nrm = tm.normalize_array(det)
    trace_nrm = tm.normalize_array(trace)
    min_eval_nrm = tm.normalize_array(min_eval)

    # inverse normalized measures
    trace_inv_nrm = tm.normalize_array(trace_inv)
    det_inv_nrm = tm.normalize_array(det_inv)
    min_eval_inv_nrm = tm.normalize_array(min_eval_inv)

    # gramian measures as a dictionary
    gram_dict = {'det': det,
                 'trace': trace,
                 'min_eval': min_eval,
                 'cond_num': cond_num,
                 'det_nrm': det_nrm,
                 'trace_nrm': trace_nrm,
                 'min_eval_nrm': min_eval_nrm,
                 'det_inv_nrm': det_inv_nrm,
                 'trace_inv_nrm': trace_inv_nrm,
                 'min_eval_inv_nrm': min_eval_inv_nrm}

    return gram_dict
