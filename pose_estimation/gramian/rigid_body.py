import numpy as np

# constants
g = 9.8     # gravity
m = 1       # mass
l = 1       # edge length of cube
J = (1/6)*m*(l**2)*np.eye(3) # inertia matrix of cube (body frame)

def cross(v):
    '''
    cross product matrix, for vectors v and w, v x w = cross(v) @ w
    '''
    mat = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    return mat


def om_mat(om):
    '''
    matrix function of angular velocity om (expressed in body coordinates)
    which when multiplied by quaternion q, produces the angular velocity of q:
    om_mat(om) @ q = dq/dt 
    eq. 3.104 (p. 110) Analytic Mechanics of Space Systems, 2nd ed.
        note omega is expressed in B frame (see eq. 3.106)
    '''
    om_mat = np.array([[    0, -om[0], -om[1], -om[2]],
                       [om[0],      0,  om[2], -om[1]],
                       [om[1], -om[2],      0,  om[0]],
                       [om[2],  om[1], -om[0],      0]])

    return .5*om_mat


def newton_euler(t, v_om):
    '''
    Newton-Euler equations for a rigid body
    assumes velocity (v_A) is in inertial frame (A) coordinates,
    and omega (om_B) is in body-frame (B) coordinates
    ref: eq. 4.16 (p. 167) in A Mathematical Introduction to Robotic
    Manipulation
    '''

    v_A = v_om[:3]
    om_B = v_om[3:]
    om_col_B = om_B[:,np.newaxis] # omega as column vector
    
    # translational dynamics: all expressed in inertial frame A
    F_A = np.array([[0, 0, -m*g]]).T # inertial frame force of gravity
    v_dot_A = np.linalg.inv(m*np.eye(3)) @ F_A

    # rotational dynamics: all expressed in body frame B
    tau_B = np.array([[0, 0, 0]]).T     # inertial frame torque
    om_dot_B = np.linalg.inv(J) @ (tau_B - cross(om_B) @ J @ om_col_B)
    v_om_dot = np.concatenate((v_dot_A, om_dot_B))
    v_om_dot = np.squeeze(v_om_dot)

    return v_om_dot


def integrate_kinematics(t, v_om, xyz_q_0):
    '''
    forward Euler integration of angular velocities v and omega,
    to x and quaternions

    inputs:
        t: time points to evaluate (evenly spaced), size (# of time points)
        v_om: translational and angular velocities
              v expressed in A frame, omega expressed in B frame
              size (6, # of time points)
        xyz_q_0: initial xyz and quaternion, size (7)
    '''
    n_pts = len(t) 
    dt = t[1] - t[0]
    v = v_om[:3,:]
    om = v_om[3:,:]

    # 1st order forward Euler integration
    xyz = np.full((3, n_pts), np.nan)
    q = np.full((4, n_pts), np.nan)
    q_dot = np.full((4, n_pts), np.nan) # tim derivative of quaternion
    xyz[:,0] = xyz_q_0[:3]
    q[:,0] = xyz_q_0[3:]
    for i in range(n_pts):
        if i > 0:
            xyz[:,i] = xyz[:,i-1] + dt*v[:,i-1]
            q[:,i] = q[:,i-1] + dt*q_dot[:,i-1]
            q[:,i] = q[:,i]/np.linalg.norm(q[:,i]) # normalize quaternion

        # time derivative of quaternion (remember omega is in B coordinates!)
        q_dot_col_i = om_mat(om[:,i]) @ q[:,i][:,np.newaxis]
        q_dot[:,i] = np.squeeze(q_dot_col_i)
    xyz_q = np.concatenate((xyz, q)) 

    return xyz_q, q_dot
