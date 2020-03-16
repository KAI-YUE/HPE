"""
Denavit-Hartenberg parameters:
1. Rotate x axis DH_theta_alpha along the current z axis
2. Take the displacement transformation along the x axis
3. Rotate z axis alpha along the x axis
"""
import numpy as np

averaged_phalanx_length = \
([40.712, 34.040, 29.417, 26.423,
79.706, 35.224, 23.270, 22.036,
75.669, 41.975, 26.329, 24.504,
75.358, 39.978, 23.513, 22.647,
74.556, 27.541, 19.826, 20.395])

def z_matrix(theta, d=0):
    """
    Return the z matrix given the D-H parameter.
    To guarantee the gradient calculation, we assign the value manually.
    """
    z = np.eye(4)
    z[0][0] = np.cos(theta)
    z[0][1] = -np.sin(theta)
    z[1][0] = np.sin(theta)
    z[1][1] = np.cos(theta)
    z[2][3] = d
    return z

def x_matrix(alpha, a=0):
    """
    Return the x matrix given the D-H parameter.
    To guarantee the gradient calculation, we assign the value manually.
    """
    x = np.eye(4)
    x[0][3] = a
    x[1][1] = np.cos(alpha)
    x[1][2] = -np.sin(alpha)
    x[2][1] = np.sin(alpha)
    x[2][2] = np.cos(alpha)
    return x

def forward_kinematics(DH_theta_alpha, DH_scale):
    
    DH_a = DH_scale * averaged_phalanx_length
    
    pos = np.zeros((20, 3))
    p0 = np.array([0., 0., 0., 1.])
    
    # For each sample
    
    # Thumb kinematics
    z = z_matrix(DH_theta_alpha[0])
    x = x_matrix(-np.pi/2, 0.)
    T = z @ x
    
    # transform to thunmb_joint0
    z = z_matrix(DH_theta_alpha[1])
    x = x_matrix(np.pi/2, DH_a[0])
    T = T @ z @ x
    pos[0] = (T @ p0) [:3]
    
    z = z_matrix(DH_theta_alpha[2])
    x = x_matrix(-np.pi/2, 0.)
    T = T @ z @ x
    
     # transform to thunmb_joint1
    z = z_matrix(DH_theta_alpha[3])
    x = x_matrix(np.pi/2, DH_a[1])
    T = T @ z @ x
    pos[1] = (T @ p0) [:3]
    
    z = z_matrix(DH_theta_alpha[4])
    x = x_matrix(-np.pi/2, 0.)
    T = T @ z @ x
    
    # transform to thunmb_joint2
    z = z_matrix(DH_theta_alpha[5])
    x = x_matrix(np.pi/2, DH_a[2])
    T = T @ z @ x
    pos[2] = (T @ p0) [:3]
    
    z = z_matrix(DH_theta_alpha[6])
    x = x_matrix(-np.pi/2, 0.)
    T = T @ z @ x
    # transform to thunmb_joint3
    z = z_matrix(DH_theta_alpha[7])
    x = x_matrix(0., DH_a[3])
    T = T @ z @ x
    pos[3] = (T @ p0) [:3]
    
    # Index finger kinematics
    z = z_matrix(DH_theta_alpha[8])
    x = x_matrix(0., DH_a[4])
    T = z @ x
    pos[4] = (T @ p0) [:3]
    
    z = z_matrix(DH_theta_alpha[9])
    x = x_matrix(-np.pi/2, 0.)
    T = T @ z @ x
    
    for l in range(3):
        z = z_matrix(DH_theta_alpha[10+l])
        x = x_matrix(0., DH_a[5+l])
        T = T @ z @ x
        pos[5+l] = (T @ p0) [:3]
    
    # middle finger
    z = z_matrix(np.pi/2)
    x = x_matrix(0., DH_a[8])
    T = z @ x
    pos[8] = (T @ p0) [:3]
    
    z = z_matrix(DH_theta_alpha[13])
    x = x_matrix(-np.pi/2, 0.)
    T = T @ z @ x
    
    for l in range(3):
        z = z_matrix(DH_theta_alpha[14+l])
        x = x_matrix(0., DH_a[9+l])
        T = T @ z @ x
        pos[9+l] = (T @ p0) [:3]
        
    
    # Ring finger kinematics
    z = z_matrix(DH_theta_alpha[17])
    x = x_matrix(-np.pi/2, 0.)
    T = z @ x
    
    z = z_matrix(DH_theta_alpha[18])
    x = x_matrix(np.pi/2, DH_a[12])
    T = T @ z @ x
    pos[12] = (T @ p0) [:3]
    
    z = z_matrix(DH_theta_alpha[19])
    x = x_matrix(-np.pi/2, 0.)
    T = T @ z @ x
    
    for l in range(3):
        z = z_matrix(DH_theta_alpha[20+l])
        x = x_matrix(0., DH_a[13+l])
        T = T @ z @ x
        pos[13+l] = (T @ p0) [:3]
    
    # Little finger kinematics
    z = z_matrix(DH_theta_alpha[23])
    x = x_matrix(-np.pi/2, 0.)
    T = z @ x
    
    z = z_matrix(DH_theta_alpha[24])
    x = x_matrix(np.pi/2, DH_a[16])
    T = T @ z @ x
    pos[16] = (T @ p0) [:3]
    
    z = z_matrix(DH_theta_alpha[25])
    x = x_matrix(-np.pi/2, 0.)
    T = T @ z @ x
    
    for l in range(3):
        z = z_matrix(DH_theta_alpha[26+l])
        x = x_matrix(0., DH_a[17+l])
        T = T @ z @ x
        pos[17+l] = (T @ p0) [:3]
    
    pos = np.vstack((np.array([0.,0,0]),pos))

    return pos