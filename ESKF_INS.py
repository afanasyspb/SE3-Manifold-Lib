"""
Optimized Error-State Extended Kalman Filter (ESKF) for INS/VO Fusion.

This module implements the industry-standard Error-State Kalman Filter 
(often referred to as Indirect Kalman Filter) on the SE(3) manifold.
It features optimized NumPy matrix operations for high performance and uses the 
numerically stable Joseph form for covariance updates to prevent divergence.
"""

import numpy as np

# ==============================================================================
# Optimized Math Utilities
# ==============================================================================

def skew(v):
    """
    Returns the skew-symmetric matrix of a 3D vector.
    Used for cross product equivalent in matrix form.
    """
    x, y, z = v
    return np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,    0.0]
    ], dtype=np.float64)

def quat_normalize(q):
    """Normalizes a quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-9: # Avoid division by zero for near-zero quaternions
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # Return identity
    return q / n

def quat_mul(q1, q2):
    """
    Performs Hamilton product of two quaternions (w, x, y, z).
    Result represents combined rotation.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def so3_exp(phi):
    """
    Exponential map from so(3) (rotation vector/axis-angle) to SO(3) (unit quaternion).
    Includes small-angle approximation for numerical stability.
    
    Args:
        phi (np.ndarray): 3D rotation vector (e.g., angular velocity * dt).
    
    Returns:
        np.ndarray: Unit quaternion representing the rotation.
    """
    theta_sq = np.dot(phi, phi)
    
    if theta_sq < 1e-8: # Small angle approximation
        k = 0.5
        # return quat_normalize(np.array([1.0, k*phi[0], k*phi[1], k*phi[2]], dtype=np.float64))
        # For small angles, q approx [1, phi/2]
        return np.array([1.0, k*phi[0], k*phi[1], k*phi[2]], dtype=np.float64)
    
    theta = np.sqrt(theta_sq)
    half_theta = 0.5 * theta
    s = np.sin(half_theta) / theta
    c = np.cos(half_theta)
    
    return np.array([c, s*phi[0], s*phi[1], s*phi[2]], dtype=np.float64)

def rot_from_quat(q):
    """
    Converts a unit quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (yy + zz)
    R[0, 1] = 2.0 * (xy - wz)
    R[0, 2] = 2.0 * (xz + wy)
    
    R[1, 0] = 2.0 * (xy + wz)
    R[1, 1] = 1.0 - 2.0 * (xx + zz)
    R[1, 2] = 2.0 * (yz - wx)
    
    R[2, 0] = 2.0 * (xz - wy)
    R[2, 1] = 2.0 * (yz + wx)
    R[2, 2] = 1.0 - 2.0 * (xx + yy)
    return R

# ==============================================================================
# ESKF Implementation
# ==============================================================================

class ESKF_INS:
    def __init__(self, dt,
                 Q_gyro=2e-4, Q_acc=4e-3,
                 Q_bg=1e-6, Q_ba=1e-5,
                 R_pos=None): # Changed default to None to allow dynamic type setting
        """
        Initializes the Error-State Kalman Filter for Inertial Navigation System.
        
        Args:
            dt (float): Time step in seconds.
            Q_gyro (float): Gyroscope noise variance.
            Q_acc (float): Accelerometer noise variance.
            Q_bg (float): Gyroscope bias random walk variance.
            Q_ba (float): Accelerometer bias random walk variance.
            R_pos (np.ndarray): Position measurement noise covariance matrix (3x3).
                                Defaults to 0.5^2 * I if not provided.
        """
        self.dt = dt
        self.g = np.array([0.0, 0.0, -9.81], dtype=np.float64) # Standard gravity vector

        # Nominal State (True state estimate)
        self.p = np.zeros(3, dtype=np.float64) # Position (m)
        self.v = np.zeros(3, dtype=np.float64) # Velocity (m/s)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # Orientation quaternion (w,x,y,z)
        self.bg = np.zeros(3, dtype=np.float64) # Gyroscope bias (rad/s)
        self.ba = np.zeros(3, dtype=np.float64) # Accelerometer bias (m/s^2)

        # Error State Covariance Matrix (15x15)
        self.P = np.eye(15, dtype=np.float64) * 1e-2 # Initial covariance (e.g., small uncertainty)

        # Process Noise Matrix (Continuous-time, for error state propagation)
        self.Q_c = np.zeros((12, 12), dtype=np.float64)
        np.fill_diagonal(self.Q_c[0:3, 0:3], Q_gyro) # Gyro noise
        np.fill_diagonal(self.Q_c[3:6, 3:6], Q_acc)  # Accelerometer noise
        np.fill_diagonal(self.Q_c[6:9, 6:9], Q_bg)   # Gyro bias random walk
        np.fill_diagonal(self.Q_c[9:12, 9:12], Q_ba) # Accelerometer bias random walk

        # Measurement Noise Covariance for position updates
        if R_pos is None:
            self.R_pos = 0.5**2 * np.eye(3, dtype=np.float64)
        else:
            self.R_pos = R_pos.astype(np.float64) # Ensure float64 if provided

        # Pre-allocate Jacobian Matrices for efficiency
        self.F = np.zeros((15, 15), dtype=np.float64) # State transition Jacobian
        self.G = np.zeros((15, 12), dtype=np.float64) # Noise Jacobian
        self.I15 = np.eye(15, dtype=np.float64)       # 15x15 Identity matrix
        
        # Static parts of G matrix (noise input mapping)
        self.G[6:9, 0:3]   = np.eye(3, dtype=np.float64)  # Angular velocity error from gyro noise
        self.G[9:12, 6:9]  = np.eye(3, dtype=np.float64)  # Gyro bias random walk input
        self.G[12:15, 9:12]= np.eye(3, dtype=np.float64)  # Accel bias random walk input

    def set_state(self, p, v, q, bg=None, ba=None, P=None):
        """
        Manually sets the nominal state and optionally the covariance.
        Ensures all inputs are converted to float64.
        """
        self.p = np.array(p, dtype=np.float64)
        self.v = np.array(v, dtype=np.float64)
        self.q = quat_normalize(np.array(q, dtype=np.float64))
        if bg is not None: self.bg = np.array(bg, dtype=np.float64)
        if ba is not None: self.ba = np.array(ba, dtype=np.float64)
        if P is not None:  self.P = np.array(P, dtype=np.float64)

    def predict(self, omega_meas, acc_meas):
        """
        IMU Prediction Step (Extended Kalman Filter).
        Propagates the nominal state using IMU measurements and updates the error covariance.
        
        Args:
            omega_meas (np.ndarray): Measured angular velocity (gyroscope).
            acc_meas (np.ndarray): Measured acceleration (accelerometer).
        """
        dt = self.dt

        # Ensure input types are float64
        omega_meas = omega_meas.astype(np.float64)
        acc_meas = acc_meas.astype(np.float64)

        # 1. Nominal State Propagation (Non-linear IMU kinematics)
        omega = omega_meas - self.bg # Corrected angular velocity
        acc   = acc_meas   - self.ba # Corrected acceleration

        # Orientation update using exponential map
        dq = so3_exp(omega * dt)
        self.q = quat_normalize(quat_mul(self.q, dq))
        R_wb = rot_from_quat(self.q) # World to body rotation matrix

        # Velocity and Position update (Semi-implicit Euler integration)
        a_world = R_wb @ acc + self.g # Acceleration in world frame + gravity
        self.v += a_world * dt
        self.p += self.v * dt

        # 2. Error State Jacobian Construction
        # F_t = d(delta_x_k+1) / d(delta_x_k)
        self.F.fill(0.0) # Reset F matrix for current step
        
        # Position error dynamics: delta_p_dot = delta_v
        self.F[0:3, 3:6] = self.I15[0:3, 0:3] # Identity 3x3

        # Velocity error dynamics: delta_v_dot = R_wb * (-acc_skew) * delta_theta - R_wb * delta_ba
        Sx_acc = skew(acc)
        R_Sx = R_wb @ Sx_acc
        self.F[3:6, 6:9]   = -R_Sx       # Rotation error influence on velocity
        self.F[3:6, 12:15] = -R_wb       # Accelerometer bias error influence

        # Orientation error dynamics: delta_theta_dot = -omega_skew * delta_theta - delta_bg
        self.F[6:9, 6:9]  = -skew(omega) # Angular velocity error influence on orientation
        self.F[6:9, 9:12] = -self.I15[0:3, 0:3] # Gyro bias error influence

        # Update G matrix (noise input mapping for current R_wb)
        self.G[3:6, 3:6] = R_wb # Accelerometer noise maps to world frame velocity error

        # 3. Covariance Propagation (using discrete-time approximation)
        # Phi = I + F*dt (First order approximation of state transition matrix for error state)
        Phi = self.I15 + self.F * dt
        
        # Qd = G_t * Q_c * G_t.T * dt (Discrete-time process noise covariance)
        Qd  = self.G @ self.Q_c @ self.G.T * dt
        
        self.P = Phi @ self.P @ Phi.T + Qd # Covariance update equation

    def update_vo(self, pos_meas):
        """
        Measurement Update (Position only) using Joseph Form for stability.
        Corrects the error state based on Visual Odometry (VO) or other position measurements.
        
        Args:
            pos_meas (np.ndarray): 3D position measurement (e.g., from VO).
        """
        # Ensure input type is float64
        pos_meas = pos_meas.astype(np.float64)
        
        z = pos_meas # Measurement
        y = z - self.p # Innovation (measurement residual)

        # H matrix for position measurement is implicitly [I_3x3 0_3x12]
        # S = H P H^T + R (Innovation covariance)
        P_pos = self.P[0:3, 0:3] # Extract position covariance block
        S = P_pos + self.R_pos
        
        # Calculate Kalman Gain: K = P H^T S^-1
        # Since H^T effectively selects the first 3 columns of P,
        # P_HT is P[:, 0:3]
        P_HT = self.P[:, 0:3]
        K = np.linalg.solve(S, P_HT.T).T # Efficiently solve K = P_HT @ S_inv

        # State Correction using Kalman Gain
        delta_x = K @ y # Error state correction vector (15x1)
        
        # Apply corrections to the nominal state
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        
        # Orientation correction (applying error rotation as a quaternion multiplication)
        dq_error = so3_exp(delta_x[6:9])
        self.q = quat_normalize(quat_mul(self.q, dq_error))
        
        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        # Covariance Update (Joseph Form: P = (I - KH) P (I - KH)^T + K R K^T)
        # Joseph form guarantees symmetry and positive definiteness, crucial for long-term stability.
        
        # Efficiently compute I_KH = I - K @ H
        # Since H is [I 0 0 0 0], K@H is just K placed into the first 3 columns of an identity matrix.
        I_KH = self.I15.copy()
        I_KH[:, 0:3] -= K # Subtract K from the identity block corresponding to position states
        
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_pos @ K.T