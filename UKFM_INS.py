"""
Vectorized Manifold Unscented Kalman Filter (UKF-M) for INS/VO Fusion.

This module implements a Manifold-aware Unscented Kalman Filter designed for 
Inertial Navigation Systems aiding with absolute position updates (e.g., Visual Odometry).
It utilizes efficient NumPy vectorization to handle sigma point propagation 
directly on the SO(3) x R3 manifold.
"""

import numpy as np

EPS = 1e-9

# ==============================================================================
# Vectorized Math Utilities
# ==============================================================================

def quat_normalize(q):
    """Normalizes a quaternion or an array of quaternions."""
    n = np.linalg.norm(q)
    return q if n < EPS else q / n

def quat_mult_vec(q1, q2):
    """
    Vectorized Hamilton product of quaternions.
    Supports broadcasting for sigma point operations.
    """
    q1_in = q1 if q1.ndim == 2 else q1[np.newaxis, :]
    q2_in = q2 if q2.ndim == 2 else q2[np.newaxis, :]
    
    w1, x1, y1, z1 = q1_in.T
    w2, x2, y2, z2 = q2_in.T
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    res = np.column_stack([w, x, y, z])
    
    if q1.ndim == 1 and q2.ndim == 1:
        return res.flatten()
    return res

def quat_conj(q):
    """Returns the conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def so3_exp_vec(rv):
    """
    Vectorized Exponential Map: Rotation Vector (so3) -> Quaternion (SO3).
    Includes small-angle approximation handling.
    """
    is_1d = (rv.ndim == 1)
    if is_1d:
        rv = rv[np.newaxis, :]
        
    ang = np.linalg.norm(rv, axis=1)
    mask_small = ang < 1e-12
    ang_safe = np.where(mask_small, 1.0, ang) 
    
    half_ang = 0.5 * ang
    s = np.sin(half_ang) / ang_safe
    c = np.cos(half_ang)
    
    q = np.zeros((len(rv), 4))
    q[:, 0] = c
    q[:, 1:] = rv * s[:, np.newaxis]
    
    # Small angle fallback
    q[mask_small] = np.array([1.0, 0.0, 0.0, 0.0])
    
    if is_1d:
        return q.flatten()
    return q

def so3_log_vec(q):
    """
    Vectorized Logarithm Map: Quaternion (SO3) -> Rotation Vector (so3).
    """
    is_1d = (q.ndim == 1)
    if is_1d:
        q = q[np.newaxis, :]

    w = q[:, 0]
    v = q[:, 1:]
    n = np.linalg.norm(v, axis=1)
    
    ang = 2.0 * np.arctan2(n, np.maximum(w, 1e-12))
    
    mask_small = ang < 1e-12
    n_safe = np.where(mask_small, 1.0, n)
    
    factor = ang / n_safe
    rv = v * factor[:, np.newaxis]
    
    rv[mask_small] = 0.0
    
    if is_1d:
        return rv.flatten()
    return rv

def rotate_vec(q, v):
    """Rotate vector v by quaternion q (Vectorized)."""
    q_in = q if q.ndim == 2 else q[np.newaxis, :]
    v_in = v if v.ndim == 2 else v[np.newaxis, :]
    
    q_w = q_in[:, 0:1]
    q_xyz = q_in[:, 1:]
    
    t = 2.0 * np.cross(q_xyz, v_in)
    v_rot = v_in + q_w * t + np.cross(q_xyz, t)
    
    if q.ndim == 1 and v.ndim == 1:
        return v_rot.flatten()
    return v_rot

# ==============================================================================
# UKF-M Implementation
# ==============================================================================

class UKF_M:
    def __init__(self, dt, Q_acc=4e-3, Q_gyro=2e-4, R_pos=0.5):
        """
        Initializes the Manifold UKF.
        
        Args:
            dt: Time step.
            Q_acc: Accelerometer process noise.
            Q_gyro: Gyroscope process noise.
            R_pos: Position measurement noise std dev (scalar) or covariance.
        """
        self.dt = dt
        
        # Nominal State: p(3), v(3), q(4), bg(3), ba(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1., 0., 0., 0.])
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        
        # Error Covariance (15x15)
        self.P = np.eye(15) * 1e-2
        
        # Process Noise Matrix
        self.Q = np.zeros((15, 15))
        self.Q[0:3, 0:3]   = 0.0
        self.Q[3:6, 3:6]   = np.eye(3) * Q_acc * dt
        self.Q[6:9, 6:9]   = np.eye(3) * Q_gyro * dt
        self.Q[9:12, 9:12] = np.eye(3) * 1e-6 * dt # Bias random walk
        self.Q[12:15, 12:15] = np.eye(3) * 1e-5 * dt

        # Measurement Noise Matrix
        self.R_pos = np.eye(3) * R_pos
        self.g = np.array([0, 0, -9.81])

        # Sigma Point Parameters (Van der Merwe)
        self.n = 15 # State dimension
        self.alpha = 1e-1
        self.beta = 2.0
        self.kappa = 0.0
        
        lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.lambda_ = lambda_
        
        # Weights
        self.Wm = np.full(2*self.n + 1, 0.5 / (self.n + lambda_))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lambda_ / (self.n + lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        
    @property
    def position(self):
        return self.p.copy()

    def set_init(self, p0, q0):
        """Initializes the filter state."""
        self.p = p0.copy()
        self.q = quat_normalize(q0.copy())
        self.P = np.eye(15) * 1e-2

    def predict(self, om, ac):
        """
        Prediction Step: Propagates sigma points through non-linear process model.
        
        Args:
            om: Angular velocity measurement [wx, wy, wz].
            ac: Acceleration measurement [ax, ay, az].
        """
        dt = self.dt
        n = self.n
        
        # 1. Sigma Point Generation (Cholesky Decomposition)
        try:
            S = np.linalg.cholesky((n + self.lambda_) * (self.P + self.Q))
        except np.linalg.LinAlgError:
            # Fallback stabilization
            self.P += np.eye(15) * 1e-6
            S = np.linalg.cholesky((n + self.lambda_) * (self.P + self.Q))
            
        dX = np.zeros((2*n + 1, n))
        dX[1:n+1] = S.T
        dX[n+1:]  = -S.T
        
        # 2. Sigma Point Propagation on Manifold
        # Orientation perturbation
        d_theta = dX[:, 6:9]
        dq = so3_exp_vec(d_theta)
        q_pts = quat_mult_vec(self.q, dq)
        
        # Euclidean perturbations
        p_pts = self.p + dX[:, 0:3]
        v_pts = self.v + dX[:, 3:6]
        bg_pts = self.bg + dX[:, 9:12]
        ba_pts = self.ba + dX[:, 12:15]
        
        # IMU Model Integration
        omega_corr = om - bg_pts
        acc_corr   = ac - ba_pts
        
        # Integrate orientation
        dq_dt = so3_exp_vec(omega_corr * dt)
        q_pred = quat_mult_vec(q_pts, dq_dt)
        
        # Integrate velocity and position
        acc_world = rotate_vec(q_pred, acc_corr)
        v_pred = v_pts + (acc_world + self.g) * dt
        p_pred = p_pts + v_pred * dt
        
        bg_pred = bg_pts
        ba_pred = ba_pts
        
        # 3. Mean Reconstruction
        self.p = np.dot(self.Wm, p_pred)
        self.v = np.dot(self.Wm, v_pred)
        self.bg = np.dot(self.Wm, bg_pred)
        self.ba = np.dot(self.Wm, ba_pred)
        
        # Quaternion Mean (Iterative approximation using one pass)
        q_mean = q_pred[0]
        e_rot = so3_log_vec(quat_mult_vec(quat_conj(q_mean), q_pred))
        e_avg = np.dot(self.Wm, e_rot)
        self.q = quat_normalize(quat_mult_vec(q_mean, so3_exp_vec(e_avg)))
        
        # 4. Covariance Reconstruction
        dP = p_pred - self.p
        dV = v_pred - self.v
        dQ = so3_log_vec(quat_mult_vec(quat_conj(self.q), q_pred))
        dBg = bg_pred - self.bg
        dBa = ba_pred - self.ba
        
        dY = np.hstack([dP, dV, dQ, dBg, dBa])
        
        self.P = np.dot(self.Wc * dY.T, dY)
        
    def update(self, pos_meas):
        """
        Measurement Update Step.
        
        Args:
            pos_meas: 3D position measurement (VO/GPS).
        """
        if pos_meas is None: return
        
        z_res = pos_meas - self.p
        
        # Innovation Covariance
        S = self.P[0:3, 0:3] + self.R_pos
        
        # Kalman Gain: K = P @ H.T @ inv(S)
        K = np.linalg.solve(S, self.P[0:3, :]).T
        
        dx = K @ z_res
        
        # State Update (Retraction)
        self.p  += dx[0:3]
        self.v  += dx[3:6]
        self.q   = quat_normalize(quat_mult_vec(self.q, so3_exp_vec(dx[6:9])))
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        
        # Covariance Update
        self.P = self.P - K @ S @ K.T