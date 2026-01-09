"""
High-Performance Dual Quaternion SCLERP Filter for 6DoF Pose Estimation.
Updated with PID-based Bias Estimation (Observer/Feedback Loop).

This module implements the Geometric State Fusion algorithm using Dual Quaternions
and Screw Linear Interpolation (SCLERP) directly on the SE(3) manifold.
It utilizes vectorized NumPy operations for maximum efficiency in real-time applications.
"""

import numpy as np

EPS = 1e-9

# ==============================================================================
# Mathematical Utilities: SO(3) and Quaternions
# ==============================================================================

def normalize_vec(v):
    """Normalize a vector with zero-division check."""
    n = np.linalg.norm(v)
    return v if n < EPS else v / n

def quat_mult(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_conj(q):
    """Conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_normalize(q):
    """Normalize a quaternion."""
    n = np.linalg.norm(q)
    return q if n < EPS else q / n

def quat_from_rotvec(rv):
    """Convert rotation vector (axis-angle) to quaternion."""
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    half_ang = 0.5 * ang
    s = np.sin(half_ang) / ang
    c = np.cos(half_ang)
    
    return np.array([c, s*rv[0], s*rv[1], s*rv[2]], dtype=np.float64)

def so3_log(q):
    """Logarithmic map of SO(3): Quaternion -> Rotation Vector."""
    w = q[0]
    n = np.linalg.norm(q[1:])
    
    ang = 2.0 * np.arctan2(n, max(w, 1e-12))
    if ang < 1e-12:
        return np.zeros(3, dtype=np.float64)
    
    factor = ang / n
    return q[1:] * factor

def rot_from_quat(q):
    """Convert quaternion to 3x3 rotation matrix."""
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
# Dual Quaternion Operations (Stateless Functions)
# ==============================================================================

def dq_normalize(qr, qd):
    """Normalize dual quaternion components."""
    qr_n = quat_normalize(qr)
    # Orthogonalize dual part: qd = qd - dot(qr, qd) * qr
    dot = np.dot(qr_n, qd)
    qd_n = qd - dot * qr_n
    return qr_n, qd_n

def dq_mul(qr1, qd1, qr2, qd2):
    """Multiply two dual quaternions."""
    qr_new = quat_mult(qr1, qr2)
    qd_new = quat_mult(qr1, qd2) + quat_mult(qd1, qr2)
    return qr_new, qd_new

def dq_from_pose(qr, p):
    """Construct DQ components from rotation quaternion and position vector."""
    qr = quat_normalize(qr)
    
    # 0.5 * (p_quat * qr) inline calculation
    w2, x2, y2, z2 = qr
    x1, y1, z1 = p
    
    qd_w = -x1*x2 - y1*y2 - z1*z2
    qd_x =  x1*w2 + y1*z2 - z1*y2
    qd_y = -x1*z2 + y1*w2 + z1*x2
    qd_z =  x1*y2 - y1*x2 + z1*w2
    
    qd = 0.5 * np.array([qd_w, qd_x, qd_y, qd_z], dtype=np.float64)
    
    return dq_normalize(qr, qd)

def dq_topose(qr, qd):
    """Extract position vector from DQ components."""
    # p = 2 * qd * qr_conj
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]], dtype=np.float64)
    pq = quat_mult(qd, qr_c)
    return 2.0 * pq[1:]

def dq_inv(qr, qd):
    """Inverse (Conjugate) of a dual quaternion."""
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]], dtype=np.float64)
    qd_c = np.array([qd[0], -qd[1], -qd[2], -qd[3]], dtype=np.float64)
    return qr_c, qd_c

def dq_log_vec(qr, qd):
    """Logarithmic map returning 6D twist vector [phi, ups]."""
    phi = so3_log(qr)
    p = dq_topose(qr, qd)
    return phi, p * 0.5

def dq_exp_vec(phi, ups):
    """Exponential map taking 6D twist and returning DQ components."""
    qr = quat_from_rotvec(phi)
    return dq_from_pose(qr, 2.0 * ups)

def dq_sclerp_kernel(qr1, qd1, qr2, qd2, alpha):
    """
    Perform Screw Linear Interpolation (SCLERP) between two poses.
    This effectively interpolates along the geodesic on the SE(3) manifold.
    """
    if alpha <= 0: return qr1, qd1
    if alpha >= 1: return qr2, qd2
    
    # 1. Relative pose: rel = q1_inv * q2
    qr1_c, qd1_c = dq_inv(qr1, qd1)
    rel_r, rel_d = dq_mul(qr1_c, qd1_c, qr2, qd2)
    rel_r, rel_d = dq_normalize(rel_r, rel_d)
    
    # 2. Log map: xi = log(rel)
    phi, ups = dq_log_vec(rel_r, rel_d)
    
    # 3. Scale twist: exp(alpha * xi)
    exp_r, exp_d = dq_exp_vec(alpha * phi, alpha * ups)
    
    # 4. Composition: result = q1 * exp(...)
    res_r, res_d = dq_mul(qr1, qd1, exp_r, exp_d)
    return dq_normalize(res_r, res_d)

def robust_weight(r, c=2.0):
    """Huber weight function for robust estimation."""
    return 1.0 if r <= c else c / r

# ==============================================================================
# Filter Class
# ==============================================================================

class DQFilter:
    def __init__(self, dt, 
                 alpha_fuse=0.35,  # Basic geometric interpolation gain
                 kp_vel=0.8,       # Proportional gain for velocity correction
                 ki_ba=0.02,       # Integral gain for Accel Bias estimation
                 huber_c=2.0, 
                 vo_sigma=1.5):    # Replaced gps_sigma with vo_sigma
        """
        Initialize the Dual Quaternion SCLERP Filter (Geometric Observer).
        
        Args:
            dt: Time step (seconds).
            alpha_fuse: Geometric interpolation strength (0.0 to 1.0).
            kp_vel: P-gain for velocity correction based on position error.
            ki_ba: I-gain for accelerometer bias estimation.
            huber_c: Robust kernel threshold for outlier rejection.
            vo_sigma: Scaling factor based on Visual Odometry uncertainty.
        """
        self.dt = dt
        self.alpha_fuse = max(alpha_fuse, 0.05)
        
        # PID / Observer Gains
        self.kp_vel = kp_vel
        self.ki_ba = ki_ba
        
        # Robustness parameters
        self.huber_c = huber_c
        self.vo_sigma = vo_sigma
        self.alpha_pose = 0.1 # For output EMA smoothing
        
        # State variables
        self.qr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # Real part (Rotation)
        self.qd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64) # Dual part (Translation)
        
        # Biases and Kinematics
        self.bw = np.zeros(3, dtype=np.float64) # Gyro Bias (rad/s)
        self.ba = np.zeros(3, dtype=np.float64) # Accel Bias (m/s^2)
        
        self.acc_world_filt = np.zeros(3, dtype=np.float64)
        self.vel_est = np.zeros(3, dtype=np.float64)
        
        self.ppose_ema = None
        
        # Standard gravity vector
        self.GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    @property
    def position(self):
        """Current estimated position."""
        return dq_topose(self.qr, self.qd)
        
    def initialize(self, q0, p0):
        """Set initial state."""
        self.qr, self.qd = dq_from_pose(q0, p0)
        self.ppose_ema = p0.copy()
        self.bw.fill(0)
        self.ba.fill(0)
        self.acc_world_filt.fill(0)
        self.vel_est.fill(0)
        
    def predict(self, om, ac):
        """
        Prediction Stage (IMU Integration).
        Applies bias correction and integrates kinematics on SE(3).
        """
        dt = self.dt
        
        # 1. Bias correction (Crucial for eliminating steady-state error)
        omega_corr = om - self.bw
        acc_corr = ac - self.ba
        
        # 2. Orientation (Rotation Matrix)
        R_wb = rot_from_quat(self.qr)
        
        # 3. Global Velocity Update
        # Project acceleration to world frame and compensate gravity
        acc_world = R_wb @ acc_corr
        lin_acc = acc_world + self.GRAVITY
        
        # Simple Low-Pass on acceleration to smooth out vibration
        self.acc_world_filt = 0.25 * lin_acc + (1 - 0.25) * self.acc_world_filt
        self.vel_est += self.acc_world_filt * dt
        
        # 4. Twist Propagation (Body Frame)
        # Transform global velocity to body frame for correct screw motion
        # Twist = [omega, v_body]
        vel_body = R_wb.T @ self.vel_est
        
        phi = omega_corr * dt
        rho = vel_body * dt
        
        # Integration on the manifold: q_new = q_old * exp(twist)
        exp_r, exp_d = dq_exp_vec(phi, rho/2.0)
        self.qr, self.qd = dq_mul(self.qr, self.qd, exp_r, exp_d)
        self.qr, self.qd = dq_normalize(self.qr, self.qd)
        
    def update(self, pos_meas, vo_quat=None):
        """
        Correction Stage using SCLERP Fusion and PID-based Feedback.
        
        Args:
            pos_meas: Position measurement from VO.
            vo_quat: Optional orientation from VO (not always available).
        """
        if pos_meas is None:
            return

        # Current position
        p_curr = dq_topose(self.qr, self.qd)
        
        # Handle cases with only position (VO) by using predicted orientation
        if vo_quat is None:
            vo_quat = self.qr 

        # 1. Innovation (Error in World Frame)
        pos_innov = pos_meas - p_curr
        norm = np.linalg.norm(pos_innov)

        # 2. Adaptive Weighting for SCLERP
        # Huber weight reduces influence of outliers
        w_rob = robust_weight(norm, self.huber_c)
        # Scale alpha by VO confidence (heuristic)
        gain_scale = min(1.0 / self.vo_sigma, 2.0)
        alpha = np.clip(self.alpha_fuse * w_rob * gain_scale, 0.05, 0.8)

        # 3. Geometric Fusion (SCLERP)
        # This smoothly pulls the pose towards the measurement on the manifold
        qr_vo, qd_vo = dq_from_pose(vo_quat, pos_meas)
        self.qr, self.qd = dq_sclerp_kernel(self.qr, self.qd, qr_vo, qd_vo, alpha)
        
        # --- 4. PID-like Feedback for Velocity and Bias ---
        
        # Clamp innovation for stability in the feedback loop
        max_innov = 1.0
        if norm > max_innov:
             pos_innov_clamped = pos_innov * (max_innov / norm)
        else:
             pos_innov_clamped = pos_innov
        
        # A. Proportional (P) Term -> Correct Velocity
        # If we are behind, speed up. If ahead, slow down.
        # gain / dt gives the correction rate.
        dv = self.kp_vel * (pos_innov_clamped / max(self.dt, 1e-3))
        dv = np.clip(dv, -2.0, 2.0) # Safety clamping
        self.vel_est = np.clip(self.vel_est + dv, -20.0, 20.0) # Apply to state

        # B. Integral (I) Term -> Correct Accel Bias
        # If position error persists, it means we have a bias in acceleration.
        # Rotate error to Body Frame because Bias is sensor-fixed.
        R = rot_from_quat(self.qr)
        pos_innov_body = R.T @ pos_innov_clamped
        
        # Update Bias: ba -= Ki * error
        # Logic: If we are behind (innov > 0), actual acc > measured acc (after bias removal).
        # We need to subtract less bias. So ba should decrease.
        self.ba -= pos_innov_body * self.ki_ba
        
        # Optional: Limit bias growth
        self.ba = np.clip(self.ba, -0.5, 0.5)

        # 5. Output Smoothing (EMA)
        # Smooths the output for visualization, does not affect internal state
        p_new = dq_topose(self.qr, self.qd)
        
        if self.ppose_ema is None:
            self.ppose_ema = p_new.copy()
        else:
            self.ppose_ema = self.alpha_pose * p_new + (1 - self.alpha_pose) * self.ppose_ema
        
        # Re-inject smoothed position into state to reduce jitter
        self.qr, self.qd = dq_from_pose(self.qr, self.ppose_ema)

    def step(self, om, ac, pos_meas=None, mag=None):
        """
        Single simulation step wrapper.
        
        Args:
            om: Angular velocity [rad/s]
            ac: Acceleration [m/s^2]
            pos_meas: Position measurement [m] (from VO/GPS)
        """
        self.predict(om, ac)
        self.update(pos_meas)
        return None