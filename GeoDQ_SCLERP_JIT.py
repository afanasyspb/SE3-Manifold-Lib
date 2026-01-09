"""
High-Performance Dual Quaternion SCLERP Filter for 6DoF Pose Estimation.
(Numba JIT Optimized Version)

This module implements the Geometric State Fusion algorithm using Dual Quaternions
and Screw Linear Interpolation (SCLERP) directly on the SE(3) manifold.

Hypothesis: Geometric Algebra operations (DQ) benefit significantly more 
from JIT compilation than Matrix operations (BLAS/LAPACK backed), as they 
involve numerous small scalar operations that are costly in interpreted Python.
"""

import numpy as np
from numba import jit, float64

EPS = 1e-9

# ==============================================================================
# JIT-Compiled Math Kernels (Stateless, No Python Overhead)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def jit_normalize_vec(v):
    """Normalize a vector with zero-division check."""
    n = np.linalg.norm(v)
    if n < EPS: return v
    return v / n

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_mult(q1, q2):
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_normalize(q):
    """Normalizes a quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < EPS: return q
    return q / n

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_from_rotvec(rv):
    """Converts a rotation vector to a unit quaternion."""
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half_ang = 0.5 * ang
    s = np.sin(half_ang) / ang
    c = np.cos(half_ang)
    return np.array([c, s*rv[0], s*rv[1], s*rv[2]], dtype=np.float64)

@jit(nopython=True, cache=True, fastmath=True)
def jit_so3_log(q):
    """Logarithmic map: Quaternion (SO3) -> Rotation Vector (so3)."""
    w = q[0]
    n = np.linalg.norm(q[1:])
    ang = 2.0 * np.arctan2(n, max(w, 1e-12))
    if ang < 1e-12:
        return np.zeros(3, dtype=np.float64)
    factor = ang / n
    return q[1:] * factor

@jit(nopython=True, cache=True, fastmath=True)
def jit_rot_from_quat(q):
    """Converts a quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = np.zeros((3, 3), dtype=np.float64)
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
# JIT Dual Quaternion Operations
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_normalize(qr, qd):
    """Normalizes a dual quaternion (unit real part, orthogonal dual part)."""
    qr_n = jit_quat_normalize(qr)
    dot = np.dot(qr_n, qd)
    qd_n = qd - dot * qr_n
    return qr_n, qd_n

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_mul(qr1, qd1, qr2, qd2):
    """Dual Quaternion multiplication."""
    qr_new = jit_quat_mult(qr1, qr2)
    qd_new = jit_quat_mult(qr1, qd2) + jit_quat_mult(qd1, qr2)
    return qr_new, qd_new

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_from_pose(qr, p):
    """Constructs a Dual Quaternion from rotation and position."""
    qr = jit_quat_normalize(qr)
    w2, x2, y2, z2 = qr
    x1, y1, z1 = p
    
    # Dual part = 0.5 * (p * qr)
    qd_w = -x1*x2 - y1*y2 - z1*z2
    qd_x =  x1*w2 + y1*z2 - z1*y2
    qd_y = -x1*z2 + y1*w2 + z1*x2
    qd_z =  x1*y2 - y1*x2 + z1*w2
    
    qd = 0.5 * np.array([qd_w, qd_x, qd_y, qd_z], dtype=np.float64)
    return jit_dq_normalize(qr, qd)

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_topose(qr, qd):
    """Extracts position from a Dual Quaternion."""
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]], dtype=np.float64)
    # p = 2 * qd * qr'
    pq = jit_quat_mult(qd, qr_c)
    return 2.0 * pq[1:]

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_inv(qr, qd):
    """Computes the inverse (conjugate) of a dual quaternion."""
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]], dtype=np.float64)
    qd_c = np.array([qd[0], -qd[1], -qd[2], -qd[3]], dtype=np.float64)
    return qr_c, qd_c

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_log_vec(qr, qd):
    """Logarithmic map: DQ -> Twist (6D vector)."""
    phi = jit_so3_log(qr)
    p = jit_dq_topose(qr, qd)
    return phi, p * 0.5

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_exp_vec(phi, ups):
    """Exponential map: Twist -> DQ."""
    qr = jit_quat_from_rotvec(phi)
    return jit_dq_from_pose(qr, 2.0 * ups)

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_sclerp_kernel(qr1, qd1, qr2, qd2, alpha):
    """
    Screw Linear Interpolation (ScLERP) between two Dual Quaternions.
    Represents the shortest path on the SE(3) manifold.
    """
    if alpha <= 0: return qr1, qd1
    if alpha >= 1: return qr2, qd2
    
    qr1_c, qd1_c = jit_dq_inv(qr1, qd1)
    rel_r, rel_d = jit_dq_mul(qr1_c, qd1_c, qr2, qd2)
    rel_r, rel_d = jit_dq_normalize(rel_r, rel_d)
    
    phi, ups = jit_dq_log_vec(rel_r, rel_d)
    exp_r, exp_d = jit_dq_exp_vec(alpha * phi, alpha * ups)
    
    res_r, res_d = jit_dq_mul(qr1, qd1, exp_r, exp_d)
    return jit_dq_normalize(res_r, res_d)

@jit(nopython=True, cache=True, fastmath=True)
def jit_robust_weight(r, c):
    """Huber weight function."""
    return 1.0 if r <= c else c / r

# ==============================================================================
# JIT Logic Kernels (Predict & Update)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def kernel_predict(qr, qd, vel_est, acc_filt, bw, ba, om, ac, dt, gravity):
    """
    IMU Prediction step on SE(3).
    Integrates angular velocity and acceleration using the manifold structure.
    """
    # Bias correction
    omega_corr = om - bw
    acc_corr = ac - ba
    
    # Orientation
    R_wb = jit_rot_from_quat(qr)
    
    # Global Velocity
    acc_world = R_wb @ acc_corr
    lin_acc = acc_world + gravity
    
    # Low-pass filter on acceleration
    acc_filt_new = 0.25 * lin_acc + 0.75 * acc_filt
    vel_est_new = vel_est + acc_filt_new * dt
    
    # Twist Propagation (Body Frame)
    vel_body = R_wb.T @ vel_est_new
    
    phi = omega_corr * dt
    rho = vel_body * dt
    
    # Manifold Integration: q_new = q_old * exp(twist)
    exp_r, exp_d = jit_dq_exp_vec(phi, rho/2.0)
    qr_new, qd_new = jit_dq_mul(qr, qd, exp_r, exp_d)
    qr_new, qd_new = jit_dq_normalize(qr_new, qd_new)
    
    return qr_new, qd_new, vel_est_new, acc_filt_new

@jit(nopython=True, cache=True, fastmath=True)
def kernel_update(qr, qd, vel_est, ba, ppose_ema, pos_meas, dt, 
                  alpha_fuse, kp_vel, ki_ba, huber_c, vo_sigma, alpha_pose):
    """
    Measurement Update step.
    Fuses position measurement using SCLERP and updates biases via PID feedback.
    """
    
    p_curr = jit_dq_topose(qr, qd)
    
    # Innovation
    pos_innov = pos_meas - p_curr
    norm = np.linalg.norm(pos_innov)
    
    # Adaptive Weighting
    w_rob = jit_robust_weight(norm, huber_c)
    # Scale alpha by VO uncertainty
    gw = min(1.0 / vo_sigma, 2.0)
    alpha = min(max(alpha_fuse * w_rob * gw, 0.05), 0.8)
    
    # SCLERP Fusion (Geometric Correction)
    qr_vo, qd_vo = jit_dq_from_pose(qr, pos_meas)
    qr_new, qd_new = jit_dq_sclerp_kernel(qr, qd, qr_vo, qd_vo, alpha)
    
    # PID Feedback Loop
    max_innov = 1.0
    if norm > max_innov:
        pos_innov_clamped = pos_innov * (max_innov / norm)
    else:
        pos_innov_clamped = pos_innov
        
    # P-term: Velocity Correction
    dv = kp_vel * (pos_innov_clamped / max(dt, 1e-3))
    dv_clamped = np.zeros(3)
    for k in range(3):
        dv_clamped[k] = min(max(dv[k], -2.0), 2.0)
    
    vel_est_new = vel_est + dv_clamped
    for k in range(3):
        vel_est_new[k] = min(max(vel_est_new[k], -20.0), 20.0)
        
    # I-term: Accelerometer Bias Correction
    R = jit_rot_from_quat(qr_new)
    pos_innov_body = R.T @ pos_innov_clamped
    
    ba_new = ba - pos_innov_body * ki_ba
    for k in range(3):
        ba_new[k] = min(max(ba_new[k], -0.5), 0.5)
        
    # EMA Smoothing for output (does not affect internal state)
    p_new = jit_dq_topose(qr_new, qd_new)
    ppose_ema_new = alpha_pose * p_new + (1.0 - alpha_pose) * ppose_ema
    
    # Re-inject smoothed position into state for display stability
    qr_final, qd_final = jit_dq_from_pose(qr_new, ppose_ema_new)
    
    return qr_final, qd_final, vel_est_new, ba_new, ppose_ema_new

# ==============================================================================
# Filter Class Wrapper
# ==============================================================================

class DQFilterJIT:
    def __init__(self, dt, alpha_fuse=0.35, kp_vel=0.8, ki_ba=0.02, huber_c=2.0):
        """
        Initializes the JIT-optimized Dual Quaternion Filter.
        """
        self.dt = dt
        self.alpha_fuse = alpha_fuse
        self.kp_vel = kp_vel
        self.ki_ba = ki_ba
        self.huber_c = huber_c
        self.vo_sigma = 1.5 # Uncertainty factor for Visual Odometry
        self.alpha_pose = 0.1
        
        # State
        self.qr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.qd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.bw = np.zeros(3, dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64)
        self.vel_est = np.zeros(3, dtype=np.float64)
        self.acc_filt = np.zeros(3, dtype=np.float64)
        self.ppose_ema = np.zeros(3, dtype=np.float64)
        
        self.GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    @property
    def position(self):
        return jit_dq_topose(self.qr, self.qd)
        
    def initialize(self, q0, p0):
        """
        Initialize state. Explicitly casts to float64 to ensure JIT compatibility.
        """
        self.qr, self.qd = jit_dq_from_pose(
            jit_quat_normalize(q0.astype(np.float64)), 
            p0.astype(np.float64)
        )
        self.ppose_ema = p0.copy().astype(np.float64)
        self.bw.fill(0)
        self.ba.fill(0)
        self.acc_filt.fill(0)
        self.vel_est.fill(0)
        
        # Warmup JIT kernels
        try:
            self.step(np.zeros(3), np.zeros(3), None)
        except:
            pass

    def step(self, om, ac, pos_meas=None):
        """
        Main filter step.
        
        Args:
            om: Angular velocity [rad/s]
            ac: Acceleration [m/s^2]
            pos_meas: Position measurement [m] (VO/GPS)
        """
        # Ensure float64
        om = om.astype(np.float64)
        ac = ac.astype(np.float64)
        
        # Predict
        self.qr, self.qd, self.vel_est, self.acc_filt = kernel_predict(
            self.qr, self.qd, self.vel_est, self.acc_filt, 
            self.bw, self.ba, om, ac, self.dt, self.GRAVITY
        )
        
        # Update (if measurement available)
        if pos_meas is not None:
            pos_meas = pos_meas.astype(np.float64)
            self.qr, self.qd, self.vel_est, self.ba, self.ppose_ema = kernel_update(
                self.qr, self.qd, self.vel_est, self.ba, self.ppose_ema, pos_meas, self.dt,
                self.alpha_fuse, self.kp_vel, self.ki_ba, self.huber_c, self.vo_sigma, self.alpha_pose
            )