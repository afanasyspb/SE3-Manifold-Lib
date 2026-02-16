"""
High-Performance Dual Quaternion SCLERP Filter for 6DoF Pose Estimation.
(Numba JIT Optimized Version)

This module implements the Geometric State Fusion algorithm using Dual Quaternions
and Screw Linear Interpolation (SCLERP) directly on the SE(3) manifold.

Key Features:
1. Direct SE(3) manifold operations (Log/Exp maps).
2. Orientation Feedback Loop (Tilt Correction).
3. Optimized Small Angle Approximation (Trigonometry-free update loop).
4. JIT-compiled kernels for high-frequency execution.

Author: Ilya Afanasyev
"""

import numpy as np
from numba import jit

EPS = 1e-9

# ==============================================================================
# JIT-Compiled Math Kernels (Helpers)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def jit_normalize_vec(v):
    n = np.linalg.norm(v)
    if n < EPS: return v
    return v / n

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_mult(q1, q2):
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
    n = np.linalg.norm(q)
    if n < EPS: return q
    return q / n

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_from_rotvec(rv):
    """Converts a rotation vector to a quaternion."""
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half_ang = 0.5 * ang
    s = np.sin(half_ang) / ang
    c = np.cos(half_ang)
    return np.array([c, s*rv[0], s*rv[1], s*rv[2]], dtype=np.float64)

@jit(nopython=True, cache=True, fastmath=True)
def jit_so3_log(q):
    """Logarithmic map SO(3) -> R3."""
    w = q[0]
    n = np.linalg.norm(q[1:])
    ang = 2.0 * np.arctan2(n, max(w, 1e-12))
    if ang < 1e-12:
        return np.zeros(3, dtype=np.float64)
    factor = ang / n
    return q[1:] * factor

@jit(nopython=True, cache=True, fastmath=True)
def jit_rot_from_quat(q):
    """Converts quaternion to 3x3 rotation matrix."""
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
    """Enforces unit condition on Dual Quaternion."""
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
    """Constructs DQ from Rotation Quaternion and Translation Vector."""
    qr = jit_quat_normalize(qr)
    w2, x2, y2, z2 = qr
    x1, y1, z1 = p
    
    # qd = 0.5 * (t * qr)
    qd_w = -x1*x2 - y1*y2 - z1*z2
    qd_x =  x1*w2 + y1*z2 - z1*y2
    qd_y = -x1*z2 + y1*w2 + z1*x2
    qd_z =  x1*y2 - y1*x2 + z1*w2
    qd = 0.5 * np.array([qd_w, qd_x, qd_y, qd_z], dtype=np.float64)
    return jit_dq_normalize(qr, qd)

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_topose(qr, qd):
    """Extracts Translation Vector from DQ."""
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]], dtype=np.float64)
    pq = jit_quat_mult(qd, qr_c)
    return 2.0 * pq[1:]

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_inv(qr, qd):
    """Dual Quaternion Conjugate (Inverse for unit DQs)."""
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]], dtype=np.float64)
    qd_c = np.array([qd[0], -qd[1], -qd[2], -qd[3]], dtype=np.float64)
    return qr_c, qd_c

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_log_vec(qr, qd):
    """Logarithmic Map SE(3) -> se(3)."""
    phi = jit_so3_log(qr)
    p = jit_dq_topose(qr, qd)
    return phi, p * 0.5

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_exp_vec(phi, ups):
    """Exponential Map se(3) -> SE(3)."""
    qr = jit_quat_from_rotvec(phi)
    return jit_dq_from_pose(qr, 2.0 * ups)

@jit(nopython=True, cache=True, fastmath=True)
def jit_dq_sclerp_kernel(qr1, qd1, qr2, qd2, alpha):
    """Screw Linear Interpolation (SCLERP) between two DQs."""
    if alpha <= 0: return qr1, qd1
    if alpha >= 1: return qr2, qd2
    
    # Calculate relative transform: DQ_rel = DQ1* @ DQ2
    qr1_c, qd1_c = jit_dq_inv(qr1, qd1)
    rel_r, rel_d = jit_dq_mul(qr1_c, qd1_c, qr2, qd2)
    rel_r, rel_d = jit_dq_normalize(rel_r, rel_d)
    
    # Log map to tangent space, scale, Exp map back
    phi, ups = jit_dq_log_vec(rel_r, rel_d)
    exp_r, exp_d = jit_dq_exp_vec(alpha * phi, alpha * ups)
    
    # Apply to start: DQ_res = DQ1 @ DQ_exp
    res_r, res_d = jit_dq_mul(qr1, qd1, exp_r, exp_d)
    return jit_dq_normalize(res_r, res_d)

# ==============================================================================
# JIT Logic Kernels (Predict & Update)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def kernel_predict(qr, qd, vel_est, acc_filt, bw, ba, om, ac, dt, gravity):
    """
    IMU Prediction Step.
    Integrates angular velocity and linear acceleration on the manifold.
    """
    # 1. Bias correction
    omega_corr = om - bw
    acc_corr = ac - ba
    
    # 2. Orientation (Quaternion to Matrix)
    R_wb = jit_rot_from_quat(qr)
    
    # 3. Global Velocity Integration
    acc_world = R_wb @ acc_corr
    lin_acc = acc_world + gravity
    
    # Low-pass filter for acceleration to reduce jitter
    acc_filt_new = 0.25 * lin_acc + 0.75 * acc_filt
    vel_est_new = vel_est + acc_filt_new * dt
    
    # 4. Twist Propagation (se(3) integration)
    vel_body = R_wb.T @ vel_est_new
    phi = omega_corr * dt
    rho = vel_body * dt
    
    exp_r, exp_d = jit_dq_exp_vec(phi, rho/2.0)
    qr_new, qd_new = jit_dq_mul(qr, qd, exp_r, exp_d)
    qr_new, qd_new = jit_dq_normalize(qr_new, qd_new)
    
    return qr_new, qd_new, vel_est_new, acc_filt_new

@jit(nopython=True, cache=True, fastmath=True)
def kernel_update(qr, qd, vel_est, ba, ppose_ema, pos_meas, dt, 
                  alpha_fuse, kp_vel, ki_ba, huber_c, vo_sigma, alpha_pose):
    """
    Measurement Update Step.
    Fuses external position measurement using SCLERP and Feedback loops.
    """
    # 1. Calculate Innovation
    p_curr = jit_dq_topose(qr, qd)
    
    # Manual unrolling for performance
    dx = pos_meas[0] - p_curr[0]
    dy = pos_meas[1] - p_curr[1]
    dz = pos_meas[2] - p_curr[2]
    
    err_sq = dx*dx + dy*dy + dz*dz
    norm = np.sqrt(err_sq)
    
    # ==========================================================================
    # OPTIMIZED TILT CORRECTION (Small Angle Approximation)
    # ==========================================================================
    # Generates a correction quaternion based on the error vector relative to UP.
    # Axis = UP x Error = [-dy, dx, 0]. Small angle q ~ [1, ang_x/2, ang_y/2, 0].
    
    tilt_gain = 0.05 
    half_gain = 0.5 * tilt_gain
    
    qx = -dy * half_gain
    qy =  dx * half_gain
    # qz = 0, qw = 1.0 (approximated)
    
    # Manual quaternion multiplication (q_corr * qr) to avoid array allocation
    w, x, y, z = qr
    
    qr_corrected_w = w - qx*x - qy*y
    qr_corrected_x = x + qx*w + qy*z
    qr_corrected_y = y - qx*z + qy*w
    qr_corrected_z = z + qx*y - qy*x
    
    # Fast re-normalization (Inverse Square Root style)
    q_norm_sq = (qr_corrected_w**2 + qr_corrected_x**2 + 
                 qr_corrected_y**2 + qr_corrected_z**2)
    
    if q_norm_sq > 1e-12:
        inv_n = 1.0 / np.sqrt(q_norm_sq)
        qr_c = np.array([
            qr_corrected_w * inv_n,
            qr_corrected_x * inv_n,
            qr_corrected_y * inv_n,
            qr_corrected_z * inv_n
        ], dtype=np.float64)
    else:
        qr_c = qr
        
    # ==========================================================================

    # 2. SCLERP Fusion with Adaptive Weighting (Huber)
    w_rob = 1.0 if norm <= huber_c else huber_c / norm
    gw = min(1.0 / vo_sigma, 2.0)
    alpha = min(max(alpha_fuse * w_rob * gw, 0.05), 0.8)
    
    qr_vo, qd_vo = jit_dq_from_pose(qr_c, pos_meas)
    qr_new, qd_new = jit_dq_sclerp_kernel(qr_c, qd, qr_vo, qd_vo, alpha)
    
    # 3. PID Feedback Loop Setup
    # Limits innovation magnitude to prevent destabilization on jumps
    max_innov = 1.0
    scale = 1.0
    if norm > max_innov:
        scale = max_innov / norm
        
    # P-term: Direct Velocity Correction
    inv_dt = 1.0 / max(dt, 1e-3)
    factor = kp_vel * scale * inv_dt
    
    dv_x = min(max(dx * factor, -5.0), 5.0)
    dv_y = min(max(dy * factor, -5.0), 5.0)
    dv_z = min(max(dz * factor, -5.0), 5.0)
    
    vel_est_new = np.array([
        vel_est[0] + dv_x,
        vel_est[1] + dv_y,
        vel_est[2] + dv_z
    ], dtype=np.float64)
    
    # I-term: Accelerometer Bias Correction
    # Rotate innovation into Body Frame
    R = jit_rot_from_quat(qr_new)
    
    c_dx = dx * scale
    c_dy = dy * scale
    c_dz = dz * scale
    
    # R.T @ clamped_innov
    innov_body_x = R[0,0]*c_dx + R[1,0]*c_dy + R[2,0]*c_dz
    innov_body_y = R[0,1]*c_dx + R[1,1]*c_dy + R[2,1]*c_dz
    innov_body_z = R[0,2]*c_dx + R[1,2]*c_dy + R[2,2]*c_dz
    
    ba_new = np.array([
        ba[0] - innov_body_x * ki_ba,
        ba[1] - innov_body_y * ki_ba,
        ba[2] - innov_body_z * ki_ba
    ], dtype=np.float64)
    
    # 4. Output Smoothing (EMA)
    p_new = jit_dq_topose(qr_new, qd_new)
    
    ppose_ema_new = np.empty(3, dtype=np.float64)
    ppose_ema_new[0] = alpha_pose * p_new[0] + (1.0 - alpha_pose) * ppose_ema[0]
    ppose_ema_new[1] = alpha_pose * p_new[1] + (1.0 - alpha_pose) * ppose_ema[1]
    ppose_ema_new[2] = alpha_pose * p_new[2] + (1.0 - alpha_pose) * ppose_ema[2]
    
    qr_final, qd_final = jit_dq_from_pose(qr_new, ppose_ema_new)
    
    return qr_final, qd_final, vel_est_new, ba_new, ppose_ema_new

# ==============================================================================
# Filter Class Wrapper
# ==============================================================================

class DQFilterJIT:
    def __init__(self, dt, alpha_fuse=0.35, kp_vel=0.6, ki_ba=0.01, huber_c=2.0):
        self.dt = dt
        self.alpha_fuse = alpha_fuse
        self.kp_vel = kp_vel 
        self.ki_ba = ki_ba
        self.huber_c = huber_c
        self.vo_sigma = 1.0
        self.alpha_pose = 0.5 
        
        # State Initialization
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
        """Returns current position from Dual Quaternion state."""
        return jit_dq_topose(self.qr, self.qd)
        
    def initialize(self, q0, p0):
        """Sets initial state."""
        self.qr, self.qd = jit_dq_from_pose(
            jit_quat_normalize(q0.astype(np.float64)), 
            p0.astype(np.float64)
        )
        self.ppose_ema = p0.copy().astype(np.float64)
        self.bw.fill(0)
        self.ba.fill(0)
        self.acc_filt.fill(0)
        self.vel_est.fill(0)
        # Priming step
        try:
            self.step(np.zeros(3), np.zeros(3), None)
        except:
            pass

    def step(self, om, ac, pos_meas=None):
        """
        Main filter loop.
        :param om: Angular velocity (rad/s)
        :param ac: Linear acceleration (m/s^2)
        :param pos_meas: Optional external position measurement [x, y, z]
        """
        om = om.astype(np.float64)
        ac = ac.astype(np.float64)
        
        # Prediction
        self.qr, self.qd, self.vel_est, self.acc_filt = kernel_predict(
            self.qr, self.qd, self.vel_est, self.acc_filt, 
            self.bw, self.ba, om, ac, self.dt, self.GRAVITY
        )
        
        # Measurement Update (if available)
        if pos_meas is not None:
            pos_meas = pos_meas.astype(np.float64)
            self.qr, self.qd, self.vel_est, self.ba, self.ppose_ema = kernel_update(
                self.qr, self.qd, self.vel_est, self.ba, self.ppose_ema, pos_meas, self.dt,
                self.alpha_fuse, self.kp_vel, self.ki_ba, self.huber_c, self.vo_sigma, self.alpha_pose
            )