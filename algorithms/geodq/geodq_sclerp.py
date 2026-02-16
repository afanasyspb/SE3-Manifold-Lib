"""
High-Performance Dual Quaternion SCLERP Filter for 6DoF Pose Estimation.
(Standard Python/NumPy Version - No JIT)

Features:
- Dual Quaternion kinematics for singularity-free orientation.
- SCLERP (Screw Linear Interpolation) for manifold-correct fusion.
- Robust Tilt Correction using Small Angle Approximation (Linearized).
- Designed for ease of use and modification without Numba dependencies.

Author: Ilya Afanasyev
"""

import numpy as np

EPS = 1e-9
GRAVITY_VEC = np.array([0.0, 0.0, -9.81], dtype=np.float64)

# ==============================================================================
# Mathematical Utilities: SO(3) and Quaternions
# ==============================================================================

def normalize_vec(v):
    n = np.linalg.norm(v)
    return v if n < EPS else v / n

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def quat_normalize(q):
    n = np.linalg.norm(q)
    return q if n < EPS else q / n

def quat_from_rotvec(rv):
    """
    Standard conversion from rotation vector to quaternion.
    Used in prediction steps where rotations might be large.
    """
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half_ang = 0.5 * ang
    s = np.sin(half_ang) / ang
    c = np.cos(half_ang)
    return np.array([c, s*rv[0], s*rv[1], s*rv[2]], dtype=np.float64)

def so3_log(q):
    """Logarithmic map for SO(3) -> R3."""
    w = q[0]
    n = np.linalg.norm(q[1:])
    ang = 2.0 * np.arctan2(n, max(w, 1e-12))
    if ang < 1e-12:
        return np.zeros(3, dtype=np.float64)
    factor = ang / n
    return q[1:] * factor

def rot_from_quat(q):
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
# Dual Quaternion Operations
# ==============================================================================

def dq_normalize(qr, qd):
    """Normalize Dual Quaternion."""
    qr_n = quat_normalize(qr)
    dot = np.dot(qr_n, qd)
    qd_n = qd - dot * qr_n
    return qr_n, qd_n

def dq_mul(qr1, qd1, qr2, qd2):
    """Multiply two Dual Quaternions."""
    qr_new = quat_mult(qr1, qr2)
    qd_new = quat_mult(qr1, qd2) + quat_mult(qd1, qr2)
    return qr_new, qd_new

def dq_from_pose(qr, p):
    """Create DQ from orientation (quat) and position (vec3)."""
    qr = quat_normalize(qr)
    w2, x2, y2, z2 = qr
    x1, y1, z1 = p
    
    qd_w = -x1*x2 - y1*y2 - z1*z2
    qd_x =  x1*w2 + y1*z2 - z1*y2
    qd_y = -x1*z2 + y1*w2 + z1*x2
    qd_z =  x1*y2 - y1*x2 + z1*w2
    
    qd = 0.5 * np.array([qd_w, qd_x, qd_y, qd_z], dtype=np.float64)
    return dq_normalize(qr, qd)

def dq_topose(qr, qd):
    """Extract position from DQ."""
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]], dtype=np.float64)
    pq = quat_mult(qd, qr_c)
    return 2.0 * pq[1:]

def dq_inv(qr, qd):
    """Conjugate of unit DQ."""
    qr_c = np.array([qr[0], -qr[1], -qr[2], -qr[3]], dtype=np.float64)
    qd_c = np.array([qd[0], -qd[1], -qd[2], -qd[3]], dtype=np.float64)
    return qr_c, qd_c

def dq_log_vec(qr, qd):
    """Logarithmic map SE(3) -> se(3)."""
    phi = so3_log(qr)
    p = dq_topose(qr, qd)
    return phi, p * 0.5

def dq_exp_vec(phi, ups):
    """Exponential map se(3) -> SE(3)."""
    qr = quat_from_rotvec(phi)
    return dq_from_pose(qr, 2.0 * ups)

def dq_sclerp_kernel(qr1, qd1, qr2, qd2, alpha):
    """
    Screw Linear Interpolation (SCLERP) between two dual quaternions.
    Manifold-correct interpolation.
    """
    if alpha <= 0: return qr1, qd1
    if alpha >= 1: return qr2, qd2
    
    qr1_c, qd1_c = dq_inv(qr1, qd1)
    rel_r, rel_d = dq_mul(qr1_c, qd1_c, qr2, qd2)
    rel_r, rel_d = dq_normalize(rel_r, rel_d)
    
    phi, ups = dq_log_vec(rel_r, rel_d)
    exp_r, exp_d = dq_exp_vec(alpha * phi, alpha * ups)
    
    res_r, res_d = dq_mul(qr1, qd1, exp_r, exp_d)
    return dq_normalize(res_r, res_d)

# ==============================================================================
# Filter Class
# ==============================================================================

class DQFilter:
    def __init__(self, dt, 
                 alpha_fuse=0.35, 
                 kp_vel=0.6,    
                 ki_ba=0.01, 
                 huber_c=2.0):
        
        self.dt = dt
        self.alpha_fuse = alpha_fuse
        self.kp_vel = kp_vel
        self.ki_ba = ki_ba
        self.huber_c = huber_c
        self.vo_sigma = 1.0
        self.alpha_pose = 0.5
        
        # State Vectors
        self.qr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.qd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        self.bw = np.zeros(3, dtype=np.float64) # Gyro bias
        self.ba = np.zeros(3, dtype=np.float64) # Accel bias
        
        self.vel_est = np.zeros(3, dtype=np.float64)
        self.acc_filt = np.zeros(3, dtype=np.float64)
        self.ppose_ema = np.zeros(3, dtype=np.float64)
        
        self.GRAVITY = GRAVITY_VEC

    @property
    def position(self):
        return dq_topose(self.qr, self.qd)
        
    def initialize(self, q0, p0):
        self.qr, self.qd = dq_from_pose(q0, p0)
        self.ppose_ema = p0.copy()
        self.bw.fill(0)
        self.ba.fill(0)
        self.acc_filt.fill(0)
        self.vel_est.fill(0)
        
    def step(self, om, ac, pos_meas=None):
        """
        Main filter step: Prediction + Update.
        """
        # ======================================================================
        # 1. PREDICTION (IMU Integration)
        # ======================================================================
        omega_corr = om - self.bw
        acc_corr = ac - self.ba
        
        R_wb = rot_from_quat(self.qr)
        
        # Integrate acceleration
        acc_world = R_wb @ acc_corr
        lin_acc = acc_world + self.GRAVITY
        
        self.acc_filt = 0.25 * lin_acc + 0.75 * self.acc_filt
        self.vel_est += self.acc_filt * self.dt
        
        # Integrate Twist (Pose) on manifold
        vel_body = R_wb.T @ self.vel_est
        phi = omega_corr * self.dt
        rho = vel_body * self.dt
        
        exp_r, exp_d = dq_exp_vec(phi, rho/2.0)
        self.qr, self.qd = dq_mul(self.qr, self.qd, exp_r, exp_d)
        self.qr, self.qd = dq_normalize(self.qr, self.qd)
        
        # ======================================================================
        # 2. UPDATE (Measurement Fusion)
        # ======================================================================
        if pos_meas is not None:
            p_curr = dq_topose(self.qr, self.qd)
            
            # Calculate Innovation (Residual)
            pos_innov = pos_meas - p_curr
            norm = np.linalg.norm(pos_innov)
            dx, dy, dz = pos_innov
            
            # ------------------------------------------------------------------
            # OPTIMIZED TILT CORRECTION (Small Angle Approximation)
            # ------------------------------------------------------------------
            # Replaces explicit trigonometry to handle large residuals more robustly.
            # Axis = UP x Error => [-dy, dx, 0]
            
            tilt_gain = 0.05
            half_gain = 0.5 * tilt_gain
            
            qx = -dy * half_gain
            qy =  dx * half_gain
            
            # Apply correction: q_new = q_corr * q_old
            # q_corr is approx [1, qx, qy, 0]
            w, x, y, z = self.qr
            
            qr_corr_w = w - qx*x - qy*y
            qr_corr_x = x + qx*w + qy*z
            qr_corr_y = y - qx*z + qy*w
            qr_corr_z = z + qx*y - qy*x
            
            # Fast normalization
            q_norm = np.sqrt(qr_corr_w**2 + qr_corr_x**2 + qr_corr_y**2 + qr_corr_z**2)
            if q_norm > 1e-12:
                inv_n = 1.0 / q_norm
                qr_c = np.array([qr_corr_w, qr_corr_x, qr_corr_y, qr_corr_z]) * inv_n
            else:
                qr_c = self.qr
            
            # ------------------------------------------------------------------
            # SCLERP FUSION
            # ------------------------------------------------------------------
            w_rob = 1.0 if norm <= self.huber_c else self.huber_c / norm
            gw = min(1.0 / self.vo_sigma, 2.0)
            alpha = min(max(self.alpha_fuse * w_rob * gw, 0.05), 0.8)
            
            qr_vo, qd_vo = dq_from_pose(qr_c, pos_meas)
            self.qr, self.qd = dq_sclerp_kernel(qr_c, self.qd, qr_vo, qd_vo, alpha)
            
            # ------------------------------------------------------------------
            # PID FEEDBACK LOOP
            # ------------------------------------------------------------------
            max_innov = 1.0
            scale = 1.0
            if norm > max_innov:
                scale = max_innov / norm
                
            # P-term: Velocity
            inv_dt = 1.0 / max(self.dt, 1e-3)
            factor = self.kp_vel * scale * inv_dt
            
            dv = pos_innov * factor
            dv = np.clip(dv, -5.0, 5.0)
            self.vel_est += dv
            
            # I-term: Bias (Rotate error into Body Frame)
            R = rot_from_quat(self.qr)
            pos_innov_body = R.T @ (pos_innov * scale)
            self.ba -= pos_innov_body * self.ki_ba
            
            # ------------------------------------------------------------------
            # OUTPUT SMOOTHING
            # ------------------------------------------------------------------
            p_new = dq_topose(self.qr, self.qd)
            
            if np.all(self.ppose_ema == 0) and np.linalg.norm(self.ppose_ema) < 1e-9:
                 self.ppose_ema = p_new.copy()
            else:
                 self.ppose_ema = self.alpha_pose * p_new + (1 - self.alpha_pose) * self.ppose_ema
            
            # Re-align state to smoothed position
            self.qr, self.qd = dq_from_pose(self.qr, self.ppose_ema)