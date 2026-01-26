"""
Optimized Error-State Extended Kalman Filter (ESKF) for INS/VO Fusion.

Numba-optimized implementation of the Error-State Kalman Filter.
This module implements the industry-standard Error-State Kalman Filter 
(often referred to as Indirect Kalman Filter) on the SE(3) manifold.
It features optimized NumPy matrix operations compiled via JIT for high performance 
and uses the numerically stable Joseph form for covariance updates.
"""

import numpy as np
from numba import jit

# ==============================================================================
# JIT Math Kernels
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def jit_skew(v):
    """Returns the skew-symmetric matrix of a 3D vector."""
    x, y, z = v
    return np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,    0.0]
    ], dtype=np.float64)

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_normalize(q):
    """Normalizes a quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-9: return q
    return q / n

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_mul(q1, q2):
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
def jit_so3_exp(phi):
    """Exponential map: Rotation Vector -> Unit Quaternion."""
    theta_sq = np.dot(phi, phi)
    if theta_sq < 1e-8:
        k = 0.5
        return np.array([1.0, k*phi[0], k*phi[1], k*phi[2]], dtype=np.float64)
    
    theta = np.sqrt(theta_sq)
    half_theta = 0.5 * theta
    s = np.sin(half_theta) / theta
    c = np.cos(half_theta)
    return np.array([c, s*phi[0], s*phi[1], s*phi[2]], dtype=np.float64)

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
# JIT Logic Kernels
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def predict_kernel(p, v, q, bg, ba, P, Q_c, omega_meas, acc_meas, dt, g):
    """JIT-compiled prediction step (IMU integration and covariance propagation)."""
    # 1. Nominal State Propagation
    omega = omega_meas - bg
    acc = acc_meas - ba
    
    dq = jit_so3_exp(omega * dt)
    q_new = jit_quat_mul(q, dq)
    q_new = jit_quat_normalize(q_new)
    R_wb = jit_rot_from_quat(q_new)
    
    a_world = R_wb @ acc + g
    v_new = v + a_world * dt
    p_new = p + v_new * dt
    
    # 2. Jacobian F Construction
    F = np.zeros((15, 15), dtype=np.float64)
    # Position block
    F[0, 3] = 1.0; F[1, 4] = 1.0; F[2, 5] = 1.0
    
    # Velocity block
    Sx_acc = jit_skew(acc)
    R_Sx = R_wb @ Sx_acc
    F[3:6, 6:9]   = -R_Sx
    F[3:6, 12:15] = -R_wb
    
    # Orientation block
    F[6:9, 6:9]   = -jit_skew(omega)
    F[6, 9] = -1.0; F[7, 10] = -1.0; F[8, 11] = -1.0 # -I
    
    # G Matrix construction (Approximation for Q_d)
    G = np.zeros((15, 12), dtype=np.float64)
    G[3:6, 3:6] = R_wb
    G[6, 0] = 1.0; G[7, 1] = 1.0; G[8, 2] = 1.0
    G[9, 6] = 1.0; G[10, 7] = 1.0; G[11, 8] = 1.0
    G[12, 9] = 1.0; G[13, 10] = 1.0; G[14, 11] = 1.0
    
    # 3. Covariance Propagation
    # Phi = I + F*dt
    Phi = np.eye(15, dtype=np.float64) + F * dt
    
    # Qd = G * Qc * G.T * dt
    Qd = G @ Q_c @ G.T * dt
    
    P_new = Phi @ P @ Phi.T + Qd
    
    return p_new, v_new, q_new, bg, ba, P_new

@jit(nopython=True, cache=True, fastmath=True)
def update_kernel(p, v, q, bg, ba, P, pos_meas, R_pos):
    """JIT-compiled measurement update step using Joseph form."""
    y = pos_meas - p
    
    # S = H P H^T + R
    # H is [I 0 0 0 0], so H P H^T is P[0:3, 0:3]
    P_pos = P[0:3, 0:3]
    S = P_pos + R_pos
    
    # K = P H^T S^-1
    P_HT = P[:, 0:3]
    # Solve S * K.T = P_HT.T -> K = (Result).T
    K_T = np.linalg.solve(S, P_HT.T)
    K = K_T.T
    
    # Error State Correction
    dx = K @ y
    
    p_new = p + dx[0:3]
    v_new = v + dx[3:6]
    
    dq = jit_so3_exp(dx[6:9])
    q_new = jit_quat_mul(q, dq)
    q_new = jit_quat_normalize(q_new)
    
    bg_new = bg + dx[9:12]
    ba_new = ba + dx[12:15]
    
    # Joseph Update: P = (I - KH) P (I - KH)^T + K R K^T
    I15 = np.eye(15, dtype=np.float64)
    # I - KH. Since H is identity for first 3 cols, KH is just K placed in first 3 cols
    I_KH = I15.copy()
    I_KH[:, 0:3] -= K
    
    P_new = I_KH @ P @ I_KH.T + K @ R_pos @ K.T
    
    return p_new, v_new, q_new, bg_new, ba_new, P_new

# ==============================================================================
# Filter Class Wrapper
# ==============================================================================

class ESKF_INS_JIT:
    def __init__(self, dt,
                 Q_gyro=2e-4, Q_acc=4e-3,
                 Q_bg=1e-6, Q_ba=1e-5,
                 R_pos=None):
        """
        Initializes the JIT-optimized Error-State Kalman Filter.
        
        Args:
            dt: Time step.
            R_pos: Position measurement covariance (Visual Odometry noise).
        """
        self.dt = dt
        self.g = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        if R_pos is None:
            R_pos = 0.5**2 * np.eye(3, dtype=np.float64)
        self.R_pos = R_pos.astype(np.float64)

        # State
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64)

        self.P = np.eye(15, dtype=np.float64) * 1e-2

        # Process Noise Matrix
        self.Q_c = np.zeros((12, 12), dtype=np.float64)
        np.fill_diagonal(self.Q_c[0:3, 0:3], Q_gyro)
        np.fill_diagonal(self.Q_c[3:6, 3:6], Q_acc)
        np.fill_diagonal(self.Q_c[6:9, 6:9], Q_bg)
        np.fill_diagonal(self.Q_c[9:12, 9:12], Q_ba)

    def set_state(self, p, v, q):
        """Sets the state and triggers JIT compilation warmup."""
        self.p = p.astype(np.float64).copy()
        self.v = v.astype(np.float64).copy()
        self.q = jit_quat_normalize(q.astype(np.float64).copy())
        
        # Warmup JIT kernels
        try:
            self.predict(np.zeros(3), np.zeros(3))
        except:
            pass
            
        # Reset state after warmup
        self.p = p.astype(np.float64).copy()
        self.v = v.astype(np.float64).copy()
        self.q = jit_quat_normalize(q.astype(np.float64).copy())
        self.P = np.eye(15, dtype=np.float64) * 1e-2

    def predict(self, omega_meas, acc_meas):
        """Executes the prediction step using JIT kernels."""
        om = omega_meas.astype(np.float64)
        ac = acc_meas.astype(np.float64)
        
        self.p, self.v, self.q, self.bg, self.ba, self.P = predict_kernel(
            self.p, self.v, self.q, self.bg, self.ba,
            self.P, self.Q_c, om, ac, self.dt, self.g
        )

    def update_vo(self, pos_meas):
        """Executes the measurement update step using JIT kernels."""
        meas = pos_meas.astype(np.float64)
        
        self.p, self.v, self.q, self.bg, self.ba, self.P = update_kernel(
            self.p, self.v, self.q, self.bg, self.ba,
            self.P, meas, self.R_pos
        )