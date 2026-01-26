"""
Vectorized Manifold Unscented Kalman Filter (UKF-M) for INS/VO Fusion.

Numba-optimized implementation of the Manifold Unscented Kalman Filter.
This module implements a Sigma-Point Kalman Filter designed for Inertial 
Navigation Systems aiding with absolute position updates (Visual Odometry).
It utilizes JIT compilation for high-performance covariance propagation 
directly on the SO(3) x R3 manifold.
"""

import numpy as np
from numba import jit

# ==============================================================================
# JIT Math Kernels
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_normalize(q):
    """Normalizes a quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-9: return q
    return q / n

@jit(nopython=True, cache=True, fastmath=True)
def jit_quat_mult(q1, q2):
    """Hamilton product for single quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

@jit(nopython=True, cache=True, fastmath=True)
def jit_so3_exp(rv):
    """
    Exponential map: Rotation Vector (so3) -> Quaternion (SO3).
    Includes small-angle optimization.
    """
    theta_sq = np.dot(rv, rv)
    if theta_sq < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    theta = np.sqrt(theta_sq)
    half_theta = 0.5 * theta
    s = np.sin(half_theta) / theta
    c = np.cos(half_theta)
    return np.array([c, s*rv[0], s*rv[1], s*rv[2]], dtype=np.float64)

@jit(nopython=True, cache=True, fastmath=True)
def jit_so3_log(q):
    """
    Logarithm map: Quaternion (SO3) -> Rotation Vector (so3).
    """
    w = q[0]
    n = np.linalg.norm(q[1:])
    ang = 2.0 * np.arctan2(n, max(w, 1e-12))
    if ang < 1e-12:
        return np.zeros(3, dtype=np.float64)
    factor = ang / n
    return q[1:] * factor

@jit(nopython=True, cache=True, fastmath=True)
def jit_rotate_vec(q, v):
    """Rotates vector v by quaternion q."""
    q_xyz = q[1:]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q[0]*t + np.cross(q_xyz, t)

# ==============================================================================
# JIT Core Logic (Sigma Points Loop)
# ==============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def predict_kernel(state_p, state_v, state_q, state_bg, state_ba,
                   P, Q, dX_matrix, Wm, Wc,
                   om_meas, ac_meas, dt, g_vec):
    """
    Propagates sigma points and reconstructs mean/covariance.
    Optimized to avoid explicit large array allocation where possible.
    """
    n_dim = 15
    n_sig = dX_matrix.shape[0] # 2*n + 1
    
    # Pre-allocate accumulators for Mean
    mean_p = np.zeros(3, dtype=np.float64)
    mean_v = np.zeros(3, dtype=np.float64)
    mean_bg = np.zeros(3, dtype=np.float64)
    mean_ba = np.zeros(3, dtype=np.float64)
    
    # Storage for propagated sigma points (needed for covariance step)
    sig_p = np.zeros((n_sig, 3), dtype=np.float64)
    sig_v = np.zeros((n_sig, 3), dtype=np.float64)
    sig_q = np.zeros((n_sig, 4), dtype=np.float64)
    sig_bg = np.zeros((n_sig, 3), dtype=np.float64)
    sig_ba = np.zeros((n_sig, 3), dtype=np.float64)
    
    # 1. Propagate Sigma Points
    for i in range(n_sig):
        # Extract perturbation
        dx = dX_matrix[i] # size 15
        
        # Recover sigma point state
        p_i = state_p + dx[0:3]
        v_i = state_v + dx[3:6]
        bg_i = state_bg + dx[9:12]
        ba_i = state_ba + dx[12:15]
        
        # Manifold part (Rotation)
        dq = jit_so3_exp(dx[6:9])
        q_i = jit_quat_mult(state_q, dq)
        q_i = jit_quat_normalize(q_i)
        
        # IMU Integration
        omega_corr = om_meas - bg_i
        acc_corr = ac_meas - ba_i
        
        dq_dt = jit_so3_exp(omega_corr * dt)
        q_pred = jit_quat_mult(q_i, dq_dt)
        q_pred = jit_quat_normalize(q_pred)
        
        acc_world = jit_rotate_vec(q_pred, acc_corr)
        v_pred = v_i + (acc_world + g_vec) * dt
        p_pred = p_i + v_pred * dt
        
        # Store
        sig_p[i] = p_pred
        sig_v[i] = v_pred
        sig_q[i] = q_pred
        sig_bg[i] = bg_i
        sig_ba[i] = ba_i
        
        # Accumulate Mean
        w = Wm[i]
        mean_p += w * p_pred
        mean_v += w * v_pred
        mean_bg += w * bg_i
        mean_ba += w * ba_i
        
    # 2. Quaternion Mean (Manifold averaging)
    q_mean = sig_q[0] 
    e_rot_avg = np.zeros(3, dtype=np.float64)
    
    for i in range(n_sig):
        q_conj = np.array([q_mean[0], -q_mean[1], -q_mean[2], -q_mean[3]], dtype=np.float64)
        q_err = jit_quat_mult(q_conj, sig_q[i])
        e_rot = jit_so3_log(q_err)
        e_rot_avg += Wm[i] * e_rot
        
    dq_avg = jit_so3_exp(e_rot_avg)
    mean_q = jit_quat_mult(q_mean, dq_avg)
    mean_q = jit_quat_normalize(mean_q)
    
    # 3. Covariance Reconstruction
    P_new = np.zeros((15, 15), dtype=np.float64)
    
    for i in range(n_sig):
        dp = sig_p[i] - mean_p
        dv = sig_v[i] - mean_v
        dbg = sig_bg[i] - mean_bg
        dba = sig_ba[i] - mean_ba
        
        # Angular error
        q_mean_conj = np.array([mean_q[0], -mean_q[1], -mean_q[2], -mean_q[3]], dtype=np.float64)
        q_err = jit_quat_mult(q_mean_conj, sig_q[i])
        dtheta = jit_so3_log(q_err)
        
        y = np.zeros(15, dtype=np.float64)
        y[0:3] = dp
        y[3:6] = dv
        y[6:9] = dtheta
        y[9:12] = dbg
        y[12:15] = dba
        
        wc = Wc[i]
        # Manual outer product accumulation
        for r in range(15):
            for c in range(15):
                P_new[r, c] += wc * y[r] * y[c]
                
    return mean_p, mean_v, mean_q, mean_bg, mean_ba, P_new

@jit(nopython=True, cache=True, fastmath=True)
def update_kernel(p, v, q, bg, ba, P, pos_meas, R_pos):
    """JIT-compiled measurement update."""
    z_res = pos_meas - p
    S = P[0:3, 0:3] + R_pos
    
    # Solve S * K.T = P_HT.T
    # H = [I 0 ...], so P @ H.T is simply the first 3 columns of P
    P_HT = P[:, 0:3]
    K_T = np.linalg.solve(S, P_HT.T)
    K = K_T.T
    
    dx = K @ z_res
    
    # State Retraction
    p_new = p + dx[0:3]
    v_new = v + dx[3:6]
    
    dq = jit_so3_exp(dx[6:9])
    q_new = jit_quat_mult(q, dq)
    q_new = jit_quat_normalize(q_new)
    
    bg_new = bg + dx[9:12]
    ba_new = ba + dx[12:15]
    
    # Covariance Update
    P_new = P - K @ S @ K.T
    
    return p_new, v_new, q_new, bg_new, ba_new, P_new

# ==============================================================================
# Filter Class Wrapper
# ==============================================================================

class UKF_M_JIT:
    def __init__(self, dt, Q_acc=4e-3, Q_gyro=2e-4, R_pos=0.5):
        """
        Initializes the JIT-optimized Manifold UKF.
        
        Args:
            dt: Time step.
            R_pos: Position measurement covariance (Visual Odometry noise).
                   Note: Previously named R_gps.
        """
        self.dt = dt
        
        self.p = np.zeros(3, dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.q = np.array([1., 0., 0., 0.], dtype=np.float64)
        self.bg = np.zeros(3, dtype=np.float64)
        self.ba = np.zeros(3, dtype=np.float64)
        
        self.P = np.eye(15, dtype=np.float64) * 1e-2
        
        # Process Noise
        self.Q = np.zeros((15, 15), dtype=np.float64)
        self.Q[0:3, 0:3]   = 0.0
        self.Q[3:6, 3:6]   = np.eye(3) * Q_acc * dt
        self.Q[6:9, 6:9]   = np.eye(3) * Q_gyro * dt
        self.Q[9:12, 9:12] = np.eye(3) * 1e-6 * dt
        self.Q[12:15, 12:15] = np.eye(3) * 1e-5 * dt

        self.R_pos = np.eye(3, dtype=np.float64) * R_pos
        self.g = np.array([0, 0, -9.81], dtype=np.float64)

        # Sigma Point Parameters
        self.n = 15
        self.alpha = 1e-1
        self.beta = 2.0
        self.kappa = 0.0
        
        lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.lambda_ = lambda_
        
        self.Wm = np.full(2*self.n + 1, 0.5 / (self.n + lambda_), dtype=np.float64)
        self.Wc = self.Wm.copy()
        self.Wm[0] = lambda_ / (self.n + lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        
    @property
    def position(self):
        return self.p.copy()

    def set_init(self, p0, q0):
        """
        Resets the filter state and covariance. 
        Explicitly casts to float64 for Numba compatibility.
        """
        self.p = p0.astype(np.float64).copy()
        self.q = jit_quat_normalize(q0.astype(np.float64).copy())
        self.P = np.eye(15, dtype=np.float64) * 1e-2
        
        # Warmup JIT kernels
        try:
            self.predict(np.zeros(3), np.zeros(3))
        except:
            pass
        
        # Reset after warmup
        self.p = p0.astype(np.float64).copy()
        self.q = jit_quat_normalize(q0.astype(np.float64).copy())
        self.P = np.eye(15, dtype=np.float64) * 1e-2

    def predict(self, om, ac):
        """Prediction step wrapper."""
        # Cast inputs to float64 just in case
        om = om.astype(np.float64)
        ac = ac.astype(np.float64)
        
        n = self.n
        # Python-side Cholesky (more robust than Numba's implementation for small epsilons)
        try:
            S = np.linalg.cholesky((n + self.lambda_) * (self.P + self.Q))
        except np.linalg.LinAlgError:
            self.P += np.eye(15) * 1e-6
            S = np.linalg.cholesky((n + self.lambda_) * (self.P + self.Q))
            
        dX = np.zeros((2*n + 1, n), dtype=np.float64)
        dX[1:n+1] = S.T
        dX[n+1:]  = -S.T
        
        # Execute JIT kernel
        self.p, self.v, self.q, self.bg, self.ba, self.P = predict_kernel(
            self.p, self.v, self.q, self.bg, self.ba,
            self.P, self.Q, dX, self.Wm, self.Wc,
            om, ac, self.dt, self.g
        )
        
    def update(self, pos_meas):
        """Measurement update wrapper."""
        if pos_meas is None: return
        
        pos_meas = pos_meas.astype(np.float64)
        
        self.p, self.v, self.q, self.bg, self.ba, self.P = update_kernel(
            self.p, self.v, self.q, self.bg, self.ba,
            self.P, pos_meas, self.R_pos
        )