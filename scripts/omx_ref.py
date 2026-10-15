"""Reference implementations of the ELEC3901 lab task functions (generated from tasks.py).
Each lab imports the functions of PREVIOUS labs from here so nobody is blocked by last week; your own definition in
the current notebook takes precedence. Do not read ahead: the point of the tasks is to write them yourself."""
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as _Rot
NOM = {'d1': 0.0975, 'l2z': 0.11315, 'l2x': 0.0415, 'l3': 0.162, 'l4': 0.123}
Q_LIM = np.array([[-2.618, 2.618], [-1.868, 1.658], [-1.571, 1.676], [-1.745, 1.745]])
def small_T(v):
    T = np.eye(4); T[:3, :3] = _Rot.from_rotvec(v[:3]).as_matrix(); T[:3, 3] = v[3:]; return T

# ---- Lab 1 Task 1: Log a joint step (open task)
def step_log(arm, joint, targets_deg, samples=50, period=0.02):
    """Command joint `joint` through targets_deg; return (t, q) arrays of the measured joint angle."""
    q_cmd = arm.read_q().copy(); t0 = time.time(); log_t, log_q = [], []
    for target in np.radians(targets_deg):
        q_cmd[joint] = target; arm.move_q(q_cmd)
        for _ in range(samples):
            log_t.append(time.time() - t0); log_q.append(arm.read_q()[joint]); time.sleep(period)
    return np.array(log_t), np.array(log_q)

# ---- Lab 1 Task 2: 2D pose from scratch
def rot2(theta):
    """2x2 rotation matrix for angle theta (rad)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def se2(x, y, theta):
    """3x3 homogeneous transform: translate (x, y) then rotate theta. Returns a NumPy array."""
    T = np.eye(3); T[:2, :2] = rot2(theta); T[:2, 2] = [x, y]
    return T

def transform_point(T, p):
    """Apply a 3x3 transform T to a 2D point p (length-2 array). Returns a length-2 array."""
    return (T @ np.append(np.asarray(p, float), 1.0))[:2]

# ---- Lab 1 Task 3: Combining transforms
def transl2(x, y):
    """3x3 homogeneous transform: a pure translation by (x, y)."""
    return np.array([[1.0, 0, x], [0, 1.0, y], [0, 0, 1.0]])

def trot2(theta):
    """3x3 homogeneous transform: a pure rotation by theta (rad) about the origin."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0, 0, 1.0]])

def rot_about(theta, c):
    """3x3 homogeneous transform: rotation by theta (rad) about the point c = (cx, cy)."""
    return transl2(c[0], c[1]) @ trot2(theta) @ transl2(-c[0], -c[1])

# ---- Lab 2 Task 1: Rotations in 3D and roll-pitch-yaw angles
def rotx(a):
    """3x3 rotation about x by a (rad)."""
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(a):
    """3x3 rotation about y by a (rad)."""
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(a):
    """3x3 rotation about z by a (rad). GIVEN - the worked example: the columns are where x and y go, z stays put."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],      # column 0: x turns to (cos a, sin a, 0)
                     [s, c, 0],       # column 1: y turns to (-sin a, cos a, 0)
                     [0, 0, 1]])      # column 2: z is the axis - it does not move

def rpy_to_R(roll, pitch, yaw):
    """zyx roll-pitch-yaw (rad) -> 3x3 rotation matrix."""
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

def R_to_rpy(R):
    """3x3 rotation matrix -> [roll, pitch, yaw] (rad), zyx convention. Assume |pitch| < 90 deg."""
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    roll = np.arctan2(R[2, 1], R[2, 2]); yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])

# ---- Lab 2 Task 2: Pose in 3D: build, invert and apply a transform
def make_T(R, t):
    """3x3 rotation, 3-vector -> 4x4 homogeneous transform."""
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T

def inv_T(T):
    """Inverse of a 4x4 homogeneous transform, from R^T and -R^T t (no general matrix inverse)."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti

def transform_points(T, P):
    """Map point(s) P ((3,) or N x 3) through the 4x4 transform T; returns the same shape."""
    P = np.asarray(P, float)
    out = (T[:3, :3] @ np.atleast_2d(P).T).T + T[:3, 3]
    return out[0] if P.ndim == 1 else out

# ---- Lab 2 Task 3: Chaining a camera measurement to the base frame - and measuring the camera mount
def chain_T_BM(T_BE, T_EC, T_CM):
    """All 4x4 NumPy arrays. Returns T_BM."""
    return T_BE @ T_EC @ T_CM

def mount_from_views(T_BEs, T_CMs, T_BM=np.eye(4)):
    """Lists of 4x4 arrays (one pair per view) -> 4x4 camera mount T_EC averaged over the views."""
    Ts = [np.linalg.inv(T_BE) @ T_BM @ np.linalg.inv(T_CM) for T_BE, T_CM in zip(T_BEs, T_CMs)]
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in Ts], axis=0))
    T = np.eye(4); T[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt; T[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return T
