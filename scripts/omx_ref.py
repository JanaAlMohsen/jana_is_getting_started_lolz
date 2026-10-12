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
