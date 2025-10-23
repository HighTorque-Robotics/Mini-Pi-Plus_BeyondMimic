import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def quatToEuler(quat):
    """ 将四元数转换为欧拉角(roll, pitch, yaw)。 """
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)
    else:
        eulerVec[1] = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return eulerVec



def quat_mul_np(x, y, scalar_first=True):
    """
    Performs quaternion multiplication on arrays of quaternions
    :param x: tensor of quaternions of shape (..., 4)
    :param y: tensor of quaternions of shape (..., 4)
    :param scalar_first: True if quaternions are in [w, x, y, z] format
    :return: quaternion multiplication result in same format
    """
    if scalar_first:
        pass
    else: # convert to scalar-first
        x = x[..., [3, 0, 1, 2]]
        y = y[..., [3, 0, 1, 2]]

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
        x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
        x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
        x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0
    ], axis=-1)

    if scalar_first:
        pass
    else:
        res = res[..., [1, 2, 3, 0]]  # back to [w, x, y, z]

    return res

def quat_rotate_inverse(q, v):
    """
    将向量 v 以四元数 q 的逆旋转进行变换。  
    为保持一致，以下代码与原脚本中的实现相同。
    """
    q = np.asarray(q)
    v = np.asarray(v)

    q_w = q[:, -1]      # w
    q_vec = q[:, :3]    # x, y, z

    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * (2.0 * q_w)[:, np.newaxis]
    dot = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * (2.0 * dot)

    return a - b + c

def quat_rotate_inverse_torch(q, v, scalar_first=True):
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def quat_rotate_inverse_np(q, v, scalar_first=True):
    q = np.asarray(q)
    v = np.asarray(v)
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c

def euler_from_quaternion_torch(quat_angle, scalar_first=True):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    if scalar_first:
        quat_angle = quat_angle[..., [1, 2, 3, 0]]
    else:
        quat_angle = quat_angle[..., [0, 1, 2, 3]]
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def euler_from_quaternion_np(quat, scalar_first=True):
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]
    else:
        quat = quat[..., [0, 1, 2, 3]]
    
    x = quat[:,0]; y = quat[:,1]; z = quat[:,2]; w = quat[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


def quat_diff_np(q1, q2, scalar_first=True):
    # Ensure quaternions are numpy arrays
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Convert to scipy Rotation object (scalar-first)
    r1 = R.from_quat(q1, scalar_first=scalar_first)
    r2 = R.from_quat(q2, scalar_first=scalar_first)

    # Relative rotation
    r_rel = r2 * r1.inv()

    # Rotation vector (axis * angle)
    rotvec = r_rel.as_rotvec()  # returns angle * axis vector

    return rotvec