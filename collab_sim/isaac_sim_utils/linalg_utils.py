# # SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: LicenseRef-NvidiaProprietary
# #
# # NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# # property and proprietary rights in and to this material, related
# # documentation and any modifications thereto. Any use, reproduction,
# # disclosure or distribution of this material and related documentation
# # without an express license agreement from NVIDIA CORPORATION or
# # its affiliates is strictly prohibited.


# """ A set of utility functions for linear algebra operations.
# """

# from typing import Optional
# import copy
# import math
# import numpy as np
# from numpy.linalg import norm
# from numpy.linalg import inv
# from scipy.spatial.transform import Rotation as R
# from .state_utils import Quaternion

# def unpack_T(T):
#     """ Returns a pose given a homogeneous transformation matrix.
#         T must be of size 4x4.
#         Legacy

#         Returns (R, p)
#     """
#     print("[linalg_utils][unpack_T] consider using 'unpack_homogeneous_transform")
#     return unpack_homogeneous_transform(T)

# def unpack_homogeneous_transform(T):
#     """ Returns a pose given a homogeneous transformation matrix.
#         T must be of size 4x4.

#         Returns (R, p)
#     """

#     return T[:3, :3], T[:3, 3]


# def pack_Rp(R, p):
#     """ Packs the provided rotation matrix (R) and position (p) into a homogeneous transform
#     matrix. 
#     Kept for backwards compatibility.
#     """
#     print(f"[Warning][pack_Rp] consider using 'homogeneous_transform' instead.")
#     return homogeneous_transform(p, matrix=R)

# def homogeneous_transform(position: np.ndarray=None, rotation: R=None, matrix: np.ndarray=None, quat: Quaternion=None):
#     """ Packs the provided position and rotation into a homogeneous transform """

#     T = np.eye(4)

#     # translation portion of the homogeneous transform will be defaulted to [0, 0, 0] if a position is not provided
#     if position is not None:
#         if position.size != 3:
#             err=f"[homogeneous_transform] 'position' argument is not of length 3: You provided position.shape={position.shape}"
#             raise ValueError(err) 
#         T[:3, 3] = position

#     # rotation is set by providing a scipy.spatial.transform.Rotation object, or by directly providing a 3x3 matrix.
#     if rotation is not None: 
#         T[:3, :3] = rotation.as_matrix()
#     elif matrix is not None:
#         if matrix.shape != (3, 3):
#             err=f"[homogeneous_transform] 'matrix' (np.ndarray) argument must be of size 3x3. You provided matrix.shape={matrix.shape}"
#             raise ValueError(err)
#         T[:3, :3] = matrix
#     elif quat is not None: 
#         T[:3, :3] = quat.scipy_rotation().as_matrix()
#     else: 
#         err=f"[homogeneous_transform] One of 'rotation' (scipy.spatial.transform.Rotation) or 'matrix' (np.ndarray, 3x3) or 'quat' (state_utils.Quaternion) arguments must be provided to set the orientation of the homogeneous transformation matrix."
#         raise ValueError(err)
#     return T 

# def inverse_homogeneous_transform(T):
#     """ Inverts the provided transform matrix using the explicit formula leveraging the
#     orthogonality of R and the sparsity of the transform.

#     Specifically, denote T = h(R, t) where h(.,.) is a function mapping the rotation R and
#     translation t to a homogeneous matrix defined by those parameters. Then

#       inv(T) = inv(h(R,t)) = h(R', -R't).
#     """
#     R, p = unpack_homogeneous_transform(T)
#     R_transpose = R.T
#     return homogeneous_transform(position = -R_transpose.dot(p), matrix = R_transpose)

# def is_homogeneous(T: np.ndarray) -> bool:
#     """ Checks if T is actually a homogeneous transformation matrix:
#         R is a rotation matrix
#         last row is [0, 0, 0, 1] or [0, 0, 1] depending if it is 3D or 2D

#     Returns:
#         True if T is legitimate.
#     """
#     m, _ = T.shape
#     if m == 3:
#         return T.shape == (3, 3) \
#             and np.allclose(T[2, 0:3], [0, 0, 1]) \
#             and math.isclose(abs(np.linalg.det(T[:2, :2])), 1)
#     if m == 4: 
#         return T.shape == (4, 4) \
#             and np.allclose(T[3, 0:4], [0, 0, 0, 1]) \
#             and math.isclose(abs(np.linalg.det(T[:3, :3])), 1)

# def is_identity(M: np.ndarray) -> bool:
#     return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

# def matrix_to_euler_angles(mat: np.ndarray) -> np.ndarray:
#     """Convert rotation matrix to Euler XYZ angles.

#     Args:
#         mat (np.ndarray): A 3x3 rotation matrix.

#     Returns:
#         np.ndarray: Euler XYZ angles (in radians).
#     """
#     cy = np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
#     singular = cy < 0.00001
#     if not singular:
#         roll = math.atan2(mat[2, 1], mat[2, 2])
#         pitch = math.atan2(-mat[2, 0], cy)
#         yaw = math.atan2(mat[1, 0], mat[0, 0])
#     else:
#         roll = math.atan2(-mat[1, 2], mat[1, 1])
#         pitch = math.atan2(-mat[2, 0], cy)
#         yaw = 0
#     return np.array([roll, pitch, yaw])

# def matrix_to_quat(mat: np.ndarray, order="wxyz") -> np.ndarray:
#     """ Converts the provided rotation matrix into a quaternion in (w, x, y, z) order.
#     """
#     r = R.from_matrix(mat)
#     quat = Quaternion(rotation=r)
#     if order == "wxyz": 
#         return quat.wxyz()
#     else: 
#         return quat.xyzw()

# def T2pq(T):
#     """ Converts a 4d homogeneous matrix to a position-quaternion representation.
#     Quaternion is defined as (w, x, y, z)
#     """

#     R, p = unpack_homogeneous_transform(T)
#     return p, matrix_to_quat(R)


# def pq2T(p, q):
#     """ Converts a pose given as (<position>,<quaternion>) to a 4x4 homogeneous transform matrix.
#     Quaternion is defined as (w, x, y, z).
#     Will normalize this before passing to quat_to_rot_matrix for safety. This does not seem to be implemented in `rotations.py`, so doing that here.
#     """
#     q_normalized = np.array(q)/norm(q)

#     # turn quaternion from wxyz to xyzw
#     quat = Quaternion(wxyz=q_normalized)
#     return homogeneous_transform(rotation=R.from_quat(quat.xyzw()), position=p)
#     # return pack_Rp(r.as_matrix(), p)

# def pos_rpy_to_T(pos, rpy, degrees=False):
#     """ Converts a pos and an euler roll pitch yaw angle to a transformation matrix.
#     """
#     r = R.from_euler('xyz', rpy, degrees)
#     quat = Quaternion(rotation=r)
#     return pq2T(pos, quat.wxyz())


# # Custom functions
# def pq_A_in_B(p_wa, q_wa, p_wb, q_wb):
#     """ Given Frame A and Frame B expressed in in the world frame,
#     want frame A expressed in the frame B. The second set of arguments (frame B) correspond to the new target (world) frame.

#     T_ba = inv(T_wb) * T_wa
#     where:
#         T_wa = frame A defined wrt world
#         T_wb = frame B defined wrt world
#         T_ba = frame A defined wrt frame B.

#     Returns (p, q) of frame A's origin expressed in frame B.
#     Quaternion is defined as (w, x, y, z)
#     """

#     # Obtain homogeneous transforms given the pose information relating to the two frames.
#     T_ba = T_BA(p_wa, q_wa, p_wb, q_wb)
#     p, q = T2pq(T_ba)

#     return p, q

# def T_BA(p_wa, q_wa, p_wb, q_wb):
#     """ Given Frame A and Frame B expressed in in the world frame,
#     want frame A expressed in the frame B. The second set of arguments (frame B) correspond to the new target (world) frame.

#     T_ba = inv(T_wb) * T_wa
#     where:
#         T_wa = frame A defined wrt world
#         T_wb = frame B defined wrt world
#         T_ba = frame A defined wrt frame B.

#     Returns (p, q) of frame A's origin expressed in frame B.
#     Quaternion is defined as (w, x, y, z)
#     """

#     # Obtain homogeneous transforms given the pose information relating to the two frames.
#     T_wa = pq2T(p_wa, q_wa)
#     T_wb = pq2T(p_wb, q_wb)

#     T_BA = inv(T_wb).dot(T_wa)


#     return T_BA

# def transform(p, q, T):
#     """ Given p, q in frame A, transform them into frame B using T.
#     """
#     T_ = pq2T(p, q)
#     T_new = np.matmul(inv(T), T_)

#     p_new, q_new = T2pq(T_new)

#     return p_new, q_new

# def transform_from_Tc_to_Tw(p_c, q_c):
#     """ Transforms a recorded pose in the consistent frame into the world frame.
#     # FIXME: This is a really weird transform that came from all of the data that Henry's code came with. The end effector orientation is aligned wrt to the world frame q = (0, 0, 1, 0). Thus, this weird hack to only transform the orientation components of the data. This needs to be changed.
#     Returns:
#         (p_w, q_w) in the consistent frame, Tc, unless Tc is identity.
#     """

#     Tc = pq2T((0, 0, 0), (0, 0, 1, 0));
#     p_w = p_c
#     _, q_w = transform((0, 0, 0), q_c, Tc)
#     return p_w, q_w

# def transform_from_Tw_to_Tc(p_w, q_w):
#     Tc = pq2T((0, 0, 0), (0, 0, 1, 0));
#     p_c = p_w
#     _, q_c = transform((0, 0, 0), q_w, inv(Tc))
#     return p_c, q_c


# def wxyz_to_xyzw(quat: Optional[np.ndarray]):
#     if len(quat) != 4:
#         raise ValueError("Quaternion is not of length 4")

#     quat_new = copy.deepcopy(quat)
#     quat_new[0] = quat[1]
#     quat_new[1] = quat[2]
#     quat_new[2] = quat[3]
#     quat_new[3] = quat[0]
#     return quat_new

# def xyzw_to_wxyz(quat: Optional[np.ndarray]):
#     if len(quat) != 4:
#         raise ValueError("Quaternion is not of length 4")

#     quat_new = copy.deepcopy(quat)
#     quat_new[0] = quat[3]
#     quat_new[1] = quat[0]
#     quat_new[2] = quat[1]
#     quat_new[3] = quat[2]
#     return quat_new

# def quat_error(q1: Quaternion, q2: Quaternion) -> Quaternion: 
#     """ see https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames 
#     """
#     dw = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z 
#     dx = q1.w*q2.x - q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
#     dy = q1.w*q2.y - q1.y*q2.w - q1.x*q1.z + q1.z*q2.x 
#     dz = q1.w*q2.z - q2.w*q1.z + q1.x*q2.y - q1.y*q2.x

#     return Quaternion(xyzw=np.array([dx, dy, dz, dw]))

# def angle_between_quat(q1: Quaternion, q2: Quaternion) -> Quaternion: 
#     """ see https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames 
#     """
#     dw = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z 
#     dx = q1.w*q2.x - q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
#     dy = q1.w*q2.y - q1.y*q2.w - q1.x*q1.z + q1.z*q2.x 
#     dz = q1.w*q2.z - q2.w*q1.z + q1.x*q2.y - q1.y*q2.x

#     return math.atan2(np.linalg.norm(np.array([dx, dy, dz])), dw)

# def angle_from_quat(x, y, z, w): 
#     """ see https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames 
#     """
#     return math.atan2(np.linalg.norm(np.array([x, y, z])), w)