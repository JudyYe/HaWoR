import copy
import numpy as np
import torch

from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_quaternion, rotation_matrix_to_angle_axis
from scipy.interpolate import interp1d


def cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, handedness):
    init_rot_mat = copy.deepcopy(data_out["init_root_orient"])
    init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w_sla, init_rot_mat)
    init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
    init_rot_quat = angle_axis_to_quaternion(init_rot)
    # data_out["init_root_orient"] = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    # data_out["init_hand_pose"] = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
    data_out_init_root_orient = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
    data_out_init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])

    init_trans = data_out["init_trans"] # (B, T, 3)
    if handedness == "right":
        outputs = run_mano(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
    elif handedness == "left":
        outputs = run_mano_left(data_out["init_trans"], data_out_init_root_orient, data_out_init_hand_pose, betas=data_out["init_betas"])
    root_loc = outputs["joints"][..., 0, :].cpu()  # (B, T, 3)
    offset = init_trans - root_loc  # It is a constant, no matter what the rotation is.
    init_trans = (
        torch.einsum("tij,btj->bti", R_c2w_sla, root_loc)
        + t_c2w_sla[None, :]
        + offset
    )

    data_world = {
        "init_root_orient": init_rot, # (B, T, 3)
        "init_hand_pose": data_out_init_hand_pose, # (B, T, 15, 3)
        "init_trans": init_trans,  # (B, T, 3)
        "init_betas": data_out["init_betas"]  # (B, T, 10)
    }

    return data_world

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def load_slam_cam(fpath):
    print(f"Loading cameras from {fpath}...")
    pred_cam = dict(np.load(fpath, allow_pickle=True))
    pred_traj = pred_cam['traj']
    t_c2w_sla = torch.tensor(pred_traj[:, :3]) * pred_cam['scale']
    pred_camq = torch.tensor(pred_traj[:, 3:])
    R_c2w_sla = quaternion_to_matrix(pred_camq[:,[3,0,1,2]])
    R_w2c_sla = R_c2w_sla.transpose(-1, -2)
    t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)

    # R_x = torch.tensor([[1,  0,  0],
    #                     [0, -1,  0],
    #                     [0,  0, -1]]).float()
    # R_c2w_sla = torch.einsum('ij,njk->nik', R_x, R_c2w_sla)
    # t_c2w_sla = torch.einsum('ij,nj->ni', R_x, t_c2w_sla)
    # R_w2c_sla = R_c2w_sla.transpose(-1, -2)
    # t_w2c_sla = -torch.einsum("bij,bj->bi", R_w2c_sla, t_c2w_sla)

    return R_w2c_sla, t_w2c_sla, R_c2w_sla, t_c2w_sla


def interpolate_bboxes(bboxes):
    T = bboxes.shape[0]  

    zero_indices = np.where(np.all(bboxes == 0, axis=1))[0]

    non_zero_indices = np.where(np.any(bboxes != 0, axis=1))[0]

    if len(zero_indices) == 0:
        return bboxes

    interpolated_bboxes = bboxes.copy()
    for i in range(5):  
        interp_func = interp1d(non_zero_indices, bboxes[non_zero_indices, i], kind='linear', fill_value="extrapolate")
        interpolated_bboxes[zero_indices, i] = interp_func(zero_indices)
    
    return interpolated_bboxes


def cam_to_img(kpts, intri):
    """
    Project points in camera coordinate system to image plane
    Input:
        kpts: (**,3)
    Output:
        new_kpts: (**,2)
    """
    shape = list(kpts.shape)
    shape[-1] = 2
    kpts = kpts.flatten(0, -2)

    new_kpts = kpts.clone()
    new_kpts = intri @ new_kpts.T  # (3,N)
    new_kpts = new_kpts / new_kpts[2, :]
    new_kpts = new_kpts[:2, :].T

    new_kpts = new_kpts.reshape(*shape)
    
    return new_kpts    