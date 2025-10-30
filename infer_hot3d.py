import subprocess
from copy import deepcopy
from glob import glob
from tqdm import tqdm
import argparse
import sys
import os
import os.path as osp
import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video

from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam

from scipy.spatial.transform import Rotation
from jutils import mesh_utils, geom_utils, hand_utils, image_utils, plot_utils
from pytorch3d.structures import Meshes
import rerun as rr
from PIL import Image

import torch.nn as nn
import smplx


class HandWrapper(nn.Module):
    def __init__(self, mano_dir='/move/u/yufeiy2/egorecon/assets/mano/'):
        super().__init__()
        sided_mano_models = {
            "left": smplx.create(
                os.path.join(mano_dir, "MANO_LEFT.pkl"),
                "mano",
                is_rhand=False,
                num_pca_comps=15,
            ),
            "right": smplx.create(
                os.path.join(mano_dir, "MANO_RIGHT.pkl"),
                "mano",
                is_rhand=True,
                num_pca_comps=15,
            ),
        }
        self.sided_mano_models = nn.ModuleDict(sided_mano_models)

    def joint2verts_faces(
        self,
        joints,
    ):
        """

        :param joints: (..., 21, 3)
        """

        D = joints.shape[-1:]
        pref_dim = joints.shape[:-1]
        joints = joints.reshape(-1, 21, 3)

        meshes = plot_utils.pc_to_cubic_meshes(joints, eps=0.01)
        verts = meshes.verts_padded()
        faces = meshes.faces_padded()

        verts = verts.reshape(*pref_dim, -1, 3)
        faces = faces.reshape(*pref_dim, -1, 3)

        return verts, faces

    @staticmethod
    def para2dict(
        hand_para,
        hand_shape=None,
    ):
        if hand_shape is None:
            assert hand_para.shape[-1] == 3+3+15+10, f"hand_para shape: {hand_para.shape}"
            hand_para, hand_shape = torch.split(hand_para, [3+3+15, 10], dim=-1)

        body_params_dict = {
            "transl": hand_para[:, 3:6],
            "global_orient": hand_para[:, :3],
            "betas": hand_shape,
            "hand_pose": hand_para[:, 6:],
        }
        return body_params_dict

    @staticmethod
    def dict2para(body_params_dict, side="right", merge=False):
        if not isinstance(body_params_dict["transl"], torch.Tensor):
            body_params_dict_pt = {k: torch.FloatTensor(v) for k, v in body_params_dict.items()}
        else:
            body_params_dict_pt = body_params_dict
        hand_para = torch.cat(
            [
                body_params_dict_pt["global_orient"],
                body_params_dict_pt["transl"],
                body_params_dict_pt["hand_pose"],
            ],
            dim=-1,
        )
        hand_shape = body_params_dict_pt["betas"]
        if merge:
            return torch.cat([hand_para, hand_shape], dim=-1)
        else:
            return hand_para, hand_shape

    def hand_para2verts_faces_joints(self, hand_para, hand_shape=None, side="right"):
        """
        :param hand_para: (B, T, 3+3+15)
        :param hand_shape: (B, T, 10)
        :param side: _description_, defaults to 'right'
        """
        if hand_shape is None:
            assert hand_para.shape[-1] == 3+3+15+10, f"hand_para shape: {hand_para.shape}"
            hand_para, hand_shape = torch.split(hand_para, [3+3+15, 10], dim=-1)

        pref_dim = hand_para.shape[:-1]
        hand_para = hand_para.reshape(-1, hand_para.shape[-1])
        hand_shape = hand_shape.reshape(-1, hand_shape.shape[-1])

        model = self.sided_mano_models[side]
        if isinstance(hand_para, torch.Tensor):
            body_params_dict = {
                "transl": hand_para[:, 3:6],
                "global_orient": hand_para[:, :3],
                "betas": hand_shape,
                "hand_pose": hand_para[:, 6:],
            }
        elif isinstance(hand_para, dict):
            body_params_dict = hand_para
        else:
            raise ValueError(f"Invalid hand_para type: {type(hand_para)}")

        mano_out = model(**body_params_dict)
        hand_verts = mano_out.vertices
        hand_faces = model.faces_tensor.repeat(hand_verts.shape[0], 1, 1)
        hand_joints = mano_out.joints

        hand_verts = hand_verts.reshape(*pref_dim, -1, 3)
        hand_faces = hand_faces.reshape(*pref_dim, -1, 3)
        hand_joints = hand_joints.reshape(*pref_dim, *hand_joints.shape[-2:])

        return hand_verts, hand_faces, hand_joints




def rr_vis(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w, t_c2w):
    rr.init(output_pth + '.rrd')
    rr.save(output_pth + '.rrd')
    
    left_dict['vertices'] = left_dict['vertices'].cpu().numpy()[0]
    right_dict['vertices'] = right_dict['vertices'].cpu().numpy()[0]
    R_c2w = R_c2w.cpu().numpy()
    t_c2w = t_c2w.cpu().numpy()
    f = img_focal
    print('faocal', f)
    W, H = Image.open(image_names[0]).size
    for i in range(len(image_names)):

        rr.set_time_sequence("frame", i)
        if i == 0:
            rr.log("world/camera", rr.Pinhole(width=W, height=H, focal_length=float(f)))

        downsample_factor = 1
        image = np.array(Image.open(image_names[i]))
        image = image[::downsample_factor, ::downsample_factor]
        rr.log('images', rr.Image(image))
        
        rr.log("world/left_hand", rr.Mesh3D(vertex_positions=left_dict['vertices'][i], triangle_indices=left_dict['faces']))
        rr.log("world/right_hand", rr.Mesh3D(vertex_positions=right_dict['vertices'][i], triangle_indices=right_dict['faces']))
        rr.log("world/camera", rr.Transform3D(translation=t_c2w[i], rotation=rr.Quaternion(xyzw=Rotation.from_matrix(R_c2w[i]).as_quat())))
    


def my_vis(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w, t_c2w):
    device = 'cuda'
    B = R_c2w.shape[0]
    H = 720
    verts_left = left_dict['vertices'][0].to(device)
    faces_left = torch.FloatTensor(left_dict['faces']).to(device)[None].repeat(B, 1, 1)
    verts_right = right_dict['vertices'][0].to(device)
    faces_right = torch.FloatTensor(right_dict['faces']).to(device)[None].repeat(B, 1, 1)
    left_hand = Meshes(verts=verts_left, faces=faces_left).to(device)
    right_hand = Meshes(verts=verts_right, faces=faces_right).to(device)
    left_hand.textures = mesh_utils.pad_texture(left_hand, 'white')
    right_hand.textures = mesh_utils.pad_texture(right_hand, 'blue')
    scene = mesh_utils.join_scene([left_hand, right_hand])


    wTc = geom_utils.rt_to_homo(R_c2w, t_c2w)

    coord = plot_utils.create_coord(device, B, size=0.2)
    verts, faces = plot_utils.vis_cam(wTc=wTc, color='red', size=0.1, focal=img_focal/H * 2, homo=True)
    camera = Meshes(verts=verts, faces=faces).to(device)
    camera.textures = mesh_utils.pad_texture(camera, 'red')

    scene = mesh_utils.join_scene([scene, coord, camera])
    verts = scene.verts_packed()

    nTw = mesh_utils.get_nTw(verts[None], new_scale=1.2)

    image_list =mesh_utils.render_geom_rot_v2(scene, 'az', nTw=nTw, time_len=3)  # (V, B, H, W, 3)
    image_list = torch.stack(image_list, axis=0)
    V, B, _, H, W = image_list.shape
    image_list = image_list.reshape(B*V, 1, 3, H, W)
    image_utils.save_gif(image_list, output_pth, fps=30, ext=".mp4")


def do_one(args):
    down_sample = 4
    seq = osp.basename(args.video_path).replace('-rot90', '')

    gt_folder = f'/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/{seq}.npz'
    print(seq, args.video_path)
    gt_data = dict(np.load(gt_folder, allow_pickle=True))
    wTc = torch.FloatTensor(gt_data['wTc'])
    if args.gt_cam:
        print(gt_data.keys(), gt_data['intrinsic'])
        intrinsics = deepcopy(gt_data['intrinsic'])
        intrinsics[:2] = intrinsics[:2] / down_sample
        print(intrinsics.shape, intrinsics, gt_data['intrinsic'])

        args.img_focal = intrinsics[0, 0]

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args, gt_folder, sided_wrapper=sided_wrapper)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    print('img_focal', img_focal)
    if args.gt_cam:
        intrinsics = intrinsics
    else:
        H, W = Image.open(imgfiles[0]).size
        intrinsics = np.array([[args.img_focal, 0, W/2], [0, args.img_focal, H/2], [0, 0, 1]])

    if args.gt_cam:
        # R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_gt_cam(wTc)
        R_c2w_sla_all = wTc[:, :3, :3]
        t_c2w_sla_all = wTc[:, :3, 3]

        cTw = geom_utils.inverse_rt_v2(wTc)
        R_w2c_sla_all = cTw[:, :3, :3]
        t_w2c_sla_all = cTw[:, :3, 3]

    else:
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

        gt_wTc = wTc
        wTc_slam = geom_utils.rt_to_homo(R_c2w_sla_all, t_c2w_sla_all)
        delta = gt_wTc @ geom_utils.inverse_rt_v2(wTc_slam)

    
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, vis_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all, R_c2w_sla_all=R_c2w_sla_all, t_c2w_sla_all=t_c2w_sla_all)
    print(pred_valid[:, 130:], pred_valid.shape)
    print(vis_valid[:, 130:], vis_valid.shape)

    data = cvt2my_data(wrapper, pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, vis_valid, R_c2w_sla_all, t_c2w_sla_all, intrinsics)
    # import ipdb; ipdb.set_trace()

    # for k, v in gt_data.items():
    #     if k not in data:
    #         data[k] = v
    # print(data.keys())
    seq = osp.basename(args.video_path).replace('-rot90', '')
    save_path = os.path.join(args.output_folder, f"{seq}.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **data)
    print(f"Saved to {save_path}")
    
    if args.vis:
        # vis sequence for this video
        hand2idx = {
            "right": 1,
            "left": 0
        }
        vis_start = 0
        vis_end = pred_trans.shape[1] - 1
                
        # get faces
        faces = get_mano_faces()
        faces_new = np.array([[92, 38, 234],
                [234, 38, 239],
                [38, 122, 239],
                [239, 122, 279],
                [122, 118, 279],
                [279, 118, 215],
                [118, 117, 215],
                [215, 117, 214],
                [117, 119, 214],
                [214, 119, 121],
                [119, 120, 121],
                [121, 120, 78],
                [120, 108, 78],
                [78, 108, 79]])
        faces_right = np.concatenate([faces, faces_new], axis=0)

        # get right hand vertices
        hand = 'right'
        hand_idx = hand2idx[hand]
        pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
        right_verts = pred_glob_r['vertices'][0]
        right_dict = {
                'vertices': right_verts.unsqueeze(0),
                'faces': faces_right,
            }

        # get left hand vertices
        faces_left = faces_right[:,[0,2,1]]
        hand = 'left'
        hand_idx = hand2idx[hand]
        pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
        left_verts = pred_glob_l['vertices'][0]
        left_dict = {
                'vertices': left_verts.unsqueeze(0),
                'faces': faces_left,
            }

        # R_x = torch.tensor([[1,  0,  0],
        #                     [0, -1,  0],
        #                     [0,  0, -1]]).float()
        # R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
        # t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
        # R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
        # t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
        # left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
        # right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
        
        
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}_gt{args.gt_cam}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        my_vis(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end])
        rr_vis(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end])
    

def cvt2my_data(wrapper: hand_utils.ManopthWrapper, pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, vis_valid, R_c2w_sla_all, t_c2w_sla_all, intrinsics):
    # print(pred_trans.shape, pred_rot.shape, pred_hand_pose.shape, pred_betas.shape, pred_valid.shape, R_c2w_sla_all.shape, t_c2w_sla_all.shape, intrinsics.shape)

    data = {}
    hand2idx = {
        "right": 1,
        "left": 0
    }
    for hand in ["left", "right"]:
        hand_idx = hand2idx[hand]
        
        pca = wrapper.pose_to_pca(pred_hand_pose[hand_idx] + wrapper.hand_mean, 15)
        param_dict = {
            "transl": pred_trans[hand_idx],
            "global_orient": pred_rot[hand_idx],
            "hand_pose": pca,
            "betas": pred_betas[hand_idx],
        }
        theta, shape = HandWrapper.dict2para(param_dict, side=hand)

        data[f"{hand}_hand_theta"] = theta
        data[f"{hand}_hand_shape"] = shape

        data[f"{hand}_hand_valid"] = pred_valid[hand_idx]
        data[f"{hand}_hand_vis"] = vis_valid[hand_idx]

    wTc = geom_utils.rt_to_homo(R_c2w_sla_all, t_c2w_sla_all)
    data["wTc"] = wTc
    data["intrinsic"] = intrinsics

    # for k, v in data.items():
    #     if isinstance(v, torch.Tensor):
    #         data[k] = v.cpu().numpy()
    #     else:
    #         data[k] = v
    return data
    


def batch_preprocess(func):
    # clips = sorted([p for p in os.listdir(args.clips_dir, 'train_aria') if p.endswith(".tar")])
    clips_aria = sorted(glob(os.path.join(args.clips_dir, "train_aria", "*.tar")))
    clips_quest = sorted(glob(os.path.join(args.clips_dir, "train_quest3", "*.tar")))
    clips = clips_aria + clips_quest
    for clip in tqdm(clips):
        
        lock_file = osp.join(args.preprocess_dir, f"lock.box{args.gt_box}", os.path.basename(clip))
        done_file = osp.join(args.preprocess_dir, f"done.box{args.gt_box}", os.path.basename(clip))

        if osp.exists(done_file) and not args.no_skip:
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            if not args.no_skip:
                continue

        # get_one_clip(clip, vis=False)
        seq = osp.basename(clip).replace('.tar', '')
        print(seq)
        video_args = config_args(args, seq)
        func(video_args)

        os.makedirs(done_file, exist_ok=True)
        os.rmdir(lock_file)


def config_args(args, seq):
    video_args = deepcopy(args)
    ws_dir = osp.join(args.preprocess_dir, 'ws', seq)
    src_img_dir = osp.join(args.clips_dir, 'extract_images-rot90', seq + '/')
    dst_img_dir = osp.join(ws_dir, 'extracted_images')

    # soft link src_img_dir to dst_img_dir
    os.makedirs(osp.dirname(dst_img_dir), exist_ok=True)
    if not os.path.exists(dst_img_dir):
        cmd = f"ln -s {src_img_dir} {dst_img_dir}"
        print(cmd)
        os.system(cmd)
        # os.symlink(src_img_dir, dst_img_dir, target_is_directory=True)
    else:
        print(f"Skipping symlinking {src_img_dir} to {dst_img_dir} because it already exists")

    video_args.video_path = ws_dir
    # video_args.video_path = os.path.join(args.clips_dir, 'extracted_images-rot90', seq)
    suf = '_gt' if args.gt_box else ''
    video_args.output_folder = os.path.join(args.preprocess_dir + suf) 

    return video_args



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=2263)
    parser.add_argument("--clips_dir", type=str, default='/move/u/yufeiy2/egorecon/data/HOT3D-CLIP')
    parser.add_argument("--preprocess_dir", type=str, default='/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/hawor')
    parser.add_argument("--img_focal", type=float, default=152)
    parser.add_argument("--video_path", type=str, default='example/clip-002354-rot90.mp4')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--output_folder", type=str, default='./example/clip-002354-rot90_out')
    parser.add_argument("--vis_mode",  type=str, default='world', help='cam | world')
    parser.add_argument("--gt_box", action='store_true')
    parser.add_argument("--gt_cam", action='store_true')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--no_skip", action='store_true')
    args = parser.parse_args()
    # args.preprocess_dir = args.preprocess_dir.rstrip('/') + f'_gtcam{args.gt_cam}'
    if args.gt_cam:
        dir_suf = "camGT"
    else:
        dir_suf = "camSLAM"
    args.preprocess_dir = args.preprocess_dir.rstrip('/') + f'_{dir_suf}'

    wrapper = hand_utils.ManopthWrapper()
    sided_wrapper = {
        "left": hand_utils.ManopthWrapper(side="left"),
        "right": hand_utils.ManopthWrapper(side="right"),
    }

    batch_preprocess(do_one)

    # video_args = config_args(args, f'clip-{args.seq:06d}')
    # do_one(video_args)

    # do_one(args)

