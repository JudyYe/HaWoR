import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import torch
import cv2
from tqdm import tqdm
import subprocess

import argparse
import numpy as np
from glob import glob
from lib.pipeline.tools import detect_track
from natsort import natsorted
from hawor.utils.process import get_imgfiles, run_mano, run_mano_left
from lib.eval_utils.custom_utils import cam_to_img
from jutils import hand_utils, geom_utils
from copy import deepcopy


if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = [
        'ffmpeg',               
        '-i', video_path,       
        '-vf', 'fps=30',         
        '-start_number', '0',
        os.path.join(output_folder, '%04d.jpg')  
    ]

    subprocess.run(command, check=True)


def detect_track_video(args, gt_file=None, sided_wrapper=None):
    imgfiles, seq_folder = get_imgfiles(args)
    print(f'Running detect_track on {seq_folder} ...')
    ##### Detection + Track #####
    print('Detect and Track ...')

    start_idx = 0
    end_idx = len(imgfiles)

    suf = '_gt' if args.gt_box else ''
    if os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes{suf}.npy'):
        print(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, imgfiles
    
    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)
    if args.gt_box:
        boxes_, tracks_ = detect_track_video_gt(args, gt_file, sided_wrapper=sided_wrapper)
    else:
        boxes_, tracks_ = detect_track(imgfiles, thresh=0.2)
    
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes{suf}.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks{suf}.npy', tracks_)

    return start_idx, end_idx, seq_folder, imgfiles


def cvtfrom_gt(theta, shape, hand_wrapper:hand_utils.ManopthWrapper=None):
    global_orient = theta[..., :3]
    transl = theta[..., 3:6]
    pca = theta[..., 6:]

    pose = hand_wrapper.pca_to_pose(torch.FloatTensor(pca), add_mean=False)  # seems like run_mano assume a pose substracted by hand mean???? 
    
    return torch.FloatTensor(transl[None]), torch.FloatTensor(global_orient[None]), pose[None], torch.FloatTensor(shape[None])



def detect_track_video_gt(args, gt_file, fix_shapedirs=True, dataset_type='hotclip', down_sample=4, sided_wrapper=None):
    # the images are already downsampled by down_sample, intrinsics has not; 
    # bbox: will be saved after downsampling
    # for HOT3D-CLIP dataset
    # gt_file = ''
    imgfiles, seq_folder = get_imgfiles(args)

    gt_data = dict(np.load(gt_file, allow_pickle=True))
    wTc = torch.FloatTensor(gt_data['wTc'])
    R_c2w_sla_all = wTc[:, :3, :3]
    t_c2w_sla_all = wTc[:, :3, 3]

    cTw = geom_utils.inverse_rt_v2(wTc)
    R_w2c_gt = cTw[:, :3, :3]
    t_w2c_gt = cTw[:, :3, 3]
    # pretend 
    intrinsics = deepcopy(gt_data['intrinsic'])
    intrinsics[:2] = intrinsics[:2] / down_sample

    K = torch.FloatTensor(intrinsics)

    gt_trans_l, gt_rot_l, gt_pose_l, gt_betas_l = cvtfrom_gt(gt_data["left_hand_theta"], gt_data["left_hand_shape"], sided_wrapper["left"])
    gt_trans_r, gt_rot_r, gt_pose_r, gt_betas_r = cvtfrom_gt(gt_data["right_hand_theta"], gt_data["right_hand_shape"], sided_wrapper["right"])
    mano_valid = True

    # get joints
    # gt_trans_l =   world_trans[0:1]  # (B, T, 3)
    # gt_rot_l = world_rot[0:1]
    # gt_pose_l = world_hand_pose[0:1]  # with hand mean or not?
    # gt_betas_l = world_betas[0:1]
    # gt_trans_r = world_trans[1:2]
    # gt_rot_r = world_rot[1:2]
    # gt_pose_r = world_hand_pose[1:2]
    # gt_betas_r = world_betas[1:2]

    target_glob_l = run_mano_left(gt_trans_l, gt_rot_l, gt_pose_l, betas=gt_betas_l, fix_shapedirs=fix_shapedirs)
    target_glob_r = run_mano(gt_trans_r, gt_rot_r, gt_pose_r, betas=gt_betas_r)
    world_joints = torch.stack((target_glob_l['joints'][0], target_glob_r['joints'][0]), dim=0).cpu() # B, T, 21, 3 

    # R_w2c_gt, t_w2c_gt, _, _ = load_gt_cam(video_root, video, dataset_type=dataset_type)

    cam_j3d = torch.einsum("tij,btnj->btni", R_w2c_gt, world_joints) + t_w2c_gt[None, :, None, :]
    img_cv2 = cv2.imread(imgfiles[0])
    H, W, _ = img_cv2.shape

    cam_j2d = cam_to_img(cam_j3d, K) # max value is H or W (B, T, 21, 2)
    x_coords = cam_j2d[..., 0]
    y_coords = cam_j2d[..., 1]
    
    valid_x = (x_coords >= 0) & (x_coords < W)
    valid_y = (y_coords >= 0) & (y_coords < H)
    valid = valid_x & valid_y
    valid_j2d = torch.sum(valid, axis=-1) >= 2 # (B,T)
    valid_j2d = valid_j2d & mano_valid

    # Run
    boxes_ = []
    handedness_ = []
    for t, imgpath in enumerate(tqdm(imgfiles)):
        with torch.no_grad():
            with autocast():
                # use GT
                boxes = []
                confs = []
                handedness = []
                if valid_j2d[0, t]: # has left hand
                    det_w = x_coords[0, t].max() - x_coords[0, t].min()
                    det_h = y_coords[0, t].max() - y_coords[0, t].min()
                    xmin = max(0, x_coords[0, t].min()-0.2*det_w)
                    ymin = max(0, y_coords[0, t].min()-0.2*det_h)
                    xmax = min(W, x_coords[0, t].max()+0.2*det_w)
                    ymax = min(H, y_coords[0, t].max()+0.2*det_h)
                    boxes.append([xmin, ymin, xmax, ymax])
                    confs.append(1)
                    handedness.append(0)
                if valid_j2d[1, t]: # has right hand
                    det_w = x_coords[1, t].max() - x_coords[1, t].min()
                    det_h = y_coords[1, t].max() - y_coords[1, t].min()
                    xmin = max(0, x_coords[1, t].min()-0.2*det_w)
                    ymin = max(0, y_coords[1, t].min()-0.2*det_h)
                    xmax = min(W, x_coords[1, t].max()+0.2*det_w)
                    ymax = min(H, y_coords[1, t].max()+0.2*det_h)
                    boxes.append([xmin, ymin, xmax, ymax])
                    confs.append(1)
                    handedness.append(1)
                if len(boxes):
                    boxes = np.array(boxes)
                    confs = np.array(confs)
                    handedness = np.array(handedness)
                else:
                    boxes = np.zeros((0,4))
                    confs = np.zeros((0,))
                    handedness = np.zeros((0,))
                boxes = np.hstack([boxes, confs[:, None]])

        boxes_.append(boxes)
        handedness_.append(handedness)

    ### --- Adapt tracks data structure ---
    tracks = {}
    for frame in range(len(boxes_)):         
        handedness = handedness_[frame]
        boxes = boxes_[frame]
        for idx in [0, 1]:  
            subj = {}
            sub_index = np.where(handedness == idx)[0][0] if np.any(handedness == idx) else None
            # add fields
            subj['frame'] = frame 
            if sub_index is None:
                subj['det'] = False
                subj['det_box'] = np.zeros([1, 5])
                subj['det_handedness'] = np.zeros([1,])
            else:
                subj['det'] = True
                subj['det_box'] = boxes[[sub_index]]
                subj['det_handedness'] = handedness[[sub_index]]
            
            if idx in tracks:
                tracks[idx].append(subj)
            else:
                tracks[idx] = [subj]

    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, tracks


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    args = parser.parse_args()

    detect_track_video(args)