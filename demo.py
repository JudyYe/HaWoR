
import argparse
import sys
import os

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

def rr_vis(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w, t_c2w):
    rr.init(output_pth + '.rrd')
    rr.save(output_pth + '.rrd')
    print("Saving rrd file to ", output_pth + '.rrd')
    
    left_dict['vertices'] = left_dict['vertices'].cpu().numpy()[0]
    right_dict['vertices'] = right_dict['vertices'].cpu().numpy()[0]
    R_c2w = R_c2w.cpu().numpy()
    t_c2w = t_c2w.cpu().numpy()
    f = img_focal
    W, H = Image.open(image_names[0]).size
    for i in range(len(image_names)):

        rr.set_time_sequence("frame", i)
        if i == 0:
            print(W, H, f)
            rr.log("world/camera", rr.Pinhole(width=W, height=H, focal_length=f))

        downsample_factor = 4
        image = np.array(Image.open(image_names[i]))
        image = image[::downsample_factor, ::downsample_factor]
        rr.log('images', rr.Image(image))
        
        rr.log("world/left_hand", rr.Mesh3D(vertex_positions=left_dict['vertices'][i], triangle_indices=left_dict['faces']))
        rr.log("world/right_hand", rr.Mesh3D(vertex_positions=right_dict['vertices'][i], triangle_indices=right_dict['faces']))
        print(t_c2w[i], R_c2w[i])
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='example/video_0.mp4')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode",  type=str, default='world', help='cam | world')
    args = parser.parse_args()

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
    print(pred_valid[0][120], pred_hand_pose[0][120])
    import ipdb; ipdb.set_trace()

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

    R_x = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
    
    
    output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    image_names = imgfiles[vis_start:vis_end]
    my_vis(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end])
    rr_vis(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end])
    # # Here we use aitviewer(https://github.com/eth-ait/aitviewer) for simple visualization.
    # if args.vis_mode == 'world': 
    #     output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
    #     if not os.path.exists(output_pth):
    #         os.makedirs(output_pth)
    #     image_names = imgfiles[vis_start:vis_end]
    #     print(f"vis {vis_start} to {vis_end}")
    #     run_vis2_on_video(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end], interactive=False)
    # elif args.vis_mode == 'cam':
    #     output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
    #     if not os.path.exists(output_pth):
    #         os.makedirs(output_pth)
    #     image_names = imgfiles[vis_start:vis_end]
    #     print(f"vis {vis_start} to {vis_end}")
    #     run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, R_w2c=R_w2c_sla_all[vis_start:vis_end], t_w2c=t_w2c_sla_all[vis_start:vis_end])

    # print("finish")


