for quant and training : use GT box, GT cam

GT box GT cam

python -m move_utils.slurm_wrapper \
 --sl_time_hr 24 --sl_name gt --sl_ngpu 1 --slurm \
 python -m infer_hot3d --gt_cam --gt_box --preprocess_dir /move/u/yufeiy2/egorecon/data/HOT3D-CLIP/hawor_v2



GT box SLAM cam
 
python -m move_utils.slurm_wrapper \
 --sl_time_hr 24 --sl_name slam_gtbox --sl_ngpu 1 --slurm \
 python -m infer_hot3d --gt_box --preprocess_dir /move/u/yufeiy2/egorecon/data/HOT3D-CLIP/hawor_v2

