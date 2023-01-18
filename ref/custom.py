# encoding: utf-8
"""This file includes necessary params, info."""
import os.path as osp
import mmcv
import numpy as np
import json
from glob import glob

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "datasets")

# ---------------------------------------------------------------- #
# CUSTOM DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(data_root, "CUSTOM/")
test_dir = osp.join(dataset_root, "test")
model_dir = osp.join(dataset_root, "models")
# model_eval_dir = osp.join(dataset_root, "models_eval")
vertex_scale = 0.001

# object info
objects = ['custom_obj']
id2obj = {i: v for i, v in enumerate(objects, start=1)}
obj2id = {_name: _id for _id, _name in id2obj.items()}
obj_num = len(id2obj)

model_paths = [osp.join(model_dir, "model_mm.ply")]
texture_paths = None
# model_colors = [((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)]  # for renderer
# 
# diameters = (
#     np.array(
#         [
#             64.0944,
#             51.4741,
#             142.15,
#             139.379,
#             158.583,
#             85.3086,
#             38.5388,
#             68.884,
#             94.8011,
#             55.7152,
#             140.121,
#             107.703,
#             128.059,
#             102.883,
#             114.191,
#             193.148,
#             77.7869,
#             108.482,
#             121.383,
#             122.019,
#             171.23,
#             267.47,
#             56.9323,
#             65,
#             48.5103,
#             66.8026,
#             55.7315,
#             24.0832,
#         ]
#     )
#     / 1000.0
# )
# 
# Camera info
camera_json = sorted(glob(osp.join(test_dir, '*', 'scene_cameras.json')))[0]
with open(camera_json, 'r') as f:
    camera_json = json.load(f)
camera_info = camera_json[list(camera_json.keys())[0]]
# width = 1280
# height = 960
zNear = 0.25
zFar = 6.0
camera_matrix = np.array([[camera_info['fx'], 0.0, camera_info['ppx']], [0.0, camera_info['fy'], camera_info['ppy']], [0, 0, 1]])
del camera_json
del camera_info
 
# def get_models_info():
#     """key is str(obj_id)"""
#     models_info_path = osp.join(model_dir, "models_info.json")
#     assert osp.exists(models_info_path), models_info_path
#     models_info = mmcv.load(models_info_path)  # key is str(obj_id)
#     return models_info
# 
# 
# # ref core/gdrn_modeling/tools/itodd/itodd_1_compute_fps.py
# def get_fps_points():
#     fps_points_path = osp.join(model_dir, "fps_points.pkl")
#     assert osp.exists(fps_points_path), fps_points_path
#     fps_dict = mmcv.load(fps_points_path)
#     return fps_dict
# 
# 
# # ref core/roi_pvnet/tools/itodd/itodd_1_compute_keypoints_3d.py
# def get_keypoints_3d():
#     keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
#     assert osp.exists(keypoints_3d_path), keypoints_3d_path
#     kpts_dict = mmcv.load(keypoints_3d_path)
#     return kpts_dict
