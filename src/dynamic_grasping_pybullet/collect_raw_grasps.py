import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import trimesh
import argparse
import grasp_utils as gu
import pybullet_utils as pu
from collections import OrderedDict
import csv
import tqdm
import tf_conversions

""" 
Collect raw grasps from GraspIt! 

Return a (num_grasps, 7) numpy array.

Each grasp is in the target object link6 reference frame.

Add to a grasp_folder_path, e.g.

raw_grasps/
    - bleach_cleanser.npy
    - cube.npy
    - mustard_bottle.npy
    - potted_meat_can.npy
    - power_drill.npy
    - sugar_box.npy
    - tomato_soup_can.npy
"""


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--grasp_folder_path', type=str, required=True,
                        help="Directory to store grasps and results. Ex: grasps_dir")
    parser.add_argument('--num_grasps', type=int, default=5000)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--robot_name', type=str, required=True)
    parser.add_argument('--uniform_grasp', action='store_true', default=False)
    parser.add_argument('--rotate_roll', action='store_true', default=False)
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')

    if not os.path.exists(args.grasp_folder_path):
        os.makedirs(args.grasp_folder_path)

    return args


if __name__ == "__main__":
    args = get_args()

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]

    raw_grasps = []

    progressbar = tqdm.tqdm(total=args.num_grasps, desc=args.object_name)
    num_grasp = 0
    while num_grasp < args.num_grasps:
        graspit_grasps, graspit_grasp_poses_in_world, graspit_grasp_poses_in_object \
            = gu.generate_grasps(robot_name=args.robot_name,
                                 object_mesh=object_mesh_filepath_ply,
                                 uniform_grasp=args.uniform_grasp,
                                 floor_offset=floor_offset,
                                 max_steps=args.max_steps,
                                 rotate_roll=args.rotate_roll)
        for g_pose in graspit_grasp_poses_in_object:
            g_2d = gu.pose_2_list(g_pose)
            raw_grasps.append(g_2d[0] + g_2d[1])
            num_grasp += 1
            progressbar.update(1)
            if num_grasp >= args.num_grasps and not args.uniform_grasp:
                break
        if args.uniform_grasp:
            break
    progressbar.close()
    raw_grasps_npy = np.array(raw_grasps)
    dest_path = os.path.join(args.grasp_folder_path, args.object_name+'.npy')
    np.save(dest_path, raw_grasps_npy)
    print("Raw grasps saved in {}!".format(dest_path))
