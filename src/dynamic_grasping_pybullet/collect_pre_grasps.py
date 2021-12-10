from __future__ import division
import os
import argparse
import numpy as np
import pandas as pd
import tqdm
import pybullet as p
import pybullet_data
import trimesh
import pybullet_utils as pu
import grasp_utils as gu


""" 
Generate pregrasps for filtered grasp database
 
Given grasp database structure:

- bleach_cleanser
    - grasps_eef.npy
    - grasps_link6_com.npy
    - grasps_link6_ref.npy
- cube
    - grasps_eef.npy
    - grasps_link6_com.npy
    - grasps_link6_ref.npy
- ...
    
output:

- bleach_cleanser
    - grasps_eef.npy
    - grasps_link6_com.npy
    - grasps_link6_ref.npy
    - pre_grasps_eef_0.05.npy
    - pre_grasps_link6_ref.npy
- cube
    - grasps_eef.npy
    - grasps_link6_com.npy
    - grasps_link6_ref.npy
    - pre_grasps_eef_0.05.npy
    - pre_grasps_link6_ref.npy
- ...

"""


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--grasp_database', type=str, required=True)
    parser.add_argument('--back_off', type=float, default=0.05)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    object_names = os.listdir(args.grasp_database)
    for object_name in object_names:
        grasps_eef = np.load(os.path.join(args.grasp_database, object_name, 'grasps_eef.npy'))
        grasps_link6_ref = np.load(os.path.join(args.grasp_database, object_name, 'grasps_link6_ref.npy'))
        grasps_link6_com = np.load(os.path.join(args.grasp_database, object_name, 'grasps_link6_com.npy'))
        pre_grasps_eef = np.zeros(grasps_eef.shape)
        pre_grasps_link6_ref = np.zeros(grasps_eef.shape)
        pre_grasps_link6_com = np.zeros(grasps_eef.shape)
        bar = tqdm.tqdm(total=len(grasps_eef))
        for i, (grasp_eef_in_object, grasp_link6_ref_in_object, grasp_link6_com_in_object) in\
                enumerate(zip(grasps_eef, grasps_link6_ref, grasps_link6_com)):
            pre_grasp_eef_in_object = gu.back_off_pose_2d(pu.split_7d(grasp_eef_in_object), -args.back_off)
            pre_grasp_link6_ref_in_object = gu.back_off_pose_2d(pu.split_7d(grasp_link6_ref_in_object), args.back_off)
            pre_grasp_link6_com_in_object = gu.back_off_pose_2d(pu.split_7d(grasp_link6_com_in_object), args.back_off)
            pre_grasps_eef[i] = pu.merge_pose_2d(pre_grasp_eef_in_object)
            pre_grasps_link6_ref[i] = pu.merge_pose_2d(pre_grasp_link6_ref_in_object)
            pre_grasps_link6_com[i] = pu.merge_pose_2d(pre_grasp_link6_com_in_object)
            bar.update(1)
            bar.set_description('{}'.format(object_name))
        bar.close()
        np.save(os.path.join(args.grasp_database, object_name, 'pre_grasps_eef_'+str(args.back_off)+'.npy'), pre_grasps_eef)
        np.save(os.path.join(args.grasp_database, object_name, 'pre_grasps_link6_ref_'+str(args.back_off)+'.npy'), pre_grasps_link6_ref)
        np.save(os.path.join(args.grasp_database, object_name, 'pre_grasps_link6_com_'+str(args.back_off)+'.npy'), pre_grasps_link6_com)