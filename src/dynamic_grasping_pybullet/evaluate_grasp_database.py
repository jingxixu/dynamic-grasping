from __future__ import division
import os
import argparse
import numpy as np
import pandas as pd
import tqdm
import pybullet as p
import pybullet_data
from collect_good_grasps import EEFOnlyStaticWorld
import trimesh
import pybullet_utils as pu
import grasp_utils as gu
import misc_utils as mu


""" Given a grasp database, evaluate the success rate of each object """


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--grasp_database', type=str, required=True)
    parser.add_argument('--back_off', type=float, default=0.05)
    parser.add_argument('--robot_config_name', type=str, default='mico',
                        help="name of robot configs to load from grasputils. Ex: mico or ur5_robotiq")
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')

    return args


if __name__ == "__main__":
    args = get_args()
    p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    pu.reset_camera(yaw=50.0, pitch=-35.0, dist=1.2)

    object_names = os.listdir(args.grasp_database)
    for object_name in object_names:
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(object_name), '{}.obj'.format(object_name))
        object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
        target_urdf = mu.create_object_urdf(object_mesh_filepath, object_name)
        target_mesh = trimesh.load_mesh(object_mesh_filepath)
        floor_offset = target_mesh.bounds.min(0)[2]
        target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]
        gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]

        world = EEFOnlyStaticWorld(target_initial_pose, gripper_initial_pose, args.robot_config_name, target_urdf, False)

        successes = []
        actual_grasps, graspit_grasps = gu.load_grasp_database_new(args.grasp_database, object_name)
        bar = tqdm.tqdm(total=len(actual_grasps))
        use_actual = False
        grasps = actual_grasps if use_actual else  graspit_grasps
        for graspit_grasp_in_object in grasps:
            world.reset()
            object_pose = p.getBasePositionAndOrientation(world.target)
            success_height_threshold = object_pose[0][2] + world.controller.LIFT_VALUE - 0.05
            graspit_grasp_in_object = pu.split_7d(graspit_grasp_in_object)
            graspit_grasp_in_world = gu.convert_grasp_in_object_to_world(object_pose, graspit_grasp_in_object)

            world.controller.execute_grasp(graspit_grasp_in_world, args.back_off)
            success = p.getBasePositionAndOrientation(world.target)[0][2] > success_height_threshold
            successes.append(success)
            bar.update(1)
            bar.set_description('{} | {:.4f}'.format(object_name, np.count_nonzero(successes)/len(successes)))

        bar.close()