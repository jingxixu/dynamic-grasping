from __future__ import division
import argparse
import pybullet as p
import os
import pybullet_data
from grasp_evaluation_eef_only import create_object_urdf, convert_grasp_in_object_to_world
import trimesh
import pandas as pd
import numpy as np
import grasp_utils as gu
import pybullet_helper as ph


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='power_drill',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--grasp_dir', type=str, default='grasp_dir_first_run/power_drill')

    args = parser.parse_args()
    args.mesh_dir = os.path.abspath('dynamic_grasping_assets/models')
    args.result_file_path = os.path.join(args.grasp_dir, 'result.csv')

    return args


def load_grasp(grasp_file_path, target_id):
    LINK6_COM = [-0.002216, -0.000001, -0.058489]
    link6_reference_to_link6_com = (LINK6_COM, [0.0, 0.0, 0.0, 1.0])
    grasp_in_object = np.load(grasp_file_path, allow_pickle=True)
    grasp_in_world = convert_grasp_in_object_to_world(p.getBasePositionAndOrientation(target_id), grasp_in_object)
    grasp_in_world = gu.change_end_effector_link(ph.list_2_pose(grasp_in_world), link6_reference_to_link6_com)
    # grasp_in_world = ph.list_2_pose(grasp_in_world)
    return grasp_in_world


if __name__ == "__main__":
    args = get_args()
    p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetSimulation()
    # p.setGravity(0, 0, -9.8)

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]

    # plane = p.loadURDF("plane.urdf")
    target = p.loadURDF(target_urdf, target_initial_pose[0], target_initial_pose[1])

    df = pd.read_csv(args.result_file_path, index_col=0)

    for index, row in df.iterrows():
        if row['num_successes'] / row['num_trials'] >= 1:
            print(index)
            grasp_file_path = os.path.join(args.grasp_dir, row['grasp_fnm'])
            grasp_in_world = load_grasp(grasp_file_path, target)
            ph.create_arrow_marker(grasp_in_world)
            print("here")

    print('here')