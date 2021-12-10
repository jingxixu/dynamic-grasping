import pprint
from collections import OrderedDict
import os
import csv
import pybullet_data
import pybullet as p
import pybullet_utils as pu
import pandas as pd
from math import radians, cos, sin
import tf_conversions as tfc
import numpy as np
import random
from shapely.geometry import Polygon, Point


def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file and write the first line if the file does not already exist """
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def configure_pybullet(rendering=False, debug=False, yaw=50.0, pitch=-35.0, dist=1.2, target=(0.0, 0.0, 0.0)):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    pu.reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)


def create_object_urdf(object_mesh_filepath, object_name,
                       urdf_template_filepath='assets/object_template.urdf',
                       urdf_target_object_filepath='assets/target_object.urdf'):
    # set_up urdf
    os.system('cp {} {}'.format(urdf_template_filepath, urdf_target_object_filepath))
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name.obj', object_mesh_filepath, urdf_target_object_filepath)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name', object_name, urdf_target_object_filepath)
    os.system(sed_cmd)
    return urdf_target_object_filepath


def get_candidate_indices(prior_csv_file_path, prior_success_rate):
    df = pd.read_csv(prior_csv_file_path, index_col=0)
    df_success = df.loc[df['success_rate'] >= prior_success_rate]
    return list(df_success.index)


def calculate_target_pose(start_pose_in_world, angle, distance):
    """ calculate end pose after translating for a distance (in meter) in the direction of angle (in degrees) """
    start_pose_in_object= [[0, 0, 0], [0, 0, 0, 1]]
    target_x = cos(radians(angle)) * distance
    target_y = sin(radians(angle)) * distance
    target_pose_in_object = [[target_x, target_y, 0], [0, 0, 0, 1]]
    target_pose_in_world = tfc.toMatrix(tfc.fromTf(start_pose_in_world)).dot(
        tfc.toMatrix(tfc.fromTf(target_pose_in_object)))
    target_pose_in_world = tfc.toTf(tfc.fromMatrix(target_pose_in_world))
    target_pose_in_world = [list(target_pose_in_world[0]), list(target_pose_in_world[1])]
    return target_pose_in_object, target_pose_in_world


def calculate_transform(frame1, frame2):
    """ given two frames (frame1 and frame2) expressed in world frame,
    calculate the transform frame1_T_frame2 """
    world_T_frame1 = tfc.toMatrix(tfc.fromTf(frame1))
    world_T_frame2 = tfc.toMatrix(tfc.fromTf(frame2))
    frame1_T_world = np.linalg.inv(world_T_frame1)
    frame1_T_frame2 = frame1_T_world.dot(world_T_frame2)
    return frame1_T_frame2


def pose_2_matrix(p):
    return tfc.toMatrix(tfc.fromTf(p))


def random_point_in_polygon(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds

    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)

    while not Point([x, y]).within(polygon):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)

    return (x, y)


def normalize_values(values):
    min_v = min(values)
    max_v = max(values)
    if min_v == max_v:
        # the values do not matter in the overall ranking any more
        return [0.0] * len(values)
    return list((np.array(values) - min_v) / (max_v - min_v))

