import tf_conversions
import tf
import tf2_ros
import tf2_kdl
import rospy
import rospkg
import os
import pickle
import numpy as np
import pybullet_utils as pu
import time
import graspit_commander
import grid_sample_client
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped

from reachability_utils.reachability_resolution_analysis import interpolate_pose_in_reachability_space_grid
from reachability_utils.process_reachability_data_from_csv import load_reachability_data_from_dir
import plyfile
import skfmm
from collections import namedtuple

mico_configs = {
    'GRASPIT_LINK_TO_MOVEIT_LINK': ([0.0, 0.0, -0.16], [-0.7071067811882787, -0.7071067811848163, 0.0, 0.0]),
    'GRASPIT_LINK_TO_PYBULLET_LINK': ([0, 0, 0], [0.0, 0.0, 0.0, 1.0]),
    'PYBULLET_LINK_TO_COM': ([-0.002216, -0.000001, -0.06], [0.0, 0.0, 0.0, 1.0]),
    'PYBULLET_LINK_COM': [-0.002216, -0.000001, -0.06],
    'MOVEIT_LINK_TO_GRASPING_POINT': ([0, 0, 0], [0.0, 0.0, 0.0, 1.0]),

    'robot_urdf': os.path.abspath('assets/mico/mico.urdf'),
    'gripper_urdf': os.path.abspath('assets/mico/mico_hand.urdf'),
    'EEF_LINK_INDEX': 0,
    'GRIPPER_JOINT_NAMES': ['m1n6s200_joint_finger_1', 'm1n6s200_joint_finger_tip_1', 'm1n6s200_joint_finger_2',
                            'm1n6s200_joint_finger_tip_2'],
    'OPEN_POSITION': [0.0, 0.0, 0.0, 0.0],
    'CLOSED_POSITION': [1.1, 0.0, 1.1, 0.0],

    'reachability_data_dir': os.path.join(rospkg.RosPack().get_path('mico_reachability_config'), 'data'),
    'graspit_approach_dir': 'z'
}

ur5_robotiq_configs = {
    'GRASPIT_LINK_TO_MOVEIT_LINK': ([0, 0, 0], [0.7071067811865475, 0.0, 0.0, 0.7071067811865476]),
    'GRASPIT_LINK_TO_PYBULLET_LINK': ([0.0, 0.0, 0.0], [0.0, 0.706825181105366, 0.0, 0.7073882691671998]),
    'PYBULLET_LINK_TO_COM': ([0.0, 0.0, 0.031451], [0.0, 0.0, 0.0, 1.0]),
    'PYBULLET_LINK_COM': [0.0, 0.0, 0.031451],
    'MOVEIT_LINK_TO_GRASPING_POINT': ([0.130, -0.000, -0.0], [0.500, -0.500, 0.500, -0.500]),

    'robot_urdf': os.path.abspath('assets/NONE'),
    'gripper_urdf': os.path.abspath('assets/robotiq_2f_85_hand/robotiq_arg2f_85_model.urdf'),
    'EEF_LINK_INDEX': 0,
    'GRIPPER_JOINT_NAMES': ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint',
                            'right_outer_knuckle_joint', 'right_inner_knuckle_joint', 'right_inner_finger_joint'],
    'OPEN_POSITION': [0] * 6,
    'CLOSED_POSITION': (0.72 * np.array([1, 1, -1, 1, 1, -1])).tolist(),

    'reachability_data_dir': os.path.join(rospkg.RosPack().get_path('ur5_robotiq_reachability_config'), 'data'),
    'graspit_approach_dir': 'x'
}

robot_configs = {'mico': namedtuple('RobotConfigs', mico_configs.keys())(*mico_configs.values()),
                 'ur5_robotiq': namedtuple('RobotConfigs', ur5_robotiq_configs.keys())(*ur5_robotiq_configs.values())
                 }


def pose_2_list(pose):
    """

    :param pose: geometry_msgs.msg.Pose
    :return: pose_2d: [[x, y, z], [x, y, z, w]]
    """
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return [position, orientation]


def list_2_pose(pose_2d):
    """

    :param pose_2d: [[x, y, z], [x, y, z, w]]
    :return: pose: geometry_msgs.msg.Pose
    """
    return Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))


def ps_2_list(pose_stamped):
    """

    :param pose: geometry_msgs.msg.PoseStamped
    :return: pose_2d: [[x, y, z], [x, y, z, w]]
    """
    pose = pose_stamped.pose
    pose_2d = pose_2_list(pose)
    return pose_2d


def list_2_ps(pose_2d):
    """

    :param pose_2d: [[x, y, z], [x, y, z, w]]
    :return: pose: geometry_msgs.msg.PoseStamped
    """
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = "/world"
    pose_stamped.pose = list_2_pose(pose_2d)
    return pose_stamped


# https://www.youtube.com/watch?v=aaDUIZVNCDM
def get_transform(reference_frame, target_frame):
    listener = tf.TransformListener()
    try:
        listener.waitForTransform(reference_frame, target_frame,
                                  rospy.Time(0), timeout=rospy.Duration(1))
        translation_rotation = listener.lookupTransform(reference_frame, target_frame,
                                                        rospy.Time())
    except Exception as e1:
        try:
            tf_buffer = tf2_ros.Buffer()
            tf2_listener = tf2_ros.TransformListener(tf_buffer)
            transform_stamped = tf_buffer.lookup_transform(reference_frame, target_frame,
                                                           rospy.Time(0), timeout=rospy.Duration(1))
            translation_rotation = tf_conversions.toTf(tf2_kdl.transform_to_kdl(transform_stamped))
        except Exception as e2:
            rospy.logerr("get_transform::\n " +
                         "Failed to find transform from %s to %s" % (
                             reference_frame, target_frame,))
    return translation_rotation


def back_off(grasp_pose, offset=-.05, approach_dir='z'):
    if approach_dir == 'x':
        pre_grasp_pose = tf_conversions.toMsg(
            tf_conversions.fromMsg(grasp_pose) * tf_conversions.fromTf(((offset, 0, 0), (0, 0, 0, 1))))
    if approach_dir == 'z':
        pre_grasp_pose = tf_conversions.toMsg(
            tf_conversions.fromMsg(grasp_pose) * tf_conversions.fromTf(((0, 0, offset), (0, 0, 0, 1))))
    return pre_grasp_pose


def back_off_pose_2d(grasp_pose_2d, offset=-.05, approach_dir='z'):
    """ back off alone negative z axis """
    grasp_pose = list_2_pose(grasp_pose_2d)
    pre_grasp_pose = back_off(grasp_pose, offset, approach_dir)
    return pose_2_list(pre_grasp_pose)


def change_end_effector_link_pose_2d(grasp_pose, old_link_to_new_link):
    """

    :param grasp_pose: pose 2d
    :param old_link_to_new_link: pose 2d
    :return:
    """
    graspit_grasp_pose_for_old_link_matrix = tf_conversions.toMatrix(tf_conversions.fromTf(grasp_pose))
    old_link_to_new_link_matrix = tf_conversions.toMatrix(tf_conversions.fromTf(old_link_to_new_link))
    graspit_grasp_pose_for_new_link_matrix = np.dot(graspit_grasp_pose_for_old_link_matrix,
                                                    old_link_to_new_link_matrix)
    graspit_grasp_pose_for_new_link = tf_conversions.toTf(
        tf_conversions.fromMatrix(graspit_grasp_pose_for_new_link_matrix))
    return graspit_grasp_pose_for_new_link


def change_end_effector_link(graspit_grasp_msg_pose, old_link_to_new_link_translation_rotation):
    """
    :param old_link_to_new_link_translation_rotation: geometry_msgs.msg.Pose,
        result of listener.lookupTransform((old_link, new_link, rospy.Time(0), timeout=rospy.Duration(1))
    :param graspit_grasp_msg_pose: The pose of a graspit grasp message i.e. g.pose
    ref_T_nl = ref_T_ol * ol_T_nl
    """
    graspit_grasp_pose_for_old_link_matrix = tf_conversions.toMatrix(
        tf_conversions.fromMsg(graspit_grasp_msg_pose)
    )

    old_link_to_new_link_tranform_matrix = tf.TransformerROS().fromTranslationRotation(
        old_link_to_new_link_translation_rotation[0],
        old_link_to_new_link_translation_rotation[1])
    graspit_grasp_pose_for_new_link_matrix = np.dot(graspit_grasp_pose_for_old_link_matrix,
                                                    old_link_to_new_link_tranform_matrix)  # ref_T_nl = ref_T_ol * ol_T_nl
    graspit_grasp_pose_for_new_link = tf_conversions.toMsg(
        tf_conversions.fromMatrix(graspit_grasp_pose_for_new_link_matrix))

    return graspit_grasp_pose_for_new_link


## TODO uniform sampling grasps


def get_grasps(robot_name='MicoGripper', object_mesh='cube', object_pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
               floor_offset=None, max_steps=35000, search_energy='GUIDED_POTENTIAL_QUALITY_ENERGY', seed_grasp=None,
               uniform_grasp=True, rotate_roll=False):
    gc = graspit_commander.GraspitCommander()
    gc.clearWorld()

    gc.importRobot(robot_name)
    gc.setRobotPose(Pose(Point(0, 0, 1), Quaternion(0, 0, 0, 1)))
    gc.importGraspableBody(object_mesh, object_pose)
    if floor_offset is not None:
        floor_offset -= 0.01
        if 'cube' in object_mesh and uniform_grasp:
            floor_offset -= 0.02
        gc.importObstacle('floor', Pose(Point(-1, -1, floor_offset + object_pose.position.z), Quaternion(0, 0, 0, 1)))

    if uniform_grasp:
        # grid_sample
        pre_grasps = []
        pre_grasps.extend(grid_sample_client.GridSampleClient.computePreGrasps(resolution=25, sampling_type=0).grasps)
        if rotate_roll:
            rot_trans = tf_conversions.fromTf(((0, 0, 0), tf_conversions.Rotation.RPY(np.pi / 2, 0, 0).GetQuaternion()))
            for pg in pre_grasps:
                pg.pose = tf_conversions.toMsg(tf_conversions.fromMsg(pg.pose) * rot_trans)
        pre_grasps.extend(grid_sample_client.GridSampleClient.computePreGrasps(resolution=4, sampling_type=1).grasps)
        grasps = grid_sample_client.GridSampleClient.evaluatePreGrasps(pre_grasps)
        good_grasps = [g for g in grasps if g.volume_quality > 0]
    else:
        # simulated annealling
        grasps = gc.planGrasps(max_steps=max_steps, search_energy=search_energy,
                               use_seed_grasp=seed_grasp is not None, seed_grasp=seed_grasp)
        grasps = grasps.grasps
        good_grasps = grasps

    # import ipdb; ipdb.set_trace()
    # import time
    # for g in good_grasps:
    #     gc.setRobotPose(g.pose)
    #     time.sleep(1)
    # import ipdb; ipdb.set_trace()
    # print('Number of good grasps: \t {} out of {}'.format(len(good_grasps), len(grasps)))

    # return grasp_poses_in_world and grasp_poses_in_object
    grasp_poses_in_world, grasp_poses_in_object = extract_grasp_poses_from_graspit_grasps(graspit_grasps=good_grasps,
                                                                                          object_pose=object_pose)
    return good_grasps, grasp_poses_in_world, grasp_poses_in_object


def extract_grasp_poses_from_graspit_grasps(graspit_grasps, object_pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))):
    grasp_poses_in_world = [g.pose for g in graspit_grasps]
    world_in_object_transform = tf_conversions.fromMsg(object_pose).Inverse()
    grasp_poses_in_object = [tf_conversions.toMsg(world_in_object_transform * tf_conversions.fromMsg(g)) for g in
                             grasp_poses_in_world]
    return grasp_poses_in_world, grasp_poses_in_object


def generate_grasps(robot_name='MicoGripper', load_fnm=None, save_fnm=None, object_mesh="cube",
                    object_pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
                    floor_offset=0, max_steps=35000, search_energy='GUIDED_POTENTIAL_QUALITY_ENERGY', seed_grasp=None,
                    uniform_grasp=True, rotate_roll=False):
    """
    :return grasps: a list of graspit Grasps
    :return grasp_poses_in_world: a list of ROS poses, poses of graspit end-effector in graspit world frame
    :return grasp_poses_in_object: a list of ROS poses, poses of graspit end-effector in graspit object frame
    """
    if load_fnm and os.path.exists(load_fnm):
        grasps = pickle.load(open(load_fnm, "rb"))
        grasp_poses_in_world, grasp_poses_in_object = extract_grasp_poses_from_graspit_grasps(graspit_grasps=grasps,
                                                                                              object_pose=object_pose)
    else:
        grasp_results = get_grasps(robot_name=robot_name, object_mesh=object_mesh, object_pose=object_pose,
                                   floor_offset=floor_offset, max_steps=max_steps, search_energy=search_energy,
                                   seed_grasp=seed_grasp, uniform_grasp=uniform_grasp, rotate_roll=rotate_roll)
        grasps, grasp_poses_in_world, grasp_poses_in_object = grasp_results
        if save_fnm:
            pickle.dump(grasps, open(save_fnm, "wb"))

    return grasps, grasp_poses_in_world, grasp_poses_in_object


def convert_graspit_pose_in_object_to_moveit_grasp_pose(graspit_grasp_pose_in_object, object_pose,
                                                        old_ee_to_new_ee_translation_rotation, pre_grasp_offset=-0.05):
    """
    :param graspit_grasp_pose_in_object: pose of graspit end-effector in graspiy object frame
    :param object_pose: the pose of the object where the graspit_grasp_pose_in_object are generated
    :param old_ee_to_new_ee_translation_rotation: transform from graspit end-effector to moveit end-effector
    :param pre_grasp_offset: how much to back off for pre-grasp
    :return ee_in_world: ROS pose, moveit end-effector pose in moveit world frame
    :return pre_grasp: ROS pose, moveit end-effector pre-grasp pose in moveit world frame
    """
    ee_in_world = tf_conversions.toMsg(tf_conversions.fromMsg(object_pose) *
                                       tf_conversions.fromMsg(graspit_grasp_pose_in_object) *
                                       tf_conversions.fromTf(old_ee_to_new_ee_translation_rotation))

    pre_grasp = back_off(ee_in_world, pre_grasp_offset)
    return ee_in_world, pre_grasp


def generate_grasps_old(load_fnm=None, save_fnm=None, body="cube", body_extents=(0.05, 0.05, 0.05)):
    """
    This method assumes the target object to be at world origin. Filter out bad grasps by volume quality.

    :param load_fnm: load file name
    :param save_fnm: save file name. save as graspit grasps.grasps
    :param body: the name of the graspable object to load in graspit, or the mesh file path
    :param body_extents: the extents of the bounding box
    :return: graspit grasps.grasps
    """
    ## NOTE, now use sim-ann anf then switch

    if load_fnm and os.path.exists(load_fnm):
        grasps = pickle.load(open(load_fnm, "rb"))
        return grasps
    else:
        gc = graspit_commander.GraspitCommander()
        gc.clearWorld()

        ## creat scene in graspit
        floor_offset = -body_extents[2] / 2 - 0.01  # half of the block size + half of the conveyor
        floor_pose = Pose(Point(-1, -1, floor_offset), Quaternion(0, 0, 0, 1))
        body_pose = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))

        gc.importRobot('MicoGripper')
        gc.importGraspableBody(body, body_pose)
        gc.importObstacle('floor', floor_pose)
        grasps = gc.planGrasps()
        grasps = grasps.grasps
        grasps = [g for g in grasps if g.volume_quality > 0]

        if save_fnm:
            pickle.dump(grasps, open(save_fnm, "wb"))
        return grasps


def display_grasp_pose_in_rviz(pose_2d_list, reference_frame):
    """

    :param pose_2d_list: a list of 2d array like poses
    :param reference_frame: which frame to reference
    """
    my_tf_manager = tf_manager.TFManager()
    for i, pose_2d in enumerate(pose_2d_list):
        pose = ut.list_2_pose(pose_2d)
        ps = PoseStamped()
        ps.pose = pose
        ps.header.frame_id = reference_frame
        my_tf_manager.add_tf('G_{}'.format(i), ps)
        # import ipdb; ipdb.set_trace()
        my_tf_manager.broadcast_tfs()


def load_reachability_params(reachability_data_dir):
    step_size = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.step'), delimiter=',')
    mins = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.mins'), delimiter=',')
    dims = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.dims'), delimiter=',', dtype=int)

    return step_size, mins, dims


def get_reachability_of_grasps_pose(grasps_in_world, sdf_reachability_space, mins, step_size, dims):
    """ grasps_in_world is a list of geometry_msgs/Pose """
    sdf_values = []
    for g_pose in grasps_in_world:
        trans, rot = tf_conversions.toTf(tf_conversions.fromMsg(g_pose))
        rpy = tf_conversions.Rotation.Quaternion(*rot).GetRPY()
        query_pose = np.concatenate((trans, rpy))
        sdf_values.append(
            interpolate_pose_in_reachability_space_grid(sdf_reachability_space, mins, step_size, dims, query_pose))

    # is_reachable = [sdf_values[i] > 0 for i in range(len(sdf_values))]
    return sdf_values


def get_reachability_of_grasps_pose_2d(grasps_in_world, sdf_reachability_space, mins, step_size, dims):
    """ grasps_in_world is a list of pose_2d """
    sdf_values = []
    for g_pose in grasps_in_world:
        trans, rot = g_pose[0], g_pose[1]
        rpy = tf_conversions.Rotation.Quaternion(*rot).GetRPY()
        query_pose = np.concatenate((trans, rpy))
        sdf_values.append(
            interpolate_pose_in_reachability_space_grid(sdf_reachability_space,
                                                        mins, step_size, dims, query_pose))

    # is_reachable = [sdf_values[i] > 0 for i in range(len(sdf_values))]
    return sdf_values


def read_vertex_points_from_ply_filepath(ply_filepath):
    # # TODO: remove dependency on plyfile, replace with trimesh
    # import trimesh
    # mesh = trimesh.load(ply_filepath)
    # return np.array(mesh.vertices)
    ply = plyfile.PlyData.read(ply_filepath)

    mesh_vertices = np.ones((ply['vertex']['x'].shape[0], 3))
    mesh_vertices[:, 0] = ply['vertex']['x']
    mesh_vertices[:, 1] = ply['vertex']['y']
    mesh_vertices[:, 2] = ply['vertex']['z']
    return mesh_vertices


def transform_points(vertices, transform):
    vertices_hom = np.ones((vertices.shape[0], 4))
    vertices_hom[:, :-1] = vertices

    # Create new 4xN transformed array
    transformed_vertices_hom = np.dot(transform, vertices_hom.T).T

    transformed_vertices = transformed_vertices_hom[:, :-1]

    return transformed_vertices


def add_obstacles_to_reachability_space_full(points, mins, step_size, dims):
    voxel_grid = np.zeros(shape=dims)

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)

    grid_points_min = np.round((bbox_min - np.array(mins)) / step_size).astype(int)
    grid_points_max = np.round((bbox_max - np.array(mins)) / step_size).astype(int)

    grid_points_min = np.clip(grid_points_min, 0, dims - 1)
    grid_points_max = np.clip(grid_points_max, 0, dims - 1)

    voxel_grid[grid_points_min[0]:grid_points_max[0] + 1, grid_points_min[1]:grid_points_max[1] + 1,
    grid_points_min[2]:grid_points_max[2] + 1] = 1

    return voxel_grid


def create_occupancy_grid_from_obstacles(obstacle_mesh_filepaths, obstacle_poses, mins_xyz, step_size_xyz, dims_xyz):
    voxel_grid = np.zeros(shape=dims_xyz)

    for filepath, pose in zip(obstacle_mesh_filepaths, obstacle_poses):
        vertices = read_vertex_points_from_ply_filepath(filepath)
        # if obstacle.type == 'box':
        #     sample_points = np.meshgrid(*[np.linspace(-sz / 2., sz / 2., 4) for sz in obstacle.box_size])
        #     vertices = np.array(sample_points).reshape(len(sample_points), -1).T
        frame = tf_conversions.fromMsg(pose)
        transform = tf_conversions.toMatrix(frame)
        vertices_transformed = transform_points(vertices, transform)

        voxel_grid += add_obstacles_to_reachability_space_full(vertices_transformed, mins_xyz, step_size_xyz,
                                                               dims_xyz)
    voxel_grid[np.where(voxel_grid > 0)] = 1
    return voxel_grid


def convert_grasp_in_object_to_world(object_pose, grasp_in_object):
    """
    :param object_pose: 2d list
    :param grasp_in_object: 2d list
    """
    object_T_grasp = tf_conversions.toMatrix(tf_conversions.fromTf(grasp_in_object))
    world_T_object = tf_conversions.toMatrix(tf_conversions.fromTf(object_pose))
    world_T_grasp = world_T_object.dot(object_T_grasp)
    grasp_in_world = tf_conversions.toTf(tf_conversions.fromMatrix(world_T_grasp))
    return grasp_in_world


def convert_grasp_in_world_to_object(object_pose, grasp_in_world):
    """
    :param object_pose: 2d list
    :param grasp_in_world: 2d list
    """
    world_T_object = tf_conversions.fromTf(object_pose)
    object_T_world = world_T_object.Inverse()
    object_T_world = tf_conversions.toMatrix(object_T_world)
    world_T_grasp = tf_conversions.toMatrix(tf_conversions.fromTf(grasp_in_world))
    object_T_grasp = object_T_world.dot(world_T_grasp)
    grasp_in_object = tf_conversions.toTf(tf_conversions.fromMatrix(object_T_grasp))
    return grasp_in_object


def visualize_grasps_with_reachability(grasp_poses, sdf_values, use_cmap_from_mpl=True, cmap_name='viridis'):
    """

    :param grasp_poses: a list of pose 2d in world frame
    :param sdf_values: a list of their corresponding sdf values
    """
    maximum = max(sdf_values)
    minimum = min(sdf_values)
    if use_cmap_from_mpl:
        cmap = pu.MplColorHelper(unicode(cmap_name), minimum, maximum)
        pu.plot_heatmap_bar(unicode(cmap_name), minimum, maximum)
    line_length = 0.1
    frame_ids_all = []
    for g, r in zip(grasp_poses, sdf_values):
        g = tf_conversions.toTf(tf_conversions.fromTf(g) * tf_conversions.fromTf(((0, 0, -line_length), (0, 0, 0, 1))))
        if use_cmap_from_mpl:
            frame_ids = pu.create_arrow_marker(g, raw_color=cmap.get_rgb(r), line_length=line_length)
        else:
            frame_ids = pu.create_arrow_marker(g, raw_color=pu.rgb(r, maximum=maximum, minimum=minimum),
                                               line_length=line_length)
        frame_ids_all.extend(frame_ids)
    return frame_ids_all


def visualize_grasp_with_reachability(grasp_pose, sdf_value, maximum, minimum, use_cmap_from_mpl=True,
                                      cmap_name='viridis'):
    """

    :param grasp_poses: a pose 2d in world frame
    :param sdf_values: the corresponding sdf values
    """
    if use_cmap_from_mpl:
        cmap = pu.MplColorHelper(unicode(cmap_name), minimum, maximum)
        pu.plot_heatmap_bar(unicode(cmap_name), minimum, maximum)
        line_length = 0.3
        grasp_pose = tf_conversions.toTf(
            tf_conversions.fromTf(grasp_pose) * tf_conversions.fromTf(((0, 0, -line_length), (0, 0, 0, 1))))
        frame_ids = pu.create_arrow_marker(grasp_pose, raw_color=cmap.get_rgb(sdf_value), line_length=0.3, line_width=3)
    else:
        line_length = 0.1
        grasp_pose = tf_conversions.toTf(
            tf_conversions.fromTf(grasp_pose) * tf_conversions.fromTf(((0, 0, -line_length), (0, 0, 0, 1))))
        frame_ids = pu.create_arrow_marker(grasp_pose, raw_color=pu.rgb(sdf_value, maximum=maximum, minimum=minimum))
    return frame_ids


def get_reachability_space(reachability_data_dir):
    rospy.loginfo("start creating sdf reachability space...")
    start_time = time.time()
    _, mins, step_size, dims, sdf_reachability_space = load_reachability_data_from_dir(reachability_data_dir)
    rospy.loginfo("sdf reachability space created, which takes {}".format(time.time() - start_time))
    return sdf_reachability_space, mins, step_size, dims


def get_reachability_space_obstacles(reachability_data_dir, obstacle_mesh_filepaths=None, obstacle_poses=None):
    rospy.loginfo("start creating sdf reachability space...")
    start_time = time.time()

    if obstacle_mesh_filepaths:
        binary_reachability_space, mins, step_size, dims, _ = load_reachability_data_from_dir(reachability_data_dir)

        obstacles_mask_3d = create_occupancy_grid_from_obstacles(obstacle_mesh_filepaths=obstacle_mesh_filepaths,
                                                                 obstacle_poses=obstacle_poses,
                                                                 mins_xyz=mins[:3],
                                                                 step_size_xyz=step_size[:3],
                                                                 dims_xyz=dims[:3])
        # embed obstacles into reachability space
        binary_reachability_space[obstacles_mask_3d > 0] = 0

        # Generate sdf
        binary_reachability_space -= 0.5
        sdf_reachability_space = skfmm.distance(binary_reachability_space, periodic=[False, False, False, True, True, True])
        binary_reachability_space += 0.5  # undo previous operation
    else:
        _, mins, step_size, dims, sdf_reachability_space = load_reachability_data_from_dir(reachability_data_dir)
    rospy.loginfo("sdf reachability space created, which takes {}".format(time.time()-start_time))
    return sdf_reachability_space, mins, step_size, dims


def load_grasp_database(grasp_database_path, object_name, back_off):
    grasps_eef = np.load(os.path.join(grasp_database_path, object_name, 'grasps_eef.npy'))
    grasps_link6_com = np.load(os.path.join(grasp_database_path, object_name, 'grasps_link6_com.npy'))
    grasps_link6_ref = np.load(os.path.join(grasp_database_path, object_name, 'grasps_link6_ref.npy'))
    pre_grasps_eef = np.load(os.path.join(grasp_database_path, object_name, 'pre_grasps_eef_'+str(back_off)+'.npy'))
    pre_grasps_link6_com = np.load(os.path.join(grasp_database_path, object_name, 'pre_grasps_link6_com_'+str(back_off)+'.npy'))
    pre_grasps_link6_ref = np.load(os.path.join(grasp_database_path, object_name, 'pre_grasps_link6_ref_'+str(back_off)+'.npy'))
    return grasps_eef, grasps_link6_ref, grasps_link6_com, pre_grasps_eef, pre_grasps_link6_ref, pre_grasps_link6_com


def load_grasp_database_new(grasp_database_path, object_name):
    if os.path.exists(os.path.join(grasp_database_path, object_name, 'actual_grasps.npy')):
        actual_grasps = np.load(os.path.join(grasp_database_path, object_name, 'actual_grasps.npy'))
        graspit_grasps = np.load(os.path.join(grasp_database_path, object_name, 'graspit_grasps.npy'))
    else:
        # This is for the mico grasps that were generated and saved the old way
        actual_grasps = np.load(os.path.join(grasp_database_path, object_name, 'grasps_eef.npy'))
        actual_grasps = [pu.merge_pose_2d(tf_conversions.toTf(
            tf_conversions.fromTf(pu.split_7d(g)) * tf_conversions.fromTf(
                mico_configs['GRASPIT_LINK_TO_MOVEIT_LINK']).Inverse())) for g in actual_grasps]
        graspit_grasps = actual_grasps
    return actual_grasps, graspit_grasps
