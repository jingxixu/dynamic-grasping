import os
import pybullet as p
import time
import trimesh
import argparse
import grasp_utils as gu
import misc_utils as mu
from mico_controller import MicoController
import rospy
import pprint
from dynamic_grasping_world import DynamicGraspingWorld
import json
import pandas as pd
import ast
from distutils.util import strtobool


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--robot_config_name', type=str, default='mico',
                        help="name of robot configs to load from grasputils. Ex: mico or ur5_robotiq")
    parser.add_argument('--motion_mode', type=str, default='dynamic_linear')
    parser.add_argument('--grasp_database_path', type=str, required=True)
    parser.add_argument('--rendering', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--realtime', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--num_trials', type=int, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--max_check', type=int, default=1)
    parser.add_argument('--back_off', type=float, default=0.05)
    parser.add_argument('--distance_low', type=float, default=0.15)
    parser.add_argument('--distance_high', type=float, default=0.4)
    parser.add_argument('--conveyor_z_low', type=float, default=0.01, help='typical half the conveyor thickness')
    parser.add_argument('--conveyor_z_high', type=float, default=0.01)
    parser.add_argument('--use_reachability', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--rank_by_manipulability', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--always_try_switching', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_joint_space_dist', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--record_videos', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--replay_trajectory', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--baseline_experiment_path', type=str, help='use motion path in this file for the run')
    parser.add_argument('--failure_only', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--exp_id', type=int)
    parser.add_argument('--num_obstacles', type=int, default=3)
    parser.add_argument('--load_obstacles', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--embed_obstacles_sdf', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--obstacle_distance_low', type=float, default=0.15)
    parser.add_argument('--obstacle_distance_high', type=float, default=0.25)
    parser.add_argument('--distance_between_region', type=float, default=0.05)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--use_motion_aware', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--motion_aware_model_path', type=str)

    # dynamic hyper parameter
    parser.add_argument('--conveyor_speed', type=float, default=0.01)
    parser.add_argument('--grasp_threshold', type=float, default=0.03)
    parser.add_argument('--lazy_threshold', type=float, default=0.3)
    parser.add_argument('--large_prediction_threshold', type=float, default=0.3)
    parser.add_argument('--small_prediction_threshold', type=float, default=0.1)
    parser.add_argument('--distance_travelled_threshold', type=float, default=0.1)
    parser.add_argument('--close_delay', type=float, default=0.5)
    parser.add_argument('--use_seed_trajectory', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_previous_jv', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--add_top_shelf', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_box', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_baseline_method', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_gt', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_kf', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--use_lstm_prediction', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--lstm_model_filepath', type=str)
    parser.add_argument('--pose_freq', type=int, default=5)
    parser.add_argument('--approach_prediction', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--approach_prediction_duration', type=float, default=1.0)
    parser.add_argument('--fix_motion_planning_time', type=float)
    parser.add_argument('--fix_grasp_ranking_time', type=float)
    args = parser.parse_args()

    if args.realtime:
        args.rendering = True

    args.mesh_dir = os.path.abspath('assets/models')
    args.conveyor_urdf = os.path.abspath('assets/conveyor.urdf')

    robot_configs = gu.robot_configs[args.robot_config_name]
    # update args with all the robot configs, including args.reachability_data_dir
    args.__dict__.update(robot_configs.__dict__)

    # timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    # args.runstr = 'static-'+timestr

    # create result folder
    args.result_dir = os.path.join(args.result_dir, args.object_name)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    args.result_file_path = os.path.join(args.result_dir, args.object_name + '.csv')

    # create a video folders
    if args.record_videos:
        assert args.rendering
        args.video_dir = os.path.join(args.result_dir, 'videos')
        if not os.path.exists(args.video_dir):
            os.makedirs(args.video_dir)
    args.trajectory_dir = os.path.join(args.result_dir, 'trajectories')
    if not os.path.exists(args.trajectory_dir):
        os.makedirs(args.trajectory_dir)
    return args


if __name__ == "__main__":
    args = get_args()
    json.dump(vars(args), open(os.path.join(args.result_dir, args.object_name + '.json'), 'w'), indent=4)
    mu.configure_pybullet(args.rendering, debug=args.debug)
    rospy.init_node('dynamic_grasping')

    print()
    print("Arguments:")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('\n')

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = mu.create_object_urdf(object_mesh_filepath, args.object_name,
                                        urdf_target_object_filepath='assets/{}_target.urdf'.format(args.object_name))
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    target_extents = target_mesh.bounding_box.extents.tolist()
    floor_offset = target_mesh.bounds.min(0)[2]
    conveyor_thickness = 0.02
    target_z = -target_mesh.bounds.min(0)[2] + conveyor_thickness
    target_initial_pose = [[0.3, 0.3, target_z], [0, 0, 0, 1]]
    robot_initial_pose = [[0, 0, 0], [0, 0, 0, 1]]
    conveyor_initial_pose = [[0.3, 0.3, conveyor_thickness/2], [0, 0, 0, 1]]
    obstacle_names = ['tomato_soup_can', 'power_drill', 'bleach_cleanser'] * (args.num_obstacles//3 + 1)
    obstacle_names = obstacle_names[:args.num_obstacles]

    dynamic_grasping_world = DynamicGraspingWorld(target_name=args.object_name,
                                                  obstacle_names=obstacle_names,
                                                  mesh_dir=args.mesh_dir,
                                                  robot_config_name=args.robot_config_name,
                                                  target_initial_pose=target_initial_pose,
                                                  robot_initial_pose=robot_initial_pose,
                                                  robot_initial_state=MicoController.HOME,
                                                  conveyor_initial_pose=conveyor_initial_pose,
                                                  robot_urdf=args.robot_urdf,
                                                  conveyor_urdf=args.conveyor_urdf,
                                                  conveyor_speed=args.conveyor_speed,
                                                  target_urdf=target_urdf,
                                                  target_mesh_file_path=object_mesh_filepath,
                                                  target_extents=target_extents,
                                                  grasp_database_path=args.grasp_database_path,
                                                  reachability_data_dir=args.reachability_data_dir,
                                                  realtime=args.realtime,
                                                  max_check=args.max_check,
                                                  use_reachability=args.use_reachability,
                                                  rank_by_manipulability=args.rank_by_manipulability,
                                                  always_try_switching=args.always_try_switching,
                                                  use_joint_space_dist=args.use_joint_space_dist,
                                                  back_off=args.back_off,
                                                  pose_freq=args.pose_freq,
                                                  use_seed_trajectory=args.use_seed_trajectory,
                                                  use_previous_jv=args.use_previous_jv,
                                                  use_gt=args.use_gt,
                                                  use_kf=args.use_kf,
                                                  use_lstm_prediction=args.use_lstm_prediction,
                                                  lstm_model_filepath=args.lstm_model_filepath,
                                                  grasp_threshold=args.grasp_threshold,
                                                  lazy_threshold=args.lazy_threshold,
                                                  large_prediction_threshold=args.large_prediction_threshold,
                                                  small_prediction_threshold=args.small_prediction_threshold,
                                                  close_delay=args.close_delay,
                                                  distance_travelled_threshold=args.distance_travelled_threshold,
                                                  distance_low=args.distance_low,
                                                  distance_high=args.distance_high,
                                                  conveyor_z_low=args.conveyor_z_low,
                                                  conveyor_z_high=args.conveyor_z_high,
                                                  add_top_shelf=args.add_top_shelf,
                                                  use_box=args.use_box,
                                                  use_baseline_method=args.use_baseline_method,
                                                  approach_prediction=args.approach_prediction,
                                                  approach_prediction_duration=args.approach_prediction_duration,
                                                  fix_motion_planning_time=args.fix_motion_planning_time,
                                                  fix_grasp_ranking_time=args.fix_grasp_ranking_time,
                                                  load_obstacles=args.load_obstacles,
                                                  embed_obstacles_sdf=args.embed_obstacles_sdf,
                                                  obstacle_distance_low=args.obstacle_distance_low,
                                                  obstacle_distance_high=args.obstacle_distance_high,
                                                  distance_between_region=args.distance_between_region,
                                                  use_motion_aware=args.use_motion_aware,
                                                  motion_aware_model_path=args.motion_aware_model_path,
                                                  alpha=args.alpha)

    # adding option to use previous experiment as config
    baseline_experiment_config_df = None
    if args.baseline_experiment_path and os.path.exists(args.baseline_experiment_path):
        args.baseline_experiment_path = os.path.join(args.baseline_experiment_path, args.object_name,
                                                     '{}.csv'.format(args.object_name))
        if os.path.exists(args.baseline_experiment_path):
            baseline_experiment_config_df = pd.read_csv(args.baseline_experiment_path, index_col=0)
            for key in ['target_quaternion', 'obstacle_poses', 'z_start_end']:
                if key in baseline_experiment_config_df.keys():
                    baseline_experiment_config_df[key] = baseline_experiment_config_df[key].apply(
                        lambda x: ast.literal_eval(x))
                    if key == 'obstacle_poses':
                        baseline_experiment_config_df[key] = baseline_experiment_config_df[key].apply(
                            lambda x: [x[p:p + 7] for p in range(0, len(x), 7)])
                else:
                    baseline_experiment_config_df[key] = None
            # TODO: remove the next line which is just for backward compatibility
            if baseline_experiment_config_df['z_start_end'].values[0] is None:
                baseline_experiment_config_df['z_start_end'] = [[args.conveyor_z_low, args.conveyor_z_low]] * \
                                                               baseline_experiment_config_df.shape[0]

            args.num_trials = len(baseline_experiment_config_df)

    for i in range(args.num_trials):
        if args.exp_id is not None and i != args.exp_id:
            continue
        reset_dict = None
        # reset_dict = {
        #     'distance': 0.2787919083152529,
        #     'length': 1.0,
        #     'theta': 110.23333162496952,
        #     'direction': 1,
        #     'target_quaternion': [0.0, 0.0, 0.8092568854035559, 0.5874549288472571]
        # }
        if baseline_experiment_config_df is not None:
            reset_dict = baseline_experiment_config_df.loc[i].to_dict()
            if args.failure_only and reset_dict['success']:
                print('skipping trial {}'.format(i))
                continue
        if args.load_obstacles:
            p.resetSimulation()
            p.setGravity(0, 0, -9.8)
            dynamic_grasping_world.load_world()
        distance, theta, length, direction, target_quaternion, obstacle_poses, z_start_end = dynamic_grasping_world.reset(
            mode=args.motion_mode, reset_dict=reset_dict)

        if args.replay_trajectory and args.baseline_experiment_path:
            trajectory_path = os.path.join(os.path.dirname(args.baseline_experiment_path), 'trajectories',
                                           '{}.json'.format(i))
            if os.path.exists(trajectory_path):
                with open(trajectory_path, 'r') as infile:
                    object_arm_trajectory = json.load(infile)
                if args.record_videos:
                    logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                                  os.path.join(args.video_dir, '{}.mp4'.format(i)))
                distance, theta, length, direction, target_quaternion, obstacle_poses, z_start_end = dynamic_grasping_world.reset(
                    mode=args.motion_mode, reset_dict=reset_dict)
                dynamic_grasping_world.replay_trajectory(object_arm_trajectory)
                if args.record_videos:
                    p.stopStateLogging(logging)
            continue

        time.sleep(2)  # for moveit to update scene, might not be necessary, depending on computing power
        if args.record_videos:
            logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join(args.video_dir, '{}.mp4'.format(i)))
        success, grasp_idx, dynamic_grasping_time, grasp_switched_list, num_ik_called_list = dynamic_grasping_world.dynamic_grasp()
        # success, grasp_idx, dynamic_grasping_time, grasp_switched_list, num_ik_called_list, object_arm_trajectory = None, None, None, None, None, None
        time.sleep(0.5)
        if args.record_videos:
            p.stopStateLogging(logging)
        with open(os.path.join(args.trajectory_dir, '{}.json'.format(i)), 'w') as outfile:
            json.dump(dynamic_grasping_world.trajectory_log, outfile)

        result = [('exp_idx', i),
                  ('grasp_idx', grasp_idx),
                  ('success', success),
                  ('dynamic_grasping_time', dynamic_grasping_time),
                  ('grasp_switched_list', grasp_switched_list),
                  ('num_ik_called_list', num_ik_called_list),
                  ('theta', theta),
                  ('length', length),
                  ('distance', distance),
                  ('direction', direction),
                  ('target_quaternion', target_quaternion),
                  ('z_start_end', z_start_end),
                  ('obstacle_poses', sum(obstacle_poses, []))]
        mu.write_csv_line(result_file_path=args.result_file_path, result=result)
