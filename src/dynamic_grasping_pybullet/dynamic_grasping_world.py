import os
import numpy as np
import pyquaternion as pyqt
import pybullet as p
import pybullet_data
import time
import grasp_utils as gu
import pybullet_utils as pu
from mico_controller import MicoController
import rospy
import threading
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from math import pi, cos, sin, sqrt, atan, radians, degrees
from kalman_filter_3d import KalmanFilter, create_kalman_filter
import random
import tf_conversions as tfc
import moveit_commander as mc
from ur5_robotiq_pybullet import load_ur_robotiq_robot, UR5RobotiqPybulletController
from shapely.geometry import Polygon, Point
import misc_utils as mu
import trimesh
from itertools import combinations
from collections import deque
import torch
import torch.nn.functional as F


class DynamicGraspingWorld:
    def __init__(self,
                 target_name,
                 obstacle_names,
                 mesh_dir,
                 robot_config_name,
                 target_initial_pose,
                 robot_initial_pose,
                 robot_initial_state,
                 conveyor_initial_pose,
                 robot_urdf,
                 conveyor_urdf,
                 conveyor_speed,
                 target_urdf,
                 target_mesh_file_path,
                 target_extents,
                 grasp_database_path,
                 reachability_data_dir,
                 realtime,
                 max_check,
                 use_reachability,
                 rank_by_manipulability,
                 back_off,
                 pose_freq,
                 use_seed_trajectory,
                 use_previous_jv,
                 use_kf,
                 use_lstm_prediction,
                 lstm_model_filepath,
                 use_gt,
                 grasp_threshold,
                 lazy_threshold,
                 large_prediction_threshold,
                 small_prediction_threshold,
                 close_delay,
                 distance_travelled_threshold,
                 distance_low,
                 distance_high,
                 conveyor_z_low,
                 conveyor_z_high,
                 use_box,
                 add_top_shelf,
                 use_baseline_method,
                 approach_prediction,
                 approach_prediction_duration,
                 fix_motion_planning_time,
                 fix_grasp_ranking_time,
                 always_try_switching,
                 use_joint_space_dist,
                 load_obstacles,
                 embed_obstacles_sdf,
                 obstacle_distance_low,
                 obstacle_distance_high,
                 distance_between_region,
                 use_motion_aware,
                 motion_aware_model_path,
                 alpha):
        self.target_name = target_name
        self.obstacle_names = obstacle_names
        self.mesh_dir = mesh_dir
        self.robot_config_name = robot_config_name
        self.target_initial_pose = target_initial_pose
        self.robot_initial_pose = robot_initial_pose
        self.initial_distance = np.linalg.norm(
            np.array(target_initial_pose[0][:2]) - np.array(robot_initial_pose[0][:2]))
        self.robot_initial_state = robot_initial_state
        self.conveyor_initial_pose = conveyor_initial_pose
        self.robot_urdf = robot_urdf
        self.conveyor_urdf = conveyor_urdf
        self.conveyor_speed = conveyor_speed
        self.target_urdf = target_urdf
        self.target_mesh_file_path = target_mesh_file_path
        self.target_extents = target_extents
        self.realtime = realtime
        self.max_check = max_check
        self.back_off = back_off
        self.use_reachability = use_reachability
        self.rank_by_manipulability = rank_by_manipulability
        self.always_try_switching = always_try_switching
        self.use_joint_space_dist = use_joint_space_dist
        self.world_steps = 0
        self.clean_trajectory_log()
        self.pose_freq = pose_freq
        self.pose_duration = 1.0 / self.pose_freq
        self.pose_steps = int(self.pose_duration * 240)
        self.use_seed_trajectory = use_seed_trajectory
        self.use_previous_jv = use_previous_jv
        self.use_kf = use_kf
        self.use_lstm_prediction = use_lstm_prediction
        self.lstm_model_filepath = lstm_model_filepath
        self.use_gt = use_gt
        self.alpha = alpha
        if use_lstm_prediction:
            self.motion_predictor_kf = LSTMMotionPredictorKF(self.pose_duration, lstm_model_filepath)
        else:
            self.motion_predictor_kf = MotionPredictorKF(self.pose_duration)
        self.distance_between_region = distance_between_region
        self.use_motion_aware = use_motion_aware
        self.motion_aware_model_path = motion_aware_model_path

        self.distance_low = distance_low  # mico 0.15  ur5_robotiq: 0.3
        self.distance_high = distance_high  # mico 0.4  ur5_robotiq: 0.7
        self.conveyor_z_low = conveyor_z_low
        self.conveyor_z_high = conveyor_z_high

        self.grasp_database_path = grasp_database_path
        actual_grasps, graspit_grasps = gu.load_grasp_database_new(grasp_database_path, self.target_name)
        use_actual = False
        self.graspit_grasps = actual_grasps if use_actual else graspit_grasps

        self.robot_configs = gu.robot_configs[self.robot_config_name]
        self.graspit_pregrasps = [
            pu.merge_pose_2d(gu.back_off_pose_2d(pu.split_7d(g), back_off, self.robot_configs.graspit_approach_dir)) for
            g in self.graspit_grasps]
        self.grasps_eef = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose_2d(pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK)) for g
            in self.graspit_grasps]
        self.grasps_link6_ref = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose_2d(pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK)) for g
            in self.graspit_grasps]
        self.pre_grasps_eef = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose_2d(pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK)) for g
            in self.graspit_pregrasps]
        self.pre_grasps_link6_ref = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose_2d(pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK)) for g
            in self.graspit_pregrasps]

        self.reachability_data_dir = reachability_data_dir
        self.sdf_reachability_space, self.mins, self.step_size, self.dims = gu.get_reachability_space(
            self.reachability_data_dir)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.load_world()
        self.reset('initial')  # the reset is needed to simulate the initial config

        self.scene = mc.PlanningSceneInterface()
        self.grasp_threshold = grasp_threshold
        self.close_delay = close_delay
        self.lazy_threshold = lazy_threshold
        self.large_prediction_threshold = large_prediction_threshold
        self.small_prediction_threshold = small_prediction_threshold
        self.distance_travelled_threshold = distance_travelled_threshold
        self.use_box = use_box
        self.add_top_shelf = add_top_shelf
        self.use_baseline_method = use_baseline_method
        self.approach_prediction = approach_prediction
        self.approach_prediction_duration = approach_prediction_duration
        self.fix_motion_planning_time = fix_motion_planning_time
        self.fix_grasp_ranking_time = fix_grasp_ranking_time
        self.value_markers = None

        # obstacles
        self.load_obstacles = load_obstacles
        self.embed_obstacles_sdf = embed_obstacles_sdf
        self.obstacle_distance_low = obstacle_distance_low
        self.obstacle_distance_high = obstacle_distance_high

        self.obstacles = []
        if self.load_obstacles:
            self.obstacle_urdfs = []
            self.obstacle_zs = []
            self.obstacle_extentss = []
            for obstacle_name in self.obstacle_names:
                mesh_filepath = os.path.join(self.mesh_dir, '{}'.format(obstacle_name), '{}.obj'.format(obstacle_name))
                self.obstacle_urdfs.append(mu.create_object_urdf(mesh_filepath, obstacle_name,
                                                                 urdf_target_object_filepath='assets/{}_obstacle.urdf'.format(
                                                                     obstacle_name)))
                obstacle_mesh = trimesh.load_mesh(mesh_filepath)
                self.obstacle_extentss.append(obstacle_mesh.bounding_box.extents.tolist())
                self.obstacle_zs.append(-obstacle_mesh.bounds.min(0)[2])

        if self.use_motion_aware:
            from train_motion_aware import MotionQualityEvaluationNet
            import torch
            from torch.nn.functional import softmax
            self.motion_aware_network = MotionQualityEvaluationNet()
            epoch_dir = os.listdir(os.path.join(self.motion_aware_model_path, self.target_name))[0]
            self.motion_aware_network.load_state_dict(torch.load(
                os.path.join(self.motion_aware_model_path, self.target_name, epoch_dir,
                             'motion_ware_net.pt')))

    def load_world(self):
        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        if 'mico' in self.robot_config_name:
            self.robot = MicoController(self.robot_initial_pose, self.robot_initial_state, self.robot_urdf)
        if 'robotiq' in self.robot_config_name:
            self.robot_id = load_ur_robotiq_robot(self.robot_initial_pose)
            self.robot = UR5RobotiqPybulletController(self.robot_id)
            p.changeDynamics(self.target, -1, mass=1,
                             frictionAnchor=1, restitution=0.1,
                             spinningFriction=0, rollingFriction=0.01, lateralFriction=0.9)
            for joint in range(p.getNumJoints(self.robot.id)):
                p.changeDynamics(self.robot.id, joint, mass=1)
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1, contactBreakingThreshold=1e-3)

        self.conveyor = Conveyor(self.conveyor_initial_pose, self.conveyor_urdf)

    def add_top_shelf_to_scene(self, dist, theta, z=0.4, width=0.1):
        theta = np.radians(theta)
        T1 = tfc.fromTf(((0, 0, 0), tfc.Rotation.RPY(0, 0, theta).GetQuaternion()))
        T2 = tfc.fromTf(((dist, 0, z), (0, 0, 0, 1)))

        T_final = T1*T2
        top_shelf_pose = tfc.toTf(T_final)

        slab_size = [width, 1, .02]
        half_extents = np.array(slab_size)/2


        vs_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=(86./255, 47./255, 14./255, 0.75))
        cs_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        object_id = p.createMultiBody(basePosition=top_shelf_pose[0], baseOrientation=top_shelf_pose[1],
                                    baseCollisionShapeIndex=cs_id, baseVisualShapeIndex=vs_id)

        pu.create_frame_marker(((top_shelf_pose[0][0], top_shelf_pose[0][1], top_shelf_pose[0][2]), top_shelf_pose[1]))


        self.scene.add_box('top_shelf', gu.list_2_ps(top_shelf_pose), size=slab_size)
        return object_id

    def reset(self, mode, reset_dict=None):
        """
        mode:
            initial: reset the target to the fixed initial pose, not moving
            static_random: reset the target to a random pose, not moving
            dynamic_linear: initialize the conveyor with a linear motion
            dynamic_linear_vary_speed: initialize the conveyor with a linear motion with variable speed
            dynamic_sinusoid: initialize the conveyor with a sinusoid motion
            dynamic_circular: initialize the conveyor with a circular motion
            hand_over: TODO
        """
        self.world_steps = 0
        self.value_markers = None
        self.clean_trajectory_log()
        if mode == 'initial':
            pu.remove_all_markers()
            target_pose, distance = self.target_initial_pose, self.initial_distance
            conveyor_pose = [[target_pose[0][0], target_pose[0][1], self.conveyor_initial_pose[0][2]],
                             [0, 0, 0, 1]] if target_pose is not None else self.conveyor_initial_pose
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            self.robot.reset()
            pu.step(2)
            return target_pose, distance

        elif mode == 'static_random':
            pu.remove_all_markers()
            target_pose, distance = self.sample_target_location()
            conveyor_pose = [[target_pose[0][0], target_pose[0][1], 0.01],
                             [0, 0, 0, 1]] if target_pose is not None else self.conveyor_initial_pose
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            self.robot.reset()
            self.scene.add_box("floor", gu.list_2_ps(((0, 0, -0.055), (0, 0, 0, 1))), size=(2, 2, 0.1))
            pu.step(2)
            return target_pose, distance

        elif mode in ['dynamic_linear', 'dynamic_linear_vary_speed', 'dynamic_sinusoid']:
            pu.remove_all_markers()
            if len(self.obstacles) != 0:
                for i in self.obstacles:
                    p.removeBody(i)
            self.motion_predictor_kf.reset_predictor()
            self.conveyor.clear_motion()

            if reset_dict is None:
                distance, theta, length, direction = self.sample_convey_linear_motion()
                target_quaternion = self.sample_target_angle()
                z_start_end = np.random.uniform(self.conveyor_z_low, self.conveyor_z_high, 2)
            else:
                distance, theta, length, direction, z_start_end = reset_dict['distance'], reset_dict['theta'], \
                                                                  reset_dict['length'], reset_dict['direction'], \
                                                                  reset_dict['z_start_end']
                target_quaternion = reset_dict['target_quaternion']
            if mode == 'dynamic_sinusoid':
                self.conveyor.initialize_sinusoid_motion(distance, theta, length, direction, self.conveyor_speed)
            else:
                self.conveyor.initialize_linear_motion(distance, theta, length, direction, self.conveyor_speed,
                                                       z_start_end[0], z_start_end[1],
                                                       variable_speed=mode == 'dynamic_linear_vary_speed')
            conveyor_pose = self.conveyor.start_pose
            target_z = self.target_initial_pose[0][2] - self.conveyor_initial_pose[0][2] + conveyor_pose[0][2]
            target_pose = [[conveyor_pose[0][0], conveyor_pose[0][1], target_z],
                           target_quaternion]
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            if self.load_obstacles:
                if reset_dict is None or reset_dict['obstacle_poses'] is None:
                    self.obstacles = self.load_obstacles_collision_free(distance, theta, length)
                else:
                    # self.get_obstacles_regions(distance, theta, length, visualize_region=True)
                    self.obstacles = self.load_obstacles_at_poses(reset_dict['obstacle_poses'])
            self.robot.reset()
            self.scene.add_box("floor", gu.list_2_ps(((0, 0, -0.055), (0, 0, 0, 1))), size=(2, 2, 0.1))
            pu.step(2)

            obstacle_poses = []
            if self.load_obstacles:
                for i, n, e in zip(self.obstacles, self.obstacle_names, self.obstacle_extentss):
                    self.scene.add_box('{}_{}'.format(n, i), gu.list_2_ps(pu.get_body_pose(i)), size=e)
                    obstacle_poses.append(pu.merge_pose_2d(pu.get_body_pose(i)))
                if self.embed_obstacles_sdf:
                    # update reachability sdf
                    obstacle_mesh_filepaths = []
                    for obstacle_name in self.obstacle_names:
                        obstacle_mesh_filepaths.append(
                            os.path.join(self.mesh_dir, '{}'.format(obstacle_name), '{}.ply'.format(obstacle_name)))

                    obstacle_poses_msg = [tfc.toMsg(tfc.fromTf(pu.split_7d(obs_pose))) for obs_pose in obstacle_poses]
                    self.sdf_reachability_space, self.mins, self.step_size, self.dims = gu.get_reachability_space_obstacles(
                        self.reachability_data_dir, obstacle_mesh_filepaths, obstacle_poses_msg)

            self.motion_predictor_kf.initialize_predictor(target_pose)

            if mode == 'dynamic_sinusoid':
                num_plot_points = 100
                idx = np.linspace(0, len(self.conveyor.discretized_trajectory) - 1, num_plot_points).astype(int)
                for i in range(len(idx) - 1):
                    pos1 = self.conveyor.discretized_trajectory[idx[i]][0]
                    pos2 = self.conveyor.discretized_trajectory[idx[i+1]][0]
                    pu.draw_line(pos1, pos2)
            else:
                pu.draw_line(self.conveyor.start_pose[0], self.conveyor.target_pose[0])

            if self.add_top_shelf:
                shelf_id = self.add_top_shelf_to_scene(distance, theta, width=0.08)
                self.obstacles.append(shelf_id)
            p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=theta + 90, cameraPitch=-35,
                                         cameraTargetPosition=(0.0, 0.0, 0.0))
            return distance, theta, length, direction, target_quaternion, obstacle_poses, np.array(z_start_end).tolist()

        elif mode == 'dynamic_circular':
            pu.remove_all_markers()
            if len(self.obstacles) != 0:
                for i in self.obstacles:
                    p.removeBody(i)
            self.motion_predictor_kf = CircularMotionPredictorKF(self.pose_duration)
            self.motion_predictor_kf.reset_predictor()
            self.conveyor.clear_motion()

            if reset_dict is None:
                distance, theta, length, direction = self.sample_convey_circular_motion()
                target_quaternion = self.sample_target_angle()
                z_start_end = np.random.uniform(self.conveyor_z_low, self.conveyor_z_high, 2)
            else:
                distance, theta, length, direction, z_start_end = reset_dict['distance'], reset_dict['theta'], \
                                                                  reset_dict['length'], reset_dict['direction'], \
                                                                  reset_dict['z_start_end']
                target_quaternion = reset_dict['target_quaternion']
            self.conveyor.initialize_circular_motion(distance, theta, length, direction, self.conveyor_speed)
            conveyor_pose = self.conveyor.start_pose
            target_pose = [[conveyor_pose[0][0], conveyor_pose[0][1], self.target_initial_pose[0][2]],
                           target_quaternion]
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            self.robot.reset()
            # self.scene.add_box("floor", gu.list_2_ps(((0, 0, -0.055), (0, 0, 0, 1))), size=(2, 2, 0.1))
            pu.step(2)

            obstacle_poses = []
            self.motion_predictor_kf.initialize_predictor(target_pose)

            # visualize circular motion, evenly pick points in the trajectory
            num_plot_points = 100
            idx = np.round(np.linspace(0, len(self.conveyor.discretized_trajectory) - 1, num_plot_points)).astype(int)
            for i in range(len(idx) - 1):
                pos1 = self.conveyor.discretized_trajectory[idx[i]][0]
                pos2 = self.conveyor.discretized_trajectory[idx[i+1]][0]
                pu.draw_line(pos1, pos2)

            p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=theta + 90, cameraPitch=-35,
                                         cameraTargetPosition=(0.0, 0.0, 0.0))

            return distance, theta, length, direction, target_quaternion, obstacle_poses, np.array(z_start_end).tolist()

        elif mode == 'hand_over':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def step(self, freeze_time, arm_motion_plan, gripper_motion_plan):
        for i in range(int(freeze_time * 240)):
            # step the robot
            self.robot.step()
            # step conveyor
            self.conveyor.step()
            # step physics
            p.stepSimulation()
            self.world_steps += 1
            if self.world_steps % self.pose_steps == 0:
                self.motion_predictor_kf.update(pu.get_body_pose(self.target))
            if self.realtime:
                time.sleep(1.0 / 240.0)
        if arm_motion_plan is not None:
            self.robot.update_arm_motion_plan(arm_motion_plan)
        if gripper_motion_plan is not None:
            self.robot.update_gripper_motion_plan(gripper_motion_plan)

    def static_grasp(self):
        target_pose = pu.get_body_pose(self.target)
        predicted_pose = target_pose

        success = False
        grasp_attempted = False  # pre_grasp and grasp is reachable and motion is found
        pre_grasp_reached = False
        grasp_reachaed = False
        comment = " "

        # planning grasp
        grasp_idx, grasp_planning_time, num_ik_called, pre_grasp, pre_grasp_jv, grasp, grasp_jv, grasp_switched = self.plan_grasp(
            predicted_pose, None)
        if grasp_jv is None or pre_grasp_jv is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, "no reachable grasp is found"

        # planning motion
        if self.use_box:
            self.scene.add_box('target', gu.list_2_ps(target_pose), size=self.target_extents)
        else:
            self.scene.add_mesh('target', gu.list_2_ps(target_pose), self.target_mesh_file_path)
        rospy.sleep(2)
        motion_planning_time, plan = self.plan_arm_motion(pre_grasp_jv)
        if plan is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, "no motion found to the planned pre grasp jv"

        # move
        self.robot.execute_arm_plan(plan, self.realtime)
        pre_grasp_reached = self.robot.equal_conf(self.robot.get_arm_joint_values(), pre_grasp_jv, tol=0.01)

        # print('self')
        # print(self.robot.get_arm_joint_values())
        # print('pre_grasp_jv')
        # print(pre_grasp_jv)
        # print('grasp_jv')
        # print(grasp_jv)

        # approach
        # plan = self.robot.plan_arm_joint_values_simple(grasp_jv)
        # self.robot.execute_arm_plan(plan, self.realtime)
        # grasp_reachaed = self.robot.equal_conf(self.robot.get_arm_joint_values(), grasp_jv, tol=0.01)
        plan, fraction = self.robot.plan_straight_line(tfc.toMsg(tfc.fromTf(grasp)), ee_step=0.01,
                                                       avoid_collisions=True)
        if plan is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, "no motion found to the planned grasp jv"
        self.robot.execute_arm_plan(plan, self.realtime)
        # print(fraction)
        grasp_attempted = True

        # close and lift
        self.robot.close_gripper(self.realtime)
        plan, fraction = self.robot.plan_cartesian_control(z=0.07)
        if fraction != 1.0:
            comment = "lift fration {} is not 1.0".format(fraction)
        if plan is not None:
            self.robot.execute_arm_plan(plan, self.realtime)
        success = self.check_success()
        pu.remove_all_markers()
        return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, comment

    def check_success(self):
        if pu.get_body_pose(self.target)[0][2] >= self.target_initial_pose[0][2] + 0.03:
            return True
        else:
            return False

    def replay_trajectory(self, object_arm_trajectory):
        visualize_reachability = False
        visualize_motion_aware = False
        visualize_final = False

        # go to grasp pose
        for grasp_idx, pre_grasp_jv, pre_grasp_pose, grasp_jv, grasp_pose, filtered_grasp_idxs, reachabilities, motion_aware_qualities, \
            arm_jv, gripper_jv, target_pose, conveyor_pose in zip(object_arm_trajectory['grasp_idx'],
                                                                  object_arm_trajectory['pre_grasp_jv'],
                                                                  object_arm_trajectory['pre_grasp_pose'],
                                                                  object_arm_trajectory['grasp_jv'],
                                                                  object_arm_trajectory['grasp_pose'],
                                                                  object_arm_trajectory['filtered_grasp_idxs'],
                                                                  object_arm_trajectory['reachabilities'],
                                                                  object_arm_trajectory['motion_aware_qualities'],
                                                                  object_arm_trajectory['arm'],
                                                                  object_arm_trajectory['gripper'],
                                                                  object_arm_trajectory['target'],
                                                                  object_arm_trajectory['conveyor']):
            self.robot.set_arm_joints(arm_jv)
            self.robot.set_gripper_joints(gripper_jv)
            self.conveyor.set_pose(conveyor_pose)
            pu.set_pose(self.target, target_pose)

            if visualize_reachability:
                show_top = 100
                pre_grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                           self.pre_grasps_eef]
                pre_grasps_eef_in_world = [
                    tfc.toTf(tfc.fromTf(g) * tfc.fromTf(self.robot_configs.MOVEIT_LINK_TO_GRASPING_POINT)) for g in
                    pre_grasps_eef_in_world]

                if self.value_markers is not None:
                    pu.remove_markers(self.value_markers)
                if show_top < 1:
                    self.value_markers = gu.visualize_grasps_with_reachability(pre_grasps_eef_in_world, reachabilities)
                else:
                    top_indices = np.argsort(reachabilities)[::-1][:show_top]
                    selected_grasps = [pre_grasps_eef_in_world[idx] for idx in top_indices]
                    selected_qualities = [reachabilities[idx] for idx in top_indices]
                    self.value_markers = gu.visualize_grasps_with_reachability(selected_grasps, selected_qualities)

            if visualize_motion_aware:
                show_top = 100
                pre_grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                           self.pre_grasps_eef]
                pre_grasps_eef_in_world = [
                    tfc.toTf(tfc.fromTf(g) * tfc.fromTf(self.robot_configs.MOVEIT_LINK_TO_GRASPING_POINT)) for g in
                    pre_grasps_eef_in_world]

                if self.value_markers is not None:
                    pu.remove_markers(self.value_markers)
                if show_top < 1:
                    self.value_markers = gu.visualize_grasps_with_reachability(pre_grasps_eef_in_world, motion_aware_qualities)
                else:
                    top_indices = np.argsort(motion_aware_qualities)[::-1][:show_top]
                    selected_grasps = [pre_grasps_eef_in_world[idx] for idx in top_indices]
                    selected_qualities = [motion_aware_qualities[idx] for idx in top_indices]
                    self.value_markers = gu.visualize_grasps_with_reachability(selected_grasps, selected_qualities)
            if visualize_final and grasp_idx is not None:
                line_length = 0.1
                pu.create_arrow_marker(tfc.toTf(tfc.fromTf(pre_grasp_pose) * tfc.fromTf(
                    self.robot_configs.MOVEIT_LINK_TO_GRASPING_POINT) * tfc.fromTf(
                    ((0, 0, -line_length), (0, 0, 0, 1)))),
                                       color_index=grasp_idx, line_length=line_length)
            time.sleep(0.1)
            # width, height, rgbPixels, depthPixels, segmentationMaskBuffer = p.getCameraImage(width=1024, height=768)

        # appraoch and grasp
        approach_traj = object_arm_trajectory['approach_traj']
        approach_jump = 10
        i = 0
        for arm_jv, gripper_jv, target_pose, conveyor_pose in approach_traj:
            if i % approach_jump == 0:
                self.robot.set_arm_joints(arm_jv)
                self.robot.set_gripper_joints(gripper_jv)
                self.conveyor.set_pose(conveyor_pose)
                pu.set_pose(self.target, target_pose)
                time.sleep(0.05)
            i += 1

        # lift
        lift_traj = object_arm_trajectory['lift_traj']
        lift_jump = 10
        i = 0
        for arm_jv, gripper_jv, target_pose, conveyor_pose in lift_traj:
            if i % lift_jump == 0:
                self.robot.set_arm_joints(arm_jv)
                self.robot.set_gripper_joints(gripper_jv)
                self.conveyor.set_pose(conveyor_pose)
                pu.set_pose(self.target, target_pose)
                time.sleep(0.05)
            i += 1
        print('finished')

    def predict(self, duration):
        if self.use_kf:
            # TODO verify that when duration is 0
            predicted_target_pose = self.motion_predictor_kf.predict(duration)
            predicted_conveyor_position = list(predicted_target_pose[0])
            predicted_conveyor_position[2] = self.conveyor_initial_pose[0][2]
            predicted_conveyor_pose = [predicted_conveyor_position, [0, 0, 0, 1]]
        elif self.use_gt:
            current_target_pose = pu.get_body_pose(self.target)
            predicted_conveyor_pose = self.conveyor.predict(duration)
            predicted_target_position = [predicted_conveyor_pose[0][0], predicted_conveyor_pose[0][1],
                                         current_target_pose[0][2]]
            predicted_target_pose = [predicted_target_position, current_target_pose[1]]
        else:
            # no prediction
            predicted_target_pose = pu.get_body_pose(self.target)
            predicted_conveyor_pose = pu.get_body_pose(self.conveyor.id)
        return predicted_target_pose, predicted_conveyor_pose

    def can_grasp(self, grasp_idx):
        planned_grasp_in_object = pu.split_7d(self.grasps_eef[grasp_idx])
        grasp_pose_tf = gu.convert_grasp_in_object_to_world(pu.get_body_pose(self.target), planned_grasp_in_object)
        current_eef_pose_tf = self.robot.get_eef_pose()

        dist_pos = np.abs(np.array(grasp_pose_tf[0]) - np.array(current_eef_pose_tf[0]))
        dist_q = pyqt.Quaternion.absolute_distance(pyqt.Quaternion(grasp_pose_tf[1]),
                                                   pyqt.Quaternion(current_eef_pose_tf[1]))
        can_grasp = np.linalg.norm(dist_pos) < np.abs(self.back_off * 1.1) and np.abs(dist_q) < np.pi / 180 * 20.
        return can_grasp

    def dynamic_grasp(self):
        """

        :return attempted_grasp_idx: the executed grasp index
        """
        grasp_idx = None
        done = False
        dynamic_grasp_time = 0
        distance = None
        initial_motion_plan_success = False  # not necessarily succeed
        grasp_switched_list = []
        num_ik_called_list = []
        object_arm_trajectory = []
        while not done:
            done = self.check_done()
            current_target_pose = pu.get_body_pose(self.target)
            duration = self.calculate_prediction_time(distance)
            predicted_target_pose, predicted_conveyor_pose = self.predict(duration)

            # update the scene. it will not reach the next line if the scene is not updated
            update_start_time = time.time()
            if self.use_box:
                self.scene.add_box('target', gu.list_2_ps(predicted_target_pose), size=self.target_extents)
            else:
                self.scene.add_mesh('target', gu.list_2_ps(predicted_target_pose), self.target_mesh_file_path)
            self.scene.add_box('conveyor', gu.list_2_ps(predicted_conveyor_pose), size=(.1, .1, .02))
            # print('Updating scene takes {} second'.format(time.time() - update_start_time))

            ############################## plan a grasp ################################
            if self.use_baseline_method:
                grasp_idx, grasp_planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv, grasp_switched \
                    = self.plan_grasp_baseline(predicted_target_pose, grasp_idx)
            else:
                grasp_idx, grasp_planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, \
                planned_grasp_jv, grasp_switched, grasp_order_idxs, reachabilities, motion_aware_qualities \
                    = self.plan_grasp(predicted_target_pose, grasp_idx)
            num_ik_called_list.append(num_ik_called)
            grasp_switched_list.append(grasp_switched)
            dynamic_grasp_time += grasp_planning_time
            if planned_grasp_jv is None or planned_pre_grasp_jv is None:
                self.step(grasp_planning_time, None, None)
                self.save_trajecotry_log(grasp_idx, planned_pre_grasp_jv, planned_pre_grasp, planned_grasp_jv, planned_grasp, grasp_order_idxs, reachabilities, motion_aware_qualities)
                continue
            self.step(grasp_planning_time, None, None)
            self.save_trajecotry_log(grasp_idx, planned_pre_grasp_jv, planned_pre_grasp, planned_grasp_jv, planned_grasp, grasp_order_idxs, reachabilities, motion_aware_qualities)

            # planned_pre_grasp = gu.convert_grasp_in_object_to_world(pu.get_body_pose(self.target), pu.split_7d(
            #     self.pre_grasps_eef[grasp_idx]))
            line_length = 0.1
            pu.create_arrow_marker(tfc.toTf(tfc.fromTf(planned_pre_grasp) * tfc.fromTf(
                self.robot_configs.MOVEIT_LINK_TO_GRASPING_POINT) * tfc.fromTf(((0, 0, -line_length), (0, 0, 0, 1)))),
                                   color_index=grasp_idx, line_length=line_length)
            # self.robot.set_arm_joints(planned_pre_grasp_jv)
            # continue

            ############################# plan a motion ############################
            distance = np.linalg.norm(np.array(self.robot.get_eef_pose()[0]) - np.array(planned_pre_grasp[0]))
            distance_travelled = np.linalg.norm(np.array(current_target_pose[0]) - np.array(
                last_motion_plan_success_pos)) if initial_motion_plan_success else 0
            if self.check_lazy_plan(distance, grasp_switched, distance_travelled):
                # print("lazy plan")
                continue
            motion_planning_time, plan = self.plan_arm_motion(planned_pre_grasp_jv)
            dynamic_grasp_time += motion_planning_time
            if plan is None:
                self.step(motion_planning_time, None, None)
                self.save_trajecotry_log(grasp_idx, planned_pre_grasp_jv, planned_pre_grasp, planned_grasp_jv, planned_grasp, grasp_order_idxs, reachabilities, motion_aware_qualities)
                continue
            self.step(motion_planning_time, plan, None)
            self.save_trajecotry_log(grasp_idx, planned_pre_grasp_jv, planned_pre_grasp, planned_grasp_jv, planned_grasp, grasp_order_idxs, reachabilities, motion_aware_qualities)
            last_motion_plan_success_pos = current_target_pose[0]
            initial_motion_plan_success = True

            # check can grasp or not
            can_grasp = self.can_grasp(grasp_idx)
            can_grasp_old = self.robot.equal_conf(self.robot.get_arm_joint_values(), planned_pre_grasp_jv, tol=self.grasp_threshold)
            if can_grasp or can_grasp_old:
                grasp_plan_found, motion_planning_time, approach_trajectory = self.approach_and_grasp_timed(grasp_idx)
                # grasp_plan_found, motion_planning_time, approach_trajectory = self.approach_and_grasp(grasp_idx, planned_pre_grasp_jv)
                self.trajectory_log['approach_traj'] = approach_trajectory
                dynamic_grasp_time += motion_planning_time
                if not grasp_plan_found:
                    print("the predicted approach motion is not reachable")
                    continue
                lift_traj = self.execute_lift()
                self.trajectory_log['lift_traj'] = lift_traj
                return self.check_success(), grasp_idx, dynamic_grasp_time, grasp_switched_list, num_ik_called_list
        return False, None, dynamic_grasp_time, grasp_switched_list, num_ik_called_list

    def save_trajecotry_log(self, grasp_idx, pre_grasp_jv, pre_grasp_pose, grasp_jv, grasp_pose, filtered_order_idxs, reachabilities, motion_aware_qualities):
        self.trajectory_log['world_step'].append(self.world_steps)
        self.trajectory_log['arm'].append(self.robot.get_arm_joint_values())
        self.trajectory_log['gripper'].append(self.robot.get_gripper_joint_values())
        self.trajectory_log['target'].append(pu.get_body_pose(self.target))
        self.trajectory_log['conveyor'].append(self.conveyor.get_pose())
        self.trajectory_log['grasp_idx'].append(grasp_idx)
        self.trajectory_log['pre_grasp_jv'].append(list(pre_grasp_jv) if pre_grasp_jv is not None else pre_grasp_jv)
        self.trajectory_log['pre_grasp_pose'].append(pre_grasp_pose)
        self.trajectory_log['grasp_jv'].append(list(grasp_jv) if grasp_jv is not None else grasp_jv)
        self.trajectory_log['grasp_pose'].append(grasp_pose)
        self.trajectory_log['filtered_grasp_idxs'].append(list(filtered_order_idxs))
        self.trajectory_log['reachabilities'].append(list(reachabilities) if reachabilities is not None else reachabilities)
        self.trajectory_log['motion_aware_qualities'].append(list(motion_aware_qualities) if motion_aware_qualities is not None else motion_aware_qualities)

    def clean_trajectory_log(self):
        self.trajectory_log = {
            'world_step': [],
            'arm': [],
            'gripper': [],
            'target': [],
            'conveyor': [],
            'grasp_idx': [],
            'pre_grasp_jv': [],
            'pre_grasp_pose': [],
            'grasp_jv': [],
            'grasp_pose': [],
            'filtered_grasp_idxs': [],
            'reachabilities': [],
            'motion_aware_qualities': [],
            'appraoch_traj': [],
            'lift_traj': []
        }

    def approach_and_grasp(self, grasp_idx, planned_grasp_jv):
        if self.approach_prediction:
            # one extra IK call, right now ignore the time because it is very small
            predicted_target_pose, predicted_conveyor_pose = self.predict(self.approach_prediction_duration)
            planned_grasp_in_object = pu.split_7d(self.grasps_eef[grasp_idx])
            planned_grasp = gu.convert_grasp_in_object_to_world(predicted_target_pose, planned_grasp_in_object)
            planned_grasp_jv = self.robot.get_arm_ik(planned_grasp, avoid_collisions=False,
                                                     arm_joint_values=self.robot.get_arm_joint_values())
            if planned_grasp_jv is None:
                print("the predicted approach motion is not reachable")
                return False, 0

        motion_planning_time, arm_motion_plan, gripper_motion_plan = self.plan_approach_motion(planned_grasp_jv,
                                                                                               self.approach_prediction_duration)
        object_arm_trajectory = self.execute_approach_and_grasp(arm_motion_plan, gripper_motion_plan)
        return True, motion_planning_time, object_arm_trajectory

    def approach_and_grasp_timed(self, grasp_idx):
        # object_velocity = np.array(self.conveyor.target_pose[0]) - np.array(self.conveyor.start_pose[0])
        # object_velocity *= self.conveyor.speed / np.linalg.norm(object_velocity)
        object_velocity = np.array(self.predict(1)[0][0]) - np.array(pu.get_body_pose(self.target)[0])
        arm_discretized_plan, gripper_discretized_plan = self.get_grasping_plan_timed_control(grasp_idx,
                                                                                              self.back_off,
                                                                                              object_velocity)
        if arm_discretized_plan is None:
            return False, 0, []
        object_arm_trajectory = self.execute_approach_and_grasp_timed(arm_discretized_plan, gripper_discretized_plan)
        return True, 0, object_arm_trajectory

    def get_grasping_plan_timed_control(self, grasp_idx, back_off, object_velocity):

        pre_grasp_in_object = pu.split_7d(self.pre_grasps_eef[grasp_idx])
        grasp_in_object = pu.split_7d(self.grasps_eef[grasp_idx])

        pregrasp_object_pose = p.getBasePositionAndOrientation(self.target)
        pre_grasp_link6_com_in_world = gu.convert_grasp_in_object_to_world(pregrasp_object_pose, pre_grasp_in_object)

        max_eef_speed = abs(back_off)    # should be dependent on the Jacobian
        # max_eef_speed = self.robot.get_current_max_eef_velocity(self.robot.get_arm_joint_values())
        # jac_t, jac_r, jacobian_moveit = self.get_pybullet_jacobian()
        # np.dot(np.array(jac_t)[:, :6], self.robot.arm_max_joint_velocities)
        # np.dot(np.abs(approach_direction), np.abs(max_eef_speed[:3]))
        approach_duration = abs(back_off) / max_eef_speed
        approach_direction = np.array(grasp_in_object[0]) - np.array(pre_grasp_in_object[0])
        approach_direction /= np.linalg.norm(approach_direction)

        object_position_at_grasp_pose = np.array(pu.get_body_pose(self.target)[0]) + object_velocity * approach_duration
        object_pose_at_grasp_pose = [object_position_at_grasp_pose, pregrasp_object_pose[1]]
        at_grasp_pose = gu.convert_grasp_in_object_to_world(object_pose_at_grasp_pose, grasp_in_object)

        gripper_close_duration = approach_duration * 0.2
        object_position_at_grasp_closed = object_position_at_grasp_pose + object_velocity * gripper_close_duration
        object_pose_at_grasp_closed = [object_position_at_grasp_closed, pregrasp_object_pose[1]]
        final_grasp_pose = gu.convert_grasp_in_object_to_world(object_pose_at_grasp_closed, grasp_in_object)

        grasping_timing = [0, approach_duration, approach_duration + gripper_close_duration]
        grasping_eef_wp = [pre_grasp_link6_com_in_world, at_grasp_pose, final_grasp_pose]
        # grasping_eef_jv = [self.robot.get_arm_ik(eef_pose, avoid_collisions=False) for eef_pose in grasping_eef_wp]
        grasping_eef_jv = []
        ik_seed = self.robot.get_arm_joint_values()
        for eef_pose in grasping_eef_wp:
            ik_result = self.robot.get_arm_ik(eef_pose, avoid_collisions=False, arm_joint_values=ik_seed)
            if ik_result is None:
                return None, None
            grasping_eef_jv.append(ik_result)
            ik_seed = grasping_eef_jv[-1]

        jv_diffs = np.abs(np.diff(np.array([self.robot.get_arm_joint_values()] + grasping_eef_jv), axis=0))
        max_joint_jump = 2.0  # there should not be a large jump in configuration during approach and grasp
        if np.any(jv_diffs > max_joint_jump):
            return None, None

        grasping_gripper_wp = [self.robot.OPEN_POSITION, self.robot.OPEN_POSITION, self.robot.CLOSED_POSITION]
        arm_discretized_plan, gripper_discretized_plan = self.discretize_grasping_plan(grasping_timing, grasping_eef_jv,
                                                                                       grasping_gripper_wp)
        return arm_discretized_plan, gripper_discretized_plan

    def discretize_grasping_plan(self, grasping_timing, grasping_eef_jv, grasping_gripper_wp):

        num_steps = 240 * np.array(grasping_timing)
        arm_discretized_plan = []
        gripper_discretized_plan = []
        for i in range(len(grasping_timing)-1):
            arm_wp = np.linspace(grasping_eef_jv[i], grasping_eef_jv[i+1], num_steps[i+1] - num_steps[i])
            arm_discretized_plan.extend(arm_wp)

            gripper_wp = np.linspace(grasping_gripper_wp[i], grasping_gripper_wp[i+1], num_steps[i+1] - num_steps[i])
            gripper_discretized_plan.extend(gripper_wp)
        return arm_discretized_plan, gripper_discretized_plan

    def execute_approach_and_grasp_timed(self, arm_discretized_plan, gripper_discretized_plan):
        """ modify the arm and gripper plans according to close delay and execute it """
        assert len(arm_discretized_plan) == len(gripper_discretized_plan)
        object_arm_trajectory = []  # arm, gripper, target, conveyor
        for arm_wp, gripper_wp in zip(arm_discretized_plan, gripper_discretized_plan):
            self.robot.control_arm_joints(arm_wp)
            self.robot.control_gripper_joints(gripper_wp)
            self.conveyor.step()
            p.stepSimulation()
            if self.realtime:
                time.sleep(1.0 / 240.0)
            self.world_steps += 1
            if self.world_steps % self.pose_steps == 0:
                self.motion_predictor_kf.update(pu.get_body_pose(self.target))
            object_arm_trajectory.append((self.robot.get_arm_joint_values(), self.robot.get_gripper_joint_values(),
                                          pu.get_body_pose(self.target), self.conveyor.get_pose()))
        return object_arm_trajectory

    def get_pybullet_jacobian(self):
        gripper_joint_values = self.robot.get_gripper_joint_values()
        arm_joint_values = self.robot.get_arm_joint_values()
        current_positions = arm_joint_values + gripper_joint_values

        zero_vec = [0.0] * len(current_positions)
        jac_t, jac_r = p.calculateJacobian(self.robot.id, self.robot.EEF_LINK_INDEX, (0, 0, 0), current_positions,
                                           zero_vec, zero_vec)
        jacobian_moveit = self.robot.moveit.arm_commander_group.get_jacobian_matrix(arm_joint_values)
        return jac_t, jac_r, jacobian_moveit

    def plan_approach_motion(self, grasp_jv, prediction_duration):
        """ Plan the discretized approach motion for both arm and gripper """
        # no need to prediction in the old trajectory because plan simple takes about 0.001
        predicted_period = 0
        start_time = time.time()

        if self.robot.arm_discretized_plan is not None:
            future_target_index = min(int(predicted_period * 240 + self.robot.arm_wp_target_index),
                                      len(self.robot.arm_discretized_plan) - 1)
            if future_target_index == -1:
                # catch error
                print(self.robot.arm_discretized_plan)
                import ipdb
                ipdb.set_trace()
            start_joint_values = self.robot.arm_discretized_plan[future_target_index]
            arm_discretized_plan = self.robot.plan_arm_joint_values_simple(grasp_jv,
                                                                           start_joint_values=start_joint_values,
                                                                           duration=prediction_duration)
        else:
            arm_discretized_plan = self.robot.plan_arm_joint_values_simple(grasp_jv, duration=prediction_duration)

        # there is no gripper discretized plan
        gripper_discretized_plan = self.robot.plan_gripper_joint_values(self.robot.CLOSED_POSITION,
                                                                        duration=prediction_duration)

        planning_time = time.time() - start_time
        print("Planning a motion takes {:.6f}".format(planning_time))
        return planning_time, arm_discretized_plan, gripper_discretized_plan

    def execute_approach_and_grasp(self, arm_plan, gripper_plan):
        """ modify the arm and gripper plans according to close delay and execute it """
        arm_len = len(arm_plan)
        num_delay_steps = int(arm_len * self.close_delay)
        gripper_len = len(gripper_plan)
        final_len = max(arm_len, gripper_len + num_delay_steps)

        arm_plan = np.vstack(
            (arm_plan, np.tile(arm_plan[-1], (final_len - arm_len, 1)))) if arm_len <= final_len else arm_plan
        gripper_plan = np.vstack((np.tile(gripper_plan[0], (num_delay_steps, 1)), gripper_plan))
        gripper_plan = np.vstack((gripper_plan, np.tile(gripper_plan[-1], (final_len - len(gripper_plan), 1)))) if len(
            gripper_plan) <= final_len else gripper_plan
        assert len(arm_plan) == len(gripper_plan)
        object_arm_trajectory = []
        for arm_wp, gripper_wp in zip(arm_plan, gripper_plan):
            self.robot.control_arm_joints(arm_wp)
            self.robot.control_gripper_joints(gripper_wp)
            self.conveyor.step()
            p.stepSimulation()
            if self.realtime:
                time.sleep(1.0 / 240.0)
            object_arm_trajectory.append((pu.get_body_pose(self.target), None,
                                          self.robot.get_arm_joint_values() + self.robot.get_gripper_joint_values()))
            self.world_steps += 1
            if self.world_steps % self.pose_steps == 0:
                self.motion_predictor_kf.update(pu.get_body_pose(self.target))
        return object_arm_trajectory

    def execute_lift(self):
        # lift twice in case the first lift attempt does not work
        object_arm_trajectory = []  # arm, gripper, target, conveyor
        for i in range(3):
            if i == 0:
                plan, fraction = self.robot.plan_cartesian_control(z=0.07)
            elif i == 1 and isinstance(self.robot, MicoController):
                current_eef_position = self.robot.get_eef_pose()[0]
                plan, fraction = self.robot.plan_cartesian_control(z=0.07, x=-np.sign(current_eef_position)[0] * 0.05,
                                                                   y=-np.sign(current_eef_position)[1] * 0.05)
            elif i == 2 and isinstance(self.robot, MicoController):
                current_eef_position = self.robot.get_eef_pose()[0]
                plan, fraction = self.robot.plan_cartesian_control(z=0.07, x=np.sign(current_eef_position)[0] * 0.05,
                                                                   y=np.sign(current_eef_position)[1] * 0.05)
            else:
                plan, fraction = self.robot.plan_cartesian_control(z=0.07)
            if fraction != 1.0:
                print('fraction {} not 1'.format(fraction))
            if plan is not None:
                # self.robot.execute_arm_plan(plan, self.realtime)
                for wp in plan:
                    self.robot.control_arm_joints(wp)
                    p.stepSimulation()
                    if self.realtime:
                        time.sleep(1. / 240.)
                    object_arm_trajectory.append((self.robot.get_arm_joint_values(), self.robot.get_gripper_joint_values(),
                                                  pu.get_body_pose(self.target), self.conveyor.get_pose()))
                pu.step(2)
                object_arm_trajectory.append((self.robot.get_arm_joint_values(), self.robot.get_gripper_joint_values(),
                                              pu.get_body_pose(self.target), self.conveyor.get_pose()))
                if fraction == 1.0:
                    break
        return object_arm_trajectory

    def get_ik_error(self, eef_pose, ik_result, coeff=0.4):

        # fk_result = self.robot.get_arm_fk(ik_result)
        fk_result = self.robot.get_arm_fk_pybullet(ik_result)

        trans_dist = np.linalg.norm(np.array(eef_pose[0]) - np.array(fk_result[0]))
        ang_dist = np.abs(np.dot(np.array(eef_pose[1]), np.array(fk_result[1])))

        return (1 - coeff) * trans_dist + coeff * ang_dist

    def get_iks_pregrasp_and_grasp_approximate(self, query_grasp_idx, target_pose):
        planned_pre_grasp_in_object = pu.split_7d(self.pre_grasps_eef[query_grasp_idx])
        planned_pre_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_pre_grasp_in_object)
        planned_pre_grasp_jv = self.robot.get_arm_ik_pybullet(planned_pre_grasp)

        planned_grasp_in_object = pu.split_7d(self.grasps_eef[query_grasp_idx])
        planned_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_grasp_in_object)
        planned_grasp_jv = self.robot.get_arm_ik_pybullet(planned_grasp, arm_joint_values=planned_pre_grasp_jv)

        return planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

    def plan_grasp_baseline(self, target_pose, old_grasp_idx):
        """ Plan a reachable pre_grasp and grasp pose"""
        # timing of the best machine
        ik_call_time = 0.01
        # rank_grasp_time = 0.135

        # optionally rank grasp based on reachability
        rank_grasp_time_start = time.time()
        grasp_order_idxs = self.rank_grasps(target_pose)
        rank_grasp_time = time.time() - rank_grasp_time_start
        print('rank grasp takes {}'.format(rank_grasp_time))

        planned_pre_grasps, planned_pre_grasp_jvs, planned_grasps, planned_grasp_jvs = [], [], [], []
        grasp_order_idxs = grasp_order_idxs[:self.max_check]
        if old_grasp_idx is not None:
            np.append(grasp_order_idxs, old_grasp_idx)  # always add old grasp index
        for i, grasp_idx in enumerate(grasp_order_idxs):
            planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv = self.get_iks_pregrasp_and_grasp_approximate(
                grasp_idx, target_pose)
            map(lambda x, y: x.append(y),
                [planned_pre_grasps, planned_pre_grasp_jvs, planned_grasps, planned_grasp_jvs],
                [planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv])

        pregrasp_ik_error = [self.get_ik_error(ee_pose, ik) for ee_pose, ik in
                             zip(planned_pre_grasps, planned_pre_grasp_jvs)]
        grasp_ik_error = [self.get_ik_error(ee_pose, ik) for ee_pose, ik in zip(planned_grasps, planned_grasp_jvs)]
        ik_error_sum = np.array(pregrasp_ik_error) + np.array(grasp_ik_error)
        min_error_idx = np.argmin(ik_error_sum)
        margin = 0.02  # TODO: check if this margin makes sense
        if old_grasp_idx and ik_error_sum[-1] < ik_error_sum[min_error_idx] + margin:   # -1: old grasp index is last in list, keep old grasp if error is not far from min
            grasp_idx = -1
        else:
            grasp_idx = min_error_idx
        grasp_switched = (grasp_idx != old_grasp_idx)

        num_ik_called = 2 * len(planned_pre_grasps)
        planning_time = rank_grasp_time + num_ik_called * ik_call_time
        # print("Planning a grasp takes {:.6f}".format(planning_time))

        return grasp_order_idxs[min_error_idx], planning_time, num_ik_called, planned_pre_grasps[grasp_idx], \
               planned_pre_grasp_jvs[grasp_idx], planned_grasps[grasp_idx], planned_grasp_jvs[grasp_idx], grasp_switched

    def get_iks_pregrasp_and_grasp(self, query_grasp_idx, target_pose):
        """ return 1 or 2 ik called; if successful, then planned_grasp_jv is not None """
        # the actual IK call maximum possible time is 0.1s
        num_ik_called = 0
        planned_pre_grasp_in_object = pu.split_7d(self.pre_grasps_eef[query_grasp_idx])
        planned_pre_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_pre_grasp_in_object)
        planned_pre_grasp_jv = self.robot.get_arm_ik(planned_pre_grasp, timeout=0.02, restarts=5)
        num_ik_called += 1

        planned_grasp, planned_grasp_jv = None, None
        if planned_pre_grasp_jv is not None:
            planned_grasp_in_object = pu.split_7d(self.grasps_eef[query_grasp_idx])
            planned_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_grasp_in_object)
            planned_grasp_jv = self.robot.get_arm_ik(planned_grasp, avoid_collisions=False,
                                                     arm_joint_values=planned_pre_grasp_jv,
                                                     timeout=0.02, restarts=5)
            num_ik_called += 1
        return num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

    def get_reachability_of_pregrasp_and_grasp(self, target_pose, grasp_idx):
        if grasp_idx is None:
            return [None, None]

        graspit_grasp_poses_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                        [self.graspit_pregrasps[grasp_idx], self.graspit_grasps[grasp_idx]]]
        sdf_values = gu.get_reachability_of_grasps_pose_2d(graspit_grasp_poses_in_world,
                                                           self.sdf_reachability_space,
                                                           self.mins,
                                                           self.step_size,
                                                           self.dims)
        return sdf_values

    def rank_grasps(self, target_pose, visualize_reachability=False, visualize_motion_aware=False, show_top=0):
        reachability_qualities = None
        motion_aware_qualities = None
        if self.use_reachability:
            graspit_pregrasps_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                          self.graspit_pregrasps]
            reachability_qualities = gu.get_reachability_of_grasps_pose_2d(graspit_pregrasps_in_world,
                                                               self.sdf_reachability_space,
                                                               self.mins,
                                                               self.step_size,
                                                               self.dims)
            if visualize_reachability:
                pre_grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                           self.pre_grasps_eef]
                pre_grasps_eef_in_world = [
                    tfc.toTf(tfc.fromTf(g) * tfc.fromTf(self.robot_configs.MOVEIT_LINK_TO_GRASPING_POINT)) for g in
                    pre_grasps_eef_in_world]

                if self.value_markers is not None:
                    pu.remove_markers(self.value_markers)
                if show_top < 1:
                    self.value_markers = gu.visualize_grasps_with_reachability(pre_grasps_eef_in_world, reachability_qualities)
                else:
                    top_indices = np.argsort(reachability_qualities)[::-1][:show_top]
                    selected_grasps = [pre_grasps_eef_in_world[idx] for idx in top_indices]
                    selected_qualities = [reachability_qualities[idx] for idx in top_indices]
                    self.value_markers = gu.visualize_grasps_with_reachability(selected_grasps, selected_qualities)

        if self.use_motion_aware:
            if self.conveyor.direction == 1:
                conveyor_angle_in_world = degrees(self.conveyor.circular_angles[self.conveyor.wp_target_index - 1]) + 90 \
                    if self.conveyor.circular_angles is not None else self.conveyor.theta + 90
            elif self.conveyor.direction == -1:
                conveyor_angle_in_world = degrees(self.conveyor.circular_angles[self.conveyor.wp_target_index - 1]) - 90 \
                    if self.conveyor.circular_angles is not None else self.conveyor.theta - 90
            else:
                raise TypeError
            target_angle_in_world = degrees(pu.get_euler_from_quaternion(target_pose[1])[2])
            conveyor_angle_in_object = conveyor_angle_in_world - target_angle_in_world
            speed = self.conveyor.speed
            motion_aware_qualities = self.get_motion_aware_qualities(self.grasps_eef,
                                                                         self.pre_grasps_eef,
                                                                         radians(conveyor_angle_in_object),
                                                                         speed)
            grasp_order_idxs = np.argsort(motion_aware_qualities)[::-1]
            # visualization
            if visualize_motion_aware:
                pre_grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                           self.pre_grasps_eef]
                pre_grasps_eef_in_world = [
                    tfc.toTf(tfc.fromTf(g) * tfc.fromTf(self.robot_configs.MOVEIT_LINK_TO_GRASPING_POINT)) for g in
                    pre_grasps_eef_in_world]
                if self.value_markers is not None:
                    pu.remove_markers(self.value_markers)
                if show_top < 1:
                    self.value_markers = gu.visualize_grasps_with_reachability(pre_grasps_eef_in_world, motion_aware_qualities)
                    print(motion_aware_qualities)
                else:
                    top_indices = np.argsort(motion_aware_qualities)[::-1][:show_top]
                    selected_grasps = [pre_grasps_eef_in_world[idx] for idx in top_indices]
                    selected_qualities = [motion_aware_qualities[idx] for idx in top_indices]
                    self.value_markers = gu.visualize_grasps_with_reachability(selected_grasps, selected_qualities)

        # rank grasps based on two qualities
        if self.use_reachability and not self.use_motion_aware:
            grasp_order_idxs = np.argsort(reachability_qualities)[::-1][:self.max_check]
        elif not self.use_reachability and self.use_motion_aware:
            grasp_order_idxs = np.argsort(motion_aware_qualities)[::-1][:self.max_check]
        elif self.use_reachability and self.use_motion_aware:
            # version 1
            # only return self.max_check grasp indices
            reachability_grasp_order_idxs = np.argsort(reachability_qualities)[::-1][:int(self.max_check/2)]
            motion_grasp_order_idxs = np.argsort(motion_aware_qualities)[::-1][:int(self.max_check/2)]
            # only include good motion aware grasps
            motion_grasp_order_idxs = np.array([i for i in motion_grasp_order_idxs if motion_aware_qualities[i] > 0.5], dtype=np.int)
            grasp_order_idxs = np.concatenate((reachability_grasp_order_idxs, motion_grasp_order_idxs))
            grasp_order_idxs = np.unique(grasp_order_idxs)

            # version 2
            # reachability_grasp_order_idxs = np.argsort(reachability_qualities)[::-1][:self.max_check]
            # filtered_motion_qualities = [motion_aware_qualities[i] for i in reachability_grasp_order_idxs]
            # if max(filtered_motion_qualities) > 0.5:
            #     grasp_order_idxs = [x for _, x in sorted(zip(filtered_motion_qualities, reachability_grasp_order_idxs))]
            #     grasp_order_idxs = grasp_order_idxs[-5:]
            # else:
            #     grasp_order_idxs = reachability_grasp_order_idxs
        else:
            grasp_order_idxs = np.random.permutation(np.arange(len(self.graspit_pregrasps)))[:self.max_check]

        return grasp_order_idxs, reachability_qualities, motion_aware_qualities

    def get_manipulabilities_eef_poses(self, eef_poses_in_object, target_pose):

        eefs_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in eef_poses_in_object]

        ik_jvs = [self.robot.get_arm_ik(pre_g, timeout=0.02, restarts=5, avoid_collisions=True) for pre_g in
                  eefs_in_world]

        manipulabilities_eef = -np.ones(len(ik_jvs))
        idx = np.where([val is not None for val in ik_jvs])[0]
        manipulabilities_eef[idx] = self.robot.get_manipulability(np.array(ik_jvs)[idx])

        return manipulabilities_eef, ik_jvs

    def get_manipulabilities_from_joint_values(self, arm_joint_values):

        manipulabilities_jv = [self.robot.get_manipulability(arm_joint_values) if jv is not None else None for jv in
                               arm_joint_values]
        return manipulabilities_jv

    def get_motion_aware_qualities(self, grasps_eef, pre_grasps_eef, angle, speed):
        """

        :param grasps_eef: a list of grasps in eef reference frame
        :param pre_grasps_eef: a list of pre grasps in eef reference frame
        :param angle: angle in radians
        :param speed: speed in m/s
        :return:
        """
        qualities = [self.compute_motion_aware_quality(g, pg, angle, speed) for g, pg in zip(grasps_eef, pre_grasps_eef)]
        return qualities

    def compute_motion_aware_quality(self, grasp_pose_7d_in_object, pre_grasp_pose_7d_in_object, angle, speed):
        """

        :param grasp_pose_7d_in_object: 7d grasp pose in object frame and uses eef reference frame
        :param pre_grasp_pose_7d_in_object: 7d pre grasp pose in object frame and uses eef reference frame
        :return:
        """
        x = torch.tensor(list(grasp_pose_7d_in_object) + list(pre_grasp_pose_7d_in_object) + [angle] + [speed])
        logits = self.motion_aware_network(x)
        probs = F.softmax(logits)
        quality = probs[1]
        return quality.item()

    def select_grasp_with_ik_from_ranked_grasp(self, target_pose, grasp_order_idxs):
        num_ik_called = 0
        grasp_idx, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv = [None] * 5
        for i, grasp_idx in enumerate(grasp_order_idxs[:self.max_check]):
            _num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv = self.get_iks_pregrasp_and_grasp(
                grasp_idx, target_pose)
            num_ik_called += _num_ik_called
            if planned_grasp_jv is not None:
                break

        if planned_pre_grasp_jv is None:
            print('pre grasp planning fails')
            grasp_idx = None
        elif planned_grasp_jv is None:
            print('pre grasp planning succeeds but grasp planning fails')
            grasp_idx = None
        return grasp_idx, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

    def select_grasp_with_ik_from_ranked_grasp_use_joint_space_dist(self, target_pose, grasp_order_idxs):
        pre_grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                   np.array(self.pre_grasps_eef)[grasp_order_idxs]]
        grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                               np.array(self.grasps_eef)[grasp_order_idxs]]

        ik_pre_grasps = [self.robot.get_arm_ik(g, timeout=0.02, restarts=2, avoid_collisions=True) for g in
                         pre_grasps_eef_in_world]
        ik_grasps = [self.robot.get_arm_ik(g, timeout=0.02, restarts=2, avoid_collisions=False,
                                           arm_joint_values=ik_pre_grasps[i]) for i, g in
                     enumerate(grasps_eef_in_world)]
        num_ik_called = 2 * len(pre_grasps_eef_in_world)
        # num_ik_called = 2

        current_arm_jv = self.robot.get_arm_joint_values()

        pre_grasp_ik_dist = np.zeros(len(ik_pre_grasps)) + np.inf
        idxs = np.where([val is not None for val in ik_pre_grasps])[0]
        for idx in idxs:
            pre_grasp_ik_dist[idx] = np.linalg.norm(np.array(ik_pre_grasps[idx]) - np.array(current_arm_jv))

        grasp_ik_dist = np.zeros(len(ik_grasps)) + np.inf
        idxs = np.where([val is not None for val in ik_grasps])[0]
        for idx in idxs:
            grasp_ik_dist[idx] = np.linalg.norm(np.array(ik_grasps[idx]) - np.array(current_arm_jv))

        ik_dist_sum = np.array(pre_grasp_ik_dist) + np.array(grasp_ik_dist)

        if np.any(np.isfinite(ik_dist_sum)):
            min_ik_dist_idx = np.argmin(ik_dist_sum)
        elif np.any(np.isfinite(pre_grasp_ik_dist)):
            min_ik_dist_idx = np.argmin(pre_grasp_ik_dist)
            print('pre grasp planning succeeds but grasp planning fails')
        else:
            print('pre grasp planning fails')
            return None, num_ik_called, None, None, None, None
        grasp_idx = grasp_order_idxs[min_ik_dist_idx]
        planned_pre_grasp = pre_grasps_eef_in_world[min_ik_dist_idx]
        planned_pre_grasp_jv = ik_pre_grasps[min_ik_dist_idx]
        planned_grasp = grasps_eef_in_world[min_ik_dist_idx]
        planned_grasp_jv = ik_grasps[min_ik_dist_idx]

        return grasp_idx, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

    def select_grasp_combining_reachability_and_motion(self, target_pose, grasp_order_idxs):
        top_ranked_grasp_idxs = grasp_order_idxs

        pre_grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                   np.array(self.pre_grasps_eef)[top_ranked_grasp_idxs]]
        grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                               np.array(self.grasps_eef)[top_ranked_grasp_idxs]]

        ik_pre_grasps = [self.robot.get_arm_ik(g, timeout=0.02, restarts=2, avoid_collisions=True) for g in
                         pre_grasps_eef_in_world]
        ik_grasps = [self.robot.get_arm_ik(g, timeout=0.02, restarts=2, avoid_collisions=False,
                                           arm_joint_values=ik_pre_grasps[i]) for i, g in
                     enumerate(grasps_eef_in_world)]
        num_ik_called = 2 * len(pre_grasps_eef_in_world)

        grasp_idx = None
        for idx in range(self.max_check):
            if ik_grasps[idx] is not None and ik_pre_grasps[idx] is not None:
                grasp_idx = top_ranked_grasp_idxs[idx]
                break
        if grasp_idx is None:
            return None, num_ik_called, None, None, None, None
        planned_pre_grasp = pre_grasps_eef_in_world[idx]
        planned_pre_grasp_jv = ik_pre_grasps[idx]
        planned_grasp = grasps_eef_in_world[idx]
        planned_grasp_jv = ik_grasps[idx]

        return grasp_idx, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

    def plan_grasp(self, target_pose, old_grasp_idx):
        """ Plan a reachable pre_grasp and grasp pose"""
        # timing of the best machine
        ik_call_time = 0.01

        # if an old grasp index is provided
        if old_grasp_idx is not None and not self.always_try_switching:
            _num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv = self.get_iks_pregrasp_and_grasp(
                old_grasp_idx, target_pose)
            grasp_switched = False
            num_ik_called = _num_ik_called
            if planned_grasp_jv is not None:
                planning_time = num_ik_called * ik_call_time
                return old_grasp_idx, planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv, grasp_switched

        # if an old grasp index is not provided or the old grasp is not reachable any more
        rank_grasp_time_start = time.time()
        grasp_order_idxs, reachabilities, motion_aware_qualities = self.rank_grasps(target_pose)
        actual_rank_grasp_time = time.time() - rank_grasp_time_start
        rank_grasp_time = actual_rank_grasp_time if self.fix_grasp_ranking_time is None else self.fix_grasp_ranking_time
        print('Rank grasp actually takes {:.6f}, fixed grasp ranking time {:.6}'.format(actual_rank_grasp_time,
                                                                                        self.fix_grasp_ranking_time))
        if self.use_joint_space_dist:
            selected_g = self.select_grasp_with_ik_from_ranked_grasp_use_joint_space_dist(target_pose, grasp_order_idxs)
        else:
            selected_g = self.select_grasp_with_ik_from_ranked_grasp(target_pose, grasp_order_idxs)
        grasp_idx, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv = selected_g

        grasp_switched = (grasp_idx != old_grasp_idx) and planned_grasp_jv is not None
        planning_time = rank_grasp_time + num_ik_called * ik_call_time
        print("Planning a grasp takes {:.6f}".format(planning_time))
        return grasp_idx, planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv, grasp_switched, grasp_order_idxs, reachabilities, motion_aware_qualities

    def plan_arm_motion(self, grasp_jv):
        """ plan a discretized motion for the arm """
        # whether we should have a fixed planning time
        predicted_period = 0.25 if self.fix_motion_planning_time is None else self.fix_motion_planning_time
        start_time = time.time()

        if self.robot.arm_discretized_plan is not None:
            future_target_index = min(int(predicted_period * 240 + self.robot.arm_wp_target_index),
                                      len(self.robot.arm_discretized_plan) - 1)
            start_joint_values = self.robot.arm_discretized_plan[future_target_index]
            start_joint_velocities = None
            if self.use_previous_jv:
                next_joint_values = self.robot.arm_discretized_plan[
                    min(future_target_index + 1, len(self.robot.arm_discretized_plan) - 1)]
                start_joint_velocities = (next_joint_values - start_joint_values) / (
                        1. / 240)  # TODO: confirm that getting joint velocity this way is right
            previous_discretized_plan = self.robot.arm_discretized_plan[
                                        future_target_index:] if self.use_seed_trajectory else None
            arm_discretized_plan = self.robot.plan_arm_joint_values(grasp_jv, start_joint_values=start_joint_values,
                                                                    previous_discretized_plan=previous_discretized_plan,
                                                                    start_joint_velocities=start_joint_velocities)
        else:
            arm_discretized_plan = self.robot.plan_arm_joint_values(grasp_jv)

        actual_planning_time = time.time() - start_time
        planning_time = actual_planning_time if self.fix_motion_planning_time is None else self.fix_motion_planning_time
        print("Planning a motion actually takes {:.6f}, fixed motion planning time {:.6}".format(actual_planning_time,
                                                                                                 self.fix_motion_planning_time))
        return planning_time, arm_discretized_plan

    def sample_target_location(self):
        r = np.random.uniform(low=self.distance_low, high=self.distance_high)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x = r * cos(theta)
        y = r * sin(theta)
        distance = np.linalg.norm(np.array([x, y]) - np.array(self.robot_initial_pose[0][:2]))
        z = self.target_initial_pose[0][2]
        angle = np.random.uniform(-pi, pi)
        pos = [x, y, z]
        orientation = p.getQuaternionFromEuler([0, 0, angle])
        return [pos, orientation], distance

    @staticmethod
    def sample_target_angle():
        """ return quaternion """
        angle = np.random.uniform(-pi, pi)
        orientation = p.getQuaternionFromEuler([0, 0, angle])
        return list(orientation)

    def sample_convey_linear_motion(self, dist=None, theta=None, length=None, direction=None):
        """ theta is in degrees """
        if dist is None:
            dist = np.random.uniform(low=self.distance_low, high=self.distance_high)
        if theta is None:
            theta = np.random.uniform(low=-90, high=90)
        if length is None:
            length = 1.0
        if direction is None:
            direction = random.sample([-1, 1], 1)[0]
        return dist, theta, length, direction

    def sample_convey_circular_motion(self, dist=None, theta=None, length=None, direction=None):
        """
        theta is in degrees

        dist: the distance from the robot,
        theta: the angle of the starting position,
        length: the length of the trajectory
        direction: 1 is counter clockwise, -1 is clockwise
        """
        # this is effectively the only difference
        if dist is None:
            dist = np.random.uniform(low=self.distance_low, high=self.distance_high)
        if theta is None:
            theta = np.random.uniform(low=0, high=360)
        if length is None:
            length = 1.0
        if direction is None:
            direction = random.sample([-1, 1], 1)[0]
        return dist, theta, length, direction


    def get_obstacles_regions(self, distance, theta, length, visualize_region=True):
        region_length = (length - 2 * self.distance_between_region) / 3
        theta = theta - 90
        translation = np.array([[1, 0, 0],
                                [0, 1, distance],
                                [0, 0, 1]])
        rotation = np.array([[cos(radians(theta)), -sin(radians(theta)), 0],
                             [sin(radians(theta)), cos(radians(theta)), 0],
                             [0, 0, 1]])
        transform_matrix = rotation.dot(translation)

        # there are 6 regions
        regions = [[(-length / 2.0, self.obstacle_distance_low),
                    (-length / 2.0, self.obstacle_distance_high),
                    (-length / 2.0 + region_length, self.obstacle_distance_high),
                    (-length / 2.0 + region_length, self.obstacle_distance_low)],

                   [(-length / 2.0 + region_length + self.distance_between_region, self.obstacle_distance_low),
                    (-length / 2.0 + region_length + self.distance_between_region, self.obstacle_distance_high),
                    (-length / 2.0 + 2 * region_length + self.distance_between_region, self.obstacle_distance_high),
                    (-length / 2.0 + 2 * region_length + self.distance_between_region, self.obstacle_distance_low)],

                   [(-length / 2.0 + 2 * region_length + 2 * self.distance_between_region, self.obstacle_distance_low),
                    (-length / 2.0 + 2 * region_length + 2 * self.distance_between_region, self.obstacle_distance_high),
                    (-length / 2.0 + 3 * region_length + 2 * self.distance_between_region, self.obstacle_distance_high),
                    (-length / 2.0 + 3 * region_length + 2 * self.distance_between_region, self.obstacle_distance_low)],

                   [(-length / 2.0 + 2 * region_length + 2 * self.distance_between_region, -self.obstacle_distance_low),
                    (-length / 2.0 + 2 * region_length + 2 * self.distance_between_region, -self.obstacle_distance_high),
                    (-length / 2.0 + 3 * region_length + 2 * self.distance_between_region, -self.obstacle_distance_high),
                    (-length / 2.0 + 3 * region_length + 2 * self.distance_between_region, -self.obstacle_distance_low)],

                   # [(-length / 2.0 + region_length + self.distance_between_region, -self.obstacle_distance_low),
                   #  (-length / 2.0 + region_length + self.distance_between_region, -self.obstacle_distance_high),
                   #  (-length / 2.0 + 2 * region_length + self.distance_between_region, -self.obstacle_distance_high),
                   #  (-length / 2.0 + 2 * region_length + self.distance_between_region, -self.obstacle_distance_low)],

                   [(-length / 2.0, -self.obstacle_distance_low),
                    (-length / 2.0, -self.obstacle_distance_high),
                    (-length / 2.0 + region_length, -self.obstacle_distance_high),
                    (-length / 2.0 + region_length, -self.obstacle_distance_low)]]

        regions_transformed = []
        for r in regions:
            r_transformed = []
            for point in r:
                r_transformed.append(tuple(transform_matrix.dot(np.array(point + (1,)))[:2]))
            regions_transformed.append(r_transformed)

        # visualize
        if visualize_region:
            v_height = 0.01
            for r in regions_transformed:
                lines = zip(r, r[1:] + [r[0]])
                for (p1, p2) in lines:
                    pu.draw_line(p1 + (v_height,), p2 + (v_height,), rgb_color=(0, 1, 0))

        # create original polygons
        polygons = [Polygon(r) for r in regions]
        return polygons, transform_matrix

    def load_obstacles_collision_free(self, distance, theta, length):
        polygons, transform_matrix = self.get_obstacles_regions(distance, theta, length)

        poses = []
        obstacles = []
        num_regions = len(polygons)
        if len(self.obstacle_names) <= num_regions:
            choices = random.choice(list(combinations(range(num_regions), len(self.obstacle_names))))
        else:
            choices = []
            for _ in range(len(self.obstacle_names) // num_regions):
                choices.extend(np.random.permutation(num_regions))
            choices.extend(random.choice(list(combinations(range(num_regions), len(self.obstacle_names) % num_regions))))
        for choice, urdf, extents, z in zip(choices, self.obstacle_urdfs, self.obstacle_extentss, self.obstacle_zs):
            position_xy = mu.random_point_in_polygon(polygons[choice])
            position_xy = tuple(transform_matrix.dot(np.array(position_xy + (1,)))[:2])
            pose = [list(position_xy) + [z], self.sample_target_angle()]
            poses.append(pose)
            obstacles.append(p.loadURDF(urdf, pose[0], pose[1]))
        return obstacles

    def load_obstacles_at_poses(self, poses):
        obstacles = []
        for pose_flattened, urdf in zip(poses, self.obstacle_urdfs):
            pose = pu.split_7d(pose_flattened)
            obstacles.append(p.loadURDF(urdf, pose[0], pose[1]))
        return obstacles

    def check_done(self):
        done = False
        if self.conveyor.wp_target_index == len(self.conveyor.discretized_trajectory):
            # conveyor trajectory finishes
            done = True
        if pu.get_body_pose(self.target)[0][2] < self.target_initial_pose[0][2] - 0.03:
            # target knocked down
            done = True
        return done

    def check_lazy_plan(self, distance, grasp_switched, distance_travelled):
        """ check whether we should do lazy plan """
        do_lazy_plan = distance > self.lazy_threshold and \
                       distance_travelled < self.distance_travelled_threshold and \
                       self.robot.arm_discretized_plan is not None and \
                       self.robot.arm_wp_target_index != len(self.robot.arm_discretized_plan) and \
                       not grasp_switched
        return do_lazy_plan

    def calculate_prediction_time(self, distance):
        if distance is None:
            # print('large')
            prediction_time = 2
        else:
            if self.small_prediction_threshold < distance <= self.large_prediction_threshold:
                # print('medium')
                prediction_time = 1
            elif distance <= self.small_prediction_threshold:
                # print('small')
                prediction_time = 0
            else:
                # print('large')
                prediction_time = 2
        return prediction_time


class Conveyor:
    def __init__(self, initial_pose, urdf_path):
        self.initial_pose = initial_pose
        self.urdf_path = urdf_path
        self.id = p.loadURDF(self.urdf_path, initial_pose[0], initial_pose[1])

        self.cid = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=initial_pose[0],
                                      childFrameOrientation=initial_pose[1])

        # motion related
        self.start_pose = None
        self.target_pose = None
        self.discretized_trajectory = None
        self.wp_target_index = 0
        self.distance = None
        self.theta = None
        self.length = None
        self.direction = None
        self.speed = None
        self.z_start = None
        self.z_end = None
        self.circular_angles = None

    def set_pose(self, pose):
        pu.set_pose(self.id, pose)
        self.control_pose(pose)

    def get_pose(self):
        return pu.get_body_pose(self.id)

    def control_pose(self, pose):
        p.changeConstraint(self.cid, jointChildPivot=pose[0], jointChildFrameOrientation=pose[1])

    def step(self):
        if self.discretized_trajectory is None or self.wp_target_index == len(self.discretized_trajectory):
            pass
        else:
            self.control_pose(self.discretized_trajectory[self.wp_target_index])
            self.wp_target_index += 1

    def initialize_linear_motion(self, dist, theta, length, direction, speed, z_start, z_end, variable_speed=False):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360)
        :param length: the length of the motion
        :param direction: the direction of the motion
            1: from smaller theta to larger theta
            -1: from larger theta to smaller theta
        :param z_start: the height of the conveyor at the start
        :param z_end: the height of the conveyor at the end
        :param speed: the speed of the conveyor
        :param variable_speed: determines if the speed of the conveyor is variable or constant
        """
        self.distance = float(dist)
        self.theta = float(theta)
        self.length = float(length)
        self.direction = float(direction)
        self.speed = float(speed)
        self.z_start = float(z_start)
        self.z_end = float(z_end)
        # use the orientation of the initial pose
        orientation = self.initial_pose[1]
        # compute start xy and end xy
        new_dist = sqrt(dist ** 2 + (length / 2.0) ** 2)
        delta_theta = atan((length / 2.0) / dist)

        theta_large = radians(self.theta) + delta_theta
        theta_small = radians(self.theta) - delta_theta

        if direction == -1:
            start_xy = [new_dist * cos(theta_large), new_dist * sin(theta_large)]
            target_xy = [new_dist * cos(theta_small), new_dist * sin(theta_small)]
        elif direction == 1:
            target_xy = [new_dist * cos(theta_large), new_dist * sin(theta_large)]
            start_xy = [new_dist * cos(theta_small), new_dist * sin(theta_small)]
        else:
            raise ValueError('direction must be in {-1, 1}')
        start_position = start_xy + [self.z_start]
        target_position = target_xy + [self.z_end]

        self.start_pose = [start_position, orientation]
        self.target_pose = [target_position, orientation]

        if variable_speed:
            # acc_levels = 2     # accelerate / decelerate level e.g. acc_levels = 3 -> [1/3, 1/2, 1, 2, 3]
            # speed_multipliers = set(np.concatenate((np.arange(1, acc_levels+1), 1./np.arange(1, acc_levels+1))))
            # speeds = np.array(sorted(speed_multipliers)) * self.speed
            n_segments = 10  # num speed switches
            # speed_multipliers = np.array(list(range(1, n_segments//2 + 1)) * 2) / (n_segments/2.)
            # rng = np.random.RandomState(2)
            # speeds = rng.permutation(speed_multipliers) * self.speed
            speed_multipliers = np.linspace(0.6, 1.0, n_segments)[::-1]
            speeds = speed_multipliers * self.speed
            segments = np.linspace(start_position, target_position, n_segments+1)
            position_trajectory = []
            for i in range(n_segments):
                # speed = np.random.choice(speeds)
                speed = speeds[i]
                dist = np.linalg.norm(segments[i] - segments[i+1])
                num_steps = int(dist / speed * 240)
                wps = np.linspace(segments[i], segments[i+1], num_steps)
                position_trajectory.extend(wps)
        else:
            num_steps = int(self.length / self.speed * 240)
            position_trajectory = np.linspace(start_position, target_position, num_steps)
        self.discretized_trajectory = [[list(pos), orientation] for pos in position_trajectory]
        self.wp_target_index = 1

    def initialize_sinusoid_motion(self, dist, theta, length, direction, speed, amp_div=8, period_div=3):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360)
        :param length: the length of the motion
        :param direction: the direction of the motion
            1: from smaller theta to larger theta
            -1: from larger theta to smaller theta
        :param speed: the speed of the conveyor
        """
        self.distance = float(dist)
        self.theta = float(theta)
        self.length = float(length)
        self.direction = float(direction)
        self.speed = float(speed)
        # uses the z value and orientation of the current pose
        z = self.get_pose()[0][-1]
        orientation = self.get_pose()[1]

        num_steps = int(self.length / self.speed * 240)

        start_position = np.array([0, -length / 2.0, 1]) * direction
        target_position = np.array([0, length / 2.0, 1]) * direction
        position_trajectory = np.linspace(start_position, target_position, num_steps)
        # Amplitude: length/4., period: self.length/3 i.e. 3 sinusoids within the length of the trajectory
        position_trajectory[:, 0] = (self.length * 1.0 / amp_div) * np.sin(2 * np.pi * position_trajectory[:, 1] /
                                                                           (self.length * 1.0 / period_div))
        T_1 = np.array([[np.cos(radians(self.theta)), -np.sin(radians(self.theta)), 0],
                        [np.sin(radians(self.theta)), np.cos(radians(self.theta)), 0],
                        [0, 0, 1]])
        T_2 = np.array([[1, 0, self.distance], [0, 1, 0], [0, 0, 1]])
        position_trajectory = np.dot(T_1, np.dot(T_2, position_trajectory.T)).T
        position_trajectory[:, -1] = z
        self.start_pose = [position_trajectory[0], orientation]
        self.target_pose = [position_trajectory[-1], orientation]

        self.discretized_trajectory = [[list(pos), orientation] for pos in position_trajectory]
        self.wp_target_index = 1

    def initialize_circular_motion(self, dist, theta, length, direction, speed):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360)
        :param length: the length of the motion
        :param direction: the direction of the motion
            1: counter clockwise
            -1: clockwise
        :param speed: the speed of the conveyor
        """
        self.distance = float(dist)
        self.theta = float(theta)
        self.length = float(length)
        self.direction = float(direction)
        self.speed = float(speed)
        # uses the z value and orientation of the current pose
        z = self.get_pose()[0][-1]
        orientation = self.get_pose()[1]

        # calculate waypoints
        num_points = int(self.length / self.speed) * 240
        delta_angle = self.length / self.distance
        angles = np.linspace(radians(theta), radians(theta)+delta_angle, num_points)
        if direction == -1:
            angles = angles[::-1]
        self.circular_angles = angles

        self.discretized_trajectory = [[(cos(ang) * self.distance, sin(ang) * self.distance, z), orientation] for ang in angles]
        self.wp_target_index = 1

        self.start_pose = self.discretized_trajectory[0]
        self.target_pose = self.discretized_trajectory[-1]

    def initialize_conveyor_motion_v2(self, angle, speed, length, start_pose=None, is_sinusoid=False):
        """
        Initialize a motion using the start pose as initial pose, in the direction of the angle.

        :param angle: the angle of the motion direction in the conveyor frame, in degrees
        :param speed: the speed of the motion
        """
        self.theta = float(angle)
        self.length = float(length)
        self.speed = float(speed)
        start_pose_in_world = conveyor_pose = self.get_pose() if start_pose is None else start_pose

        num_steps = int(length / speed * 240)

        start_position = np.array([0, 0, 1])
        target_position = np.array([length, 0, 1])
        position_trajectory = np.linspace(start_position, target_position, num_steps)

        if is_sinusoid:
            # Amplitude: dist/2., period: self.length/3 i.e. 3 sinusoids within the length of the trajectory
            position_trajectory[:, 1] = self.length/20. * np.sin(2*np.pi * position_trajectory[:, 0] / (self.length/4))

        # rotate direction of motion according to angle
        T_1 = np.array([[np.cos(radians(self.theta)), -np.sin(radians(self.theta)), 0],
                        [np.sin(radians(self.theta)), np.cos(radians(self.theta)), 0],
                        [0, 0, 1]])
        position_trajectory = np.dot(T_1, position_trajectory.T).T
        position_trajectory[:, -1] = 0

        # adjust motion to the reference start position
        position_trajectory = np.dot(tfc.toMatrix(tfc.fromTf(start_pose_in_world)),
                                     np.concatenate((position_trajectory, np.ones((position_trajectory.shape[0], 1))),
                                                    axis=-1).T)[:-1].T

        self.discretized_trajectory = [[list(p), start_pose_in_world[1]] for p in position_trajectory]
        self.wp_target_index = 1
        return self.discretized_trajectory[0], self.discretized_trajectory[-1]

    def clear_motion(self):
        self.start_pose = None
        self.target_pose = None
        self.discretized_trajectory = None
        self.wp_target_index = 0
        self.distance = None
        self.theta = None
        self.length = None
        self.direction = None
        self.circular_angles = None

    def predict(self, duration):
        # predict the ground truth future pose of the conveyor
        num_predicted_steps = int(duration * 240)
        predicted_step_index = self.wp_target_index - 1 + num_predicted_steps
        if predicted_step_index < len(self.discretized_trajectory):
            return self.discretized_trajectory[predicted_step_index]
        else:
            return self.discretized_trajectory[-1]


class MotionPredictorKF:
    def __init__(self, time_step):
        # the predictor takes a pose estimation once every time step
        self.time_step = time_step
        self.target_pose = None
        self.kf = None
        self.initialized = False

    def initialize_predictor(self, initial_pose):
        self.target_pose = initial_pose
        x0 = np.zeros(9)
        x0[:3] = initial_pose[0]
        # x0[3] = 0.03
        x0 = x0[:, None]
        self.kf = create_kalman_filter(x0=x0)
        self.initialized = True

    def reset_predictor(self):
        self.target_pose = None
        self.kf = None

    def update(self, current_pose):
        # TODO quaternion is not considered yet
        if not self.initialized:
            raise ValueError("predictor not initialized!")
        self.target_pose = current_pose
        current_position = current_pose[0]
        self.kf.predict(dt=self.time_step)
        self.kf.update(np.array(current_position)[:, None])

    def predict(self, duration):
        """ return just a predicted pose """
        if not self.initialized:
            raise ValueError("predictor not initialized!")
        # print("current position: {}".format(self.target_pose[0]))
        future_estimate = np.dot(self.kf.H, self.kf.predict(dt=duration, predict_only=True))
        future_position = list(np.squeeze(future_estimate))
        # print("future position: {}\n".format(future_position))
        future_orientation = self.target_pose[1]
        return [future_position, future_orientation]


class CircularMotionPredictorKF:
    def __init__(self, time_step):
        # the predictor takes a pose estimation once every time step
        self.time_step = time_step
        self.target_pose = None
        self.kf = None
        self.radius = None
        self.initialized = False
        self.reference_angle = 0

    def adjust_angle_range(self, angle):
        # temporary function to handle wrap around angle
        diff = pu.wrap_angle(angle-self.reference_angle)
        return self.reference_angle + diff

    def initialize_predictor(self, initial_pose):
        self.target_pose = initial_pose
        x0 = np.zeros(3)
        radius, angle = np.linalg.norm(initial_pose[0][:2]), np.arctan2(initial_pose[0][1], initial_pose[0][0])
        self.radius = radius
        self.reference_angle = angle
        x0[:1] = angle
        # x0[3] = 0.03
        x0 = x0[:, None]
        self.kf = create_kalman_filter(x0=x0, ndim=1)
        self.initialized = True

    def reset_predictor(self):
        self.target_pose = None
        self.kf = None

    def update(self, current_pose):
        # TODO quaternion is not considered yet
        if not self.initialized:
            raise ValueError("predictor not initialized!")
        self.target_pose = current_pose
        current_position = current_pose[0]
        current_angle = np.arctan2(current_pose[0][1], current_pose[0][0])
        current_angle = self.adjust_angle_range(current_angle)
        self.reference_angle = current_angle
        self.kf.predict(dt=self.time_step)
        self.kf.update(np.array([current_angle])[:, None])

    def predict(self, duration):
        """ return just a predicted pose """
        if not self.initialized:
            raise ValueError("predictor not initialized!")
        # print("current position: {}".format(self.target_pose[0]))
        future_estimate = np.dot(self.kf.H, self.kf.predict(dt=duration, predict_only=True))
        angle = np.squeeze(future_estimate)
        future_position = [self.radius*np.cos(angle), self.radius*np.sin(angle), self.target_pose[0][2]]
        # print("future position: {}\n".format(future_position))
        future_orientation = self.target_pose[1]
        return [future_position, future_orientation]


class LSTMMotionPredictorKF:
    def __init__(self, time_step, model_weight_path, is_lstm_model=False, history=5):
        # the predictor takes a pose estimation once every time step
        self.time_step = time_step      # measurement period (T) i.e. 1/frequency
        self.target_pose = None
        self.kf = None
        self.initialized = False
        self.future_position = None

        self.future_horizons = (1.0,  2.0)
        self.history = history
        self.dim = 3
        self.input_shape = (self.history, self.dim)
        self.output_shape = (len(self.future_horizons), self.dim)
        self.is_lstm_model = is_lstm_model
        self.data_gen_sampling_frequency = 1.0 / time_step  # 50 not really needed
        self.measurement_sampling_frequency = 1.0 / time_step  # 5
        self.subsample_ratio = int(self.data_gen_sampling_frequency / self.measurement_sampling_frequency)

        import motion_prediction_model
        self.prediction_model = motion_prediction_model.load_model(model_weight_path, self.input_shape,
                                                                   self.output_shape, is_lstm_model=self.is_lstm_model)
        self.position_history = deque(maxlen=int(self.data_gen_sampling_frequency / self.measurement_sampling_frequency * self.history + 1))
        self.update_counter = 1

    def initialize_predictor(self, initial_pose):
        for _ in range(self.position_history.maxlen):
            self.position_history.append(initial_pose[0])
        self.target_pose = initial_pose
        self.future_position = np.array(initial_pose[0])
        self.initialized = True

    def reset_predictor(self):
        self.prediction_model.reset_states()
        self.target_pose = None
        self.update_counter = 1

    def update(self, current_pose):
        # TODO quaternion is not considered yet
        if not self.initialized:
            raise ValueError("predictor not initialized!")
        self.position_history.append(current_pose[0])
        if self.update_counter == 0:
            data = np.array(self.position_history)[- self.history * self.subsample_ratio:: self.subsample_ratio]
            if self.is_lstm_model:
                self.future_position = self.prediction_model.predict((data - data[0])[None, None, ...]).squeeze() + data[0]
            else:
                # note that the model predicts pose relative to the start of the history
                self.future_position = self.prediction_model.predict((data - data[0])[None, ...]).squeeze() + data[0]
        self.target_pose = current_pose
        self.update_counter = (self.update_counter + 1) % self.subsample_ratio

    def predict(self, duration):
        """ return just a predicted pose """
        if not self.initialized:
            raise ValueError("predictor not initialized!")
        if duration == 0.0:
            return self.target_pose

        if len(self.future_position.shape) > 1:
            assert duration in self.future_horizons, 'motion prediction network was trained on future_horizons: {}'.format(self.future_horizons)
            future_idx = np.where(np.array(self.future_horizons) == duration)[0][0]
            future_position = self.future_position[future_idx]
        else:
            future_position = self.future_position

        # TODO: make this dependent on duration
        future_orientation = self.target_pose[1]
        return [future_position, future_orientation]
