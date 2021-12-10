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
import moveit_commander as mc
from moveit_msgs.srv import GetPositionIK, GetPositionFK

import rospy
from moveit_msgs.msg import DisplayTrajectory, PositionIKRequest, RobotState, GenericTrajectory, RobotTrajectory
from moveit_msgs.msg import TrajectoryConstraints, Constraints, JointConstraint
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from collections import namedtuple

np.set_printoptions(suppress=True)

Motion = namedtuple('Motion', ['position_trajectory', 'time_trajectory', 'velocity_trajectory'])


class MicoController:
    GROUPS = {
        'arm': ['m1n6s200_joint_1',
                'm1n6s200_joint_2',
                'm1n6s200_joint_3',
                'm1n6s200_joint_4',
                'm1n6s200_joint_5',
                'm1n6s200_joint_6'],
        'gripper': ["m1n6s200_joint_finger_1",
                    "m1n6s200_joint_finger_tip_1",
                    "m1n6s200_joint_finger_2",
                    "m1n6s200_joint_finger_tip_2"]
    }

    GROUP_INDEX = {
        'arm': [2, 3, 4, 5, 6, 7],
        'gripper': [9, 10, 11, 12]
    }

    INDEX_NAME_MAP = {
        0: 'connect_root_and_world',
        1: 'm1n6s200_joint_base',
        2: 'm1n6s200_joint_1',
        3: 'm1n6s200_joint_2',
        4: 'm1n6s200_joint_3',
        5: 'm1n6s200_joint_4',
        6: 'm1n6s200_joint_5',
        7: 'm1n6s200_joint_6',
        8: 'm1n6s200_joint_end_effector',
        9: 'm1n6s200_joint_finger_1',
        10: 'm1n6s200_joint_finger_tip_1',
        11: 'm1n6s200_joint_finger_2',
        12: 'm1n6s200_joint_finger_tip_2',
    }

    EEF_LINK_INDEX = 8
    OPEN_POSITION = [0.0, 0.0, 0.0, 0.0]
    CLOSED_POSITION = [1.1, 0.0, 1.1, 0.0]
    LINK6_COM = [-0.002216, -0.000001, -0.058489]
    LIFT_VALUE = 0.2
    HOME = [4.80469, 2.92482, 1.002, 4.20319, 1.4458, 1.3233]
    EEF_LINK = "m1n6s200_end_effector"
    BASE_LINK = "root"
    ARM = 'arm'

    # this is read from moveit_configs joint_limits.yaml
    MOVEIT_ARM_MAX_VELOCITY = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35]

    def __init__(self,
                 initial_pose,
                 initial_joint_values,
                 urdf_path,
                 planner_id='PRM',   # PRM, RRTConnect
                 use_manipulability=False):
        self.initial_pose = initial_pose
        self.initial_joint_values = initial_joint_values
        self.urdf_path = urdf_path
        self.use_manipulability = use_manipulability

        self.id = p.loadURDF(self.urdf_path,
                             basePosition=self.initial_pose[0],
                             baseOrientation=self.initial_pose[1],
                             flags=p.URDF_USE_SELF_COLLISION)

        self.arm_ik_svr = rospy.ServiceProxy('compute_ik', GetPositionIK)
        self.arm_fk_svr = rospy.ServiceProxy('compute_fk', GetPositionFK)

        self.arm_commander_group = mc.MoveGroupCommander('arm')
        self.arm_commander_group.set_planner_id(planner_id=planner_id)
        self.robot = mc.RobotCommander()
        self.scene = mc.PlanningSceneInterface()
        rospy.sleep(2)

        if use_manipulability:
            from manipulability_computation.srv import GetManipulabilityIndex
            self.compute_manipulability_svr = rospy.ServiceProxy('compute_manipulability_index', GetManipulabilityIndex)
            self.compute_manipulability_svr.wait_for_service()

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=1)

        self.arm_difference_fn = pu.get_difference_fn(self.id, self.GROUP_INDEX['arm'])
        self.reset()
        self.robot_state_template = self.robot.get_current_state()

        self.arm_max_joint_velocities = [pu.get_max_velocity(self.id, j_id) for j_id in self.GROUP_INDEX['arm']]

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['arm'], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def control_arm_joints(self, joint_values):
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def compute_next_action(self, object_pose, ):
        pass

    def step(self):
        """ step the robot for 1/240 second """
        # calculate the latest conf and control array
        if self.arm_discretized_plan is None or self.arm_wp_target_index == len(self.arm_discretized_plan):
            pass
        else:
            self.control_arm_joints(self.arm_discretized_plan[self.arm_wp_target_index])
            self.arm_wp_target_index += 1

        if self.gripper_discretized_plan is None or self.gripper_wp_target_index == len(self.gripper_discretized_plan):
            pass
        else:
            self.control_gripper_joints(self.gripper_discretized_plan[self.gripper_wp_target_index])
            self.gripper_wp_target_index += 1

    def reset(self):
        self.set_arm_joints(self.initial_joint_values)
        self.set_gripper_joints(self.OPEN_POSITION)
        self.clear_scene()
        self.arm_discretized_plan = None
        self.gripper_discretized_plan = None
        self.arm_wp_target_index = 0
        self.gripper_wp_target_index = 0

    def update_arm_motion_plan(self, arm_discretized_plan):
        self.arm_discretized_plan = arm_discretized_plan
        self.arm_wp_target_index = 1

    def update_gripper_motion_plan(self, gripper_discretized_plan):
        self.gripper_discretized_plan = gripper_discretized_plan
        self.gripper_wp_target_index = 1

    def get_arm_joint_values(self):
        return pu.get_joint_positions(self.id, self.GROUP_INDEX['arm'])

    def get_gripper_joint_values(self):
        return pu.get_joint_positions(self.id, self.GROUP_INDEX['gripper'])

    def get_eef_pose(self):
        return pu.get_link_pose(self.id, self.EEF_LINK_INDEX)

    def get_current_max_eef_velocity(self, arm_joint_values):
        arm_joint_values = self.get_arm_joint_values() if arm_joint_values is None else arm_joint_values
        jacobian = self.arm_commander_group.get_jacobian_matrix(arm_joint_values)
        max_eef_velocity = np.dot(jacobian, self.arm_max_joint_velocities)
        return np.squeeze(np.array(max_eef_velocity))

    def get_manipulability_srv(self, list_of_joint_values):
        assert self.use_manipulability, 'self.use_manipulability flag is set to false, check constructor and start manipulability ros service'
        manipulability_indexes = []
        for jvs in list_of_joint_values:
            if jvs is None:
                manipulability_indexes.append(None)
            else:
                self.robot_state_template.joint_state.name = self.robot_state_template.joint_state.name[:len(jvs)]
                self.robot_state_template.joint_state.position = jvs
                result = self.compute_manipulability_svr(self.robot_state_template, self.ARM)

                manipulability_iget_manipulability_srvndexes.append(result.manipulability_index)
        return manipulability_indexes

    def get_manipulability(self, list_of_joint_values, translation_only=True):
        manipulability_indexes = []
        for jvs in list_of_joint_values:
            if jvs is None:
                manipulability_indexes.append(None)
            else:
                jacobian = self.arm_commander_group.get_jacobian_matrix(list(jvs))
                if translation_only:
                    matrix = np.dot(jacobian[:3], jacobian[:3].T)
                else:
                    matrix = np.dot(jacobian, jacobian.T)
                manipulability_index = np.sqrt(np.linalg.det(matrix))
                manipulability_indexes.append(manipulability_index)

                # # u_, s_, vh_ = np.linalg.svd(jacobian, full_matrices=True)
                # # index2 = np.sqrt(np.prod(s_))

        return manipulability_indexes

    def get_arm_ik_pybullet(self, pose_2d, arm_joint_values=None, gripper_joint_values=None):
        gripper_joint_values = self.get_gripper_joint_values() if gripper_joint_values is None else gripper_joint_values
        arm_joint_values = self.get_arm_joint_values() if arm_joint_values is None else arm_joint_values

        joint_values = p.calculateInverseKinematics(self.id,
                                                    self.EEF_LINK_INDEX,  # self.JOINT_INDICES_DICT[self.EEF_LINK],
                                                    pose_2d[0],
                                                    pose_2d[1],
                                                    currentPositions=arm_joint_values + gripper_joint_values,
                                                    # maxNumIterations=100,
                                                    # residualThreshold=.01
                                                    )
        ik_result = list(joint_values[:6])
        # handle joint limit violations. TODO: confirm that this logic makes sense
        for i in range(len(self.GROUP_INDEX['arm'])):
            if pu.violates_limit(self.id, self.GROUP_INDEX['arm'][i], ik_result[i]):
                lower, upper = pu.get_joint_limits(self.id, self.GROUP_INDEX['arm'][i])
                if ik_result[i] < lower and ik_result[i] + 2*np.pi > upper:
                    ik_result[i] = lower
                if ik_result[i] > upper and ik_result[i] - 2*np.pi < lower:
                    ik_result[i] = upper
                if ik_result[i] < lower:
                    ik_result[i] += 2 * np.pi
                if ik_result[i] > upper:
                    ik_result[i] -= 2 * np.pi

        return ik_result

    def get_arm_ik(self, pose_2d, timeout=0.1, restarts=1, avoid_collisions=True, arm_joint_values=None,
                   gripper_joint_values=None):
        gripper_joint_values = self.get_gripper_joint_values() if gripper_joint_values is None else gripper_joint_values
        arm_joint_values = self.get_arm_joint_values() if arm_joint_values is None else arm_joint_values
        j = None
        for i in range(restarts):
            # s = time.time()
            j = self.get_arm_ik_ros(pose_2d, timeout, avoid_collisions, arm_joint_values, gripper_joint_values)
            # print('ik call takes {}'.format(time.time() - s))
            if j is not None:
                break
        if j is None:
            # print("No ik exists!")
            return None
        else:
            return j

    def get_arm_fk(self, arm_joint_values):
        pose = self.get_arm_fk_ros(arm_joint_values)
        return gu.pose_2_list(pose) if pose is not None else None

    def get_arm_fk_pybullet(self, joint_values):
        return pu.forward_kinematics(self.id, self.GROUP_INDEX['arm'], joint_values, self.EEF_LINK_INDEX)

    def get_arm_ik_ros(self, pose_2d, timeout, avoid_collisions, arm_joint_values, gripper_joint_values):
        """
        Compute arm IK.
        :param pose_2d: 2d list, [[x, y, z], [x, y, z, w]]
        :param timeout: timeout in seconds
        :param avoid_collisions: whether to avoid collisions when computing ik
        :param arm_joint_values: arm joint values to seed the IK
        :param gripper_joint_values: gripper joint values for computing IK
        :return: a list of joint values if success; None if no ik
        """
        # when there is collision, we need timeout to control the time to search
        rospy.wait_for_service('compute_ik')

        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = self.BASE_LINK
        gripper_pose_stamped.header.stamp = rospy.Time.now()
        gripper_pose_stamped.pose = Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))

        service_request = PositionIKRequest()
        service_request.group_name = "arm"
        service_request.ik_link_name = self.EEF_LINK
        service_request.pose_stamped = gripper_pose_stamped
        service_request.timeout.nsecs = timeout * 1e9
        service_request.avoid_collisions = avoid_collisions

        seed_robot_state = self.robot.get_current_state()
        seed_robot_state.joint_state.name = self.GROUPS['arm'] + self.GROUPS['gripper']
        seed_robot_state.joint_state.position = arm_joint_values + gripper_joint_values
        service_request.robot_state = seed_robot_state

        try:
            resp = self.arm_ik_svr(ik_request=service_request)
            if resp.error_code.val == -31:
                # print("No ik exists!")
                return None
            elif resp.error_code.val == 1:
                return self.parse_joint_state_arm(resp.solution.joint_state)
            else:
                print("Other errors!")
                return None
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def get_arm_fk_ros(self, arm_joint_values):
        """ return a ros pose """
        rospy.wait_for_service('compute_fk')

        header = Header(frame_id="world")
        fk_link_names = [self.EEF_LINK]
        robot_state = RobotState()
        robot_state.joint_state.name = self.GROUPS['arm']
        robot_state.joint_state.position = arm_joint_values

        try:
            resp = self.arm_fk_svr(header=header, fk_link_names=fk_link_names, robot_state=robot_state)
            if resp.error_code.val != 1:
                print("error ({}) happens when computing fk".format(resp.error_code.val))
                return None
            else:
                return resp.pose_stamped[0].pose
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def parse_joint_state_arm(self, joint_state):
        d = {n: v for (n, v) in zip(joint_state.name, joint_state.position)}
        return [d[n] for n in self.GROUPS['arm']]

    @staticmethod
    def convert_range(joint_values):
        """ Convert continuous joint to range [-pi, pi] """
        circular_idx = [0, 3, 4, 5]
        new_joint_values = []
        for i, v in enumerate(joint_values):
            if i in circular_idx:
                new_joint_values.append(pu.wrap_angle(v))
            else:
                new_joint_values.append(v)
        return new_joint_values

    @staticmethod
    def process_plan(moveit_plan, start_joint_values):
        """
        convert position trajectory to work with current joint values
        :param moveit_plan: MoveIt plan
        :return plan: Motion
        """
        diff = np.array(start_joint_values) - np.array(moveit_plan.joint_trajectory.points[0].positions)
        for p in moveit_plan.joint_trajectory.points:
            p.positions = (np.array(p.positions) + diff).tolist()
        plan = MicoController.extract_plan(moveit_plan)
        return plan

    @staticmethod
    def process_discretized_plan(discretized_plan, start_joint_values):
        """
        convert discretized plan to work with current joint values
        :param discretized_plan: discretized plan, list of waypoints
        :return plan: Motion
        """
        diff = np.array(start_joint_values) - np.array(discretized_plan[0])
        new_discretized_plan = []
        for wp in discretized_plan:
            new_discretized_plan.append((np.array(wp) + diff).tolist())
        return new_discretized_plan

    @staticmethod
    def extract_plan(moveit_plan):
        """
        Extract numpy arrays of position, velocity and time trajectories from moveit plan,
        and return Motion object
        """
        points = moveit_plan.joint_trajectory.points
        position_trajectory = []
        velocity_trajectory = []
        time_trajectory = []
        for p in points:
            position_trajectory.append(list(p.positions))
            velocity_trajectory.append(list(p.velocities))
            time_trajectory.append(p.time_from_start.to_sec())
        return Motion(np.array(position_trajectory), np.array(time_trajectory), np.array(velocity_trajectory))

    def set_gripper_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['gripper'], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values)

    def control_gripper_joints(self, joint_values):
        pu.control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values)

    def close_gripper(self, realtime=False):
        waypoints = self.plan_gripper_joint_values(self.CLOSED_POSITION)
        self.execute_gripper_plan(waypoints, realtime)

    def open_gripper(self, realtime=False):
        waypoints = self.plan_gripper_joint_values(self.OPEN_POSITION)
        self.execute_gripper_plan(waypoints, realtime)

    def plan_gripper_joint_values(self, goal_joint_values, start_joint_values=None, duration=None):
        if start_joint_values is None:
            start_joint_values = self.get_gripper_joint_values()
        num_steps = 240 if duration is None else int(duration*240)
        discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        return discretized_plan

    def create_seed_trajectory(self, seed_discretized_plan, start_joint_values, goal_joint_values):

        skip = max(1, len(seed_discretized_plan) // 100)
        seed_discretized_plan_trimmed = seed_discretized_plan[::skip]

        # waypoints = [start_joint_values] + seed_discretized_plan_trimmed[::skip].tolist() + [goal_joint_values]
        seed_start_adapted = self.adapt_conf(seed_discretized_plan_trimmed[0], start_joint_values)
        seed_discretized_plan_trimmed = seed_discretized_plan_trimmed + (seed_start_adapted - seed_discretized_plan_trimmed[0])

        diff_first_3_joints = np.abs(self.arm_difference_fn(goal_joint_values, seed_discretized_plan_trimmed[-1]))[:3]
        if any(diff_first_3_joints[:2] > 1):
            print('Discarding seed trajectory, end of seed trajectory is too far from new goal joint values')
            return None

        end_idx = max(int(len(seed_discretized_plan_trimmed) * 0.75), 2)
        seed_discretized_plan_trimmed = seed_discretized_plan_trimmed[:end_idx]   # remove last portion of trajectory
        diffs = self.arm_difference_fn(goal_joint_values, seed_discretized_plan_trimmed[-1])
        final_waypoints = seed_discretized_plan_trimmed[-1] + np.linspace(0, 1, 30)[:, None] * diffs
        waypoints = [start_joint_values] + seed_discretized_plan_trimmed.tolist() + final_waypoints.tolist()

        planner_description = self.arm_commander_group.get_interface_description()
        if 'CHOMP' in planner_description.planner_ids:
            reference_trajectories = self.encode_seed_trajectory_chomp(waypoints)
        elif 'STOMP' in planner_description.planner_ids:
            reference_trajectories = self.encode_seed_trajectory_stomp(waypoints)
        else:
            reference_trajectories = None
        return reference_trajectories

    def encode_seed_trajectory_chomp(self, joint_wps):
        joint_trajectory_seed = JointTrajectory()
        # joint_trajectory_seed.header.frame_id = ''
        joint_trajectory_seed.joint_names = self.GROUPS['arm']
        joint_trajectory_seed.points = []
        for wp in joint_wps:
            point = JointTrajectoryPoint()
            point.positions = wp
            joint_trajectory_seed.points.append(point)

        generic_trajectory = GenericTrajectory()
        generic_trajectory.joint_trajectory.append(joint_trajectory_seed)
        reference_trajectories = [generic_trajectory]

        return reference_trajectories

    def encode_seed_trajectory_stomp(self, joint_wps):
        if joint_wps is None or len(joint_wps) == 0:
            return None
        tcs = TrajectoryConstraints()

        for joint_wp in joint_wps:
            cs = Constraints()
            for j_idx in range(len(joint_wp)):
                jc = JointConstraint()
                # jc.joint_name = plan.joint_trajectory.joint_names[j_idx]
                jc.joint_name = self.GROUPS['arm'][j_idx]
                jc.position = joint_wp[j_idx]
                cs.joint_constraints.append(jc)
            tcs.constraints.append(cs)
        return tcs

    def display_trajectory(self, plan):
        # can display plan directly or seed trajectory created with GenericTrajectory
        if isinstance(plan, list):
            if isinstance(plan[0], GenericTrajectory):
                moveit_plan = RobotTrajectory()
                moveit_plan.joint_trajectory = plan[0].joint_trajectory[0]
                plan = moveit_plan
        if isinstance(plan, TrajectoryConstraints):
            moveit_plan = RobotTrajectory()
            moveit_plan.joint_trajectory.joint_names = [jc.joint_name for jc in plan.constraints[0].joint_constraints]
            for cs in plan.constraints:
                point = JointTrajectoryPoint()
                for jc in cs.joint_constraints:
                    point.positions.append(jc.position)
                    moveit_plan.joint_trajectory.points.append(point)
            plan = moveit_plan

        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)

        self.display_trajectory_publisher.publish(display_trajectory)

    def plan_arm_joint_values(self, goal_joint_values, start_joint_values=None, maximum_planning_time=0.5, previous_discretized_plan=None, start_joint_velocities=None):
        """
        Plan a trajectory from current joint values to goal joint values
        :param goal_joint_values: a list of goal joint values
        :param start_joint_values: a list of start joint values; if None, use current values
        :return plan: Motion
        """
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()

        start_joint_values_converted = self.convert_range(start_joint_values)
        goal_joint_values_converted = self.convert_range(goal_joint_values)
        seed_trajectory = None
        if previous_discretized_plan is not None and len(previous_discretized_plan) > 2:
            seed_discretized_plan = previous_discretized_plan  # TODO: is there a need to normalize range of joint values? i.e. undo process_plan
            seed_trajectory = self.create_seed_trajectory(seed_discretized_plan, start_joint_values_converted, goal_joint_values_converted)
            # if seed_trajectory is not None:
            #     self.display_trajectory(seed_trajectory)
            #     import ipdb; ipdb.set_trace()

        moveit_plan = self.plan_arm_joint_values_ros(start_joint_values_converted, goal_joint_values_converted,
                                                     maximum_planning_time=maximum_planning_time,
                                                     start_joint_velocities=start_joint_velocities,
                                                     seed_trajectory=seed_trajectory)  # STOMP does not convert goal joint values
        if isinstance(moveit_plan, tuple):
            # if using the chomp branch
            moveit_plan = moveit_plan[1]
        # check if there exists a plan
        if len(moveit_plan.joint_trajectory.points) == 0:
            print('plan_arm_joint_values fails')
            return None

        motion_plan = MicoController.process_plan(moveit_plan, start_joint_values)
        discretized_plan = MicoController.discretize_plan(motion_plan)
        if len(discretized_plan) == 0:
            print('start joint values is the same as goal joint values, so plan_arm_joint_values fails')
            return None
        return discretized_plan

    def plan_arm_joint_values_ros(self, start_joint_values, goal_joint_values, maximum_planning_time=0.5, seed_trajectory=None, start_joint_velocities=None):
        """ No matter what start and goal are, the returned plan start and goal will
            make circular joints within [-pi, pi] """
        # setup moveit_start_state
        start_robot_state = self.robot.get_current_state()
        start_robot_state.joint_state.name = self.GROUPS['arm']
        start_robot_state.joint_state.position = start_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)
        self.arm_commander_group.set_joint_value_target(goal_joint_values)
        self.arm_commander_group.set_planning_time(maximum_planning_time)

        # takes around 0.11 second
        if seed_trajectory is not None:
            planner_description = self.arm_commander_group.get_interface_description()
            if 'CHOMP' in planner_description.planner_ids:
                # self.arm_commander_group.set_reference_trajectories(seed_trajectory)
                plan = self.arm_commander_group.plan(reference_trajectories=seed_trajectory)
                self.arm_commander_group.clear_reference_trajectories()
            elif 'STOMP' in planner_description.planner_ids:
                self.arm_commander_group.set_joint_value_target([jc.position for jc in seed_trajectory.constraints[-1].joint_constraints])
                self.arm_commander_group.set_trajectory_constraints(seed_trajectory)
                plan = self.arm_commander_group.plan()
                self.arm_commander_group.clear_trajectory_constraints()
            else:
                assert False, 'seed trajectory not handled by planner {}'.format(planner_description.planner_ids)
        else:
            plan = self.arm_commander_group.plan()
        if isinstance(plan, tuple):
            # if using the chomp branch
            plan = plan[1]
        if start_joint_velocities is not None and len(plan.joint_trajectory.points) > 0:
            plan.joint_trajectory.points[0].velocities = start_joint_velocities
        plan = self.arm_commander_group.retime_trajectory(start_robot_state, plan)
        return plan

    def plan_straight_line(self, goal_eef_pose, start_joint_values=None, ee_step=0.05, jump_threshold=3.0,
                           avoid_collisions=True):
        """

        :return plan: Motion
        """
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()

        # moveit will do the conversion internally
        moveit_plan, fraction = self.plan_straight_line_ros(start_joint_values, goal_eef_pose, ee_step=ee_step,
                                                            jump_threshold=jump_threshold,
                                                            avoid_collisions=avoid_collisions)

        # print("plan length: {}, fraction: {}".format(len(plan.joint_trajectory.points), fraction))

        # check if there exists a plan
        if len(moveit_plan.joint_trajectory.points) == 0:
            return None, fraction
        plan = MicoController.process_plan(moveit_plan, start_joint_values)
        discretized_plan = MicoController.discretize_plan(plan)
        return discretized_plan, fraction

    def plan_straight_line_ros(self, start_joint_values, end_eef_pose, ee_step=0.05, jump_threshold=3.0,
                               avoid_collisions=True):
        """
        :param start_joint_values: start joint values
        :param end_eef_pose: goal end effector pose
        :param ee_step: float. The distance in meters to interpolate the path.
        :param jump_threshold: The maximum allowable distance in the arm's
            configuration space allowed between two poses in the path. Used to
            prevent "jumps" in the IK solution.
        :param avoid_collisions: bool. Whether to check for obstacles or not.
        :return:
        """
        # set moveit start state
        # TODO plan should take in gripper joint values for start state
        # TODO reduce step

        # from scratch
        # joint_state = JointState()
        # joint_state.name = self.ARM_JOINT_NAMES
        # joint_state.position = start_joint_values
        # moveit_robot_state = RobotState()
        # moveit_robot_state.joint_state = joint_state

        # using current state, including all other joint info
        start_robot_state = self.robot.get_current_state()
        start_robot_state.joint_state.name = self.GROUPS['arm']
        start_robot_state.joint_state.position = start_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)

        start_eef_pose = gu.list_2_pose(self.get_arm_fk(start_joint_values))
        plan, fraction = self.arm_commander_group.compute_cartesian_path(
            [start_eef_pose, end_eef_pose],
            ee_step,
            jump_threshold,
            avoid_collisions)
        # remove the first redundant point
        plan.joint_trajectory.points = plan.joint_trajectory.points[1:]
        # speed up the trajectory
        # for p in plan.joint_trajectory.points:
        #     p.time_from_start = rospy.Duration.from_sec(p.time_from_start.to_sec() / 1.5)
        return plan, fraction

    def violate_limits(self, joint_values):
        return pu.violates_limits(self.id, self.GROUP_INDEX['arm'], joint_values)

    @staticmethod
    def discretize_plan(motion_plan):
        """ return np array """
        discretized_plan = np.zeros((0, 6))
        for i in range(len(motion_plan.position_trajectory) - 1):
            num_steps = (motion_plan.time_trajectory[i + 1] - motion_plan.time_trajectory[i]) * 240
            segment = np.linspace(motion_plan.position_trajectory[i], motion_plan.position_trajectory[i + 1], num_steps)
            if i + 1 == len(motion_plan.position_trajectory) - 1:
                discretized_plan = np.vstack((discretized_plan, segment))
            else:
                discretized_plan = np.vstack((discretized_plan, segment[:-1]))
        # if len(discretized_plan) == 0 or len(motion_plan.position_trajectory) == 1:
        #     print(len(discretized_plan))
        #     print(len(motion_plan.position_trajectory))
        #     import ipdb
        #     ipdb.set_trace()
        return discretized_plan

    def clear_scene(self):
        for obj_name in self.get_attached_object_names():
            self.scene.remove_attached_object(self.EEF_LINK_INDEX, obj_name)
        for obj_name in self.get_known_object_names():
            self.scene.remove_world_object(obj_name)

    def get_known_object_names(self):
        return self.scene.get_known_object_names()

    def get_attached_object_names(self):
        return self.scene.get_attached_objects().keys()

    def plan_cartesian_control(self, x=0.0, y=0.0, z=0.0, frame="world"):
        """
        Only for small motion, do not check friction
        :param frame: "eef" or "world"
        """
        if frame == "eef":
            pose_2d = self.get_eef_pose()
            world_T_old = tf_conversions.toMatrix(tf_conversions.fromTf(pose_2d))
            old_T_new = tf_conversions.toMatrix(tf_conversions.fromTf(((x, y, z), (0, 0, 0, 1))))
            world_T_new = world_T_old.dot(old_T_new)
            pose_2d_new = tf_conversions.toTf(tf_conversions.fromMatrix(world_T_new))
        elif frame == "world":
            pose_2d_new = self.get_eef_pose()
            pose_2d_new[0][0] += x
            pose_2d_new[0][1] += y
            pose_2d_new[0][2] += z
        else:
            raise TypeError("not supported frame: {}".format(frame))
        discretized_plan, fraction = self.plan_straight_line(gu.list_2_pose(pose_2d_new),
                                                             ee_step=0.01,
                                                             avoid_collisions=False)
        return discretized_plan, fraction

    def plan_arm_joint_values_simple(self, goal_joint_values, start_joint_values=None, duration=None):
        """ Linear interpolation between joint_values """
        start_joint_values = self.get_arm_joint_values() if start_joint_values is None else start_joint_values

        diffs = self.arm_difference_fn(goal_joint_values, start_joint_values)
        steps = np.abs(np.divide(diffs, self.MOVEIT_ARM_MAX_VELOCITY)) * 240
        num_steps = int(max(steps))
        if duration is not None:
            num_steps = int(duration * 240)
            # num_steps = max(int(duration * 240), steps)     # this should ensure that it satisfies the max velocity of the end-effector
        waypoints = MicoController.refine_path(start_joint_values, diffs, num_steps)
        # print(self.adapt_conf(goal_joint_values, waypoints[-1]))
        return waypoints

    @staticmethod
    def refine_path(start_joint_values, diffs, num_steps):
        goal_joint_values = np.array(start_joint_values) + np.array(diffs)
        waypoints = np.linspace(start_joint_values, goal_joint_values, num_steps)
        return waypoints

    def adapt_conf(self, conf2, conf1):
        """ adapt configuration 2 to configuration 1"""
        diff = self.arm_difference_fn(conf2, conf1)
        adapted_conf2 = np.array(conf1) + np.array(diff)
        return adapted_conf2.tolist()

    def equal_conf(self, conf1, conf2, tol=0):
        adapted_conf2 = self.adapt_conf(conf2, conf1)
        return np.allclose(conf1, adapted_conf2, atol=tol)

    # execution
    def execute_arm_plan(self, plan, realtime=False):
        """
        execute a discretized arm plan (list of waypoints)
        """
        for wp in plan:
            self.control_arm_joints(wp)
            p.stepSimulation()
            if realtime:
                time.sleep(1. / 240.)
        pu.step(2)

    def execute_gripper_plan(self, plan, realtime=False):
        """
        execute a discretized gripper plan (list of waypoints)
        """
        for wp in plan:
            self.control_gripper_joints(wp)
            p.stepSimulation()
            if realtime:
                time.sleep(1. / 240.)
        pu.step(2)


