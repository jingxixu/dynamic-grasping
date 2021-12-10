from collections import namedtuple
import pybullet as p

import rospy
import rospkg
import tf_conversions as tfc

from ur5_robotiq_moveit import UR5RobotiqMoveIt, normalize_angle

import numpy as np
import os
import time

import pybullet_utils as pu

def load_ur_robotiq_robot(robot_initial_pose):
    # load robot
    urdf_dir = os.path.join(rospkg.RosPack().get_path('ur5_robotiq_description'), 'urdf')
    urdf_filepath = os.path.join(urdf_dir, 'ur5_robotiq.urdf')
    xacro_filepath = os.path.join(urdf_dir, 'ur5_robotiq_robot.xacro')
    if not os.path.exists(urdf_filepath):
        cmd = 'rosrun xacro xacro --inorder {} -o {}'.format(xacro_filepath, urdf_filepath)
        os.system(cmd)
        robotiq_description_dir = rospkg.RosPack().get_path('robotiq_2f_85_gripper_visualization')
        sed_cmd = "sed -i 's|{}|{}|g' {}".format('package://robotiq_2f_85_gripper_visualization',
                                                 robotiq_description_dir, urdf_filepath)
        os.system(sed_cmd)
        ur5_description_dir = rospkg.RosPack().get_path('ur_description')
        sed_cmd = "sed -i 's|{}|{}|g' {}".format('package://ur_description', ur5_description_dir, urdf_filepath)
        os.system(sed_cmd)
        # adjust the gripper effort for stable grasping in pybullet
        sed_cmd = "sed -i 's|{}|{}|g' {}".format('limit effort="1000"', 'limit effort="200"', urdf_filepath)
        os.system(sed_cmd)

    robot_id = p.loadURDF(urdf_filepath, basePosition=robot_initial_pose[0], baseOrientation=robot_initial_pose[1],
                          flags=p.URDF_USE_SELF_COLLISION)
    return robot_id

Motion = namedtuple('Motion', ['position_trajectory', 'time_trajectory', 'velocity_trajectory'])

class UR5RobotiqPybulletController(object):
    JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                         'qIndex', 'uIndex', 'flags',
                                         'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                         'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                         'parentFramePos', 'parentFrameOrn', 'parentIndex'])

    JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                           'jointReactionForces', 'appliedJointMotorTorque'])

    LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                         'localInertialFramePosition', 'localInertialFrameOrientation',
                                         'worldLinkFramePosition', 'worldLinkFrameOrientation'])

    # movable joints for each moveit group
    GROUPS = {
        'arm': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
        'gripper': ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint',
                    'right_inner_knuckle_joint', 'right_inner_finger_joint']
    }
    HOME = [0, -0.8227210029571718, -0.130, -0.660, 0, 1.62]
    HOME = [0, -1.15, 0.9, -0.660, 0, 0.0]
    OPEN_POSITION = [0] * 6
    CLOSED_POSITION = 0.72 * np.array([1, 1, -1, 1, 1, -1])

    JOINT_INDICES_DICT = {}
    EE_LINK_NAME = 'ee_link'

    TIP_LINK = "ee_link"
    BASE_LINK = "base_link"
    ARM = "manipulator"
    GRIPPER = "gripper"
    ARM_JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    GRIPPER_JOINT_NAMES = ['finger_joint']

    # this is read from moveit_configs joint_limits.yaml
    MOVEIT_ARM_MAX_VELOCITY = [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]

    def __init__(self, robot_id):
        self.id = robot_id
        self.initial_joint_values = self.HOME
        self.num_joints = p.getNumJoints(self.id)

        joint_infos = [p.getJointInfo(robot_id, joint_index) for joint_index in range(p.getNumJoints(robot_id))]
        self.JOINT_INDICES_DICT = {entry[1]: entry[0] for entry in joint_infos}
        self.GROUP_INDEX = {key: [self.JOINT_INDICES_DICT[joint_name] for joint_name in self.GROUPS[key]] for key in
                            self.GROUPS}
        self.EEF_LINK_INDEX = pu.link_from_name(robot_id, self.EE_LINK_NAME)

        self.moveit = UR5RobotiqMoveIt()
        rospy.sleep(2)

        self.arm_difference_fn = pu.get_difference_fn(self.id, self.GROUP_INDEX['arm'])

        self.arm_max_joint_velocities = [pu.get_max_velocity(self.id, j_id) for j_id in self.GROUP_INDEX['arm']]
        self.attach_cid = None
        self.attach_object_id = None
        self.reset()

    def reset(self):
        self.set_arm_joints(self.initial_joint_values)
        self.set_gripper_joints(self.OPEN_POSITION)
        self.clear_scene()
        self.arm_discretized_plan = None
        self.gripper_discretized_plan = None
        self.arm_wp_target_index = 0
        self.gripper_wp_target_index = 0
        if self.attach_cid is not None:
            p.removeConstraint(self.attach_cid)
        self.attach_cid = None
        self.attach_object_id = None

    def attach_object(self, target_id):
        target_pose = pu.get_body_pose(target_id)
        eef_pose = self.get_eef_pose()
        eef_P_world = p.invertTransform(eef_pose[0], eef_pose[1])
        eef_P_target = p.multiplyTransforms(
            eef_P_world[0], eef_P_world[1], target_pose[0], target_pose[1])
        self.attach_cid = p.createConstraint(
            parentBodyUniqueId=target_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.id,
            childLinkIndex=self.EEF_LINK_INDEX,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=eef_P_target[0],
            childFrameOrientation=eef_P_target[1])
        self.attach_object_id = target_id

    def detach(self):
        p.removeConstraint(self.attach_cid)
        self.attach_cid = None
        self.attach_object_id = None

    def update_arm_motion_plan(self, arm_discretized_plan):
        self.arm_discretized_plan = arm_discretized_plan
        self.arm_wp_target_index = 1

    def update_gripper_motion_plan(self, gripper_discretized_plan):
        self.gripper_discretized_plan = gripper_discretized_plan
        self.gripper_wp_target_index = 1

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['arm'], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def control_arm_joints(self, joint_values):
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def set_gripper_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['gripper'], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values)

    def control_gripper_joints(self, joint_values):
        pu.control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values)

    def close_gripper(self, realtime=False):
        waypoints = self.plan_gripper_joint_values(self.CLOSED_POSITION)
        self.execute_gripper_plan(waypoints, realtime)

    def plan_gripper_joint_values(self, goal_joint_values, start_joint_values=None, duration=None):
        if start_joint_values is None:
            start_joint_values = self.get_gripper_joint_values()
        num_steps = 240 if duration is None else int(duration*240)
        discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        return discretized_plan

    def get_current_max_eef_velocity(self, arm_joint_values):
        arm_joint_values = self.get_arm_joint_values() if arm_joint_values is None else arm_joint_values
        return self.moveit.get_current_max_eef_velocity(arm_joint_values)

    def get_manipulability(self, list_of_joint_values):
        return self.moveit.get_manipulability(list_of_joint_values)

    def get_jacobian_pybullet(self, arm_joint_values):

        gripper_joint_values = self.get_gripper_joint_values()
        current_positions = arm_joint_values + gripper_joint_values

        zero_vec = [0.0] * len(current_positions)
        jac_t, jac_r = p.calculateJacobian(self.id, self.EEF_LINK_INDEX, (0, 0, 0),
                                           current_positions, zero_vec, zero_vec)
        jacobian = np.concatenate((np.array(jac_t)[:, :6], np.array(jac_r)[:, :6]), axis=0)
        return jacobian

    def clear_scene(self):
        self.moveit.clear_scene()

    def get_arm_fk(self, arm_joint_values):
        pose = self.moveit.get_arm_fk(arm_joint_values)
        return tfc.toTf(tfc.fromMsg(pose)) if pose is not None else None

    def get_arm_fk_pybullet(self, joint_values):
        return pu.forward_kinematics(self.id, self.GROUP_INDEX['arm'], joint_values, self.EEF_LINK_INDEX)

    def get_arm_ik(self, pose_2d, timeout=0.1, restarts=1, avoid_collisions=True, arm_joint_values=None,
                   gripper_joint_values=None):

        start_joint_values = self.get_arm_joint_values() if arm_joint_values is None else arm_joint_values
        return self.get_ik_fast(pose_2d, avoid_collisions=avoid_collisions, arm_joint_values=start_joint_values)
        # gripper_joint_values = self.get_gripper_joint_values() if gripper_joint_values is None else gripper_joint_values
        if gripper_joint_values is None:
            gripper_joint_values = [self.get_joint_state(self.JOINT_INDICES_DICT[joint_name]).jointPosition for
                                    joint_name in self.GRIPPER_JOINT_NAMES]

        for _ in range(restarts):
            js = self.moveit.get_arm_ik(pose_2d, timeout, avoid_collisions, start_joint_values, gripper_joint_values)
            if js is not None:
                break
        return js

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

    def get_ik_fast(self, eef_pose, avoid_collisions=True, arm_joint_values=None, ignore_last_joint=True,
                    use_grasp_roll_duality=True):
        ik_results = self.get_ik_fast_full(eef_pose)

        if avoid_collisions:
            # avoid all collision
            collision_free = [self.moveit.check_arm_collision(ik).valid for ik in ik_results]
            ik_results = np.array(ik_results)[np.where(collision_free)]
        else:
            # allow collision with target object except floor
            collision_results_full = [self.moveit.check_arm_collision(ik) for ik in ik_results]
            collision_free = []
            for col_result in collision_results_full:
                is_free = True
                for contact in col_result.contacts:
                    if contact.body_type_1 == 0 and contact.contact_body_2 == 0: #self_collision
                        is_free = False
                        break
                    elif (contact.body_type_1 == 0 and contact.body_type_2 == 1 and contact.contact_body_2 == 'floor')\
                            or (contact.body_type_2 == 0 and contact.body_type_1 == 1 and contact.contact_body_1 == 'floor'):
                        is_free = False  # allows collision with other bodies asides floor
                        break
                collision_free.append(is_free)
            ik_results = np.array(ik_results)[np.where(collision_free)]

        if not ik_results.any():
            return None

        if arm_joint_values is not None:
            if ignore_last_joint:
                jv_dists = np.linalg.norm(ik_results[:, :-1] - np.array(arm_joint_values)[:-1], axis=1)
                # jv_dists = np.max(np.abs(ik_results[:, :-1] - np.array(arm_joint_values)[:-1]), axis=1)
            else:
                jv_dists = np.linalg.norm(ik_results - np.array(arm_joint_values), axis=1)
                # jv_dists = np.max(np.abs(ik_results - np.array(arm_joint_values)), axis=1)
            ik_result = ik_results[np.argsort(jv_dists)[0]]

            if use_grasp_roll_duality:
                # parrallel jaw grasp roll duality
                reference = normalize_angle(arm_joint_values[-1])
                original = normalize_angle(ik_result[-1])
                dual = normalize_angle(ik_result[-1] + np.pi)
                if np.abs(dual - reference) < np.abs(original - reference):
                    ik_result[-1] = dual
        else:
            ik_result = ik_results[0]
        return ik_result

    def get_ik_fast_full(self, eef_pose):

        # base_2_shoulder = gu.get_transform('base_link', 'shoulder_link')
        # base_2_shoulder = ([0.0, 0.0, 0.089159], [0.0, 0.0, 1.0, 0.0])
        # the z-offset (0.089159) is from kinematics_file config in ur_description
        base_correction = ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])

        # ee_2_wrist3 = gu.get_transform('ee_link', 'wrist_3_link')
        ee_2_wrist3 = ([0.0, 0.0, 0.0], [-0.5, 0.5, -0.5, 0.5])
        wrist_3_pose_in_shoulder = tfc.toTf(
            tfc.fromTf(base_correction).Inverse() * tfc.fromTf(eef_pose) * tfc.fromTf(ee_2_wrist3))

        if not hasattr(self, 'ur5_kin'):
            from ikfastpy import ikfastpy
            self.ur5_kin = ikfastpy.PyKinematics()
        wrist_3_pose_in_shoulder = tfc.toMatrix(tfc.fromTf(wrist_3_pose_in_shoulder))[:3]
        joint_configs = self.ur5_kin.inverse(wrist_3_pose_in_shoulder.reshape(-1).tolist())

        n_joints = self.ur5_kin.getDOF()
        n_solutions = int(len(joint_configs) / n_joints)
        joint_configs = np.asarray(joint_configs).reshape(n_solutions, n_joints)
        # print("%d solutions found:" % (n_solutions))
        # for joint_config in joint_configs:
        #     print(joint_config)
        return joint_configs

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

        return self.moveit.create_seed_trajectory(waypoints)

    def plan_arm_joint_values(self, goal_joint_values, start_joint_values=None, maximum_planning_time=0.5,
                              previous_discretized_plan=None, start_joint_velocities=None):
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()

        start_joint_values_converted = UR5RobotiqPybulletController.convert_range(start_joint_values)
        goal_joint_values_converted = UR5RobotiqPybulletController.convert_range(goal_joint_values)
        seed_trajectory = None
        if previous_discretized_plan is not None and len(previous_discretized_plan) > 2:
            seed_discretized_plan = previous_discretized_plan  # TODO: is there a need to normalize range of joint values? i.e. undo process_plan
            seed_trajectory = self.create_seed_trajectory(seed_discretized_plan, start_joint_values_converted,
                                                          goal_joint_values_converted)
            # if seed_trajectory:
            #     # for visualizing seed trajectory
            #     self.moveit.display_trajectory(seed_trajectory)
            #     import ipdb; ipdb.set_trace()

        # STOMP does not convert goal joint values
        moveit_plan = self.moveit.plan(start_joint_values_converted, goal_joint_values_converted,
                                       maximum_planning_time=maximum_planning_time,
                                       start_joint_velocities=start_joint_velocities,
                                       seed_trajectory=seed_trajectory)
        if isinstance(moveit_plan, tuple):
            # if using the chomp branch
            moveit_plan = moveit_plan[1]
        # check if there exists a plan
        if len(moveit_plan.joint_trajectory.points) == 0:
            return None

        motion_plan = UR5RobotiqPybulletController.process_plan(moveit_plan, start_joint_values)
        discretized_plan = UR5RobotiqPybulletController.discretize_plan(motion_plan)
        return discretized_plan

    def plan_arm_joint_values_simple(self, goal_joint_values, start_joint_values=None, duration=None):
        """ Linear interpolation between joint_values """
        start_joint_values = self.get_arm_joint_values() if start_joint_values is None else start_joint_values

        diffs = self.arm_difference_fn(goal_joint_values, start_joint_values)
        steps = np.abs(np.divide(diffs, self.MOVEIT_ARM_MAX_VELOCITY)) * 240
        num_steps = int(max(steps))
        if duration is not None:
            num_steps = int(duration * 240)
            # num_steps = max(int(duration * 240), steps)     # this should ensure that it satisfies the max velocity of the end-effector

        goal_joint_values = np.array(start_joint_values) + np.array(diffs)
        waypoints = np.linspace(start_joint_values, goal_joint_values, num_steps)
        return waypoints

    def plan_straight_line(self, eef_pose, start_joint_values=None, ee_step=0.05,
                           jump_threshold=3.0, avoid_collisions=True):
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()
        # start_joint_values_converted = self.convert_range(start_joint_values)
        start_joint_values_converted = start_joint_values

        # TODO: avoid_collisions should be allow touch object
        moveit_plan, fraction = self.moveit.plan_straight_line(start_joint_values_converted, eef_pose, ee_step=ee_step,
                                                               jump_threshold=jump_threshold,
                                                               avoid_collisions=avoid_collisions)

        # check if there exists a plan
        if len(moveit_plan.joint_trajectory.points) == 0:
            return None, fraction

        plan = self.process_plan(moveit_plan, start_joint_values)
        discretized_plan = UR5RobotiqPybulletController.discretize_plan(plan)
        return discretized_plan, fraction

    def plan_cartesian_control(self, x=0.0, y=0.0, z=0.0, frame="world"):
        """
        Only for small motion, do not check friction
        :param frame: "eef" or "world"
        """
        if frame == "eef":
            pose_2d = self.get_eef_pose()
            pose_2d_new = tfc.toTf(
                tfc.fromTf(pose_2d) * tfc.fromTf(((x, y, z), (0, 0, 0, 1))))
        elif frame == "world":
            pose_2d_new = self.get_eef_pose()
            pose_2d_new[0][0] += x
            pose_2d_new[0][1] += y
            pose_2d_new[0][2] += z
        else:
            raise TypeError("not supported frame: {}".format(frame))
        discretized_plan, fraction = self.plan_straight_line(tfc.toMsg(tfc.fromTf(pose_2d_new)),
                                                             ee_step=0.01,
                                                             avoid_collisions=False)
        return discretized_plan, fraction

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
        plan = UR5RobotiqPybulletController.extract_plan(moveit_plan)
        return plan

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
        return discretized_plan

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

    def equal_conf(self, conf1, conf2, tol=0):
        adapted_conf2 = self.adapt_conf(conf2, conf1)
        return np.allclose(conf1, adapted_conf2, atol=tol)

    def adapt_conf(self, conf2, conf1):
        """ adapt configuration 2 to configuration 1"""
        diff = self.arm_difference_fn(conf2, conf1)
        adapted_conf2 = np.array(conf1) + np.array(diff)
        return adapted_conf2.tolist()



    def reset_joint_values(self, joint_indices, joint_values):
        for i, v in zip(joint_indices, joint_values):
            p.resetJointState(self.id, i, v)

    def reset_arm_joint_values(self, joint_values):
        self.reset_joint_values(self.GROUP_INDEX['arm'], joint_values)

    def reset_gripper_joint_values(self, joint_values):
        self.reset_joint_values(self.GROUP_INDEX['gripper'], joint_values)

    def set_group_joint_values(self, group_joint_indices, joint_values):
        p.setJointMotorControlArray(self.id, group_joint_indices, p.POSITION_CONTROL, joint_values,
                                    forces=[500] * len(joint_values))

    def set_arm_joint_values(self, joint_values):
        self.set_group_joint_values(self.GROUP_INDEX['arm'], joint_values)

    def set_gripper_joint_values(self, joint_values=(0,)*6):
        self.set_group_joint_values(self.GROUP_INDEX['gripper'], joint_values)

    def get_joint_state(self, joint_index):
        return self.JointState(*p.getJointState(self.id, joint_index))

    def get_arm_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['arm']]

    def get_gripper_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['gripper']]

    def get_eef_pose(self):
        return pu.get_link_pose(self.id, self.EEF_LINK_INDEX)

    def execute_arm_motion_plan(self, motion_plan):
        try:
            rospy.loginfo("length of trajectory is: {}".format(len(motion_plan.joint_trajectory.points)))

            jt_points = motion_plan.joint_trajectory.points
            frequency = 50.
            rate = rospy.Rate(frequency)
            start_time = rospy.Time.now()
            next_point_idx = 1  # TODO what if trajectory has only 1 point?
            rospy.loginfo('moving to next trajectory point {}'.format(next_point_idx))
            while True:
                time_since_start = rospy.Time.now() - start_time
                time_diff = (jt_points[next_point_idx].time_from_start - time_since_start).to_sec()

                # handle redundant first points
                if time_diff == 0:
                    rospy.loginfo("ignore trajectory point {}".format(next_point_idx))
                    next_point_idx += 1
                    rospy.loginfo('moving to next trajectory point {}'.format(next_point_idx))
                    continue

                num_steps = max((time_diff * frequency), 1)

                current_jv = self.get_arm_joint_values()
                next_jv = current_jv + (np.array(jt_points[next_point_idx].positions) - current_jv) / num_steps
                self.set_arm_joint_values(next_jv)
                if (rospy.Time.now() - start_time).to_sec() > jt_points[next_point_idx].time_from_start.to_sec():
                    next_point_idx += 1
                    rospy.loginfo('moving to next trajectory point {}'.format(next_point_idx))
                    # import ipdb; ipdb.set_trace()
                if next_point_idx == len(jt_points):
                    break
                rate.sleep()
            rospy.loginfo('Trajectory took {} secs instead of {} secs'.format((rospy.Time.now() - start_time).to_sec(),
                                                                              jt_points[-1].time_from_start.to_sec()))
        except Exception as e:
            print("exception in execute_motion_plan catched")
            print(e)

    def move_gripper_joint_values(self, joint_values, duration=1.0, num_steps=10):
        """ this method has nothing to do with moveit """
        start_joint_values = self.get_gripper_joint_values()
        goal_joint_values = joint_values
        position_trajectory = np.linspace(start_joint_values, goal_joint_values, num_steps)
        for i in range(num_steps):
            p.setJointMotorControlArray(self.id, self.GROUP_INDEX['gripper'], p.POSITION_CONTROL,
                                        position_trajectory[i], forces=[200] * len(joint_values))
            rospy.sleep(duration / num_steps)
