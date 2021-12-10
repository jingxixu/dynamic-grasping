import numpy as np
import os

import rospy
import moveit_commander as mc
from moveit_msgs.msg import DisplayTrajectory, PositionIKRequest, RobotState
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import GenericTrajectory, RobotTrajectory
from moveit_msgs.msg import TrajectoryConstraints, Constraints, JointConstraint
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header


def normalize_angle_positive(angle):
    two_pi = 2.0 * np.pi
    return ((angle % two_pi) + two_pi) % two_pi


def shortest_angular_distance(start, end):
    diff = normalize_angle_positive(end) - normalize_angle_positive(start)
    if diff > np.pi:
        diff = - (2 * np.pi - diff)
    return normalize_angle(diff)


def normalize_angle(angle):
    angle_normalized = normalize_angle_positive(angle)
    if angle_normalized > np.pi:
        angle_normalized -= 2 * np.pi
    return angle_normalized


class UR5RobotiqMoveIt(object):
    TIP_LINK = "ee_link"
    BASE_LINK = "base_link"
    ARM = "manipulator"
    GRIPPER = "gripper"
    ARM_JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    GRIPPER_JOINT_NAMES = ['finger_joint']
    JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

    OPEN_POSITION = [0]
    CLOSED_POSITION = [0.72]
    FINGER_MAXTURN = 1.3
    MOVEIT_ARM_MAX_VELOCITY = [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]

    def __init__(self, use_manipulability=False, planner_id='PRM'): # PRM, RRTConnect
        # the service names have to be this
        self.arm_ik_svr = rospy.ServiceProxy('compute_ik', GetPositionIK)
        self.arm_fk_svr = rospy.ServiceProxy('compute_fk', GetPositionFK)

        self.arm_commander_group = mc.MoveGroupCommander(self.ARM)
        self.arm_commander_group.set_planner_id(planner_id)

        self.robot = mc.RobotCommander()
        self.scene = mc.PlanningSceneInterface()
        rospy.sleep(2)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            DisplayTrajectory,
                                                            queue_size=20)
        self.eef_link = self.arm_commander_group.get_end_effector_link()

        rospy.wait_for_service('check_state_validity', timeout=10)
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)

        self.robot_state_template = self.robot.get_current_state()
        self.use_manipulability = use_manipulability
        if use_manipulability:
            from manipulability_computation.srv import GetManipulabilityIndex
            rospy.wait_for_service('compute_manipulability_index', timeout=10)
            self.compute_manipulability_svr = rospy.ServiceProxy('compute_manipulability_index', GetManipulabilityIndex)
            self.compute_manipulability_svr.wait_for_service()

    def plan(self, start_joint_values, goal_joint_values, maximum_planning_time=0.5,
             seed_trajectory=None, start_joint_velocities=None):
        """ No matter what start and goal are, the returned plan start and goal will
            make circular joints within [-pi, pi] """
        # setup moveit_start_state
        start_robot_state = self.robot.get_current_state()
        start_robot_state.joint_state.name = self.ARM_JOINT_NAMES
        start_robot_state.joint_state.position = start_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)
        self.arm_commander_group.set_joint_value_target(goal_joint_values)
        self.arm_commander_group.set_planning_time(maximum_planning_time)
        # takes around 0.11 second
        if seed_trajectory is not None:
            plan = self.arm_commander_group.plan(reference_trajectories=seed_trajectory)
        else:
            plan = self.arm_commander_group.plan()
        if isinstance(plan, tuple):
            # if using the chomp branch
            plan = plan[1]
        if start_joint_velocities is not None and len(plan.joint_trajectory.points) > 0:
            plan.joint_trajectory.points[0].velocities = start_joint_velocities
            velocity_scaling_factor = acceleration_scaling_factor = 0.1   # moveit uses 0.1 by default
            plan = self.arm_commander_group.retime_trajectory(start_robot_state, plan,
                                                              velocity_scaling_factor=velocity_scaling_factor,
                                                              acceleration_scaling_factor=acceleration_scaling_factor)
        return plan

    def plan_ee_pose(self, start_joint_values, goal_ee_pose, maximum_planning_time=0.5, gripper_joint_values=[]):
        """ using set_pose_target instead """
        # setup moveit_start_state
        start_robot_state = self.robot.get_current_state()
        start_robot_state.joint_state.name = self.ARM_JOINT_NAMES + self.GRIPPER_JOINT_NAMES[:len(gripper_joint_values)]
        start_robot_state.joint_state.position = start_joint_values + gripper_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)
        self.arm_commander_group.set_pose_target(goal_ee_pose)
        self.arm_commander_group.set_planning_time(maximum_planning_time)

        plan = self.arm_commander_group.plan()
        return plan

    def plan_straight_line(self, start_joint_values, end_eef_pose, ee_step=0.05, jump_threshold=3.0,
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
        start_robot_state.joint_state.name = self.ARM_JOINT_NAMES
        start_robot_state.joint_state.position = start_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)
        # self.arm_commander_group.set_planner_id('RRTConnect')

        start_eef_pose = self.get_arm_fk(start_joint_values)
        plan, fraction = self.arm_commander_group.compute_cartesian_path(
            [start_eef_pose, end_eef_pose],
            ee_step,
            jump_threshold,
            avoid_collisions)
        # remove the first redundant point
        plan.joint_trajectory.points = plan.joint_trajectory.points[1:]
        # speed up the trajectory
        for p in plan.joint_trajectory.points:
            p.time_from_start = rospy.Duration.from_sec(p.time_from_start.to_sec() / 1.5)
        return plan, fraction

    def get_current_max_eef_velocity(self, arm_joint_values):
        jacobian = self.arm_commander_group.get_jacobian_matrix(arm_joint_values)
        max_eef_velocity = np.dot(jacobian, self.MOVEIT_ARM_MAX_VELOCITY)
        return np.squeeze(np.array(max_eef_velocity))

    def get_manipulability_srv(self, list_of_joint_values):
        assert self.use_manipulability, 'self.use_manipulability flag is set to false, ' \
                                        'check constructor and start manipulability ros service'
        manipulability_indexes = []
        for jvs in list_of_joint_values:
            if jvs is None:
                manipulability_indexes.append(None)
            else:
                self.robot_state_template.joint_state.name = self.robot_state_template.joint_state.name[:len(jvs)]
                self.robot_state_template.joint_state.position = jvs
                result = self.compute_manipulability_svr(self.robot_state_template, self.ARM)

                manipulability_indexes.append(result.manipulability_index)
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

    def get_arm_eff_pose(self):
        """ Return [[x, y, x], [x, y, z, w]]"""
        pose_stamped = self.arm_commander_group.get_current_pose()
        pose = pose_stamped.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return [position, orientation]

    def get_arm_ik(self, pose_2d, timeout, avoid_collisions, arm_joint_values, gripper_joint_values):
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
        service_request.group_name = self.ARM
        service_request.ik_link_name = self.TIP_LINK
        service_request.pose_stamped = gripper_pose_stamped
        service_request.timeout.nsecs = timeout * 1e9
        service_request.avoid_collisions = avoid_collisions

        seed_robot_state = self.robot.get_current_state()
        seed_robot_state.joint_state.name = self.JOINT_NAMES
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

    def get_arm_fk(self, arm_joint_values):
        """ return a ros pose """
        rospy.wait_for_service('compute_fk')

        header = Header(frame_id="world")
        fk_link_names = [self.TIP_LINK]
        robot_state = RobotState()
        robot_state.joint_state.name = self.ARM_JOINT_NAMES
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
        return [d[n] for n in self.ARM_JOINT_NAMES]

    ''' scene and collision '''

    def check_arm_collision(self, arm_joint_values, constraints=None):

        robot_state = RobotState()
        robot_state.joint_state.name = self.ARM_JOINT_NAMES
        robot_state.joint_state.position = arm_joint_values

        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = self.ARM
        if constraints is not None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)
        return result

    def clear_scene(self):
        for obj_name in self.get_attached_object_names():
            self.scene.remove_attached_object(self.eef_link, obj_name)
        for obj_name in self.get_known_object_names():
            self.scene.remove_world_object(obj_name)

    def add_box(self, name, pose_2d, size=(1, 1, 1)):
        pose_stamped = PoseStamped()
        pose = Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))
        pose_stamped.pose = pose
        pose_stamped.header.frame_id = "/world"
        self.scene.add_box(name, pose_stamped, size)

    def add_mesh(self, name, pose_2d, mesh_filepath, size=(1, 1, 1)):
        pose_stamped = PoseStamped()
        pose = Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))
        pose_stamped.pose = pose
        pose_stamped.header.frame_id = "/world"
        self.scene.add_mesh(name, pose_stamped, mesh_filepath, size)

    def get_known_object_names(self):
        return self.scene.get_known_object_names()

    def get_attached_object_names(self):
        return self.scene.get_attached_objects().keys()

    # def create_seed_trajectory(self, seed_discretized_plan_trimmed, start_joint_values, goal_joint_values):
    def create_seed_trajectory(self, waypoints):

        planner_description = self.arm_commander_group.get_interface_description()
        if 'CHOMP' in planner_description.planner_ids:
            reference_trajectories = self.encode_seed_trajectory_chomp(waypoints)
        if planner_description.name == 'OMPL':
            # seed trajectory is densely sampled for chomp, sampling based methods might not need as much
            # waypoints = np.array(waypoints)[::10]
            reference_trajectories = self.encode_seed_trajectory_chomp(waypoints)
        elif 'STOMP' in planner_description.planner_ids:
            reference_trajectories = self.encode_seed_trajectory_stomp(waypoints)
        else:
            reference_trajectories = None
        return reference_trajectories

    def encode_seed_trajectory_chomp(self, joint_wps):
        joint_trajectory_seed = JointTrajectory()
        # joint_trajectory_seed.header.frame_id = ''
        joint_trajectory_seed.joint_names = self.ARM_JOINT_NAMES
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


if __name__ == "__main__":
    rospy.init_node('mico_moveit')

    mm = UR5RobotiqMoveIt()
    print(mm.get_arm_eff_pose())
