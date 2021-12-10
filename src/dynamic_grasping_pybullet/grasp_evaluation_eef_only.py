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


""" Use Graspit as backend to generate grasps and test in pybullet,
    saved as pose lists in link6 reference link"""


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--grasp_folder_path', type=str, default='grasp_dir',
                        help="Directory to store grasps and results. Ex: grasps_dir")
    parser.add_argument('--num_grasps', type=int, default=1000)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--disable_gui', action='store_true', default=False)
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')
    args.gripper_urdf = os.path.abspath('assets/mico_hand/mico_hand.urdf')

    args.grasp_folder_path = os.path.join(args.grasp_folder_path, args.object_name)
    args.result_file_path = os.path.join(args.grasp_folder_path, 'result.csv')
    if not os.path.exists(args.grasp_folder_path):
        os.makedirs(args.grasp_folder_path)

    return args


def write_csv_line(result_file_path, index, num_trials, num_successes, volume_quality, epsilon_quality, grasp_fnm):
    result = [('index', index),
              ('num_trials', num_trials),
              ('num_successes', num_successes),
              ('volume_quality', volume_quality),
              ('epsilon_quality', epsilon_quality),
              ('grasp_fnm', grasp_fnm)]
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


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


class Controller:
    EEF_LINK_INDEX = 0
    GRIPPER_INDICES = [1, 3]
    OPEN_POSITION = [0.0, 0.0]
    CLOSED_POSITION = [1.1, 1.1]
    LINK6_COM = [-0.002216, -0.000001, -0.058489]
    LIFT_VALUE = 0.2

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.cid = None

    def reset_to(self, pose):
        """ the pose is for the link6 center of mass """
        p.resetBasePositionAndOrientation(self.robot_id, pose[0], pose[1])
        self.move_to(pose)

    def move_to(self, pose):
        """ the pose is for the link6 center of mass """
        num_steps = 240
        current_pose = self.get_pose()
        positions = np.linspace(current_pose[0], pose[0], num_steps)
        angles = np.linspace(p.getEulerFromQuaternion(current_pose[1]), p.getEulerFromQuaternion(pose[1]), num_steps)
        quaternions = np.array([p.getQuaternionFromEuler(angle) for angle in angles])
        if self.cid is None:
            self.cid = p.createConstraint(parentBodyUniqueId=self.robot_id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                     childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=current_pose[0], childFrameOrientation=current_pose[1])
        for pos, ori in zip(positions, quaternions):
            p.changeConstraint(self.cid, jointChildPivot=pos, jointChildFrameOrientation=ori)
            p.stepSimulation()
        pu.step()

    def close_gripper(self):
        num_steps = 240
        waypoints = np.linspace(self.OPEN_POSITION, self.CLOSED_POSITION, num_steps)
        for wp in waypoints:
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                        jointIndices=self.GRIPPER_INDICES,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=wp,
                                        forces=[10, 10]
                                        )
            p.stepSimulation()
        pu.step()

    def execute_grasp(self, graspit_pose_msg):
        """ High level grasp interface using graspit pose in world frame (link6_reference_frame)"""
        link6_reference_to_link6_com = (self.LINK6_COM, [0.0, 0.0, 0.0, 1.0])
        link6_com_pose_msg = gu.change_end_effector_link(graspit_pose_msg, link6_reference_to_link6_com)
        self.reset_to(gu.pose_2_list(link6_com_pose_msg))
        self.close_gripper()
        self.lift()

    def open_gripper(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.GRIPPER_INDICES,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.OPEN_POSITION)
        pu.step()

    def lift(self, z=LIFT_VALUE):
        target_pose = self.get_pose()
        target_pose[0][2] += z
        self.move_to(target_pose)

    def get_pose(self):
        "the pose is for the link6 center of mass"
        return [list(p.getBasePositionAndOrientation(self.robot_id)[0]), list(p.getBasePositionAndOrientation(self.robot_id)[1])]


class World:

    def __init__(self, target_initial_pose, gripper_initial_pose, gripper_urdf, target_urdf):
        self.target_initial_pose = target_initial_pose
        self.gripper_initial_pose = gripper_initial_pose
        self.gripper_urdf = gripper_urdf
        self.target_urdf = target_urdf

        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        self.robot = p.loadURDF(self.gripper_urdf, self.gripper_initial_pose[0], self.gripper_initial_pose[1], flags=p.URDF_USE_SELF_COLLISION)

        self.controller = Controller(self.robot)

    def reset(self):
        p.resetBasePositionAndOrientation(self.target, self.target_initial_pose[0], self.target_initial_pose[1])
        p.resetBasePositionAndOrientation(self.robot, self.gripper_initial_pose[0], self.gripper_initial_pose[1])
        self.controller.move_to(self.gripper_initial_pose)
        self.controller.open_gripper()


if __name__ == "__main__":
    args = get_args()
    if args.disable_gui:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    # p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=-24.4, cameraPitch=-47.0,
    #                              cameraTargetPosition=(-0.2, -0.30, 0.0))

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]
    gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]

    world = World(target_initial_pose, gripper_initial_pose, args.gripper_urdf, target_urdf)
    link6_reference_to_ee = ([0.0, 0.0, -0.16], [1.0, 0.0, 0.0, 0])
    ee_to_link6_reference = ([0.0, -3.3091697137634315e-14, -0.16], [-1.0, 0.0, 0.0, -1.0341155355510722e-13])

    num_grasps = 0 if not os.path.exists(args.result_file_path) else len(os.listdir(args.grasp_folder_path)) - 1
    progressbar = tqdm.tqdm(initial=num_grasps, total=args.num_grasps)
    while num_grasps < args.num_grasps:
        # start sampling grasps and evaluate
        world.reset()
        object_pose = p.getBasePositionAndOrientation(world.target)
        success_threshold = object_pose[0][2] + world.controller.LIFT_VALUE - 0.05
        object_pose_msg = gu.list_2_pose(object_pose)
        graspit_grasps, graspit_grasp_poses_in_world, graspit_grasp_poses_in_object \
            = gu.generate_grasps(object_mesh=object_mesh_filepath_ply,
                                 object_pose=object_pose_msg,
                                 uniform_grasp=False,
                                 floor_offset=floor_offset,
                                 max_steps=40000)
        volume_qualities = [g.volume_quality for g in graspit_grasps]
        epsilon_qualities = [g.epsilon_quality for g in graspit_grasps]
        for g_pose_msg, g_pose_object_msg, vq, eq in zip(graspit_grasp_poses_in_world, graspit_grasp_poses_in_object, volume_qualities, epsilon_qualities):
            successes = []
            for t in range(args.num_trials):  # test a single grasp
                world.controller.execute_grasp(g_pose_msg)
                success = p.getBasePositionAndOrientation(world.target)[0][2] > 0.2
                successes.append(success)
                # print(success)    # the place to put a break point
                world.reset()
            # write results
            grasp_file_name = "grasp"+"_"+"{:04d}".format(num_grasps)+".npy"
            num_successes = np.count_nonzero(successes)
            write_csv_line(result_file_path=args.result_file_path,
                           index=num_grasps,
                           num_trials=args.num_trials,
                           num_successes=num_successes,
                           volume_quality=vq,
                           epsilon_quality=eq,
                           grasp_fnm=grasp_file_name)
            grasp_list = gu.pose_2_list(g_pose_object_msg)
            grasp_array = np.array(grasp_list[0] + grasp_list[1])
            np.save(os.path.join(args.grasp_folder_path, grasp_file_name), grasp_array)
            progressbar.update(1)
            progressbar.set_description("grasp index: {} | success rate {}/{}".format(num_grasps, num_successes, args.num_trials))
            num_grasps += 1
            if num_grasps == args.num_grasps:
                break
    progressbar.close()
    print("finished")

