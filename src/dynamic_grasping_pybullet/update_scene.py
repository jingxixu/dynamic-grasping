import moveit_commander as mc
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='update scene for dynamic grasping')
    parser.add_argument('--use_box', action='store_true', default=False, help="Use box to approximate target mesh")
    args = parser.parse_args()
    return args


class SceneUpdater:
    def __init__(self):
        self.target_pose_stamped = None
        self.conveyor_pose_stamped = None
        self.target_mesh_file_path = None
        rospy.Subscriber("target_pose", PoseStamped, self.target_listen_cb, queue_size=1)
        rospy.Subscriber("conveyor_pose", PoseStamped, self.conveyor_listen_cb, queue_size=1)

    def target_listen_cb(self, pose_stamped):
        self.target_pose_stamped = pose_stamped

    def conveyor_listen_cb(self, pose_stamped):
        self.conveyor_pose_stamped = pose_stamped


if __name__ == "__main__":
    args = get_args()
    rospy.init_node('update_scene', anonymous=True)
    scene_updater = SceneUpdater()
    scene = mc.PlanningSceneInterface()

    print('Waiting for the first msg to be published...')
    while scene_updater.target_pose_stamped is None or \
            scene_updater.conveyor_pose_stamped is None:
        rospy.sleep(0.1)
    print('The first msg received!')

    while not rospy.is_shutdown():
        target_mesh_file_path = rospy.get_param('target_mesh_file_path')
        target_extents = rospy.get_param('target_extents')
        if args.use_box:
            scene.add_box('target', scene_updater.target_pose_stamped, size=target_extents)
        else:
            scene.add_mesh('target', scene_updater.target_pose_stamped, target_mesh_file_path)
        scene.add_box('conveyor', scene_updater.conveyor_pose_stamped, size=(.1, .1, .02))
