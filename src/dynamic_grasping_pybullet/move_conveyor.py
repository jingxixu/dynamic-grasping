""" start moving conveyor and also update the scene (target + conveyor) """
## having two shared memory causes one of them stopped after some time

import time
import rospy
import pybullet as p
import mico_moveit
import utils as ut
import rospy
from geometry_msgs.msg import Pose

p.connect(p.SHARED_MEMORY)
mico_moveit = mico_moveit.MicoMoveit()
ut.remove_all_constraints()

MOVE = True # if not MOVE, just update scene
BACK_AND_FORCE = False # if false, quit after exp finishes
PRINT_SPEED = True # print speed only once

if __name__ == "__main__":

    rospy.init_node("update_scene")
    mico_moveit.clear_scene()
    time.sleep(1) # need some time for clear scene to finish, otherwise floor does not show up
    mico_moveit.add_box("floor", ((0, 0, -0.005), (0, 0, 0, 1)), size=(2, 2, 0.01))
    pub = rospy.Publisher('target_pose', Pose, queue_size=1)

    target = ut.get_body_id("cube_small_modified")
    conveyor = ut.get_body_id("conveyor")

    speed = 0.03 # m/s
    rate = 10
    r = rospy.Rate(rate)  # publishing frequency

    # distance along x to travel
    max_x = 0.8
    min_x = -0.8
    step = speed/float(rate) # meters per step

    pivot = ut.get_body_pose(conveyor)[0]

    if MOVE:
        cid = p.createConstraint(parentBodyUniqueId=conveyor, parentLinkIndex=-1, childBodyUniqueId=-1,
                                 childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                 parentFramePosition=[0, 0, 0], childFramePosition=pivot)
        direction = '+' # for moving back and force
        c = time.time()
        while True:
            # print(direction)
            # print(pivot[0])
            if direction == "+":
                pivot[0] = pivot[0] + step
            else:
                pivot[0] = pivot[0] - step
            # p.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)
            p.changeConstraint(cid, pivot, maxForce=5000)

            target_pose_2d = ut.get_body_pose(target)
            conveyor_pose_2d = ut.get_body_pose(conveyor)
            target_pose = ut.list_2_pose(target_pose_2d)
            pub.publish(target_pose)

            mico_moveit.add_box("cube", target_pose_2d, size=(0.05, 0.05, 0.05))
            mico_moveit.add_box("conveyor", conveyor_pose_2d, size=(.1, .1, .02))

            if pivot[0] > max_x:
                if PRINT_SPEED:
                    time_spent = time.time() - c
                    speed = (conveyor_pose_2d[0][0] - min_x) / time_spent
                    rospy.loginfo("real speed: {} m/s".format(speed))
                    PRINT_SPEED = False  # only print once
                if BACK_AND_FORCE:
                    direction = "-"
                else:
                    exit(0)
            elif pivot[0] < min_x:
                direction = "+"
            r.sleep()
    else:
        while True:
            target_pose_2d = ut.get_body_pose(target)
            conveyor_pose_2d = ut.get_body_pose(conveyor)
            target_pose = ut.list_2_pose(target_pose_2d)
            pub.publish(target_pose)
            mico_moveit.add_box("cube", target, size=(0.05, 0.05, 0.05))
            mico_moveit.add_box("conveyor", conveyor, size=(.1, .1, .02))
            time.sleep(.1)


