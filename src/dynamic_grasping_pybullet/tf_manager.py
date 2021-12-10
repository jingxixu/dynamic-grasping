import rospy
import tf
import tf_conversions.posemath as pm


class TFManager(object):

    def __init__(self):
        self._tfs = {}
        self._tf_broadcaster = tf.TransformBroadcaster()

    def add_tf(self, frame_name, ps):
        """
        :type frame_name: str
        :type ps: geometry_msgs.msg.PoseStamped
        :param frame_name: the name of the new frame
        :param ps: the pose stamped for the new frame
        :return: None
        """
        translation, rotation = pm.toTf(pm.fromMsg(ps.pose))
        self._tfs[frame_name] = (ps, translation, rotation)

    def broadcast_tfs(self):
        for frame_name, (ps, translation, rotation) in self._tfs.items():
            self._tf_broadcaster.sendTransform(
                translation, rotation,
                rospy.Time.now(),
                frame_name,
                ps.header.frame_id)

    def clear_tfs(self):
        self._tfs.clear()