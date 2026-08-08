#!/usr/bin/env python

import rospy
import time
from geometry_msgs.msg import PoseStamped
positions_1 = [
        (0, 0, 0, 1),
        (10, -3, 1, 1),
        (7, 4, 0, 1)]
positions_0 = [
    (0,0, 0, 1),
    (3, 0, 1, -1),
    (5, -3, 0, 1),
    (10, -3, 1, 1),
    (11, 3, -1, 0),
    (7, 4, 1, -1)]

positions_4 = [
        (0, 0, 0, 1),
        (-1, 2, 1, 1),
        (4, 7, 1, 1),
        (4, 11, 1, -1),
        (-1, 6, 1, -1)]

positions_2 = [
        (0, 0, 0, 1),
        (3, 0, -1, 0),
        (-1, 2, 1, -1),
        (-1, -3, 1, 1)]
positions_3 = [
        (0, 0, 0, 1),
        (-1, 6, 0, 1),
        (4, 7, -1, 0)]
positions_5 = [
        (0, 0, 0, 1),
        (-1, -3, 1, -1),
        (-1, -12, 1, 1),
        (1, -7, 1, 1)]
positions_8 = [
        (0, 0, 0, 1),
        (1, -7, 0, 1),
        (6, -7, 0, 1)]
positions_6 = [
        (0, 0, 0, 1),
        (4, 11, 1, 1),
        (4, 16, 1, -1)]
positions_7 = [
        (0, 0, 0, 1),
        (-1, -12, 1, -1),
        (-1, -17, 1, 1)]
positions_02 = [
        (0, 0, 0, 1),
        (5, -3, 0, 1),
        (10, -3, 0, 1)]
positions_012 = [
        (0, 0, 0, 1),
        (4, 7, 1, 1),
        (4, 11, 1, -1)]
positions = positions_6
if __name__ == '__main__':
    rospy.init_node('pubpose')
    turtle_vel_pub = rospy.Publisher('send_mark_goal', PoseStamped, queue_size=1)

    mypose = PoseStamped()
    mypose.header.frame_id = 'map'  # or "base_link"

    for (x, y, z, w) in positions:
        mypose.pose.position.x = float(x)
        mypose.pose.position.y = float(y)
        mypose.pose.position.z = 0
        mypose.pose.orientation.x = 0
        mypose.pose.orientation.y = 0
        mypose.pose.orientation.z = float(z)
        mypose.pose.orientation.w = float(w)

        turtle_vel_pub.publish(mypose)
        
        # Wait for a while before publishing the next position
        time.sleep(3)  # Adjust the sleep time as needed

