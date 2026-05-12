#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import json
import subprocess

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class VoiceCmdExecutor(object):
    def __init__(self):
        rospy.init_node("voice_cmd_executor", anonymous=False)

        self.poi_file = rospy.get_param("~poi_file", "/home/jetbot/jetbot_ws/config/poi.yaml")

        # parameter tuning sesuai command Anda
        self.forward_speed = rospy.get_param("~forward_speed", "0.018")
        self.turn_left_speed = rospy.get_param("~turn_left_speed", "0.035")
        self.turn_right_speed = rospy.get_param("~turn_right_speed", "0.25")
        self.goal_tolerance = rospy.get_param("~goal_tolerance", "0.18")
        self.heading_turn_start = rospy.get_param("~heading_turn_start", "0.28")
        self.heading_turn_stop = rospy.get_param("~heading_turn_stop", "0.14")
        self.obstacle_stop_distance = rospy.get_param("~obstacle_stop_distance", "0.14")
        self.obstacle_release_distance = rospy.get_param("~obstacle_release_distance", "0.18")
        self.min_forward_burst = rospy.get_param("~min_forward_burst", "0.80")
        self.min_turn_burst = rospy.get_param("~min_turn_burst", "0.25")
        self.forward_angular_bias = rospy.get_param("~forward_angular_bias", "0.0")
        self.forward_heading_kp = rospy.get_param("~forward_heading_kp", "0.0")
        self.max_forward_right_angular = rospy.get_param("~max_forward_right_angular", "0.0")
        self.max_forward_left_angular = rospy.get_param("~max_forward_left_angular", "0.0")
        self.break_forward_error = rospy.get_param("~break_forward_error", "0.20")
        self.break_forward_distance_increase = rospy.get_param("~break_forward_distance_increase", "0.015")

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.cmd_sub = rospy.Subscriber("/voice_cmd", String, self.on_voice_cmd, queue_size=10)

        self.current_proc = None

        rospy.loginfo("voice_cmd_executor ready")

    def stop_robot(self):
        msg = Twist()
        self.cmd_pub.publish(msg)
        self.cmd_pub.publish(msg)
        self.cmd_pub.publish(msg)

    def stop_active_goal(self):
        if self.current_proc is not None:
            try:
                if self.current_proc.poll() is None:
                    rospy.logwarn("Stopping active go_to_poi_minimal process")
                    self.current_proc.terminate()
                    self.current_proc.wait(timeout=2)
            except Exception:
                try:
                    self.current_proc.kill()
                except Exception:
                    pass
            self.current_proc = None

        # jaga-jaga kill node kalau masih hidup
        try:
            subprocess.call(["rosnode", "kill", "/go_to_poi_minimal"])
        except Exception:
            pass

        self.stop_robot()

    def start_goal(self, poi_name):
        self.stop_active_goal()

        cmd = [
            "rosrun", "jetbot_tools", "go_to_poi_minimal.py",
            "_poi_name:=" + poi_name,
            "_poi_file:=" + self.poi_file,
            "_forward_speed:=" + self.forward_speed,
            "_turn_left_speed:=" + self.turn_left_speed,
            "_turn_right_speed:=" + self.turn_right_speed,
            "_goal_tolerance:=" + self.goal_tolerance,
            "_heading_turn_start:=" + self.heading_turn_start,
            "_heading_turn_stop:=" + self.heading_turn_stop,
            "_obstacle_stop_distance:=" + self.obstacle_stop_distance,
            "_obstacle_release_distance:=" + self.obstacle_release_distance,
            "_min_forward_burst:=" + self.min_forward_burst,
            "_min_turn_burst:=" + self.min_turn_burst,
            "_forward_angular_bias:=" + self.forward_angular_bias,
            "_forward_heading_kp:=" + self.forward_heading_kp,
            "_max_forward_right_angular:=" + self.max_forward_right_angular,
            "_max_forward_left_angular:=" + self.max_forward_left_angular,
            "_break_forward_error:=" + self.break_forward_error,
            "_break_forward_distance_increase:=" + self.break_forward_distance_increase,
        ]

        rospy.loginfo("Starting goal to POI: %s", poi_name)
        rospy.loginfo("CMD: %s", " ".join(cmd))

        self.current_proc = subprocess.Popen(cmd)

    def on_voice_cmd(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logerr("Invalid /voice_cmd JSON: %s", str(e))
            return

        intent = data.get("intent")
        target = data.get("target")

        rospy.loginfo("Received /voice_cmd: %s", data)

        if intent == "go_to":
            if not target:
                rospy.logerr("go_to without target")
                return
            self.start_goal(target)
            return

        if intent == "return_home":
            self.start_goal("home")
            return

        if intent == "stop":
            self.stop_active_goal()
            return

        if intent == "cancel_goal":
            self.stop_active_goal()
            return

        rospy.logwarn("Unknown intent: %s", intent)


if __name__ == "__main__":
    try:
        VoiceCmdExecutor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
