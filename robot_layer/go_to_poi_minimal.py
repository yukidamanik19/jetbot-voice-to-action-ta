#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import rospy
import tf
import yaml

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value, low, high):
    return max(low, min(high, value))


class MinimalPoiControllerV26(object):
    def __init__(self):
        rospy.init_node("go_to_poi_minimal")

        # ===== Parameters =====
        self.poi_name = rospy.get_param("~poi_name", "home")
        self.poi_file = rospy.get_param(
            "~poi_file",
            os.path.expanduser("~/jetbot_ws/config/poi.yaml")
        )

        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.map_frame = rospy.get_param("~map_frame", "map")

        # Translasi
        self.forward_speed = rospy.get_param("~forward_speed", 0.015)

        # Turn asymmetry
        self.turn_left_speed = rospy.get_param("~turn_left_speed", 0.10)
        self.turn_right_speed = rospy.get_param("~turn_right_speed", 0.13)

        # Goal
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.18)
        self.slowdown_distance = rospy.get_param("~slowdown_distance", 0.30)

        # Heading hysteresis
        self.heading_turn_start = rospy.get_param("~heading_turn_start", 0.12)
        self.heading_turn_stop = rospy.get_param("~heading_turn_stop", 0.06)

        # Obstacle hysteresis
        self.obstacle_stop_distance = rospy.get_param("~obstacle_stop_distance", 0.17)
        self.obstacle_release_distance = rospy.get_param("~obstacle_release_distance", 0.24)

        # Front lidar sector
        self.front_sector_deg = rospy.get_param("~front_sector_deg", 20.0)

        # Yaw correction 180 deg
        self.yaw_offset = rospy.get_param("~yaw_offset", math.pi)

        # Burst durations
        self.min_forward_burst = rospy.get_param("~min_forward_burst", 0.50)
        self.min_turn_burst = rospy.get_param("~min_turn_burst", 0.40)

        # ===== Forward compensation =====
        # Robot cenderung belok kiri saat maju -> perlu bias kanan (angular negatif)
        self.forward_angular_bias = rospy.get_param("~forward_angular_bias", -0.070)

        # Penting:
        # Untuk robot kamu, saat heading_error membesar positif ketika maju,
        # robot justru perlu koreksi lebih ke kanan.
        # Maka dipakai tanda "minus" pada koreksi heading.
        self.forward_heading_kp = rospy.get_param("~forward_heading_kp", 0.45)

        # Batas angular saat maju dipisah kiri/kanan
        self.max_forward_right_angular = rospy.get_param("~max_forward_right_angular", 0.12)
        self.max_forward_left_angular = rospy.get_param("~max_forward_left_angular", 0.04)

        # Kalau saat forward hold heading error membesar, paksa keluar
        self.break_forward_error = rospy.get_param("~break_forward_error", 0.12)

        # Kalau distance malah memburuk saat forward burst, paksa keluar
        self.break_forward_distance_increase = rospy.get_param("~break_forward_distance_increase", 0.015)

        # Loop
        self.control_rate = rospy.get_param("~control_rate", 10.0)

        # ===== State =====
        self.latest_scan = None
        self.front_blocked = False
        self.in_turn_mode = True
        self.last_mode = "INIT"
        self.mode_until = rospy.Time(0)

        # simpan jarak saat mulai forward burst
        self.forward_burst_start_distance = None

        # ROS
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.listener = tf.TransformListener()

        # Target
        self.target_x, self.target_y = self.load_poi(self.poi_file, self.poi_name)

        rospy.loginfo("==== go_to_poi_minimal v2.6 ====")
        rospy.loginfo("POI target: %s", self.poi_name)
        rospy.loginfo("Target x=%.3f y=%.3f", self.target_x, self.target_y)
        rospy.loginfo("poi_file=%s", self.poi_file)
        rospy.loginfo("forward_speed=%.4f", self.forward_speed)
        rospy.loginfo("turn_left_speed=%.4f turn_right_speed=%.4f",
                      self.turn_left_speed, self.turn_right_speed)
        rospy.loginfo("goal_tolerance=%.3f slowdown_distance=%.3f",
                      self.goal_tolerance, self.slowdown_distance)
        rospy.loginfo("heading_turn_start=%.3f heading_turn_stop=%.3f",
                      self.heading_turn_start, self.heading_turn_stop)
        rospy.loginfo("obstacle_stop=%.3f obstacle_release=%.3f front_sector_deg=%.1f",
                      self.obstacle_stop_distance, self.obstacle_release_distance, self.front_sector_deg)
        rospy.loginfo("yaw_offset=%.3f rad", self.yaw_offset)
        rospy.loginfo("min_forward_burst=%.2f min_turn_burst=%.2f",
                      self.min_forward_burst, self.min_turn_burst)
        rospy.loginfo("forward_angular_bias=%.4f forward_heading_kp=%.3f max_forward_right_angular=%.3f max_forward_left_angular=%.3f",
                      self.forward_angular_bias, self.forward_heading_kp,
                      self.max_forward_right_angular, self.max_forward_left_angular)
        rospy.loginfo("break_forward_error=%.3f break_forward_distance_increase=%.3f",
                      self.break_forward_error, self.break_forward_distance_increase)

    def load_poi(self, poi_file, poi_name):
        if not os.path.exists(poi_file):
            rospy.logerr("File POI tidak ditemukan: %s", poi_file)
            raise IOError("POI file not found")

        with open(poi_file, "r") as f:
            data = yaml.safe_load(f)

        if poi_name not in data:
            rospy.logerr("POI '%s' tidak ada di file: %s", poi_name, poi_file)
            raise KeyError("POI not found")

        return float(data[poi_name]["x"]), float(data[poi_name]["y"])

    def scan_callback(self, msg):
        self.latest_scan = msg

    def get_robot_pose(self):
        self.listener.waitForTransform(
            self.map_frame,
            self.base_frame,
            rospy.Time(0),
            rospy.Duration(1.0)
        )
        (trans, rot) = self.listener.lookupTransform(
            self.map_frame,
            self.base_frame,
            rospy.Time(0)
        )

        x = trans[0]
        y = trans[1]
        yaw = tf.transformations.euler_from_quaternion(rot)[2]
        return x, y, yaw

    def get_front_min_distance(self):
        if self.latest_scan is None:
            return None

        scan = self.latest_scan
        sector_rad = math.radians(self.front_sector_deg)

        valid_ranges = []
        angle = scan.angle_min

        for r in scan.ranges:
            if -sector_rad <= angle <= sector_rad:
                if (not math.isnan(r)) and (not math.isinf(r)) and r > 0.0:
                    valid_ranges.append(r)
            angle += scan.angle_increment

        if not valid_ranges:
            return None

        return min(valid_ranges)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def set_mode_with_burst(self, mode_name, burst_duration):
        now = rospy.Time.now()
        self.last_mode = mode_name
        self.mode_until = now + rospy.Duration.from_sec(burst_duration)

    def burst_active(self):
        return rospy.Time.now() < self.mode_until

    def update_heading_mode(self, abs_heading_error):
        if self.in_turn_mode:
            if abs_heading_error < self.heading_turn_stop:
                self.in_turn_mode = False
        else:
            if abs_heading_error > self.heading_turn_start:
                self.in_turn_mode = True

    def choose_turn_cmd(self, heading_error):
        cmd = Twist()
        if heading_error > 0.0:
            cmd.angular.z = self.turn_left_speed
            mode = "TURN_LEFT"
        else:
            cmd.angular.z = -self.turn_right_speed
            mode = "TURN_RIGHT"
        cmd.linear.x = 0.0
        return cmd, mode

    def choose_forward_cmd(self, distance, heading_error):
        cmd = Twist()

        if distance < self.slowdown_distance:
            cmd.linear.x = min(self.forward_speed, 0.015)
        else:
            cmd.linear.x = self.forward_speed

        # KOREKSI PENTING:
        # sign dibalik agar saat heading_error positif, robot makin diarahkan ke kanan
        ang = self.forward_angular_bias - (self.forward_heading_kp * heading_error)

        # clamp asimetris: koreksi kanan boleh lebih kuat daripada kiri
        if ang < 0.0:
            ang = clamp(ang, -self.max_forward_right_angular, 0.0)
        else:
            ang = clamp(ang, 0.0, self.max_forward_left_angular)

        cmd.angular.z = ang
        return cmd

    def should_break_forward(self, distance, abs_heading_error, obstacle_ahead):
        if obstacle_ahead:
            return True

        if abs_heading_error > self.break_forward_error:
            return True

        if self.forward_burst_start_distance is not None:
            if distance > (self.forward_burst_start_distance + self.break_forward_distance_increase):
                return True

        return False

    def run(self):
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():
            try:
                robot_x, robot_y, robot_yaw = self.get_robot_pose()
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn_throttle(1.0, "TF map -> base_footprint belum siap")
                self.stop_robot()
                rate.sleep()
                continue

            dx = self.target_x - robot_x
            dy = self.target_y - robot_y
            distance = math.sqrt(dx * dx + dy * dy)

            target_yaw = math.atan2(dy, dx)
            control_yaw = wrap_to_pi(robot_yaw + self.yaw_offset)
            heading_error = wrap_to_pi(target_yaw - control_yaw)
            abs_heading_error = abs(heading_error)

            front_min = self.get_front_min_distance()
            cmd = Twist()

            if distance <= self.goal_tolerance:
                rospy.loginfo("Target tercapai. distance=%.3f", distance)
                self.stop_robot()
                break

            if front_min is None:
                rospy.logwarn_throttle(1.0, "Belum ada data /scan valid di sektor depan")
                self.stop_robot()
                rate.sleep()
                continue

            # Obstacle hysteresis
            if self.front_blocked:
                if front_min > self.obstacle_release_distance:
                    self.front_blocked = False
            else:
                if front_min < self.obstacle_stop_distance:
                    self.front_blocked = True

            obstacle_ahead = self.front_blocked

            # Heading hysteresis
            self.update_heading_mode(abs_heading_error)

            # ===== Burst hold =====
            if self.burst_active():
                if self.last_mode == "FORWARD":
                    if self.should_break_forward(distance, abs_heading_error, obstacle_ahead):
                        cmd, base_mode = self.choose_turn_cmd(heading_error if heading_error != 0.0 else 1.0)
                        mode = "BREAK_" + base_mode
                        self.forward_burst_start_distance = None
                        self.set_mode_with_burst("TURN", self.min_turn_burst)
                    else:
                        cmd = self.choose_forward_cmd(distance, heading_error)
                        mode = "FORWARD_HOLD"
                else:
                    cmd, base_mode = self.choose_turn_cmd(heading_error if heading_error != 0.0 else 1.0)
                    mode = base_mode + "_HOLD"

                rospy.loginfo_throttle(
                    1.0,
                    "%s | dist=%.3f | heading_error=%.3f | robot_yaw=%.3f | control_yaw=%.3f | front_min=%.3f | cmd_v=%.3f | cmd_w=%.3f | blocked=%s",
                    mode, distance, heading_error, robot_yaw, control_yaw, front_min,
                    cmd.linear.x, cmd.angular.z, str(obstacle_ahead)
                )

                self.cmd_pub.publish(cmd)
                rate.sleep()
                continue

            # ===== Mode selection =====
            if obstacle_ahead:
                cmd, base_mode = self.choose_turn_cmd(heading_error if heading_error != 0.0 else 1.0)
                mode = "OBSTACLE_" + base_mode
                self.forward_burst_start_distance = None
                self.set_mode_with_burst("TURN", self.min_turn_burst)

                rospy.logwarn_throttle(
                    1.0,
                    "%s | dist=%.3f | heading_error=%.3f | robot_yaw=%.3f | control_yaw=%.3f | front_min=%.3f | stop=%.3f release=%.3f",
                    mode, distance, heading_error, robot_yaw, control_yaw, front_min,
                    self.obstacle_stop_distance, self.obstacle_release_distance
                )

            elif self.in_turn_mode:
                cmd, mode = self.choose_turn_cmd(heading_error)
                self.forward_burst_start_distance = None
                self.set_mode_with_burst("TURN", self.min_turn_burst)

                rospy.loginfo_throttle(
                    1.0,
                    "%s | dist=%.3f | heading_error=%.3f | robot_yaw=%.3f | control_yaw=%.3f | front_min=%.3f | cmd_v=%.3f | cmd_w=%.3f",
                    mode, distance, heading_error, robot_yaw, control_yaw, front_min,
                    cmd.linear.x, cmd.angular.z
                )

            else:
                cmd = self.choose_forward_cmd(distance, heading_error)
                mode = "FORWARD"
                self.forward_burst_start_distance = distance
                self.set_mode_with_burst("FORWARD", self.min_forward_burst)

                rospy.loginfo_throttle(
                    1.0,
                    "%s | dist=%.3f | heading_error=%.3f | robot_yaw=%.3f | control_yaw=%.3f | front_min=%.3f | cmd_v=%.3f | cmd_w=%.3f",
                    mode, distance, heading_error, robot_yaw, control_yaw, front_min,
                    cmd.linear.x, cmd.angular.z
                )

            self.cmd_pub.publish(cmd)
            rate.sleep()

        self.stop_robot()


if __name__ == "__main__":
    try:
        node = MinimalPoiControllerV26()
        node.run()
    except rospy.ROSInterruptException:
        pass
