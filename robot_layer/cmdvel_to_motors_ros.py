#!/usr/bin/env python2
# ROS Melodic: /cmd_vel (Twist) -> PCA9685 motor control + auto stop timeout

import rospy
from geometry_msgs.msg import Twist
from pca9685 import PCA9685


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class MotorBridge(object):
    def __init__(self):
        self.i2c_bus  = rospy.get_param("~i2c_bus", 1)
        self.i2c_addr = int(rospy.get_param("~i2c_addr", 0x60))
        self.freq_hz  = rospy.get_param("~freq_hz", 100)

        self.L_PWM = rospy.get_param("~left_pwm", 8)
        self.L_IN1 = rospy.get_param("~left_in1", 9)
        self.L_IN2 = rospy.get_param("~left_in2", 10)

        self.R_PWM = rospy.get_param("~right_pwm", 12)
        self.R_IN1 = rospy.get_param("~right_in1", 13)
        self.R_IN2 = rospy.get_param("~right_in2", 14)

        self.max_linear  = float(rospy.get_param("~max_linear", 0.25))
        self.max_angular = float(rospy.get_param("~max_angular", 1.2))

        self.left_scale  = float(rospy.get_param("~left_scale", 1.0))
        self.right_scale = float(rospy.get_param("~right_scale", 1.0))

        self.min_pwm_left  = float(rospy.get_param("~min_pwm_left", 0.0))
        self.min_pwm_right = float(rospy.get_param("~min_pwm_right", 0.0))

        self.left_invert  = bool(rospy.get_param("~left_invert", False))
        self.right_invert = bool(rospy.get_param("~right_invert", False))

        self.cmd_timeout = float(rospy.get_param("~cmd_timeout", 0.2))

        self.last_cmd_time = rospy.Time.now()
        self.last_left = 0.0
        self.last_right = 0.0

        self.pca = PCA9685(bus=self.i2c_bus, address=self.i2c_addr, freq_hz=self.freq_hz)

        self.sub = rospy.Subscriber("/cmd_vel", Twist, self.cb_cmd, queue_size=1)
        rospy.on_shutdown(self.stop_all)

        rospy.loginfo("Listening on /cmd_vel ...")
        rospy.loginfo("cmd_timeout=%.2f sec", self.cmd_timeout)

    def stop_motor(self, pwm_ch, in1_ch, in2_ch):
        self.pca.set_duty(pwm_ch, 0.0)
        self.pca.set_duty(in1_ch, 0.0)
        self.pca.set_duty(in2_ch, 0.0)

    def stop_all(self):
        self.stop_motor(self.L_PWM, self.L_IN1, self.L_IN2)
        self.stop_motor(self.R_PWM, self.R_IN1, self.R_IN2)

    def apply_motor(self, pwm_ch, in1_ch, in2_ch, value, invert=False, min_pwm=0.0):
        value = clamp(value, -1.0, 1.0)

        if invert:
            value = -value

        if abs(value) < 1e-6:
            self.stop_motor(pwm_ch, in1_ch, in2_ch)
            return

        duty = min_pwm + (1.0 - min_pwm) * abs(value)
        duty = clamp(duty, 0.0, 1.0)

        if value > 0.0:
            self.pca.set_duty(in1_ch, 1.0)
            self.pca.set_duty(in2_ch, 0.0)
        else:
            self.pca.set_duty(in1_ch, 0.0)
            self.pca.set_duty(in2_ch, 1.0)

        self.pca.set_duty(pwm_ch, duty)

    def set_motors(self, left, right):
        self.apply_motor(self.L_PWM, self.L_IN1, self.L_IN2,
                         left, invert=self.left_invert, min_pwm=self.min_pwm_left)
        self.apply_motor(self.R_PWM, self.R_IN1, self.R_IN2,
                         right, invert=self.right_invert, min_pwm=self.min_pwm_right)

    def cb_cmd(self, msg):
        v = msg.linear.x / self.max_linear if self.max_linear > 0 else 0.0
        w = msg.angular.z / self.max_angular if self.max_angular > 0 else 0.0

        left  = clamp((v - w) * self.left_scale, -1.0, 1.0)
        right = clamp((v + w) * self.right_scale, -1.0, 1.0)

        self.last_left = left
        self.last_right = right
        self.last_cmd_time = rospy.Time.now()

        self.set_motors(left, right)


if __name__ == "__main__":
    rospy.init_node("cmdvel_to_motors_ros")
    bridge = MotorBridge()
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        if (rospy.Time.now() - bridge.last_cmd_time).to_sec() > bridge.cmd_timeout:
            bridge.stop_all()
        rate.sleep()
