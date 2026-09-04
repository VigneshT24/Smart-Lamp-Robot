# base_yaw_joint
# shoulder_pitch_joint
# elbow_pitch_joint
# neck_yaw_joint
# head_pitch_joint
# these joints ^ are controllable

from simulator import LampSimulator
import time

lamp = LampSimulator()

NEUTRAL = {
    "base_yaw_joint": 0.0,
    "shoulder_pitch_joint": 0.15,
    "elbow_pitch_joint": -0.6,
    "neck_yaw_joint": 0.0,
    "head_pitch_joint": -0.1,
}

GREET = {
    "base_yaw_joint": 0.2,
    "shoulder_pitch_joint": 0.35,
    "elbow_pitch_joint": -0.9,
    "neck_yaw_joint": 0.25,
    "head_pitch_joint": 0.2,
}

LISTEN = {
    "base_yaw_joint": 0.0,
    "shoulder_pitch_joint": 0.3,
    "elbow_pitch_joint": -0.8,
    "neck_yaw_joint": 0.0,
    "head_pitch_joint": -0.25,
}

lamp.move_pose(NEUTRAL, 0.5)
time.sleep(1)
lamp.move_pose(GREET, 0.5)
time.sleep(1)
lamp.move_pose(LISTEN, 0.5)
time.sleep(1)

input("Press Enter to exit...")
lamp.close()