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

lamp.move_pose(NEUTRAL, 1.0)
time.sleep(4)
lamp.move_pose(GREET, 1.0)

try:    
    time.sleep(4)

    print("Testing LOOK_LEFT")
    lamp.perform_action("LOOK_LEFT")
    time.sleep(2)

    print("Testing LOOK_RIGHT")
    lamp.perform_action("LOOK_RIGHT")
    time.sleep(2)

    print("Testing LOOK_UPPER_RIGHT")
    lamp.perform_action("LOOK_UPPER_RIGHT")
    time.sleep(2)

    print("Testing LOOK_UPPER_LEFT")
    lamp.perform_action("LOOK_UPPER_LEFT")
    time.sleep(2)

    print("Testing LOOK_LOWER_RIGHT")
    lamp.perform_action("LOOK_LOWER_RIGHT")
    time.sleep(2)

    print("Testing LOOK_LOWER_LEFT")
    lamp.perform_action("LOOK_LOWER_LEFT")
    time.sleep(2)

    print("Testing LOOK_CENTER")
    lamp.perform_action("LOOK_CENTER")
    time.sleep(2)

    print("Testing NOD")
    lamp.perform_action("NOD")
    time.sleep(2)

    print("Testing LOOK_UP")
    lamp.perform_action("LOOK_UP")
    time.sleep(2)

    print("Testing LOOK_DOWN")
    lamp.perform_action("LOOK_DOWN")
    time.sleep(2)

finally:
    lamp.close()