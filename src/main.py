import time 

from simulator import LampSimulator
from perception import Perception

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

lamp = LampSimulator()
percep = Perception()

state = "neutral"

lamp.move_pose(NEUTRAL, 0.5)

try:
    while True:
        person = percep.person_present()
        speaking = percep.human_speaking()

        if not person:
            if state != "neutral":
                print("Person Gone, Therefore Neutral")
                lamp.move_pose(NEUTRAL, 0.5)
                state = "neutral"
        elif person and state == "neutral":
            print("Person Appeared. Therefore Greet")
            lamp.move_pose(GREET, 0.5)
            state = "greet"
        elif person and speaking and state != "listen":
            print("Person is Speaking, Therefore Listen")
            lamp.move_pose(LISTEN, 0.5)
            state = "listen"

        time.sleep(0.05)
finally:
    percep.close()
    lamp.close()