import time 

from simulator import LampSimulator
from perception import Perception
from speech import SpeechAgent
from vision import VisionAgent

NEUTRAL = {
    "base_yaw_joint": 0.0,
    "shoulder_pitch_joint": 0.05,
    "elbow_pitch_joint": -0.45,
    "neck_yaw_joint": 0.0,
    "head_pitch_joint": -0.05,
}

GREET = {
    "base_yaw_joint": 0.45,
    "shoulder_pitch_joint": 0.65,
    "elbow_pitch_joint": -1.15,
    "neck_yaw_joint": 0.35,
    "head_pitch_joint": 0.0,
}

LISTEN = {
    "base_yaw_joint": 0.0,
    "shoulder_pitch_joint": 0.45,
    "elbow_pitch_joint": -1.05,
    "neck_yaw_joint": 0.0,
    "head_pitch_joint": -0.50,
}

lamp = LampSimulator()
percep = Perception()
speech = SpeechAgent()
vision = VisionAgent(speech.client)

state = "neutral"

lamp.move_pose(NEUTRAL, 0.5)

try:
    person_frames = 0
    missing_frames = 0
    while True:
        person = percep.person_present()

        if person:
            person_frames += 1
            missing_frames = 0
        else:
            missing_frames += 1
            person_frames = 0

        person_confirmed = person_frames >= 4
        person_gone = missing_frames >= 10

        print("\nperson_confirmed: ", person_confirmed, "\nperson_gone: ", person_gone)
        print("\nstate: ", state)

        speaking = percep.human_speaking()

        if person_gone:
            if state != "neutral":
                print("Person Gone, Therefore Neutral")
                lamp.move_pose(NEUTRAL, 0.35)
                state = "neutral"
        elif person_confirmed and state == "neutral":
            print("Person Appeared. Therefore Greet")
            lamp.move_pose(GREET, 0.25)
            state = "greet"
        elif person_confirmed  and speaking and state != "listen":
            print("Person Speaking, Therefore Listen")
            lamp.move_pose(LISTEN, 0.30)
            state = "listen"

            percep.pause_audio()

            try:
                audio = speech.listen()

                if audio:
                    result = speech.process_audio(audio, vision.get_memory())

                    print("Human: ", result["transcript"])

                transcript_lower = result["transcript"].lower()

                vision_request = any(phrase in transcript_lower for phrase in ["what do you see", "look at",
                                                                                "remember this", "remember that",
                                                                                "what is this",])

                if vision_request:
                    frame = percep.get_frame()

                    if frame is not None:
                        objects = vision.observe(frame)

                        print("Updated scene memory:", objects)

                        if objects:
                            names = ", ".join(obj["name"] for obj in objects)
                            response_text = (f"Yep, I'll remember the {names} too.")
                        elif len(vision.get_memory()) != 0:
                            response_text = "I already remember what I'm seeing."
                        else:
                            response_text = "I couldn't clearly identify anything."
                    else:
                        response_text = "I couldn't get a clear view."

                else:
                    response_text = result["response"]

                speech.speak(response_text)
                time.sleep(0.5)

            finally:
                percep.resume_audio()

            state = "greet"

        time.sleep(0.05)
finally:
    percep.close()
    lamp.close()