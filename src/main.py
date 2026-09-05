import time 

from simulator import LampSimulator
from perception import Perception
from speech import SpeechAgent
from vision import VisionAgent
from planner import GoalPlanner
from audio import AudioController

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
planner = GoalPlanner(speech.client)
audio_ctrl = AudioController()

state = "neutral"

lamp.move_pose(NEUTRAL, 0.5)

try:
    person_frames = 0
    disengaged_since = None

    DISENGAGE_DELAY = 2.0  # in seconds
    light_stay_off = False
    while True:
        person = percep.person_attending()

        if person:
            person_frames += 1
            disengaged_since = None
        else:
            person_frames = 0

            if disengaged_since is None:
                disengaged_since = time.monotonic()

        person_confirmed = person_frames >= 4

        person_gone = (disengaged_since is not None and time.monotonic() - disengaged_since >= DISENGAGE_DELAY)

        print("\nperson_confirmed: ", person_confirmed, "\nperson_gone: ", person_gone)
        print("\nstate: ", state)

        speaking = percep.human_speaking()

        if person_gone:
            lamp.set_light(False)
            if state != "neutral":
                print("Person Gone, Therefore Neutral")
                lamp.move_pose(NEUTRAL, 0.35)
                state = "neutral"
        elif person_confirmed and state == "neutral":
            print("Person Appeared. Therefore Greet")
            percep.pause_audio()
            lamp.set_light(True)
            lamp.move_pose(GREET, 0.25)
            audio_ctrl.play_chime()
            percep.resume_audio()
            state = "greet"
        elif person_confirmed  and speaking and state != "listen":
            if not light_stay_off:
                lamp.set_light(True)
            print("Person Speaking, Therefore Listen")
            lamp.move_pose(LISTEN, 0.30)
            state = "listen"

            percep.pause_audio()

            result = {}

            try:
                audio = speech.listen()

                if audio:
                    result = speech.process_audio(audio, vision.get_memory())

                    print("Human: ", result["transcript"])

                transcript_lower = result["transcript"].lower() if result else ""

                light_on_request = "turn on" in transcript_lower
                light_off_request = "turn off" in transcript_lower

                goal_request = any(phrase in transcript_lower for phrase in ["look at", "look toward", "look to",
                                                                             "turn toward", "turn to", "face the", "nod"])

                vision_request = any(phrase in transcript_lower for phrase in ["what do you see", "remember this", "remember that", "what is this",])

                if light_on_request:
                    response_text = "Sure, I will turn on the light"
                    lamp.set_light(True)
                elif light_off_request:
                    response_text = "Sure, I will turn off the light"
                    lamp.set_light(False)
                    light_stay_off = True
                elif vision_request:
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
                elif goal_request:
                    print("Goal-directed action requested")

                    frame = percep.get_frame()

                    if frame is not None:
                        # observe before acting
                        vision.observe(frame)

                        current_scene = vision.get_current_scene()

                        print("Current scene:", current_scene)

                        plan = planner.plan(result["transcript"], current_scene)

                        print("Plan:", plan)

                        for action in plan["actions"]:
                            print("Executing:", action)
                            lamp.perform_action(action)

                        # observe again after performing actions
                        verify_frame = percep.get_frame()

                        target = plan.get("target", "").strip().lower()
                        verified = False

                        if verify_frame is not None:
                            vision.observe(verify_frame)

                            verified_scene = vision.get_current_scene()
                            print("Scene after action:", verified_scene)

                            for obj in verified_scene:
                                name = obj.get("name", "").strip().lower()

                                if target in name or name in target:
                                    verified = True
                                    break

                        if verified:
                            lamp.perform_action("NOD")
                            audio_ctrl.play_music_cue()

                            response_text = (f"Done. I found the {plan.get('target', 'target')} " f"and completed the action.")
                        else:
                            response_text = (f"I performed the action, but I couldn't verify the " f"{plan.get('target', 'target')} afterward.")

                    else:
                        response_text = "I couldn't get a clear view."

                else:
                    response_text = result["response"] if result else ""

                speech.speak(response_text)
                time.sleep(0.5)

            finally:
                percep.resume_audio()

            state = "greet"

        time.sleep(0.05)
finally:
    percep.close()
    lamp.close()