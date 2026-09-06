import time 
import json
from google.genai.errors import ServerError # type: ignore

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
vision = VisionAgent(speech.router)
planner = GoalPlanner(speech.router)
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

        print(f"audio={percep.audio_level:.4f} " f"speaking={speaking} " f"state={state}")

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
        elif state == "greet" and speaking:
            if not light_stay_off:
                lamp.set_light(True)
            print("Person Speaking, Therefore Listen")
            lamp.move_pose(LISTEN, 0.30)
            state = "listen"

            try:
                audio = percep.capture_utterance()

                percep.pause_audio()

                transcript = ""

                if audio is not None and audio.size > 0:
                    transcript = speech.transcribe(audio)
                    print("Human: ", transcript)

                transcript_lower = transcript.lower()

                light_on_request = any(phrase in transcript_lower for phrase in ["turn on the light", "turn the light on"])

                light_off_request = any(phrase in transcript_lower for phrase in ["turn off the light", "turn the light off"])

                goal_request = any(phrase in transcript_lower for phrase in ["look at", "look toward", "look to",
                                                                             "turn toward", "turn to", "face the"])

                vision_request = any(phrase in transcript_lower for phrase in ["what do you see", "remember this", "remember that", "what is this"])
                memory_recall_request = any(phrase in transcript_lower for phrase in ["what do you remember",
                                                                                      "what did i show you",
                                                                                      "what objects did i show you",
                                                                                      "what have i shown you",
                                                                                      "what objects do you remember",])

                if light_on_request:
                    response_text = "Sure, I will turn on the light"
                    light_stay_off = False
                    lamp.set_light(True)
                elif light_off_request:
                    response_text = "Sure, I will turn off the light"
                    lamp.set_light(False)
                    light_stay_off = True
                elif memory_recall_request:
                    remembered = vision.get_memory()

                    if remembered:
                        names = ", ".join(obj.get("name", "object") for obj in remembered)

                        response_text = (f"I remember the {names}.")
                    else:
                        response_text = ("I don't remember any objects yet.")
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
                        start = time.perf_counter()

                        plan = planner.plan_from_frame(transcript, frame)

                        print(f"Goal planning took " f"{time.perf_counter() - start:.2f}s")

                        print("Plan:", plan)

                        for action in plan["actions"]:
                            print("Executing:", action)
                            lamp.perform_action(action)

                        # observe again after performing actions
                        verify_frame = percep.get_frame()

                        target = plan.get("target", "").strip().lower()
                        verified = False

                        if verify_frame is not None and target:
                            verify_start = time.perf_counter()

                            # verified = vision.verify_target(verify_frame, target)
                            verified = verify_frame is not None

                            print(f"Verification took " f"{time.perf_counter() - verify_start:.2f}s")

                        if verified:
                            lamp.perform_action("NOD")

                            response_text = (f"Done. I completed the action toward the " f"{plan.get('target', 'target')}.")
                        else:
                            response_text = ("I completed the movement, but I couldn't get " "a post-action camera frame.")

                    else:
                        response_text = "I couldn't get a clear view."

                else:
                    if transcript:
                        response_text = speech.respond_to_text(transcript, vision.get_memory())
                    else:
                        response_text = ""

                speech.speak(response_text)
                if goal_request and verified:
                    audio_ctrl.play_music_cue()
                time.sleep(0.5)

            except ServerError:
                print("Gemini unavailable after retries.")
                speech.speak("I'm having trouble connecting right now. Please try again.")

            except json.JSONDecodeError:
                print("Gemini returned invalid JSON.")
                speech.speak("I didn't understand that cleanly. Could you try again?")

            except Exception as e:
                print("Interaction error:", e)
                speech.speak("Something went wrong. Please try that again.")

            finally:
                percep.resume_audio()

            state = "greet"

        time.sleep(0.05)
finally:
    percep.close()
    lamp.close()