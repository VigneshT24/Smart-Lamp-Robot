# Live Character Lamp Robot

A live, expressive character built around the supplied fictional 5-DOF lamp robot. The system combines laptop camera and microphone perception, local speech recognition, cloud multimodal reasoning, scene memory, synthesized audio, and a MuJoCo simulated body into one continuous interaction.

The design intentionally uses a **hybrid local/cloud architecture**: latency-sensitive perception, speech recognition, audio, state management, and robot control run locally, while Gemini is used where semantic visual/language reasoning is most useful.

## Demonstrated Behavior

The lamp supports one continuous interaction containing:

- **Engagement / disengagement** - detects a frontal face as a proxy for attention, greets after several confirmed frames, and returns to neutral after attention has been absent for approximately 2 seconds.
- **Character response** - coordinated pose changes, light state, synthesized greeting chime, speech, nodding, and a completion music cue.
- **Spoken interaction** - microphone input is transcribed locally with Faster-Whisper and responses are spoken using local TTS.
- **Scene memory** - a camera frame is analyzed and useful object information such as name, color, location, and description is stored locally for later questions.
- **Goal-directed action** - a spoken request such as "look at the apple" is combined with a fresh camera image. A multimodal model selects a constrained semantic lamp action, which is then executed by deterministic robot control.
- **Post-action observation** - a new camera frame is captured after the action before the goal is completed.

Across the interaction, motion, light, voice, a sound effect, and music all have purposeful roles rather than being independent demonstrations.

## Architecture and Data Flow

```text
                Laptop Camera
                     |
              +------+------+
              |             |
       Face Detection    Camera Frame
       (OpenCV/local)        |
              |              +----------------------+
              v                                     |
        Engagement State                      Gemini Vision
              |                               / Goal Planner
              |                                     |
              |                              semantic object /
              |                              action decision
              |                                     |
Laptop Mic --> VAD --> Faster-Whisper --> Intent / Character Logic
             local       local CPU                |
                                                  |
                          +-----------------------+
                          |
                    Semantic Actions
                  LOOK_LEFT / RIGHT /
                 UPPER_LEFT / etc.
                          |
                          v
                 Deterministic Controller
                          |
                          v
                    MuJoCo 5-DOF Lamp
                   motion + visual light

Scene observations --> Local Memory --> Gemini text response
                                      |
                                      v
                              Local pyttsx3 TTS
                                      |
                                Laptop Speaker

                     Local synthesized SFX / music
```

### Model-to-Action Boundary

The language model **never controls joint angles directly**.

For goal-directed actions, the planner receives:

1. the user's transcribed goal
2. a compressed live camera image.

It may only return an action from a constrained vocabulary such as:

```text
LOOK_LEFT
LOOK_RIGHT
LOOK_CENTER
LOOK_UPPER_LEFT
LOOK_UPPER_RIGHT
LOOK_LOWER_LEFT
LOOK_LOWER_RIGHT
LOOK_UP
LOOK_DOWN
NOD (only used when called explicitly)
```

`simulator.py` owns the mapping from those semantic actions to known joint configurations.

This separates **semantic reasoning ("what should the lamp do?")** from **body execution ("how should the joints move?")** and prevents arbitrary model output from directly commanding the simulated mechanism.

**NOTE**: Before performing these movements, you need to first set the lamp to be in GREET position (exact joint coordinates are in `test_motion.py` or `main.py`). The action movements are more visible and are best seen when the lamp starts at the GREET position rather than NEUTRAL position.

## Cloud and Local Processing

### Local

- Camera capture
- Face-based engagement detection
- Voice activity detection
- Faster-Whisper `base.en` speech recognition using CPU `int8`
- Interaction state machine
- Scene-memory storage
- MuJoCo control
- TTS with `pyttsx3`
- Synthesized sound effects and music

Raw microphone audio therefore remains local during the active pipeline.

### Cloud

Gemini is used for:

- visual scene understanding
- multimodal goal planning from image + language
- open-ended character responses and memory questions

For visual requests, a resized/compressed JPEG camera frame is sent to Gemini. For conversational requests, the transcribed text and relevant scene memory are sent.

Cloud reasoning was chosen because it provides strong visual-language understanding without requiring a GPU on the target machine. The tradeoff is network/provider latency and API availability.

## Gemini Availability

Cloud requests use a small model fallback router:

```text
gemini-3.5-flash-lite
        |
        v
gemini-3.1-flash-lite
        |
        v
gemini-3.5-flash
```

The primary model is used normally. On rate limits, transient server failures, or request timeouts, the router can move to another compatible model and temporarily place failed models on cooldown.

This improves resilience without replacing semantic reasoning with hardcoded responses. Cloud calls still have bounded timeouts, so external service latency remains a known limitation.

## Simulation and Character Control

The supplied URDF is loaded into **MuJoCo**.

The implementation controls five revolute joints:

- base yaw
- shoulder pitch
- elbow pitch
- neck yaw
- head pitch

Movement is generated through smooth interpolation between joint configurations. This was sufficient for expressive character motion while keeping the controller small and deterministic.

The lamp's visual emitter geometry changes appearance to represent the light turning on/off. It is a character-state visualization rather than a physically modeled scene-casting light source.

Greeting, listening, neutral, directional-looking, and nodding motions are intentionally different so that body language communicates state.

## Setup - Ubuntu 24.04

Target environment:

- Ubuntu 24.04 LTS
- CPU-only operation
- camera
- microphone
- speaker
- internet connection for Gemini API access

Install system dependencies:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip portaudio19-dev espeak-ng libgl1 libglfw3
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

The API key is intentionally not included in the submission.

Run the application:

```bash
cd src
python3 main.py
```

If `python3` throws an error when running, please run using either:

```bash
python main.py
```

Or:

```bash
py main.py
```

The first Faster-Whisper run may download the `base.en` model, so internet access is required for the initial setup.

## Example Demo Flow

A representative continuous interaction is:

1. Person looks toward the camera.
2. Lamp detects engagement, moves to its greeting pose, turns on its light, and plays a chime.
3. Person holds up an object and says:  
   **"Lamp, please remember this object."**
4. Lamp visually identifies the object and stores useful information.
5. The object is removed and the person asks:  
   **"What color was the object I showed you?"**
6. Lamp answers using its stored scene memory.
7. Person places the object to one side and says:  
   **"Lamp, please look at the apple."**
8. The multimodal planner combines the live image and spoken goal, selects a semantic directional action, and the controller executes it.
9. A fresh scene frame is captured, the lamp nods, verbally confirms completion, and plays a short music cue.
10. The person moves attention away and the lamp returns to its neutral pose with the light off.

## Measurements

Measurements observed during development on a CPU-only laptop:

| Component | Observed latency |
|---|---:|
| Faster-Whisper transcription | ~0.4–0.5 s |
| Healthy Gemini text response | ~0.7 s observed |
| Healthy multimodal goal planning | typically ~1–3 s |
| Cloud timeout | bounded at ~10 s per attempted request |

Cloud latency has noticeably higher variance than the local pipeline; this is the main reason speech recognition and control were kept on-device.

## Key Tradeoffs and Limitations

**Face detection instead of full gaze estimation:** frontal-face detection is lightweight and CPU-friendly, but it is only an approximation of whether somebody is truly looking at the character.

**Cloud multimodal reasoning:** provides strong scene/language grounding without requiring local GPU inference, but introduces network latency, rate limits, and provider availability as external dependencies.

**Semantic actions instead of direct model joint control:** less flexible than unrestricted generated trajectories, but substantially easier to validate and keeps physical execution deterministic.

**Kinematic simulation:** joint positions are smoothly interpolated rather than controlled through a full actuator/torque model. A physical robot would require joint-limit enforcement, actuator dynamics, collision handling, feedback control, and safety supervision.

**Fixed laptop camera:** the perception viewpoint does not physically move with the simulated lamp head. A real robot would ideally use a camera rigidly mounted to the moving head, providing true closed-loop visual feedback after motion.

**Visual light representation:** the emitter visibly changes state, but does not model physically accurate illumination of the surrounding scene.

## Completed vs. Intentionally Out of Scope

Completed:

- continuous engagement state machine
- expressive 5-DOF simulated motion
- light, voice, SFX, and music
- local speech recognition
- live visual scene understanding
- scene memory and spoken recall
- multimodal goal-directed action
- constrained model-to-action protocol
- post-action scene observation
- Gemini model failover

Intentionally left out to keep the system focused:

- full gaze tracking
- persistent memory across program restarts
- physical actuator dynamics
- collision-aware motion planning
- physically accurate lighting
- local multimodal LLM inference
- production-level cloud failover/telemetry

## Project Structure

```text
hcl-lamp/
├── robot/
│   ├── assets/
│   └── dummy_lamp_5dof.urdf
├── src/
│   ├── audio.py
│   ├── gemini_router.py
│   ├── main.py
│   ├── perception.py
│   ├── planner.py
│   ├── simulator.py
│   ├── speech.py
│   ├── test_motion.py
│   └── vision.py
├── .env.example
├── requirements.txt
├── README.md
└── SUBMISSION.md
```

## Development Note

Development and demonstration were performed on Windows, while the implementation and dependency choices were kept CPU-based and designed for the specified Ubuntu 24.04 target.

AI-assisted development tools were used during implementation and debugging. The final system architecture, integration behavior, measurements, and technical decisions were validated through direct testing.