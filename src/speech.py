import io
import os
import time
import wave
import json

from google.genai.errors import ServerError # type: ignore
import numpy as np # type: ignore
import sounddevice as sd # type: ignore
import pyttsx3 # type: ignore

from dotenv import load_dotenv # type: ignore
from google import genai # type: ignore
from google.genai import types # type: ignore

load_dotenv()


class SpeechAgent:
    def __init__(self):
        self.sample_rate = 16000
        self.threshold = 0.015

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

    def _generate_with_retry(self, **kwargs):
        for attempt in range(3):
            try:
                return self.client.models.generate_content(**kwargs)

            except ServerError as e:
                if e.code == 503 and attempt < 2:
                    print("Gemini busy, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def listen(self):
        block_size = 1024
        frames = []

        speech_started = False
        silent_blocks = 0

        silence_needed = int(0.8 / (block_size / self.sample_rate))

        max_blocks = int(10 / (block_size / self.sample_rate))

        print("Listening...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
        ) as stream:

            for _ in range(max_blocks):
                data, _ = stream.read(block_size)

                level = np.sqrt(np.mean(data ** 2))

                if level > self.threshold:
                    speech_started = True
                    silent_blocks = 0

                elif speech_started:
                    silent_blocks += 1

                if speech_started:
                    frames.append(data.copy())

                if speech_started and silent_blocks >= silence_needed:
                    break

        if not frames:
            return None

        audio = np.concatenate(frames)

        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    def process_audio(self, audio_bytes, scene_memory=None):
        memory_text = json.dumps(scene_memory or [])

        response = self._generate_with_retry(
            model="gemini-3.5-flash-lite",
            contents=[
                f"""
                    Listen to the person speaking.

                    The lamp's remembered scene objects are:
                    {memory_text}

                    Return ONLY valid JSON:

                    {{
                        "transcript": "exactly what the person said",
                        "response": "short natural response"
                    }}

                    If the person asks about something previously seen, use the scene memory.
                    Do not invent objects that are not in memory.

                    The response must be 1-2 sentences.
                    Only include words that should be spoken aloud.
                    Do not describe movements.
                """,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav",
                ),
            ],
        )

        text = response.text.strip() # type: ignore
        text = text.replace("```json", "").replace("```", "").strip()

        return json.loads(text)

    def speak(self, text):
        print("Lamp:", text)

        engine = pyttsx3.init()
        engine.setProperty("rate", 175)

        engine.say(text)
        engine.runAndWait()

        engine.stop()
        del engine