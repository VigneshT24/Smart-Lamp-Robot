import json
import cv2 # type: ignore
import time

from google.genai.errors import ServerError # type: ignore
from google import genai
from google.genai import types # type: ignore


class VisionAgent:
    def __init__(self, client):
        self.client = client
        self.memory = {}
        self.current_scene = []

    def _generate_with_retry(self, **kwargs):
        for attempt in range(3):
            try:
                return self.client.models.generate_content(**kwargs)

            except ServerError as e:
                if e.code == 503 and attempt < 2:
                    print("Gemini vision busy, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def observe(self, frame):
        cv2.imshow("Gemini Vision Input", frame)
        cv2.waitKey(1)

        frame = cv2.resize(frame, (640, 480))
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

        if not success:
            return None

        image_bytes = encoded.tobytes()

        response = self._generate_with_retry(
            model="gemini-3.5-flash-lite",
            contents=[
                """
                    Look at this scene.

                    Identify the important visible objects a person might later ask about.

                    Return ONLY valid JSON:

                    {
                        "objects": [
                            {
                                "name": "object name",
                                "color": "color if visible",
                                "location": "rough location in scene",
                                "description": "short useful description"
                            }
                        ]
                    }

                    Keep it concise. Include at most 5 important objects.
                """,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
            ],
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )

        text = response.text.strip() # type: ignore

        # gemini sometimes wraps in json
        text = text.replace("```json", "").replace("```", "").strip()

        scene = json.loads(text)
        new_objects = scene.get("objects", [])
        self.current_scene = new_objects
        newly_added = []

        for obj in new_objects:
            name = obj.get("name", "").strip().lower()

            if not name:
                continue

            # only count it as new if we haven't seen this name before
            if name not in self.memory:
                newly_added.append(obj)

            # still update stored details
            existing = self.memory.get(name, {})
            self.memory[name] = {**existing, **obj}

        return newly_added

    def verify_target(self, frame, target):
        frame = cv2.resize(frame, (320, 240))

        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

        if not success:
            return False

        response = self._generate_with_retry(
            model="gemini-3.5-flash-lite",
            contents=[
                f"""
                Is the "{target}" visible in this image?

                Return ONLY valid JSON:

                {{
                    "visible": true
                }}
                """,
                types.Part.from_bytes(
                    data=encoded.tobytes(),
                    mime_type="image/jpeg"
                ),
            ],
            config=types.GenerateContentConfig(
                automatic_function_calling=
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            ),
        )

        text = response.text.strip() # type: ignore
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        return bool(result.get("visible", False))

    def get_memory(self):
        return list(self.memory.values())

    def get_current_scene(self):
        return self.current_scene