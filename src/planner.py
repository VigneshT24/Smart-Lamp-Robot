import json
import time
from google.genai.errors import ServerError # type: ignore

class GoalPlanner:
    ALLOWED_ACTIONS = [
        "LOOK_LEFT",
        "LOOK_RIGHT",
        "LOOK_CENTER",
        "LOOK_UPPER_LEFT",
        "LOOK_UPPER_RIGHT",
        "LOOK_LOWER_LEFT",
        "LOOK_LOWER_RIGHT"
    ]

    def __init__(self, client):
        self.client = client

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

    def plan(self, goal, scene_objects):
        prompt = f"""
        You control an articulated desk lamp robot.

        The user gave this goal:
        "{goal}"

        Current visible objects:
        {json.dumps(scene_objects)}

        You may ONLY use these actions:
        {self.ALLOWED_ACTIONS}

        Choose a short sequence of actions that best satisfies the goal.

        Object locations such as left, right, or center, upper left,
        upper right, bottom left, bottom right should determine which direction the lamp looks.

        Return ONLY valid JSON:

        {{
            "actions": ["LOOK_LEFT"],
            "target": "red bottle",
            "reason": "The red bottle is on the left side of the scene."
        }}

        Use at most 3 actions.
        """

        response = self._generate_with_retry(model="gemini-3.5-flash-lite", contents=prompt)

        text = response.text.strip() # type: ignore
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        # reject anything outside our action vocabulary
        result["actions"] = [action for action in result.get("actions", []) if action in self.ALLOWED_ACTIONS]

        return result