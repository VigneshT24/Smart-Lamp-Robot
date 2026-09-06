import time
import httpx # type: ignore

from google.genai.errors import ClientError, ServerError # type: ignore


MODEL_FALLBACKS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash"
]


class GeminiRouter:
    def __init__(self, client):
        self.client = client

        # if a model fails, temporarily stop choosing it
        self.cooldown_until = {model: 0.0 for model in MODEL_FALLBACKS}

    def _put_on_cooldown(self, model, seconds):
        self.cooldown_until[model] = time.monotonic() + seconds

    def generate(self, *, contents, config=None):
        last_error = None
        timeout_count = 0

        for model in MODEL_FALLBACKS:
            # skip models that recently failed
            if time.monotonic() < self.cooldown_until[model]:
                print(f"Skipping {model} - cooling down")
                continue

            print(f"Trying Gemini model: {model}")

            start = time.perf_counter()

            try:
                response = self.client.models.generate_content(model=model, contents=contents, config=config)

                elapsed = time.perf_counter() - start

                print(f"Gemini model {model} succeeded " f"in {elapsed:.2f}s")

                return response

            except httpx.TimeoutException as e:
                elapsed = time.perf_counter() - start

                print(f"{model} timed out after " f"{elapsed:.2f}s")

                self._put_on_cooldown(model, 30)

                last_error = e
                timeout_count += 1

                # do not allow 5 sequential 10-second timeouts
                if timeout_count >= 2:
                    break

            except ServerError as e:
                code = getattr(e, "code", None)

                if code in (500, 502, 503, 504):
                    print(f"{model} returned {code}. " f"Trying fallback...")

                    self._put_on_cooldown(model, 20)
                    last_error = e
                    continue

                raise

            except ClientError as e:
                code = getattr(e, "code", None)

                if code == 429:
                    print(f"{model} hit rate limit. " f"Trying fallback...")

                    self._put_on_cooldown(model, 60)
                    last_error = e
                    continue

                if code == 404:
                    print(f"{model} unavailable. " f"Trying fallback...")

                    # don't repeatedly retry a model this session
                    self._put_on_cooldown(model, 3600)
                    last_error = e
                    continue

                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("No Gemini fallback models are currently available.")