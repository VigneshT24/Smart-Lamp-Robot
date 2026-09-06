import time
from collections import deque
import cv2 # type: ignore
import numpy as np # type: ignore
import sounddevice as sd # type: ignore


class Perception:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.sample_rate = 16000
        self.block_size = 512
        self.threshold = 0.01

        self.loud_blocks = 0

        self.audio_level = 0.0

        # keep about 1.5 second of recent microphone audio
        self.audio_buffer = deque(maxlen=48)

        self.recording = False
        self.recording_frames = []
        self.silent_blocks = 0

        self.audio_stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._audio_callback,
        )

        self.audio_stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata.copy()

        self.audio_level = float(np.sqrt(np.mean(chunk ** 2)))

        self.audio_buffer.append(chunk)

        if self.audio_level > self.threshold:
            self.loud_blocks += 1
        else:
            self.loud_blocks = 0

        if self.recording:
            self.recording_frames.append(chunk)

            if self.audio_level > self.threshold:
                self.silent_blocks = 0
            else:
                self.silent_blocks += 1
    def capture_utterance(self, timeout=6.0):
        print("Capturing speech...")

        # include audio from just before speech was detected
        self.recording_frames = list(self.audio_buffer)

        self.silent_blocks = 0
        self.recording = True

        silence_needed = int(1.3 / (self.block_size / self.sample_rate))

        start = time.monotonic()

        while time.monotonic() - start < timeout:
            # keeps the opencv window responsive while recording
            cv2.waitKey(1)

            if self.silent_blocks >= silence_needed:
                break

            time.sleep(0.01)

        self.recording = False

        if not self.recording_frames:
            return None

        audio = np.concatenate(self.recording_frames, axis=0)

        # faster-whisper wants a 1-D float32 audio array
        audio = audio.reshape(-1).astype(np.float32)

        return audio

    def person_attending(self):
        success, frame = self.camera.read()

        if not success:
            return False

        # downscale for faster face detection
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        attending = len(faces) > 0

        # debug boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(small, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status = "ATTENDING" if attending else "NOT ATTENDING"

        cv2.putText(small, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if attending else (0, 0, 255), 2)

        cv2.imshow("Live Camera Feed", small)
        cv2.waitKey(1)

        return attending

    def get_frame(self):
        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    def human_speaking(self):
        return self.audio_level > self.threshold

    def pause_audio(self):
        if self.audio_stream.active:
            self.audio_stream.stop()
        self.audio_level = 0.0


    def resume_audio(self):
        self.audio_level = 0.0
        self.audio_buffer.clear()

        if not self.audio_stream.active:
            self.audio_stream.start()

    def close(self):
        self.camera.release()
        self.audio_stream.stop()
        self.audio_stream.close()
        cv2.destroyAllWindows()