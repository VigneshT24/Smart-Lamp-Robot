import cv2
import numpy as np
import sounddevice as sd


class Perception:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.audio_level = 0.0

        self.audio_stream = sd.InputStream(
            channels=1,
            samplerate=16000,
            callback=self._audio_callback,
        )
        self.audio_stream.start()

    def _audio_callback(self, indata, frames, time, status):
        self.audio_level = np.sqrt(np.mean(indata ** 2))

    def person_present(self):
        success, frame = self.camera.read()

        if not success:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        return len(faces) > 0

    def human_speaking(self):
        return self.audio_level > 0.015

    def close(self):
        self.camera.release()
        self.audio_stream.stop()
        self.audio_stream.close()