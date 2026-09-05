import cv2 # type: ignore
import numpy as np # type: ignore
import sounddevice as sd # type: ignore


class Perception:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

        self.audio_level = 0.0

        self.audio_stream = sd.InputStream(
            channels=1,
            samplerate=16000,
            callback=self._audio_callback,
        )
        self.audio_stream.start()

    def _audio_callback(self, indata, frames, time, status):
        self.audio_level = np.sqrt(np.mean(indata ** 2))

    def person_attending(self):
        success, frame = self.camera.read()

        if not success:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        attending = False

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]

            # only search roughly in upper part of face
            eye_region = face_gray[0:int(h * 0.65), :]

            eyes = self.eye_detector.detectMultiScale( eye_region, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20),)

            if len(eyes) >= 2:
                attending = True
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # debug visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        status = "ATTENDING" if attending else "NOT ATTENDING"

        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if attending else (0, 0, 255), 2)

        cv2.imshow("Live Camera Feed", frame)
        cv2.waitKey(1)

        return attending

    def get_frame(self):
        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    def human_speaking(self):
        return self.audio_level > 0.015

    def pause_audio(self):
        self.audio_stream.stop()
        self.audio_level = 0.0

    def resume_audio(self):
        self.audio_level = 0.0
        self.audio_stream.start()

    def close(self):
        self.camera.release()
        self.audio_stream.stop()
        self.audio_stream.close()
        cv2.destroyAllWindows()