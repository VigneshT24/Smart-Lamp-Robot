import numpy as np # type: ignore
import sounddevice as sd # type: ignore


class AudioController:
    def __init__(self):
        self.sample_rate = 44100

    def _bell_tone(self, frequency, duration, volume=0.25):
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

        # layer harmonics so it sounds less like a pure beep
        wave = ( np.sin(2 * np.pi * frequency * t) + 0.45 * np.sin(2 * np.pi * frequency * 2 * t) + 0.20 * np.sin(2 * np.pi * frequency * 3 * t))

        # natural bell-like decay
        envelope = np.exp(-4 * t / duration)

        return volume * wave * envelope


    def play_chime(self):
        note1 = self._bell_tone(523.25, 0.45)   # C
        note2 = self._bell_tone(659.25, 0.50)   # E
        note3 = self._bell_tone(783.99, 0.65)   # G

        gap = np.zeros(int(self.sample_rate * 0.04))

        audio = np.concatenate([ note1, gap, note2, gap, note3])

        sd.play(audio, self.sample_rate)
        sd.wait()

    def play_music_cue(self):
        # tiny musical phrase
        notes = [
            (523.25, 0.20),
            (659.25, 0.20),
            (783.99, 0.20),
            (1046.50, 0.35),
        ]

        audio = np.concatenate([ self._bell_tone(freq, duration) for freq, duration in notes])

        sd.play(audio, self.sample_rate)
        sd.wait()