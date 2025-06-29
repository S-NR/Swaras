import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050  # CD quality

# Frequencies based on C4 = 261.63 Hz
SWARA_FREQS = {
    "Sa": 261.63,
    "Re": 294.33,
    "Ga": 329.63,
    "Ma": 349.23,
    "Pa": 392.00,
    "Dha": 440.00,
    "Ni": 493.88,
    "Sa (higher)": 523.25
}

def generate_swara_wave(swara, duration=0.5):
    freq = SWARA_FREQS.get(swara, 0)
    if freq == 0:
        return np.zeros(int(SAMPLE_RATE * duration))  # silence
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave

def generate_swara_sequence(swaras):
    audio = np.concatenate([generate_swara_wave(s) for s in swaras])
    return audio

def save_swara_audio(swaras, out_path="static/swara_output.wav"):
    audio = generate_swara_sequence(swaras)
    sf.write(out_path, audio, SAMPLE_RATE)
    return out_path
