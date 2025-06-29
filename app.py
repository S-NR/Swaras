from flask import Flask, render_template, request, redirect, send_file, url_for
import librosa
import numpy as np
import soundfile as sf
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)  # Ensure static folder exists for WAV output

# Swara frequency ratios (relative to base freq, Sa = 261.63 Hz)
SCALE_RATIOS = {
    "Sa": 1.0,
    "Re": 9/8,
    "Ga": 5/4,
    "Ma": 4/3,
    "Pa": 3/2,
    "Dha": 5/3,
    "Ni": 15/8,
    "Sa (higher)": 2.0
}

SAMPLE_RATE = 22050

# Swara synthesis functions
def generate_swara_wave(swara, duration=0.5):
    base_freq = 261.63  # Sa in Hz
    freq = base_freq * SCALE_RATIOS.get(swara, 0)
    if freq == 0:
        return np.zeros(int(SAMPLE_RATE * duration))
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    return 0.5 * np.sin(2 * np.pi * freq * t)

def generate_swara_sequence(swaras):
    return np.concatenate([generate_swara_wave(s) for s in swaras])

def save_swara_audio(swaras, out_path="static/swara_output.wav"):
    audio = generate_swara_sequence(swaras)
    sf.write(out_path, audio, SAMPLE_RATE)
    return out_path

# Pitch to swara mapping
def closest_swara(freq, base_freq=261.63):
    min_diff = float('inf')
    closest = "Unknown"
    for swara, ratio in SCALE_RATIOS.items():
        target_freq = base_freq * ratio
        diff = abs(freq - target_freq)
        if diff < min_diff:
            min_diff = diff
            closest = swara
    return closest

# Swara detection using librosa
def detect_swaras(filepath):
    y, sr = librosa.load(filepath)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    swaras = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            swara = closest_swara(pitch)
            swaras.append(swara)
    from itertools import groupby
    simplified = [k for k, _ in groupby(swaras)]
    return simplified

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['audiofile']
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        swaras = detect_swaras(filepath)

        # Save swaras to text file
        with open("swaras_output.txt", "w") as f:
            f.write("\n".join(swaras))

        # ✅ Generate audio playback
        output_wav = save_swara_audio(swaras)

        return render_template('index.html', swaras=swaras)

    return redirect('/')

@app.route('/download')
def download():
    return send_file("swaras_output.txt", as_attachment=True)

# ✅ Correct entry point for platforms like Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
