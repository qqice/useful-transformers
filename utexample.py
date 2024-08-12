from useful_transformers import whisper,WhisperModel
import sys
import wave
import audioread
import numpy as np
from pydub import AudioSegment

filename = "/nvme/recordings/mystream/2024-08-09_23-35-35-223146.mp4"
model_name = "base.wenet"
src_lang = "zh"
task = "transcribe"
# Load the mp4 file
with audioread.audio_open(filename) as f:
    # Convert the audio to mono and resample it to 16000Hz
    audio = AudioSegment.from_file(filename)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    audio = audio.set_frame_rate(16000)

    # Save the audio to a temporary wav file
    audio.export("temp.wav", format="wav")

# Read the wav file
w = wave.open("temp.wav")
assert w.getnchannels() == 1, f'Only one channel supported'
assert w.getsampwidth() == 2, f'Datatype should be int16'
assert w.getframerate() == 16000, f'Only 16kHz supported'
frames = w.readframes(w.getnframes())
audio = np.frombuffer(frames, dtype=np.int16)
# Decode the audio using the WhisperModel
model = WhisperModel(model_name)
text = whisper.decode_pcm(audio, model, task, src_lang)

print(text)

