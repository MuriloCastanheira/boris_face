import pyaudio
import json
from vosk import Model, KaldiRecognizer

# Caminho do modelo baixado
MODEL_PATH = "/home/castanheira/ros2_ws/src/boris_face/boris_face/vosk-model-small-pt-0.3"

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# Microfone
p = pyaudio.PyAudio()
stream = p.open(
    rate=16000,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=8000
)

stream.start_stream()

print("🎤 Fale algo...")

while True:
    data = stream.read(4000, exception_on_overflow=False)

    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        print("Você disse:", result.get("text", ""))
