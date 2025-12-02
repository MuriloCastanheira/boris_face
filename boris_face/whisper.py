import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import time

# ---------------------
# CONFIGURAÇÕES
# ---------------------
MODEL_SIZE = "tiny"   # tiny, base, small, medium, large
SAMPLE_RATE = 16000
BLOCK_DURATION = 5.0   # segundos de áudio por bloco
DEVICE = "cpu"         # "cpu" ou "cuda"

# ---------------------
# INICIALIZA O MODELO
# ---------------------
print("Carregando modelo Whisper...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")
print("Modelo carregado!")

audio_queue = queue.Queue()
running = True

# ---------------------
# THREAD: captura áudio
# ---------------------
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def microphone_thread():
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        dtype='float32'
    ):
        while running:
            time.sleep(0.1)

# ---------------------
# THREAD: processamento
# ---------------------
def transcriber_thread():
    buffer_audio = np.zeros((0, 1), dtype=np.float32)

    while running:
        # Espera bloco de áudio
        block = audio_queue.get()

        # Acumula no buffer
        buffer_audio = np.concatenate((buffer_audio, block), axis=0)

        # Processo: transcrição parcial
        if len(buffer_audio) >= SAMPLE_RATE * 2:  # processa a cada 2 segundos acumulados
            # Converte para áudio mono
            audio_data = buffer_audio.flatten()

            segments, info = model.transcribe(audio_data, beam_size=1, language="pt")

            text = "".join([seg.text for seg in segments]).strip()

            if text:
                print(f"🗣️ {text}")

            # Limpa buffer para próxima rodada
            buffer_audio = np.zeros((0, 1), dtype=np.float32)

# ---------------------
# EXECUÇÃO
# ---------------------
print("🎤 Iniciando áudio... fale algo!")
audio_thread = threading.Thread(target=microphone_thread)
transcribe_thread = threading.Thread(target=transcriber_thread)

audio_thread.start()
transcribe_thread.start()

try:
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    print("Encerrando...")
    running = False
    audio_thread.join()
    transcribe_thread.join()
