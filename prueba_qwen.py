"""Genera la misma frase con cada voz en español y mide tiempos (CPU: Kokoro, Piper; GPU: Qwen3-TTS)."""
import os
import sys
import time
import wave

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(BASE, "models")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "muestras")
os.makedirs(OUT, exist_ok=True)
TEXTO = ("Hola, bienvenidos al servidor de inteligencia artificial de Plan Juárez. "
         "Este es un ejemplo de voz generada localmente, sin enviar datos a internet.")


def informe(nombre, t, audio, sr):
    dur = len(audio) / sr
    print(f"{nombre:34} {t:6.2f}s para {dur:5.2f}s de audio (x{dur / t:5.1f} tiempo real)", flush=True)


# Qwen3-TTS 0.6B (GPU)
import torch
from qwen_tts import Qwen3TTSModel


def vram():
    return torch.cuda.max_memory_allocated() / 2**20


t = time.time()
cv = Qwen3TTSModel.from_pretrained(f"{M}/qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda:0",
                                   dtype=torch.bfloat16, attn_implementation="sdpa")
print(f"qwen CustomVoice cargado en {time.time() - t:.1f}s; idiomas={cv.get_supported_languages()}", flush=True)
print("hablantes:", cv.get_supported_speakers(), flush=True)
for spk in ["Ryan", "Serena"]:
    t = time.time()
    wavs, sr = cv.generate_custom_voice(text=TEXTO, speaker=spk, language="Spanish")
    informe(f"qwen custom {spk}", time.time() - t, wavs[0], sr)
    sf.write(f"{OUT}/qwen_custom_{spk}.wav", wavs[0], sr)
print(f"VRAM máx CustomVoice: {vram():.0f} MB", flush=True)
del cv
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Clonación con Base usando como referencia la voz de Kokoro em_alex (voz en español nativa)
base = Qwen3TTSModel.from_pretrained(f"{M}/qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map="cuda:0",
                                     dtype=torch.bfloat16, attn_implementation="sdpa")
t = time.time()
wavs, sr = base.generate_voice_clone(text="Esta voz fue clonada a partir de unos segundos de audio de referencia.",
                                     language="Spanish", ref_audio=f"{OUT}/kokoro_em_alex.wav", ref_text=TEXTO)
informe("qwen base clon (ref em_alex)", time.time() - t, wavs[0], sr)
sf.write(f"{OUT}/qwen_base_clon.wav", wavs[0], sr)
print(f"VRAM máx Base: {vram():.0f} MB", flush=True)
