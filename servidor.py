"""Servidor de voz (TTS) compatible con la API de OpenAI: Kokoro y Piper en CPU, Qwen3-TTS 0.6B en GPU.

El motor se elige por la voz pedida, así que cualquier valor de `model` funciona (tts-1, kokoro, qwen3-tts...).
Qwen3-TTS se carga en la GPU solo al usarse y se descarga tras QWEN_IDLE_SECONDS sin uso.

Voces clonadas (Qwen3-TTS Base): coloca en voces/ un par <nombre>.wav + <nombre>.txt (transcripción del audio,
5-15 s), o súbelo con POST /v1/voces (también desde la página web); aparece como la voz "clon_<nombre>".
"""
import base64
import gc
import io
import json
import os
import re
import threading
import time
import uuid
import wave

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

BASE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(BASE, "models")
VOCES_CLON = os.path.join(BASE, "voces")
API_KEY = os.environ.get("TTS_API_KEY", "")
QWEN_IDLE_SECONDS = int(os.environ.get("QWEN_IDLE_SECONDS", "300"))

KOKORO_VOCES = {"ef_dora": "Kokoro · Dora (mujer, es)", "em_alex": "Kokoro · Alex (hombre, es)",
                "em_santa": "Kokoro · Santa (hombre, es)"}
PIPER_VOCES = {"es_MX-claude-high": "Piper · Claude (mujer, es-MX)", "es_MX-ald-medium": "Piper · Ald (hombre, es-MX)"}
QWEN_VOCES = {s: f"Qwen3 · {s.capitalize()}" for s in
              ["serena", "vivian", "ryan", "aiden", "dylan", "eric", "uncle_fu", "ono_anna", "sohee"]}
QWEN_IDIOMAS = {"es": "spanish", "en": "english", "fr": "french", "de": "german", "it": "italian",
                "pt": "portuguese", "ja": "japanese", "ko": "korean", "zh": "chinese", "ru": "russian"}

app = FastAPI(title="TTS local")


def requiere_clave(request: Request):
    if API_KEY and request.headers.get("authorization", "") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="API key inválida")


# --- Motores ---------------------------------------------------------------------------------
class Kokoro:
    def __init__(self):
        from kokoro_onnx import Kokoro as K
        self.k = K(f"{M}/kokoro/kokoro-v1.0.onnx", f"{M}/kokoro/voices-v1.0.bin")
        self.lock = threading.Lock()

    PAUSA_ORACION, PAUSA_FRASE = 0.25, 0.1
    CORTE_ORACION = re.compile(r"(?<=[.!?…])\s+")
    CORTE_FRASE = re.compile(r"(?<=[.!?…,;:])\s+")

    def generar(self, texto, p):
        # kokoro-onnx solo alarga pausas si el modelo da tiempos por fonema, y este no los da:
        # si se piden pausas distintas, se genera cada fragmento por separado y se inserta el silencio.
        lang = "es" if p.language == "es" else "en-us"
        cambia_frase = p.clause_pause != self.PAUSA_FRASE
        if not cambia_frase and p.sentence_pause == self.PAUSA_ORACION:
            with self.lock:
                return self.k.create(texto, voice=p.voice, speed=p.speed, lang=lang)
        corte = self.CORTE_FRASE if cambia_frase else self.CORTE_ORACION
        partes = []
        for t in (t.strip() for t in corte.split(texto)):
            # Un fragmento de una o dos palabras se pronuncia mal aislado ("Hola," → "Gola"): se une al siguiente
            if partes and len(partes[-1].split()) <= 2 and partes[-1][-1] not in ".!?…":
                partes[-1] += " " + t
            elif t:
                partes.append(t)
        trozos, sr = [], 24000
        with self.lock:
            for i, parte in enumerate(partes):
                a, sr = self.k.create(parte, voice=p.voice, speed=p.speed, lang=lang)
                trozos.append(a)
                if i < len(partes) - 1:
                    pausa = p.sentence_pause if parte.rstrip()[-1] in ".!?…" else p.clause_pause
                    trozos.append(np.zeros(int(pausa * sr), dtype=np.float32))
        return np.concatenate(trozos), sr


class Piper:
    def __init__(self):
        from piper import PiperVoice, SynthesisConfig
        self.voces = {v: PiperVoice.load(f"{M}/piper/{v}.onnx") for v in PIPER_VOCES}
        self.config = SynthesisConfig
        self.lock = threading.Lock()

    def generar(self, texto, p):
        voz = self.voces[p.voice]
        # None = valores por defecto de la voz (de su .onnx.json)
        conf = self.config(length_scale=voz.config.length_scale / p.speed,
                           noise_scale=p.expressiveness, noise_w_scale=p.rhythm)
        buf = io.BytesIO()
        with self.lock, wave.open(buf, "wb") as w:
            voz.synthesize_wav(texto, w, syn_config=conf)
        buf.seek(0)
        audio, sr = sf.read(buf, dtype="float32")
        return audio, sr


class Qwen:
    """Carga perezosa en GPU; un hilo la descarga tras QWEN_IDLE_SECONDS sin uso."""

    def __init__(self):
        self.modelos = {}
        self.ultimo_uso = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._vigilar, daemon=True).start()

    def _modelo(self, variante):
        if variante not in self.modelos:
            import torch
            from qwen_tts import Qwen3TTSModel
            # Solo una variante a la vez en la GPU
            self._descargar()
            self.modelos[variante] = Qwen3TTSModel.from_pretrained(
                f"{M}/qwen/Qwen3-TTS-12Hz-0.6B-{variante}", device_map="cuda:0",
                dtype=torch.bfloat16, attn_implementation="sdpa")
        return self.modelos[variante]

    def _descargar(self):
        if self.modelos:
            import torch
            self.modelos.clear()
            gc.collect()
            torch.cuda.empty_cache()

    def _vigilar(self):
        while True:
            time.sleep(15)
            with self.lock:
                if self.modelos and time.time() - self.ultimo_uso > QWEN_IDLE_SECONDS:
                    self._descargar()

    def generar(self, texto, p):
        voz, lang = p.voice, QWEN_IDIOMAS.get(p.language, "spanish")
        muestreo = {"temperature": p.temperature} if p.temperature is not None else {}
        with self.lock:
            try:
                if voz.startswith("clon_"):
                    nombre = voz[len("clon_"):]
                    ref_txt = open(os.path.join(VOCES_CLON, f"{nombre}.txt"), encoding="utf-8").read().strip()
                    wavs, sr = self._modelo("Base").generate_voice_clone(
                        text=texto, language=lang, ref_audio=os.path.join(VOCES_CLON, f"{nombre}.wav"),
                        ref_text=ref_txt, **muestreo)
                else:
                    wavs, sr = self._modelo("CustomVoice").generate_custom_voice(
                        text=texto, speaker=voz, language=lang, instruct=p.instructions or None, **muestreo)
            finally:
                self.ultimo_uso = time.time()
        return wavs[0], sr


kokoro, piper, qwen = Kokoro(), Piper(), Qwen()


def voces_clon():
    if not os.path.isdir(VOCES_CLON):
        return {}
    return {f"clon_{n[:-4]}": f"Qwen3 clon · {n[:-4]}" for n in sorted(os.listdir(VOCES_CLON))
            if n.endswith(".wav") and os.path.exists(os.path.join(VOCES_CLON, n[:-4] + ".txt"))}


def todas_las_voces():
    return {**KOKORO_VOCES, **PIPER_VOCES, **QWEN_VOCES, **voces_clon()}


def motor_de(voz):
    if voz in KOKORO_VOCES:
        return kokoro
    if voz in PIPER_VOCES:
        return piper
    if voz in QWEN_VOCES or voz in voces_clon():
        return qwen
    return None


# --- API -------------------------------------------------------------------------------------
class Peticion(BaseModel):
    input: str
    voice: str = "ef_dora"
    model: str = "tts-1"
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    language: str = "es"
    volume: float = Field(1.0, ge=0.1, le=3.0, description="Ganancia (todas las voces)")
    pronunciation: bool = Field(True, description="Aplicar el diccionario de pronunciación")
    paragraph_pause: float = Field(0.7, ge=0.0, le=5.0, description="Silencio entre párrafos (todas las voces)")
    # Qwen3-TTS
    instructions: str = Field("", description="Estilo en lenguaje natural (solo voces Qwen3 predefinidas)")
    temperature: float | None = Field(None, ge=0.1, le=1.5)
    # Kokoro
    sentence_pause: float = Field(0.25, ge=0.0, le=3.0)
    clause_pause: float = Field(0.1, ge=0.0, le=2.0)
    # Piper (None = valor de la voz)
    expressiveness: float | None = Field(None, ge=0.0, le=1.5, description="noise_scale")
    rhythm: float | None = Field(None, ge=0.0, le=1.5, description="noise_w_scale")


DICCIONARIO = os.path.join(BASE, "pronunciacion.json")
PAUSA = re.compile(r"\[\s*pausa\s+(\d+(?:[.,]\d+)?)\s*s?\s*\]", re.IGNORECASE)


def leer_diccionario():
    try:
        with open(DICCIONARIO, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def aplicar_diccionario(texto, dic):
    # Palabra completa y respetando mayúsculas, para que "IA" no cambie "ía" dentro de otras palabras
    for palabra, lectura in sorted(dic.items(), key=lambda kv: -len(kv[0])):
        texto = re.sub(rf"(?<!\w){re.escape(palabra)}(?!\w)", lectura, texto)
    return texto


def segmentos(texto):
    """Divide el texto en [(texto, segundos_de_silencio_después)] según las marcas [pausa Ns]."""
    partes, inicio = [], 0
    for m in PAUSA.finditer(texto):
        partes.append((texto[inicio:m.start()].strip(), min(float(m.group(1).replace(",", ".")), 10.0)))
        inicio = m.end()
    partes.append((texto[inicio:].strip(), 0.0))
    return partes


FORMATOS = {"mp3": ("MP3", None, "audio/mpeg"), "wav": ("WAV", "PCM_16", "audio/wav"),
            "flac": ("FLAC", None, "audio/flac"), "opus": ("OGG", "OPUS", "audio/ogg"),
            "ogg": ("OGG", "VORBIS", "audio/ogg")}
# Qwen es lento (~tiempo real) y ocupa la GPU todo ese tiempo; Kokoro y Piper corren en CPU
QWEN_MAX_CHARS = int(os.environ.get("QWEN_MAX_CHARS", "3000"))
BLOQUE_MAX = {"qwen": 300, "cpu": 800}  # caracteres por bloque; bloques cortos = progreso fino y Qwen más estable
FIN_ORACION = re.compile(r"(?<=[.!?…])\s+")


class Cancelado(Exception):
    pass


def bloques(parrafo, maximo):
    """Agrupa oraciones completas en bloques de hasta `maximo` caracteres."""
    salida = []
    for oracion in (o.strip() for o in FIN_ORACION.split(parrafo)):
        if not oracion:
            continue
        if salida and len(salida[-1]) + 1 + len(oracion) <= maximo:
            salida[-1] += " " + oracion
        else:
            salida.append(oracion)
    return salida


def preparar(p: Peticion):
    """Valida la petición y la convierte en un plan [(texto, silencio_después)] para el motor."""
    texto = p.input.strip()
    if not texto:
        raise HTTPException(400, "Texto vacío")
    motor = motor_de(p.voice)
    if motor is None:
        raise HTTPException(400, f"Voz desconocida: {p.voice}. Ver GET /v1/audio/voices")
    if p.response_format not in FORMATOS:
        raise HTTPException(400, f"Formato no soportado: {p.response_format}")
    if p.pronunciation:
        texto = aplicar_diccionario(texto, leer_diccionario())
    es_qwen = motor is qwen
    if es_qwen and len(PAUSA.sub("", texto)) > QWEN_MAX_CHARS:
        raise HTTPException(400, f"Las voces de Qwen3-TTS aceptan hasta {QWEN_MAX_CHARS} caracteres "
                                 f"(este texto tiene {len(PAUSA.sub('', texto))}). Usa Kokoro o Piper, o divide el texto.")
    maximo = BLOQUE_MAX["qwen" if es_qwen else "cpu"]
    pausa_oracion = p.sentence_pause if motor is kokoro else 0.25
    plan = []
    for frag, silencio in segmentos(texto):
        parrafos = [b for b in (bloques(par, maximo) for par in re.split(r"\n\s*", frag)) if b]
        for i, bs in enumerate(parrafos):
            for j, b in enumerate(bs):
                ultimo_parrafo, ultimo_bloque = i == len(parrafos) - 1, j == len(bs) - 1
                pausa = (silencio if ultimo_parrafo else p.paragraph_pause) if ultimo_bloque else pausa_oracion
                plan.append([b, pausa])
        if not parrafos and plan:  # [pausa] seguida de otra [pausa]: se suman
            plan[-1][1] += silencio
    if not plan:
        raise HTTPException(400, "El texto solo contiene pausas")
    plan[-1][1] = 0.0
    return motor, plan


def sintetizar(p: Peticion, motor, plan, avance=None, cancelado=None):
    trozos, sr = [], None
    for i, (texto, silencio) in enumerate(plan):
        if cancelado and cancelado():
            raise Cancelado()
        a, sr = motor.generar(texto, p)
        trozos.append(np.asarray(a, dtype="float32").reshape(-1))
        if silencio:
            trozos.append(np.zeros(int(silencio * sr), dtype="float32"))
        if avance:
            avance(i + 1, sum(len(t) for t in trozos) / sr)
    audio = np.clip(np.concatenate(trozos) * p.volume, -1.0, 1.0)
    fmt, subtipo, mime = FORMATOS[p.response_format]
    if fmt == "OGG" and sr not in (8000, 12000, 16000, 24000, 48000):
        fmt, subtipo, mime = FORMATOS["mp3"]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=fmt, subtype=subtipo)
    return buf.getvalue(), mime, len(audio) / sr


@app.post("/v1/audio/speech", dependencies=[Depends(requiere_clave)])
def speech(p: Peticion):
    motor, plan = preparar(p)
    datos, mime, _ = sintetizar(p, motor, plan)
    return Response(datos, media_type=mime)


# --- Trabajos en segundo plano (textos largos con progreso) -------------------------------------
TRABAJOS = {}
TRABAJOS_LOCK = threading.Lock()
TRABAJO_TTL = 3600  # se borran de memoria una hora después de terminar


class Trabajo:
    def __init__(self, p, motor, plan):
        self.id = uuid.uuid4().hex
        self.estado, self.hecho, self.total = "en_cola", 0, len(plan)
        self.segundos_audio, self.error, self.audio, self.mime = 0.0, None, None, None
        self.creado, self.inicio, self.fin, self.cancelar = time.time(), None, None, False
        threading.Thread(target=self._correr, args=(p, motor, plan), daemon=True).start()

    def _avance(self, hecho, segundos):
        self.hecho, self.segundos_audio = hecho, segundos

    def _correr(self, p, motor, plan):
        self.estado, self.inicio = "procesando", time.time()
        try:
            self.audio, self.mime, self.segundos_audio = sintetizar(
                p, motor, plan, self._avance, lambda: self.cancelar)
            self.estado = "listo"
        except Cancelado:
            self.estado = "cancelado"
        except Exception as e:
            self.estado, self.error = "error", str(e) or e.__class__.__name__
        finally:
            self.fin = time.time()

    def resumen(self):
        ahora = self.fin or time.time()
        return {"id": self.id, "estado": self.estado, "hecho": self.hecho, "total": self.total,
                "segundos_audio": round(self.segundos_audio, 1), "error": self.error,
                "transcurrido": round(ahora - (self.inicio or ahora), 1), "bytes": len(self.audio or b"")}


def trabajo(id_):
    t = TRABAJOS.get(id_)
    if t is None:
        raise HTTPException(404, "No existe ese trabajo (o ya expiró)")
    return t


@app.post("/v1/trabajos", dependencies=[Depends(requiere_clave)])
def crear_trabajo(p: Peticion):
    motor, plan = preparar(p)
    with TRABAJOS_LOCK:
        for k in [k for k, t in TRABAJOS.items() if t.fin and time.time() - t.fin > TRABAJO_TTL]:
            del TRABAJOS[k]
        t = Trabajo(p, motor, plan)
        TRABAJOS[t.id] = t
    return t.resumen()


@app.get("/v1/trabajos/{id_}", dependencies=[Depends(requiere_clave)])
def ver_trabajo(id_: str):
    return trabajo(id_).resumen()


@app.get("/v1/trabajos/{id_}/audio", dependencies=[Depends(requiere_clave)])
def audio_trabajo(id_: str):
    t = trabajo(id_)
    if t.estado != "listo":
        raise HTTPException(409, f"El trabajo está {t.estado}")
    return Response(t.audio, media_type=t.mime)


@app.delete("/v1/trabajos/{id_}", dependencies=[Depends(requiere_clave)])
def cancelar_trabajo(id_: str):
    t = trabajo(id_)
    t.cancelar = True
    return t.resumen()


@app.get("/v1/config")
def config():
    return {"qwen_max_chars": QWEN_MAX_CHARS}


# --- Voces clonadas ----------------------------------------------------------------------------
NOMBRE_VOZ = re.compile(r"^[a-z0-9_-]{1,40}$")


class NuevaVoz(BaseModel):
    nombre: str = Field(..., description="Solo minúsculas, números, _ y -; la voz será clon_<nombre>")
    transcripcion: str = Field(..., min_length=10, max_length=1000)
    audio_wav_base64: str = Field(..., description="Grabación en WAV codificada en base64")
    reemplazar: bool = False


@app.get("/v1/voces")
def listar_clones():
    return {"voces": [{"id": k, "nombre": k[len("clon_"):]} for k in voces_clon()]}


@app.post("/v1/voces", dependencies=[Depends(requiere_clave)])
def crear_clon(v: NuevaVoz):
    nombre = v.nombre.strip().lower()
    if not NOMBRE_VOZ.match(nombre):
        raise HTTPException(400, "Nombre inválido: usa solo minúsculas, números, _ o - (máx. 40)")
    wav, txt = os.path.join(VOCES_CLON, f"{nombre}.wav"), os.path.join(VOCES_CLON, f"{nombre}.txt")
    if os.path.exists(wav) and not v.reemplazar:
        raise HTTPException(409, f"Ya existe la voz clon_{nombre}")
    try:
        audio, sr = sf.read(io.BytesIO(base64.b64decode(v.audio_wav_base64, validate=True)), dtype="float32")
    except Exception:
        raise HTTPException(400, "No se pudo leer el audio (se espera WAV)")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duracion = len(audio) / sr
    if not 3 <= duracion <= 30:
        raise HTTPException(400, f"La grabación dura {duracion:.1f} s; debe durar entre 3 y 30 s (ideal 8-15 s)")
    pico = float(np.abs(audio).max())
    if pico < 0.01:
        raise HTTPException(400, "La grabación está en silencio")
    audio = audio * (0.9 / pico)  # normaliza el volumen
    os.makedirs(VOCES_CLON, exist_ok=True)
    sf.write(wav, audio, sr, subtype="PCM_16")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(v.transcripcion.strip())
    return {"id": f"clon_{nombre}", "duracion": round(duracion, 1)}


@app.delete("/v1/voces/{nombre}", dependencies=[Depends(requiere_clave)])
def borrar_clon(nombre: str):
    if not NOMBRE_VOZ.match(nombre) or f"clon_{nombre}" not in voces_clon():
        raise HTTPException(404, "No existe esa voz")
    for ext in (".wav", ".txt"):
        os.remove(os.path.join(VOCES_CLON, nombre + ext))
    return {"borrada": f"clon_{nombre}"}


@app.get("/v1/pronunciacion")
def ver_diccionario():
    return leer_diccionario()


@app.put("/v1/pronunciacion", dependencies=[Depends(requiere_clave)])
def guardar_diccionario(dic: dict[str, str]):
    limpio = {k.strip(): v.strip() for k, v in dic.items() if k.strip() and v.strip()}
    tmp = DICCIONARIO + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(limpio, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DICCIONARIO)
    return limpio


@app.get("/v1/audio/voices")
def voices():
    return {"voices": [{"id": k, "name": v} for k, v in todas_las_voces().items()]}


@app.get("/v1/audio/models")
def audio_models():
    return {"models": [{"id": m, "name": m} for m in ("kokoro", "piper", "qwen3-tts")]}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in ("kokoro", "piper", "qwen3-tts")]}


@app.get("/")
def pagina():
    """Página web para escribir un texto, generar el audio y descargarlo."""
    return FileResponse(os.path.join(BASE, "pagina.html"), media_type="text/html")


@app.get("/health")
def health():
    return {"ok": True, "qwen_en_gpu": list(qwen.modelos)}
