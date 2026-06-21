import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import whisper
from whisper.audio import load_audio, SAMPLE_RATE  # SAMPLE_RATE = 16000
from silero_vad import load_silero_vad, VADIterator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("WHISPER_MODEL_PATH", "models/small.pt"))
# DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except:
    DEVICE = "cpu"
LANGUAGE = "en"  # set to None for auto-detect

VAD_WINDOW_SAMPLES = 512          # Silero requires exactly this at 16kHz
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 300          # pause length that counts as a safe word/utterance boundary
VAD_SPEECH_PAD_MS = 100

PARTIAL_EVERY_N_CHUNKS = 3        # cosmetic "still listening" partial cadence while mid-utterance
# ─────────────────────────────────────────────────────────────────────────────

# model: whisper.Whisper | None = None
model: Optional[whisper.Whisper] = None

vad_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vad_model
    print(f"[whisper] loading '{MODEL_PATH}' on {DEVICE.upper()} …")
    model = whisper.load_model(str(MODEL_PATH), device=DEVICE)
    print("[whisper] model ready ✓")

    print("[silero-vad] loading …")
    vad_model = load_silero_vad()
    print("[silero-vad] ready ✓")

    yield


app = FastAPI(title="Whisper STT", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return HTMLResponse(Path("static/index.html").read_text())


# ── Decode + transcribe helpers ────────────────────────────────────────────────
def _decode_to_pcm(raw_bytes: bytes) -> np.ndarray:
    """Decode the FULL accumulated WebM/Opus buffer to 16kHz mono float32 PCM.

    Must always decode from the start of the buffer — a MediaRecorder timeslice
    after the first one is just a WebM *cluster*, not a standalone file, so it
    can't be decoded on its own.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(raw_bytes)
        tmp_path = f.name
    try:
        return load_audio(tmp_path, sr=SAMPLE_RATE)
    finally:
        os.unlink(tmp_path)


def _transcribe_pcm(pcm: np.ndarray) -> str:
    if pcm.size == 0:
        return ""
    result = model.transcribe(pcm, fp16=(DEVICE == "cuda"), language=LANGUAGE)
    return result["text"].strip()


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    raw_buffer = bytearray()        # encoded WebM/Opus bytes — grows for the whole session
    decoded_len = 0                 # how many PCM samples we've already extracted
    committed_sample = 0            # PCM index up to which text has already been finalized
    vad_leftover = np.array([], dtype=np.float32)  # PCM samples not yet a full VAD window
    chunks_since_partial = 0

    vad_iterator = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )

    # Serializes actual whisper.transcribe() calls so a "partial" and a "final"
    # never run concurrently against the (non-thread-safe-under-overlap) model.
    transcribe_lock = asyncio.Lock()

    async def run_blocking(fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    async def emit_final(pcm_segment: np.ndarray):
        async with transcribe_lock:
            text = await run_blocking(_transcribe_pcm, pcm_segment)
        if text:
            await websocket.send_json({"type": "final", "text": text})

    async def emit_partial(pcm_segment: np.ndarray):
        async with transcribe_lock:
            text = await run_blocking(_transcribe_pcm, pcm_segment)
        if text:
            await websocket.send_json({"type": "partial", "text": text})

    try:
        while True:
            message = await websocket.receive()

            # ── Binary: raw audio chunk ───────────────────────────────────
            if message.get("bytes"):
                raw_buffer.extend(message["bytes"])

                pcm_full = await run_blocking(_decode_to_pcm, bytes(raw_buffer))
                new_samples = pcm_full[decoded_len:]
                decoded_len = len(pcm_full)

                # Feed Silero VAD in fixed windows; stash any remainder for next round
                feed = (
                    np.concatenate([vad_leftover, new_samples])
                    if vad_leftover.size
                    else new_samples
                )
                usable_len = (len(feed) // VAD_WINDOW_SAMPLES) * VAD_WINDOW_SAMPLES
                vad_leftover = feed[usable_len:]

                boundary_sample = None
                for i in range(0, usable_len, VAD_WINDOW_SAMPLES):
                    window = feed[i:i + VAD_WINDOW_SAMPLES]
                    speech_dict = vad_iterator(torch.from_numpy(window), return_seconds=False)
                    if speech_dict and "end" in speech_dict:
                        boundary_sample = speech_dict["end"]

                if boundary_sample is not None and boundary_sample > committed_sample:
                    # Silence found -> safe cut point BETWEEN words/utterances.
                    # Commit everything up to here as final text.
                    segment = pcm_full[committed_sample:boundary_sample]
                    committed_sample = boundary_sample
                    chunks_since_partial = 0
                    asyncio.create_task(emit_final(segment))
                else:
                    # Still mid-utterance — show a non-committed live partial only
                    chunks_since_partial += 1
                    if chunks_since_partial >= PARTIAL_EVERY_N_CHUNKS and not transcribe_lock.locked():
                        chunks_since_partial = 0
                        tail = pcm_full[committed_sample:]
                        asyncio.create_task(emit_partial(tail))

            # ── Text: control message ─────────────────────────────────────
            elif message.get("text"):
                ctrl = json.loads(message["text"])

                if ctrl.get("type") == "stop":
                    if raw_buffer:
                        pcm_full = await run_blocking(_decode_to_pcm, bytes(raw_buffer))
                        tail = pcm_full[committed_sample:]
                        if tail.size:
                            # Waits for any in-flight partial/final to release the lock first
                            async with transcribe_lock:
                                text = await run_blocking(_transcribe_pcm, tail)
                            if text:
                                await websocket.send_json({"type": "final", "text": text})

                    # Reset for next recording session
                    raw_buffer = bytearray()
                    decoded_len = 0
                    committed_sample = 0
                    vad_leftover = np.array([], dtype=np.float32)
                    chunks_since_partial = 0
                    vad_iterator.reset_states()

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[ws] unexpected error: {exc}")




        
