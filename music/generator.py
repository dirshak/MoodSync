"""Wraps Hugging Face transformers' MusicGen (facebook/musicgen-small,
pretrained, frozen) to turn a text prompt into a wav file.

The model is loaded lazily on first use and cached as a module-level
singleton, since loading it takes several seconds and it must not be
reloaded per-request. Generation is serialized behind a lock so only one
clip renders at a time (CPU-bound; concurrent generations would just
thrash and be slower overall).
"""

import threading
from pathlib import Path

import numpy as np
import torch

_model = None
_processor = None
_device = None
_load_lock = threading.Lock()
_generate_lock = threading.Lock()

MODEL_ID = "facebook/musicgen-small"
FRAME_RATE_TOKENS_PER_SEC = 50  # musicgen-small's audio codec runs at ~50 Hz


def _ensure_loaded():
    global _model, _processor, _device
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
        _model.to(_device)
        _model.eval()


def is_loaded() -> bool:
    return _model is not None


def preload() -> None:
    """Download and load the model without generating anything.

    Used by the ZeroGPU deployment to pay the one-off ~1.5GB download and
    model load at startup rather than inside a time-limited GPU window."""
    _ensure_loaded()


def move_to(device: str) -> None:
    """Move the loaded model onto `device`.

    On ZeroGPU, CUDA only becomes available inside an @spaces.GPU call, so the
    device chosen at load time (CPU) has to be revised once we are inside one."""
    global _device
    _ensure_loaded()
    if _device != device:
        _model.to(device)
        _device = device


def generate_music(prompt: str, duration_seconds: float, out_path: Path) -> None:
    """Generates `duration_seconds` of audio from `prompt` and writes a wav
    file to `out_path`, creating parent directories as needed."""
    from scipy.io import wavfile

    _ensure_loaded()

    with _generate_lock:
        inputs = _processor(text=[prompt], padding=True, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        max_new_tokens = int(duration_seconds * FRAME_RATE_TOKENS_PER_SEC)
        with torch.no_grad():
            audio_values = _model.generate(**inputs, max_new_tokens=max_new_tokens)

        sample_rate = _model.config.audio_encoder.sampling_rate
        audio = audio_values[0, 0].to(torch.float32).cpu().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize to int16 PCM for broad browser <audio> compatibility.
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    pcm = (audio * 32767).astype(np.int16)
    wavfile.write(str(out_path), sample_rate, pcm)
