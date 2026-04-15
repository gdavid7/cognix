"""Wrapper around TRIBE v2 for brain-response prediction and caching."""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


def _text_hash(text: str) -> str:
    """Stable hash for a text string, used as cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class TribeWrapper:
    """Loads TRIBE v2 and provides text -> pooled brain vector with disk caching.

    Usage:
        wrapper = TribeWrapper(cache_dir="artifacts/cached_vectors")
        vector = wrapper.encode("The desert stretched flat and empty.")
        # vector.shape == (20484,)
    """

    def __init__(self, cache_dir: str = "artifacts/cached_vectors", model_cache: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache = model_cache
        self._model = None

    def _load_model(self):
        """Lazy-load TRIBE v2 (expensive — requires GPU + ~40GB VRAM)."""
        if self._model is not None:
            return
        try:
            from tribev2 import TribeModel
        except ImportError:
            raise ImportError(
                "TRIBE v2 not installed. Install it with:\n"
                "  git clone https://github.com/facebookresearch/tribev2.git\n"
                "  cd tribev2 && pip install -e '.[plotting]'\n"
                "  huggingface-cli login  # needs access to meta-llama/Llama-3.2-3B"
            )
        self._model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=self.model_cache)

    def predict_raw(self, text: str) -> np.ndarray:
        """Run TRIBE v2 on text and return the raw brain tensor (T, 20484)."""
        self._load_model()

        # TRIBE expects a file path — write text to a temp file.
        # Flush + fsync to avoid buffering issues.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
            tmp_path = f.name

        try:
            events = self._model.get_events_dataframe(text_path=tmp_path)
            preds, _segments = self._model.predict(events=events)
            return np.asarray(preds)  # (T, 20484)
        finally:
            os.unlink(tmp_path)

    def pool(self, brain_tensor: np.ndarray) -> np.ndarray:
        """Temporal mean pooling: (T, 20484) -> (20484,)."""
        return brain_tensor.mean(axis=0)

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Text -> pooled brain vector (20484,), with disk caching."""
        key = _text_hash(text)
        cache_path = self.cache_dir / f"{key}.npy"

        if use_cache and cache_path.exists():
            return np.load(cache_path)

        raw = self.predict_raw(text)
        pooled = self.pool(raw)

        if use_cache:
            np.save(cache_path, pooled)

        return pooled

    def encode_batch(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """Encode multiple texts, returning (N, 20484) array."""
        vectors = [self.encode(text, use_cache=use_cache) for text in texts]
        return np.stack(vectors)
