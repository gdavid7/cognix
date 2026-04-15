"""Wrapper around TRIBE v2 for brain-response prediction and caching.

Supports two caching modes:
  - Pooled only: saves (20484,) mean-pooled vectors (lightweight, Round 1 style)
  - Raw + pooled: saves both (T, 20484) raw tensors and pooled vectors (Round 2+)

Raw tensors are needed for fingerprinting experiments (variance pooling, region-specific
analysis, temporal methods). Pooled vectors are used for immediate similarity computation.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def text_hash(text: str) -> str:
    """Stable hash for a text string, used as cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class TribeWrapper:
    """Loads TRIBE v2 and provides text -> brain tensor with disk caching.

    Usage (pooled only, lightweight):
        wrapper = TribeWrapper(cache_dir="artifacts/cached_vectors")
        vector = wrapper.encode("The desert stretched flat and empty.")
        # vector.shape == (20484,)

    Usage (raw + pooled, for Round 2):
        wrapper = TribeWrapper(
            cache_dir="/content/drive/MyDrive/cognix_cache",
            save_raw=True,
        )
        wrapper.encode_and_cache("The desert stretched flat and empty.")
        # Saves both raw tensor and pooled vector to cache_dir
    """

    def __init__(
        self,
        cache_dir: str = "artifacts/cached_vectors",
        model_cache: str = "./cache",
        save_raw: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.raw_dir = self.cache_dir / "raw_tensors"
        self.pooled_dir = self.cache_dir / "pooled_vectors"
        self.index_path = self.cache_dir / "text_index.json"
        self.progress_path = self.cache_dir / "progress.json"
        self.save_raw = save_raw
        self.model_cache = model_cache
        self._model = None

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if save_raw:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.pooled_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize text index
        if self.index_path.exists():
            with open(self.index_path) as f:
                self._text_index = json.load(f)
        else:
            self._text_index = {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._text_index, f)

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

    def is_cached(self, text: str) -> bool:
        """Check if a text has already been processed."""
        key = text_hash(text)
        pooled_path = self.pooled_dir / f"{key}.npy"
        return pooled_path.exists()

    def encode_and_cache(self, text: str) -> np.ndarray:
        """Run TRIBE, cache raw tensor + pooled vector, return pooled.

        Skips if already cached (resumable).
        """
        key = text_hash(text)
        pooled_path = self.pooled_dir / f"{key}.npy"

        # Skip if already done
        if pooled_path.exists():
            return np.load(pooled_path)

        # Run TRIBE
        raw = self.predict_raw(text)
        pooled = self.pool(raw)

        # Save raw tensor if requested
        if self.save_raw:
            raw_path = self.raw_dir / f"{key}.npz"
            np.savez_compressed(raw_path, tensor=raw)

        # Save pooled vector
        np.save(pooled_path, pooled)

        # Update text index
        self._text_index[key] = text
        self._save_index()

        return pooled

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Text -> pooled brain vector (20484,), with disk caching."""
        key = text_hash(text)
        pooled_path = self.pooled_dir / f"{key}.npy"

        if use_cache and pooled_path.exists():
            return np.load(pooled_path)

        raw = self.predict_raw(text)
        pooled = self.pool(raw)

        if use_cache:
            np.save(pooled_path, pooled)
            self._text_index[key] = text
            self._save_index()

        return pooled

    def load_pooled(self, text: str) -> np.ndarray | None:
        """Load a cached pooled vector without running TRIBE."""
        key = text_hash(text)
        pooled_path = self.pooled_dir / f"{key}.npy"
        if pooled_path.exists():
            return np.load(pooled_path)
        return None

    def load_raw(self, text: str) -> np.ndarray | None:
        """Load a cached raw tensor without running TRIBE."""
        key = text_hash(text)
        raw_path = self.raw_dir / f"{key}.npz"
        if raw_path.exists():
            return np.load(raw_path)["tensor"]
        return None

    def encode_batch(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """Encode multiple texts, returning (N, 20484) array of pooled vectors."""
        vectors = [self.encode(text, use_cache=use_cache) for text in texts]
        return np.stack(vectors)

    def get_progress(self, total_texts: list[str]) -> dict:
        """Check how many texts are already cached."""
        done = sum(1 for t in total_texts if self.is_cached(t))
        return {"done": done, "total": len(total_texts), "remaining": len(total_texts) - done}
