"""Embedding backends behind one small interface.

- fastembed:<model>  ONNX Runtime, no PyTorch. Default is BAAI/bge-small-en-v1.5 (384-d).
- wordllama          static 256-d token embeddings bundled in a 16 MB package. Very fast,
                     noticeably weaker; kept for the embedding ablation in the eval.
- hashing            deterministic feature hashing. Only for tests and offline smoke runs.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Protocol

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def model_cache_dir() -> str:
    """fastembed defaults to the system temp dir, which macOS cleans out. Keep models somewhere stable."""
    env = os.getenv("FASTEMBED_CACHE_PATH")
    return str(Path(env).expanduser()) if env else str(Path.home() / ".cache" / "fastembed")


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model}"
        self._model = TextEmbedding(model_name=model, cache_dir=model_cache_dir())
        self._batch = batch_size
        # bge v1.5 retrieves better when short queries carry its instruction prefix.
        self._query_prefix = BGE_QUERY_PREFIX if "bge-" in model and "v1.5" in model else ""
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack(list(self._model.embed(texts, batch_size=self._batch))))

    def embed_query(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.embed([self._query_prefix + text])))
        return _normalize(vec)


class WordLlamaEmbedder:
    def __init__(self, dim: int = 256) -> None:
        from wordllama import WordLlama

        self.name = f"wordllama:l2_supercat_{dim}"
        self._wl = WordLlama.load(dim=dim)
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(self._wl.embed(texts, norm=True))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


TOKEN = re.compile(r"[a-z0-9]+")


_STOP = frozenset(
    "a an and are as at be by can do does for from how i if in is it of on or the to we what when who will with you your".split()
)


class HashingEmbedder:
    """Bag of (crudely stemmed) words hashed into a fixed-size vector. Deterministic, no download."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing:{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in TOKEN.findall(text.lower()):
            if tok in _STOP or len(tok) < 2:
                continue
            feat = tok[:5]  # crude stemming: "receipts" and "receipt" land in the same bucket
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text))


def get_embedder(spec: str) -> Embedder | None:
    spec = spec.strip()
    if spec in ("", "none"):
        return None
    if spec.startswith("fastembed:"):
        return FastEmbedEmbedder(spec.split(":", 1)[1])
    if spec.startswith("wordllama"):
        return WordLlamaEmbedder()
    if spec.startswith("hashing"):
        dim = int(spec.split(":", 1)[1]) if ":" in spec else 256
        return HashingEmbedder(dim)
    raise ValueError(f"Unknown embedding model spec: {spec!r}")
