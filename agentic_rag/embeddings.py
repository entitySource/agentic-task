"""Embedding models for retrieval, behind LangChain's `Embeddings` interface.

* `FastEmbedEmbeddings` - a small local neural model (BAAI/bge-small-en-v1.5,
  ONNX, ~70 MB, downloaded once). Matches paraphrases: "live accounts" finds
  a note about "active customers".
* `HashingEmbeddings` - no download, no network, deterministic. Hashes words
  and character trigrams into a fixed-size vector. Good for exact and
  near-exact wording; used in tests and as the fallback.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re

from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)

STOPWORDS = set(
    "a an the of in on for to by and or is are was were be what which who how many much show me give "
    "list with from at this that these those our my per each all any do does did it its as than then vs "
    "versus between over under about tell please can could would i we you".split()
)


def singular(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def words(text: str) -> list[str]:
    return [singular(t) for t in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")) if t not in STOPWORDS]


class HashingEmbeddings(Embeddings):
    name = "hashing"
    relevance_floor = 0.12  # cosine below this is noise for hashed features

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for w in words(text):
            i, sign = self._bucket("w:" + w)
            vec[i] += 2.0 * sign
            padded = f"^{w}$"
            for k in range(len(padded) - 2):
                i, sign = self._bucket("c:" + padded[k:k + 3])
                vec[i] += 0.5 * sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbeddings(Embeddings):
    name = "bge-small-en-v1.5"
    relevance_floor = 0.62  # bge scores unrelated English text around 0.5

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None):
        import os

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


def make_embeddings(kind: str = "auto", cache_dir: str | None = None) -> Embeddings:
    kind = (kind or "auto").lower()
    if kind == "hash":
        return HashingEmbeddings()
    try:
        return FastEmbedEmbeddings(cache_dir=cache_dir)
    except Exception as e:  # not installed, or the model could not be downloaded
        if kind == "fastembed":
            raise
        log.warning("Falling back to hashing embeddings: %s", e)
        return HashingEmbeddings()
