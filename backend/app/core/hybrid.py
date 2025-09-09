"""Hybrid retrieval combining TF-IDF and Embedding scores via linear fusion.
Requires embedding index built (embed/build). Falls back gracefully if embeddings absent.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from .vector_store import TfidfStore, Document
from .embedding_store import EmbeddingStore, EmbeddingDocument

@dataclass
class HybridHit:
    id: str
    text: str
    meta: dict
    score_tfidf: float
    score_embed: float
    score_final: float

class HybridRetriever:
    def __init__(self, index_dir: Path, alpha: float = 0.5, emb_model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = Path(index_dir)
        self.alpha = alpha  # weight for embeddings (0..1)
        self.emb_model_name = emb_model_name

    def retrieve(self, query: str, k: int = 5) -> List[HybridHit]:
        tf_store = TfidfStore(self.index_dir)
        tf_results = tf_store.query(query, k=max(k, 10))  # grab a bit more for fusion
        tf_map: Dict[str, tuple[Document, float]] = {d.id: (d, s) for d, s in tf_results}

        # Try embedding retrieval
        emb_hits: Dict[str, tuple[EmbeddingDocument, float]] = {}
        try:
            emb_store = EmbeddingStore(self.index_dir, model_name=self.emb_model_name)
            emb_results = emb_store.query(query, k=max(k, 10))
            emb_hits = {d.id: (d, s) for d, s in emb_results}
        except Exception:
            # embeddings unavailable, treat all embed scores as 0
            pass

        hybrid: List[HybridHit] = []
        ids = set(tf_map.keys()) | set(emb_hits.keys())
        # Normalize scores locally per modality for fair fusion
        def _norm(values):
            if not values:
                return {}
            mx = max(values.values()) or 1e-9
            mn = min(values.values())
            rng = (mx - mn) or 1e-9
            return {k: (v - mn) / rng for k, v in values.items()}

        tf_norm = _norm({i: s for i, (_, s) in tf_map.items()})
        emb_norm = _norm({i: s for i, (_, s) in emb_hits.items()})

        for _id in ids:
            doc = tf_map.get(_id, (None, None))[0] or emb_hits.get(_id, (None, None))[0]  # type: ignore
            if doc is None:
                continue
            st = tf_norm.get(_id, 0.0)
            se = emb_norm.get(_id, 0.0)
            final = (1 - self.alpha) * st + self.alpha * se
            hybrid.append(HybridHit(id=_id, text=doc.text, meta=getattr(doc, 'meta', {}), score_tfidf=st, score_embed=se, score_final=final))

        hybrid.sort(key=lambda h: h.score_final, reverse=True)
        return hybrid[:k]
