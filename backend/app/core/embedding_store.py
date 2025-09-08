"""Sentence-transformer embedding store with simple cosine retrieval.
Optional install: pip install .[embeddings]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

@dataclass
class EmbeddingDocument:
    id: str
    text: str
    meta: dict

class EmbeddingStore:
    def __init__(self, persist_dir: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.docs: List[EmbeddingDocument] = []

    def add(self, docs: List[EmbeddingDocument]):
        self.docs.extend(docs)

    def build(self):
        if SentenceTransformer is None:
            raise RuntimeError("Install embeddings extras: pip install .[embeddings]")
        self.model = SentenceTransformer(self.model_name)
        texts = [d.text for d in self.docs]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        self._save()

    def query(self, q: str, k: int = 5) -> List[Tuple[EmbeddingDocument, float]]:
        if self.embeddings is None:
            self._load()
        if SentenceTransformer is None:
            raise RuntimeError("Install embeddings extras: pip install .[embeddings]")
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        q_vec = self.model.encode([q], convert_to_numpy=True, show_progress_bar=False)[0]
        sims = self._cosine(self.embeddings, q_vec)
        idx = sims.argsort()[::-1][:k]
        return [(self.docs[i], float(sims[i])) for i in idx]

    def _cosine(self, mat, vec):  # type: ignore
        import numpy as np
        denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(vec) + 1e-9)) + 1e-9
        return (mat @ vec) / denom

    def _save(self):
        import numpy as np
        meta = [{"id": d.id, "meta": d.meta} for d in self.docs]
        (self.persist_dir / "emb_docs.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (self.persist_dir / "emb_texts.txt").write_text("\n\n".join(d.text for d in self.docs))
        np.save(self.persist_dir / "embeddings.npy", self.embeddings)
        (self.persist_dir / "model_name.txt").write_text(self.model_name)

    def _load(self):
        import numpy as np
        meta = json.loads((self.persist_dir / "emb_docs.json").read_text())
        texts = (self.persist_dir / "emb_texts.txt").read_text().split("\n\n")
        self.docs = [EmbeddingDocument(id=m["id"], meta=m["meta"], text=t) for m, t in zip(meta, texts)]
        self.embeddings = np.load(self.persist_dir / "embeddings.npy")
        self.model_name = (self.persist_dir / "model_name.txt").read_text().strip()
