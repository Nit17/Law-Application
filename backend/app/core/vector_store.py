from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    id: str
    text: str
    meta: dict


class TfidfStore:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.docs: List[Document] = []

    def add_texts(self, docs: List[Document]):
        self.docs.extend(docs)

    def build(self):
        texts = [d.text for d in self.docs]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self._save()

    def query(self, q: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self.vectorizer or self.matrix is None:
            self._load()
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [(self.docs[i], float(sims[i])) for i in top_idx]

    # Persistence as simple JSON + sklearn internal pickles via vectorizer vocabulary
    def _save(self):
        meta = [{"id": d.id, "meta": d.meta} for d in self.docs]
        (self.persist_dir / "docs.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (self.persist_dir / "texts.txt").write_text("\n\n".join(d.text for d in self.docs))
        # Save vectorizer vocabulary
        import joblib

        joblib.dump(self.vectorizer, self.persist_dir / "vectorizer.joblib")
        joblib.dump(self.matrix, self.persist_dir / "matrix.joblib")

    def _load(self):
        import joblib

        docs_meta = json.loads((self.persist_dir / "docs.json").read_text())
        texts = (self.persist_dir / "texts.txt").read_text().split("\n\n")
        self.docs = [Document(id=m["id"], meta=m["meta"], text=t) for m, t in zip(docs_meta, texts)]
        self.vectorizer = joblib.load(self.persist_dir / "vectorizer.joblib")
        self.matrix = joblib.load(self.persist_dir / "matrix.joblib")
