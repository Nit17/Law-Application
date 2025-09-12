from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
	from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
	SentenceTransformer = None  # type: ignore


@dataclass
class TextRecord:
	id: str
	text: str
	meta: dict


class EmbeddingStore:
	def __init__(self, persist_dir: Path, model_name: str = "all-MiniLM-L6-v2"):
		self.persist_dir = Path(persist_dir)
		self.persist_dir.mkdir(parents=True, exist_ok=True)
		self.model_name = model_name
		self.model = SentenceTransformer(model_name) if SentenceTransformer is not None else None
		self.records: List[TextRecord] = []
		self.embs = None

	def build(self, force: bool = False) -> int:
		texts_file = self.persist_dir / "texts.txt"
		if not texts_file.exists():
			return 0
		texts = texts_file.read_text(encoding="utf-8").split("\n\n")
		# Load docs metadata if present
		meta_file = self.persist_dir / "docs.json"
		metas = []
		if meta_file.exists():
			import json

			metas = json.loads(meta_file.read_text(encoding="utf-8"))

		self.records = [TextRecord(id=m["id"] if metas else str(i), text=t, meta=(m["meta"] if metas else {})) for i, (m, t) in enumerate(zip(metas or [{}] * len(texts), texts))]

		if self.model is None:
			# Nothing to compute; return count
			return len(self.records)

		# compute embeddings
		self.embs = self.model.encode([r.text for r in self.records], convert_to_numpy=True)
		# persist
		import joblib

		joblib.dump(self.embs, self.persist_dir / "embeddings.joblib")
		return len(self.records)

	def search(self, q: str, k: int = 5) -> List[Tuple[TextRecord, float]]:
		if self.embs is None:
			# try to load
			import joblib

			if (self.persist_dir / "embeddings.joblib").exists():
				self.embs = joblib.load(self.persist_dir / "embeddings.joblib")
		if self.model is None:
			raise RuntimeError("sentence-transformers not installed; install to use embedding features")
		q_emb = self.model.encode([q], convert_to_numpy=True)[0]
		sims = (self.embs @ q_emb) / (np.linalg.norm(self.embs, axis=1) * np.linalg.norm(q_emb) + 1e-12)
		idx = sims.argsort()[::-1][:k]
		return [(self.records[i], float(sims[i])) for i in idx]

