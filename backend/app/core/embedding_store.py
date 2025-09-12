from __future__ import annotations

from dataclasses import dataclass
import logging
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
		self.logger = logging.getLogger("embedding_store")
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

		# Ensure metas length matches texts; fallback to empty meta entries when missing
		if metas and len(metas) == len(texts):
			self.records = [TextRecord(id=metas[i].get("id", str(i)), text=texts[i], meta=metas[i].get("meta", {})) for i in range(len(texts))]
		else:
			# build default metas for each text
			self.records = [TextRecord(id=str(i), text=texts[i], meta={}) for i in range(len(texts))]

		if self.model is None:
			# Nothing to compute; return count
			self.logger.info("sentence-transformers not installed; skipping embedding computation")
			return len(self.records)

		# compute embeddings
		self.embs = self.model.encode([r.text for r in self.records], convert_to_numpy=True)
		self.logger.info("Computed embeddings for %d records", len(self.records))
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
		# Ensure records are loaded (texts/docs may be present even if embeddings were loaded earlier)
		if not self.records:
			# try to reconstruct records from docs/texts
			texts_file = self.persist_dir / "texts.txt"
			meta_file = self.persist_dir / "docs.json"
			if texts_file.exists():
				texts = texts_file.read_text(encoding="utf-8").split("\n\n")
			else:
				texts = []
			metas = []
			if meta_file.exists():
				import json

				metas = json.loads(meta_file.read_text(encoding="utf-8"))
			if metas and len(metas) == len(texts):
				self.records = [TextRecord(id=metas[i].get("id", str(i)), text=texts[i], meta=metas[i].get("meta", {})) for i in range(len(texts))]
			else:
				self.records = [TextRecord(id=str(i), text=texts[i], meta={}) for i in range(len(texts))]

		if self.model is None:
			raise RuntimeError("sentence-transformers not installed; install to use embedding features")
		q_emb = self.model.encode([q], convert_to_numpy=True)[0]
		# guard against mismatch in lengths between embeddings and records
		if self.embs is None or len(self.records) == 0:
			self.logger.info("No embeddings or records found; returning empty list from search")
			return []
		# If embeddings length mismatches records, trim/pad safely
		n_emb = self.embs.shape[0]
		n_rec = len(self.records)
		n = min(n_emb, n_rec)
		emb_matrix = self.embs[:n]
		sims = (emb_matrix @ q_emb) / (np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_emb) + 1e-12)
		idx = sims.argsort()[::-1][:k]
		return [(self.records[i], float(sims[j])) for j, i in enumerate(idx)]

