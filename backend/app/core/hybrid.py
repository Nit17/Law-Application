from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .vector_store import TfidfStore, Document

try:
	from .embedding_store import EmbeddingStore, TextRecord
except Exception:
	EmbeddingStore = None  # type: ignore


class HybridRetriever:
	def __init__(self, persist_dir: Path):
		self.persist_dir = Path(persist_dir)
		self.tf = TfidfStore(persist_dir)
		self.emb = EmbeddingStore(persist_dir) if EmbeddingStore is not None else None

	def query(self, q: str, k: int = 5) -> List[Tuple[Document, float]]:
		# Get TF-IDF results
		tf_res = self.tf.query(q, k=k)
		# Try embeddings if available and merge scores by simple average where ids match
		if self.emb is None:
			return tf_res

		try:
			emb_res = self.emb.search(q, k=k)
		except RuntimeError:
			return tf_res

		# Map by id
		combined = {}
		for d, s in tf_res:
			combined[d.id] = (d, float(s), 1)
		for r, s in emb_res:
			# r is TextRecord - try to match by id
			rid = r.id
			if rid in combined:
				d, s0, cnt = combined[rid]
				combined[rid] = (d, (s0 + float(s)) / (cnt + 1), cnt + 1)
			else:
				# create a Document-like wrapper
				d = Document(id=r.id, text=r.text, meta=r.meta)
				combined[rid] = (d, float(s), 1)

		# Return top-k by score
		items = sorted(combined.values(), key=lambda x: x[1], reverse=True)[:k]
		return [(d, score) for d, score, _ in items]

