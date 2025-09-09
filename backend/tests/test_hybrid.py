from pathlib import Path
from backend.app.core.hybrid import HybridRetriever
from backend.app.core.vector_store import TfidfStore, Document

INDEX_DIR = Path(__file__).resolve().parents[2] / 'app' / 'index'

# Prepare minimal TF-IDF index fixture quickly (no pytest fixtures to keep lightweight)

def _ensure_index():
    store = TfidfStore(INDEX_DIR)
    if not (INDEX_DIR / 'docs.json').exists():
        store.add_texts([
            Document(id='a', text='contract law agreement obligation', meta={}),
            Document(id='b', text='arbitration clause dispute resolution', meta={}),
            Document(id='c', text='termination breach remedy damages', meta={}),
        ])
        store.build()


def test_hybrid_retrieval_basic():
    _ensure_index()
    retriever = HybridRetriever(INDEX_DIR, alpha=0.5)
    hits = retriever.retrieve('arbitration dispute', k=2)
    assert len(hits) <= 2
    # Ensure scores present
    for h in hits:
        assert 0.0 <= h.score_tfidf <= 1.0
        assert 0.0 <= h.score_embed <= 1.0 or h.score_embed == 0.0
        assert 0.0 <= h.score_final <= 1.0
