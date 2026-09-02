from rank_bm25 import BM25Okapi
from retrieve import load_index_and_chunks, retrieve
from sentence_transformers import SentenceTransformer

def build_bm25_index(chunks):
    tokenized_chunks = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenized_chunks)

def hybrid_retrieve(query, index, chunks, model, bm25, top_k=5, alpha=0.5):
    """alpha=1.0 is pure vector, alpha=0.0 is pure BM25, 0.5 is balanced."""
    # Vector search — get more candidates than top_k to allow reranking
    vector_results = retrieve(query, index, chunks, model, top_k=len(chunks))
    vector_scores = {r["chunk"]: 1 / (1 + r["distance"]) for r in vector_results}

    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    bm25_scores = {chunks[i]: score for i, score in enumerate(bm25_scores_raw)}

    # Normalize both score sets to 0-1 range
    def normalize(scores_dict):
        vals = list(scores_dict.values())
        if max(vals) == min(vals):
            return {k: 0.0 for k in scores_dict}
        lo, hi = min(vals), max(vals)
        return {k: (v - lo) / (hi - lo) for k, v in scores_dict.items()}

    vector_norm = normalize(vector_scores)
    bm25_norm = normalize(bm25_scores)

    # Combine
    combined = {}
    for chunk in chunks:
        combined[chunk] = alpha * vector_norm.get(chunk, 0) + (1 - alpha) * bm25_norm.get(chunk, 0)

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"chunk": c, "score": s} for c, s in ranked]

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_index_and_chunks()
    bm25 = build_bm25_index(chunks)

    query = "How many full-time equivalent employees did Apple have?"
    results = hybrid_retrieve(query, index, chunks, model, bm25, top_k=5)

    for i, r in enumerate(results):
        print(f"--- Rank {i+1} (score: {r['score']:.4f}) ---")
        print(r["chunk"][:200])
        print()