import json
from sentence_transformers import SentenceTransformer
from retrieve import load_index_and_chunks
from hybrid_retrieve import build_bm25_index, hybrid_retrieve

def load_eval_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_hit(retrieved_chunks, keywords):
    combined_text = " ".join([r["chunk"] for r in retrieved_chunks]).lower()
    return all(kw.lower() in combined_text for kw in keywords)

def run_eval_hybrid(eval_set, index, chunks, model, bm25, top_k=5, alpha=0.5):
    results = []
    hits = 0

    for item in eval_set:
        retrieved = hybrid_retrieve(item["question"], index, chunks, model, bm25, top_k=top_k, alpha=alpha)
        hit = check_hit(retrieved, item["answer_location_keywords"])
        hits += hit
        results.append({"question": item["question"], "hit": hit})

    recall_at_k = hits / len(eval_set)
    return recall_at_k, results

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_index_and_chunks()
    bm25 = build_bm25_index(chunks)
    eval_set = load_eval_set("docs/eval_set.json")

    TOP_K = 5
    recall, results = run_eval_hybrid(eval_set, index, chunks, model, bm25, top_k=TOP_K, alpha=0.5)

    print(f"\n=== Hybrid Recall@{TOP_K}: {recall:.1%} ({int(recall * len(eval_set))}/{len(eval_set)}) ===\n")
    for r in results:
        status = "✓ HIT " if r["hit"] else "✗ MISS"
        print(f"[{status}] {r['question']}")