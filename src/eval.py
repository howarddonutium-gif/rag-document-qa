import json
from sentence_transformers import SentenceTransformer
from retrieve import load_index_and_chunks, retrieve

def load_eval_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_hit(retrieved_chunks, keywords):
    """A 'hit' means ALL keywords appear somewhere in the combined retrieved text
    (case-insensitive). Adjust to 'any' if you want a looser check."""
    combined_text = " ".join([r["chunk"] for r in retrieved_chunks]).lower()
    return all(kw.lower() in combined_text for kw in keywords)

def run_eval(eval_set, index, chunks, model, top_k=5):
    results = []
    hits = 0

    for item in eval_set:
        retrieved = retrieve(item["question"], index, chunks, model, top_k=top_k)
        hit = check_hit(retrieved, item["answer_location_keywords"])
        hits += hit

        results.append({
            "question": item["question"],
            "hit": hit,
            "top_chunk_preview": retrieved[0]["chunk"][:150]
        })

    recall_at_k = hits / len(eval_set)
    return recall_at_k, results

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_index_and_chunks()
    eval_set = load_eval_set("docs/eval_set.json")

    TOP_K = 5
    recall, results = run_eval(eval_set, index, chunks, model, top_k=TOP_K)

    print(f"\n=== Recall@{TOP_K}: {recall:.1%} ({int(recall * len(eval_set))}/{len(eval_set)}) ===\n")

    for r in results:
        status = "✓ HIT " if r["hit"] else "✗ MISS"
        print(f"[{status}] {r['question']}")
        if not r["hit"]:
            print(f"         Top chunk: {r['top_chunk_preview']}...")
    print()