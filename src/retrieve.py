from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_index_and_chunks():
    index = faiss.read_index("docs/apple_10k.faiss")
    with open("docs/apple_10k_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve(query, index, chunks, model, top_k=5):
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "chunk": chunks[idx],
            "distance": float(distances[0][rank])
        })
    return results

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_index_and_chunks()

    query = "What was Apple's total revenue?"
    results = retrieve(query, index, chunks, model, top_k=5)

    print(f"Query: {query}\n")
    for r in results:
        print(f"--- Rank {r['rank']} (distance: {r['distance']:.4f}) ---")
        print(r["chunk"][:300])
        print()