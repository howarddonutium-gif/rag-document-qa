from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

from chunking import load_text, chunk_text

def build_index(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    # Convert all chunks into vectors
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # Build a FAISS index (flat = exact search, fine for our scale)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))

    return index, model

if __name__ == "__main__":
    text = load_text("docs/apple_10k.txt")
    chunks = chunk_text(text)

    index, model = build_index(chunks)

    print(f"Index built with {index.ntotal} vectors")
    print(f"Vector dimension: {index.d}")

    # Save everything so we don't have to re-embed every time
    faiss.write_index(index, "docs/apple_10k.faiss")
    with open("docs/apple_10k_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Saved index and chunks to docs/")