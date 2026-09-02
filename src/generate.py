import ollama
from sentence_transformers import SentenceTransformer
from retrieve import load_index_and_chunks, retrieve

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([r["chunk"] for r in retrieved_chunks])
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def answer_question(query, index, chunks, embed_model, top_k=5):
    retrieved = retrieve(query, index, chunks, embed_model, top_k=top_k)
    prompt = build_prompt(query, retrieved)

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"], retrieved

if __name__ == "__main__":
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_index_and_chunks()

    query = "What was Apple's total net sales for the fiscal year?"
    answer, retrieved = answer_question(query, index, chunks, embed_model)

    print(f"Question: {query}\n")
    print(f"Answer: {answer}\n")
    print("--- Sources used ---")
    for r in retrieved:
        print(f"[{r['rank']}] {r['chunk'][:150]}...")