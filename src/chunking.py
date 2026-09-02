from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    text = load_text("docs/apple_10k.txt")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
    print("\n--- Sample chunk (#10) ---")
    print(chunks[10])