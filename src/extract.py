from pypdf import PdfReader

reader = PdfReader("docs/apple_10k.pdf")

full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

print(f"Total characters extracted: {len(full_text)}")
print(f"Total pages: {len(reader.pages)}")

with open("docs/apple_10k.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Saved to docs/apple_10k.txt")