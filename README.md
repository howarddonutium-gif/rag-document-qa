# RAG Document QA — SEC 10-K Filings

A Retrieval-Augmented Generation (RAG) system that answers natural language questions over SEC 10-K filings, combining dense vector search with keyword-based (BM25) retrieval to improve factual accuracy.

## Overview

Traditional RAG pipelines rely purely on dense vector similarity for retrieval, which can miss exact facts (specific numbers, percentages, named entities) that don't embed distinctly from surrounding text. This project implements a hybrid retrieval approach that combines semantic (vector) search with keyword (BM25) search, and evaluates the improvement on a hand-built question set.

**Case study document:** Apple Inc. FY2025 Form 10-K (108 pages, SEC EDGAR)

## Results

| Retrieval Method | Recall@5 |
|---|---|
| Vector search only (baseline) | 50% (5/10) |
| Hybrid (BM25 + vector) | 70% (7/10) |

Hybrid retrieval improved Recall@5 by 20 percentage points on a 10-question evaluation set built from the filing's financial statements, risk factors, and business overview sections.

**Why the improvement:** Several baseline misses were exact-fact lookups (e.g. "how many employees," "what percentage split between direct/indirect sales") where dense embeddings retrieved semantically-similar but factually-wrong chunks — often converging on the same generic "financial statements" section regardless of the specific question. BM25's keyword matching directly catches specific terms and numbers that vector search alone missed.

## Architecture
PDF Document
↓
Text Extraction (pypdf)
↓
Chunking (recursive character splitting, 500 chars, 100 overlap)
↓
├── Dense Embeddings (sentence-transformers, all-MiniLM-L6-v2) → FAISS index
└── Sparse Index (BM25, rank_bm25)
↓
Hybrid Retrieval (normalized score fusion, tunable alpha weight)
↓
Generation (Ollama, llama3.2:3b) — answers grounded in retrieved context only
## Tech Stack

| Component | Technology |
|---|---|
| PDF extraction | pypdf |
| Chunking | LangChain text splitters |
| Dense embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS |
| Sparse retrieval | BM25 (rank_bm25) |
| Generation | Ollama (llama3.2:3b, local) |
| Evaluation | Custom Recall@k harness |

## Project Structure

src/
extract.py # PDF text extraction
chunking.py # Text chunking with overlap
embeddings.py # Embedding generation + FAISS index build
retrieve.py # Dense vector retrieval
hybrid_retrieve.py # BM25 + vector hybrid retrieval
generate.py # LLM answer generation grounded in retrieved context
eval.py # Recall@k evaluation (vector-only) 
eval_hybrid.py # Recall@k evaluation (hybrid)
docs/
eval_set.json # 10 hand-written Q&A pairs with source keywords
apple_10k.pdf # Source document


## Setup & Usage

```bash
python -m venv rag-env
rag-env\Scripts\activate      # Windows
pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull a model
ollama pull llama3.2:3b

# Run the pipeline
python src/extract.py
python src/chunking.py
python src/embeddings.py
python src/generate.py         # ask a question
python src/eval.py             # baseline Recall@5
python src/eval_hybrid.py      # hybrid Recall@5
```

## Evaluation Methodology

A 10-question evaluation set was built by manually reading the filing and recording ground-truth answers alongside keywords expected to appear in the correct source chunk. Recall@5 measures whether all expected keywords for a question appear within the top-5 retrieved chunks — a proxy for "did retrieval surface the right information," independent of generation quality. -- proportion of actual positive cases a model actually identifies

## Known Limitations

- Small evaluation set (10 questions): sufficient to demonstrate the retrieval improvement, but not a comprehensive benchmark
- Fixed-size chunking (500 chars) doesn't respect document structure (tables, section boundaries): structure-aware chunking is a likely next improvement
- Single document tested: generalization to multi-document corpora untested
- Hybrid fusion uses a fixed alpha weight (0.5): could be tuned per query type

## Possible Extensions

- Cross-encoder re-ranking after hybrid retrieval
- Semantic/structure-aware chunking (split on section headers, tables)
- Multi-document support with per-document metadata filtering
- Source citation in generated answers
