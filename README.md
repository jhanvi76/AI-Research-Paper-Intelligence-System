# AI Research Paper Intelligence System

An end-to-end pipeline that turns 100,000+ arXiv machine-learning papers into a searchable, summarized, and entity-tagged knowledge base — ask a natural-language question and get back the most relevant papers, each with an auto-generated summary, extracted keywords, and identified tech-stack entities (frameworks, algorithms, datasets).

## What it does

Given a query like *"transformer-based models for time series forecasting"*, the system:

1. Retrieves the most semantically relevant papers from the corpus (not just keyword matches)
2. Summarizes each paper's abstract
3. Extracts its key phrases
4. Identifies the ML tech stack mentioned (languages, frameworks, libraries, algorithms, datasets)

## Pipeline

```
Dataset (CShorten/ML-ArXiv-Papers, 100K+ papers)
        │
        ▼
1. Extract        → pull title + abstract
2. Clean           → regex cleaning (LaTeX, URLs, symbols)
3. Preprocess       → tokenization + stopword removal (NLTK)
4. Embed            → Sentence-Transformers (contextual embeddings)
5. Index             → FAISS (cosine similarity search)
6. Search             → top-k semantic retrieval for a query
7. Summarize           → BART (facebook/bart-large-cnn)
8. Extract keywords     → KeyBERT
9. Tech-stack NER         → spaCy NER + curated ML vocabulary matcher
```

## Tech stack

`sentence-transformers` · `FAISS` · `transformers` (BART) · `KeyBERT` · `spaCy` · `NLTK` · `pandas` · `huggingface_hub`

## Project structure

```
.
├── system.py          # Core pipeline: Config, EmbeddingIndex, Summarizer,
│                       # KeywordExtractor, TechStackNER, ArxivPaperIntelligenceSystem
├── demo.ipynb          # End-to-end demo notebook (build index, run queries)
├── requirements.txt     # Pinned dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```python
from system import ArxivPaperIntelligenceSystem

system = ArxivPaperIntelligenceSystem()

# First run: builds embeddings + FAISS index from scratch, saves to disk
system.build(sample_size=2000)   # drop sample_size to index the full 100K+ dataset

# Later runs: skip re-embedding, just load the saved index
# system.load()

results = system.query("transformer-based models for time series forecasting", k=5, analyze_top=3)
system.explain(results)
```

### Example output

```
[1] Attention-based Transformer Networks for Multivariate Time Series Forecasting  (score=0.812)
    Summary : This paper proposes a transformer-based architecture for multivariate
              time series forecasting, using self-attention to capture long-range
              temporal dependencies more effectively than recurrent models.
    Keywords: time series forecasting, transformer architecture, self-attention, multivariate
    Tech stack: {'FRAMEWORK': ['PyTorch'], 'ALGORITHM': ['transformer', 'attention mechanism']}
```

## Design notes

- **Why FAISS with normalized inner product instead of cosine directly?** Normalizing embeddings at encode time and using `IndexFlatIP` gives cosine similarity at FAISS's fastest search speed.
- **Why a curated ML vocabulary matcher on top of spaCy NER?** General-purpose spaCy models (`en_core_web_sm`) are trained on newswire text and don't recognize domain terms like "PyTorch" as a framework or "BERT" as an algorithm — the hybrid approach (general NER + a `PhraseMatcher` over categorized ML vocabulary) is the standard practical workaround when no off-the-shelf model covers a specialized technical domain.
- **Why separate `build()` and `load()`?** Embedding 100K+ papers is the expensive step. `build()` runs it once and persists the FAISS index + metadata to disk; `load()` reuses them on subsequent runs.

## Possible extensions

- Swap `all-MiniLM-L6-v2` for a paper-specific embedding model (e.g. SPECTER) for stronger domain semantics
- Add a paper-comparison agent (side-by-side objectives/methodology/contributions across two papers)
- Wrap the system in a LangChain/LangGraph agent with tool-calling for conversational querying
- Serve as a small API (FastAPI) or a Streamlit demo UI

