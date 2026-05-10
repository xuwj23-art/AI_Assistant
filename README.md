# 📚 AI Research Assistant

> An interactive literature exploration system for researchers — search, cluster, visualise, and chat about academic papers in real time.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [FAQ](#faq)

---

## Overview

An end-to-end intelligent research literature platform that helps researchers discover topics, identify trends, and hold multi-turn AI conversations grounded in paper evidence.

**Core workflow**:

```
User enters keywords
    ↓
Online paper fetching (arXiv + OpenAlex, parallel)
    ↓
Semantic embedding (Sentence Transformers all-mpnet-base-v2)
    ↓
Automatic topic clustering (BERTopic)
    ↓
Interactive visualisation (Topic Distribution + Trends)
    ↓
RAG Q&A (FAISS retrieval + DeepSeek multi-turn chat)
```

---

## Features

### 🔍 Multi-source Online Fetching
- Supports **arXiv** (CS/AI preprints) and **OpenAlex** (250M+ cross-discipline papers)
- Parallel fetching with 3-level deduplication: DOI → arXiv ID → normalised title
- **Smart source routing** (`smart_router.py`): embeds the query, matches it to 16 discipline prototypes, and dynamically adjusts per-source fetch weights for better recall

### 🧠 Automatic Topic Clustering
- Unsupervised topic discovery with **BERTopic** on paper abstracts
- Dynamic `min_topic_size` adapts to dataset scale
- **LLM topic naming** (Gemini Flash Lite free tier): rewrites keyword sequences into readable academic titles (e.g. `Cross-Lingual Transfer Using Multilingual BERT`); falls back to a rule-based formatter when no API key is set

### 📊 Interactive Visualisation
- **Topic Distribution**: horizontal bar chart of paper counts per topic, click to browse papers
- **Research Trends**: line chart of annual paper counts per topic, auto-highlights fastest-growing areas
- **Related Topics**: cosine similarity on topic centroids (Jaccard keyword fallback when vectors unavailable)

### 💬 RAG Multi-turn Chat (DeepSeek)
- **FAISS** vector index with L2-normalised cosine similarity
- True multi-turn context: last 6 conversation turns passed to DeepSeek in OpenAI message format
- Fact-grounded answers with `[1] [2]` citation numbers
- Automatic follow-up query rewriting for context-dependent questions
- Session management: clear chat, new session, example questions
- Local T5 fallback when DeepSeek is unavailable

### 📥 PDF Download
- Backend proxy endpoint `/api/papers/{id}/pdf` — fetches and caches to `data/pdfs/`, then streams as an attachment
- Also supports `mode=redirect` for direct 302 redirect to source PDF
- 50 MB single-file cap

### 📂 Search History
- All previous search sessions (CSV + embeddings) are retained and listed in the sidebar
- One-click switch between historical datasets — no re-search needed
- Session context automatically updates when switching datasets

---

## Tech Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Frontend** | Streamlit | 1.31+ | Interactive Web UI |
| **Frontend** | Plotly | 5.18+ | Data visualisation |
| **Backend** | FastAPI | 0.109+ | REST API server |
| **Backend** | Uvicorn | 0.27+ | ASGI server |
| **NLP** | Sentence Transformers | 2.3+ | Text embedding (all-mpnet-base-v2) |
| **NLP** | BERTopic | 0.16+ | Topic modelling |
| **NLP** | UMAP | 0.5+ | Dimensionality reduction |
| **NLP** | HDBSCAN | 0.8+ | Density clustering |
| **Vector Search** | FAISS | 1.7+ | High-speed vector similarity search |
| **Data Sources** | arxiv SDK | 2.1+ | arXiv API client |
| **Data Sources** | httpx | 0.26+ | OpenAlex HTTP client |
| **Data Processing** | Pandas | 2.2+ | DataFrame operations |
| **AI Summarisation** | Transformers | 4.41+ | Local T5/BART summariser |
| **Deep Learning** | PyTorch | 2.1+ | Model inference backend |
| **LLM** | DeepSeek API | — | Multi-turn RAG chat |
| **LLM** | Gemini API | — | Topic naming (free tier) |

---

## Project Structure

```
AIassistant_v2/
├── app.py                          # Streamlit frontend entry point
├── requirements.txt                # pip dependencies
├── requirements-conda.txt          # conda-managed dependencies
├── env.example                     # environment variable template
│
├── src/
│   ├── main.py                     # FastAPI backend entry point
│   │
│   └── core/
│       ├── config.py               # Global path constants
│       ├── auto_fetch.py           # On-demand pipeline orchestrator (core)
│       │
│       ├── sources/                # Multi-source data layer
│       │   ├── base.py             # Unified Paper model + PaperSource interface
│       │   ├── arxiv_adapter.py    # arXiv source adapter
│       │   ├── aggregator.py       # Parallel fetch + 3-level dedup + weight merge
│       │   └── smart_router.py     # Semantic query → discipline → source weights
│       │
│       ├── arxiv/                  # arXiv SDK wrapper
│       │   ├── client.py
│       │   ├── model.py
│       │   └── pipeline.py
│       │
│       ├── openalex/               # OpenAlex REST API client
│       │   └── client.py
│       │
│       ├── llm/                    # Unified LLM interface
│       │   └── provider.py         # Gemini + DeepSeek + rule-based fallback
│       │
│       ├── nlp/                    # NLP processing modules
│       │   ├── embeddings.py       # Sentence Transformer embedding generator
│       │   ├── topic_modeling.py   # BERTopic wrapper
│       │   ├── topic_namer.py      # Topic naming facade (delegates to LLMProvider)
│       │   ├── topic_similarity.py # Topic centroid + cosine similarity
│       │   ├── trend_analysis.py   # Year-over-year trend computation
│       │   ├── rag.py              # RAG Q&A service (FAISS + DeepSeek multi-turn)
│       │   └── summarizer.py       # Local AI summariser (T5/BART)
│       │
│       ├── downloader/             # PDF download & local cache
│       │   └── pdf_downloader.py
│       │
│       └── api/                    # FastAPI route layer
│           ├── models.py           # Pydantic request/response models
│           ├── routes.py           # Paper management (/api/papers)
│           ├── search_routes.py    # Search entry + history (/api/search, /api/history)
│           ├── topic_routes.py     # Topic management (/api/topics)
│           ├── rag_routes.py       # RAG chat (/api/chat)
│           ├── topic_service.py    # Topic business logic
│           ├── chat_service.py     # Chat session business logic
│           ├── services.py         # Paper business logic
│           ├── paper_utils.py      # Shared paper utilities
│           └── dependencies.py     # FastAPI dependency injection
│
├── scripts/
│   └── regenerate_topic_labels.py  # Utility: re-run LLM naming on existing data
│
├── data/
│   ├── raw/                        # Raw paper CSVs (gitignored)
│   ├── processed/                  # Topic-labelled CSVs + embeddings .npy (gitignored)
│   └── pdfs/                       # Downloaded PDF cache (gitignored)
│
├── models/                         # BERTopic model files (gitignored)
└── tests/                          # Test suite
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Anaconda / Miniconda (recommended)
- RAM: ≥ 8 GB (embedding + topic modelling)
- Disk: ≥ 5 GB (model files ~420 MB + data)

### 1. Create the Conda Environment

```bash
conda create -n literature_review python=3.10 -y

# Install conda-managed core packages first (avoids dependency conflicts)
conda install -n literature_review -c conda-forge hdbscan umap-learn scikit-learn numpy pandas -y

conda activate literature_review
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: On first run, `sentence-transformers` will download `all-mpnet-base-v2` (~420 MB).  
> The app is pre-configured to use `hf-mirror.com` as a HuggingFace mirror if direct access is slow.

### 3. Configure API Keys (Optional but Recommended)

Copy the template and fill in your keys:

```bash
cp env.example .env
```

| Key | Service | Purpose |
|-----|---------|---------|
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) | Multi-turn RAG chat |
| `GEMINI_API_KEY` | [ai.google.dev](https://ai.google.dev) | LLM topic naming (free tier) |

Both keys are optional — the system falls back gracefully to local models.

### 4. Start the Backend

```bash
cd src
python main.py
# API server: http://127.0.0.1:8000
# Swagger docs: http://127.0.0.1:8000/docs
```

### 5. Start the Frontend

```bash
# In a new terminal, from the project root
streamlit run app.py
# UI: http://localhost:8501
```

### 6. First Search

1. Open `http://localhost:8501`
2. Enter a research keyword (e.g. `transformer attention mechanism`)
3. Click **🔍 Search** and wait ~1–3 minutes for fetching + analysis
4. Explore the three tabs: **Topic Distribution** / **Research Trends** / **AI Chat**

---

## Usage Guide

### Search & Data Fetching

The pipeline runs 5 steps automatically after you click Search:

| Step | Action | Output |
|------|--------|--------|
| 1 | Fetch papers online | Parallel arXiv + OpenAlex |
| 2 | Save raw data | `data/raw/arxiv_<query>.csv` |
| 3 | Generate embeddings | `all-mpnet-base-v2` on abstracts |
| 4 | Train topic model | BERTopic clusters + LLM naming |
| 5 | Save processed data | `data/processed/arxiv_<query>_with_topics.csv` + `.npy` |

**Advanced Options** (click ⚙️):
- **Max papers**: 10–200 (default 50; more = better topics but slower)
- **Data sources**: arXiv (default), OpenAlex (optional, broader coverage)

### Search History

All previous searches are saved locally. Use the **Search History** panel in the sidebar to:
- See all previous datasets (keyword, paper count, topic count, date)
- Click **📂 Switch to This Dataset** to instantly load any past result

### Topic Distribution

- Click **🔄 Load Topic Distribution** to render the bar chart
- Select a topic from the dropdown to browse its papers on the right
- Sort by **Topic Relevance** (cosine distance to centroid) or **Date (Newest)**
- Navigate pages and view related topic suggestions at the bottom

### Research Trends

- Click **🔄 Load Trend Data** to render the line chart
- Orange badges at the top show the fastest-growing research directions
- Adjust the number of topics shown with the slider
- Expand **📋 View Raw Data Table** for exact counts by year

### AI Chat

- Chat session is automatically initialised on first visit to the tab
- Ask any question about the papers — answers cite specific papers with `[1]` numbers
- Each cited paper is expandable with abstract, authors, and PDF link
- Use **🗑️ Clear Chat** to reset messages, **🔄 New Session** for a fresh context

---

## API Reference

Full Swagger UI available at `http://127.0.0.1:8000/docs` when the backend is running.

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/search` | Run full pipeline (fetch → embed → cluster) |
| `GET` | `/api/search/status` | Current data status |
| `GET` | `/api/history` | List all saved search datasets |
| `POST` | `/api/history/switch` | Switch active dataset |
| `GET` | `/api/topics` | All topics (sorted by paper count) |
| `GET` | `/api/topics/sunburst` | Topic distribution chart data |
| `GET` | `/api/topics/trends` | Year-over-year trend data |
| `GET` | `/api/topics/{id}/papers` | Paginated paper list for a topic |
| `GET` | `/api/topics/{id}/similar` | Related topic recommendations |
| `GET` | `/api/papers/{id}/pdf` | Download / redirect to paper PDF |
| `POST` | `/api/chat/init` | Initialise a chat session |
| `POST` | `/api/chat/message` | Send message, get RAG answer |
| `DELETE` | `/api/chat/history` | Clear session history |
| `GET` | `/api/chat/llm-status` | Active LLM model info |

---

## FAQ

**Q: How long does the first search take?**  
The first run downloads `all-mpnet-base-v2` (~420 MB). After that, a 50-paper pipeline takes 1–3 minutes.

**Q: OpenAlex fetching fails?**  
OpenAlex requires access to `api.openalex.org`. If the network is blocked, uncheck OpenAlex and use arXiv only.

**Q: Very few or zero topics?**  
BERTopic needs enough papers to form clusters. Try increasing the fetch count to 50+.

**Q: Chat shows "initialisation failed"?**  
A search must be completed first — the RAG service needs data to query. Run a search, then switch to the AI Chat tab.

**Q: `conda activate` doesn't work in PowerShell?**  
```powershell
# Run once as administrator
conda init powershell
# Then close and reopen the terminal
conda activate literature_review
```

**Q: DeepSeek/Gemini keys — where do I get them?**  
- DeepSeek: [platform.deepseek.com](https://platform.deepseek.com) — paid, ~$0.001/1K tokens
- Gemini: [ai.google.dev](https://ai.google.dev) — free tier available (Gemini Flash Lite)

---

## References

- [BERTopic paper](https://arxiv.org/abs/2203.05794) — Grootendorst, M. (2022)
- [Sentence-BERT paper](https://arxiv.org/abs/1908.10084) — Reimers & Gurevych (2019)
- [OpenAlex API docs](https://docs.openalex.org/)
- [FAISS docs](https://faiss.ai/)

---

## Supervisor

Dr. XU Lingling

## License

MIT
