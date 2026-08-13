# NCERT Biology Teacher — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that teaches NCERT Class 12
Biology — grounded in the actual textbook (text **and** diagrams), not
generic knowledge. Built on Gemini via the free Google AI Studio Developer
API — no cloud billing account needed, just a free API key.

Ask it a question and it retrieves the relevant textbook passages (and
diagrams, when helpful), then explains the answer like a patient teacher —
citing exactly which chapter and page it used, and honestly saying so when
the book doesn't cover something instead of guessing.

## Features

- **Grounded answers, not hallucinations** — every answer cites the
  chapter/page it came from; the model is instructed to say so honestly
  when the retrieved content doesn't actually answer the question.
- **Diagrams shown automatically** — when an answer centers on a labeled
  structure or process, the relevant textbook diagram is shown alongside
  the explanation, without the student needing to ask for it. When a page
  has several diagrams, the specific one shown is re-ranked against the
  question's own wording, not just "whichever image is listed first."
- **Conversation memory** — follow-ups like "explain that more simply"
  correctly resolve to what was just discussed.
- **Streaming responses** — answers appear as they're generated, not after
  a long silent wait.
- **Chapter-scoped search** — optionally restrict retrieval to one chapter.
- **Three independent image-retrieval strategies** (`image_handling/`),
  each in its own file, each addressing a different weak point of the last:
  1. **Caption retrieval** — Gemini captions every diagram once at index
     time; retrieval ranks by caption-text similarity.
  2. **Direct image embedding** — no captions at all; the diagram's raw
     pixels are embedded into a joint text+image space and compared
     directly against the question.
  3. **Multimodal answer generation** — the actual retrieved image is
     handed to Gemini alongside the question at answer time, so it reasons
     freshly about that specific image for that specific question.
  4. **Combined pipeline** — merges 1 and 2 for retrieval, then verifies
     each candidate with 3 before committing to an answer ("retrieve, then
     verify"), recovering from cases where either signal alone is wrong.
- **True incremental indexing** — re-indexing a changed chapter adds new
  chunks, skips unchanged ones, and deletes stale ones (via LangChain's
  `SQLRecordManager`), not just a blind re-embed-everything.
- **Objectively evaluated**, not just eyeballed — see [Evaluation](#evaluation).

## Architecture

```
NCERT chapter PDFs
      │
      ▼
┌─────────────────┐   PyMuPDF: per-page text + precisely-cropped
│  Ingestion       │   diagram images (noise-filtered, not whole pages)
└─────────────────┘
      │
      ▼
┌─────────────────┐   Gemini captions every diagram once, cached to
│  Captioning      │   disk (a second-pass "USEFUL: yes/no" marker drops
└─────────────────┘   fragments that slipped past the geometric filters)
      │
      ▼
┌─────────────────┐   Page text + folded-in captions → ~1000-char
│  Chunking        │   overlapping chunks, tagged with book/chapter/page/
└─────────────────┘   image-path metadata
      │
      ▼
┌─────────────────┐   gemini-embedding-2 → Chroma (persisted vector DB),
│  Embed + Index   │   tracked by SQLRecordManager for true incremental
└─────────────────┘   re-indexing
      │
      ▼
┌─────────────────┐   question → retrieve top-k chunks → teacher-persona
│  RAG Chain       │   prompt → Gemini (streaming) → grounded, cited answer
└─────────────────┘   + the relevant diagram, shown automatically
      │
      ▼
┌─────────────────┐   Flask + vanilla JS, streams the answer live
│  Web UI          │
└─────────────────┘
```

## Tech stack

Everything runs on **Gemini, via the free Google AI Studio Developer
API** — one provider, one free API key, no cloud billing account required:

| Purpose | Model / Tool |
|---|---|
| Chat + captioning + judge (eval) | `gemini-flash-latest` |
| Text + image embeddings (one unified model) | `gemini-embedding-2` |
| Vector database | ChromaDB (persisted locally) |
| Incremental indexing | LangChain `SQLRecordManager` |
| PDF parsing / rendering | PyMuPDF |
| Web framework | Flask (+ gunicorn for production) |
| Evaluation | DeepEval (Faithfulness, Answer Relevancy, Contextual Precision/Recall) |
| Environment / packaging | `uv` |

`gemini-embedding-2` is natively multimodal — text chunks, image captions,
and raw diagram pixels all embed into the same 3072-dim space, so a single
model/config covers both plain-text retrieval and Approach 2's direct
image-pixel search (an earlier Vertex AI version of this project needed two
separate, never-comparable embedding models for that; the Developer API
doesn't have that limitation).

## Setup

### 1. Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- A free Gemini API key — get one at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (no
  credit card, not tied to any cloud billing account)

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY to your key
```

### 4. Get the textbook data

Place the 13 NCERT Class 12 Biology chapter PDFs under
`data/raw_pdfs/class12_biology/chapter_01.pdf` … `chapter_13.pdf`
(`chapters.json` in that folder is already tracked in this repo and maps
each filename to its real chapter title).

### 5. Build the index (one-time, or after changing the PDFs)

```bash
uv run python scripts/index_all_chapters.py
```

This extracts diagrams, captions them with Gemini, chunks the text, embeds
everything, and indexes it into Chroma. It's resumable — re-running after an
interruption picks up from cached progress rather than starting over or
re-doing anything already done. Genuinely free on the Developer API's free
tier (rate-limited, not billed) — expect it to pace itself around those
limits rather than run instantly.

### 6. Run the app

```bash
uv run python app/server.py
```

Then open **http://127.0.0.1:5000/**.

## Evaluation

Run the objective quality check (5 real questions through the actual RAG
chain, scored by DeepEval using Gemini as the judge):

```bash
uv run python -m rag_ncert_biology_teacher.eval.run_eval
```

## Project structure

```
src/rag_ncert_biology_teacher/
├── config.py              # single source of truth for paths/settings/models
├── retry_utils.py         # shared exponential-backoff retry for every Google API call
├── ingestion/
│   ├── pdf_loader.py       # per-page text + precisely-cropped diagram extraction
│   ├── text_extraction.py  # cached, cheap page-text-only extraction (for chunking)
│   ├── captioning.py       # Gemini diagram captioning + USEFUL marker
│   └── chunking.py         # page text + captions → retrieval-sized chunks
├── image_handling/         # the 3 independent image-retrieval approaches + combined pipeline
├── indexing/
│   ├── embeddings.py       # Gemini Developer API text + image embedding wrappers (one unified model)
│   ├── vectorstore.py      # Chroma persistence
│   └── indexer.py          # SQLRecordManager incremental indexing
├── rag/
│   ├── retriever.py         # thin Chroma similarity-search wrapper
│   ├── prompts.py           # teacher-persona system prompt, marker parsing
│   └── chain.py              # retrieval + generation, streaming, conversation memory
├── bootstrap.py            # optional: downloads a pre-built index from GCS at startup (deploy-only)
└── eval/
    ├── gemini_judge.py      # DeepEval judge model (Gemini, free API key)
    └── run_eval.py           # the evaluation script above

app/                        # Flask web UI (server.py, templates/, static/)
scripts/                    # batch scripts (index_all_chapters.py)
data/raw_pdfs/               # source PDFs + chapters.json manifest (PDFs not tracked in git)
data/extracted/              # generated: extracted text/images/captions (not tracked, regenerable)
data/chroma/                  # generated: the vector database (not tracked, regenerable)
```

## Deployment (Render)

The Flask app is a standard WSGI app (`gunicorn app.server:app`, reads
`$PORT`). Auth is trivial now — no service account, no IAM roles, no cloud
billing setup: just set the same `GEMINI_API_KEY` env var Render as any
other config value.

**The one real thing to solve: data.** `data/raw_pdfs/*.pdf`,
`data/extracted/`, `data/chroma/`, and `data/record_manager.sqlite3` are all
git-ignored on purpose (large and/or regenerable) — so a fresh `git`-based
deploy won't have them, and a PaaS's container filesystem is otherwise
wiped on every redeploy (and, on a free tier with no persistent disk, on
every cold start too). Three ways to solve that:

- **Rebuild directly on the server** (simplest now that it's free): upload
  the raw PDFs however you like, then run
  `uv run python scripts/index_all_chapters.py` once after deploying. No
  cost beyond the free tier's rate limits, since captioning/embedding no
  longer needs paid Vertex AI calls.
- **Paid tier with a persistent disk**: add a Render disk, set `DATA_DIR` to
  its mount path, and copy your already-built `data/` folder onto it once
  (e.g. via Render's Shell + any storage you have access to).
- **Free tier, download at every cold start** (what `bootstrap.py`
  implements): downloads a pre-built `data/chroma/` + `data/extracted/` +
  `record_manager.sqlite3` from a GCS bucket on startup — entirely optional,
  unrelated to the AI features, and skippable if you'd rather not touch GCP
  at all (see `.env.example`'s `GCS_DATA_BUCKET` comment for the full
  tradeoff). Leave it unset to skip this path entirely.

**Service setup:**
- New → Web Service → connect this GitHub repo
- Environment: Python, with env var `PYTHON_VERSION=3.13.1` (or newer 3.13.x)
- Build command: `pip install uv && uv sync --frozen`
- Start command: `uv run gunicorn --bind 0.0.0.0:$PORT app.server:app`
- Environment variables: `GEMINI_API_KEY`, `GEMINI_CAPTIONING_MODEL`,
  `LLM_PROVIDER`, `LLM_MODEL` (see `.env.example`), plus `DATA_DIR` or
  `GCS_DATA_BUCKET` depending on which data option you picked above

Running locally is unaffected by any of this — `DATA_DIR` and
`GCS_DATA_BUCKET` are both optional and only need setting in production.
