# NCERT Biology Teacher — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that teaches NCERT Class 12
Biology — grounded in the actual textbook (text **and** diagrams), not
generic knowledge. Built entirely on Google Cloud (Vertex AI / Gemini).

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
  the explanation, without the student needing to ask for it.
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
┌─────────────────┐   Vertex AI text-embedding-005 → Chroma (persisted
│  Embed + Index   │   vector DB), tracked by SQLRecordManager for true
└─────────────────┘   incremental re-indexing
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

Everything runs on **Google Cloud / Vertex AI** — one provider throughout,
no other API keys needed:

| Purpose | Model / Tool |
|---|---|
| Chat + captioning + judge (eval) | `gemini-2.5-flash` |
| Text embeddings | `text-embedding-005` |
| Multimodal (image) embeddings | `gemini-embedding-2` |
| Vector database | ChromaDB (persisted locally) |
| Incremental indexing | LangChain `SQLRecordManager` |
| PDF parsing / rendering | PyMuPDF |
| Web framework | Flask (+ gunicorn for production) |
| Evaluation | DeepEval (Faithfulness, Answer Relevancy, Contextual Precision/Recall) |
| Environment / packaging | `uv` |

## Setup

### 1. Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- A Google Cloud project with the Vertex AI API enabled and billing linked
- `gcloud auth application-default login` run once on your machine (no API
  key files needed anywhere — auth is via Application Default Credentials)

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure

```bash
cp .env.example .env
# then edit .env and set GOOGLE_CLOUD_PROJECT to your GCP project id
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
re-paying for anything already done.

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

Latest results:

| Metric | Avg Score | Pass Rate |
|---|---|---|
| Faithfulness | 0.98 | 100% |
| Answer Relevancy | 0.95 | 100% |
| Contextual Precision | 0.92 | 100% |
| Contextual Recall | 0.93 | 80% |

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
│   ├── embeddings.py       # Vertex AI text + multimodal embedding wrappers
│   ├── vectorstore.py      # Chroma persistence
│   └── indexer.py          # SQLRecordManager incremental indexing
├── rag/
│   ├── retriever.py         # thin Chroma similarity-search wrapper
│   ├── prompts.py           # teacher-persona system prompt, marker parsing
│   └── chain.py              # retrieval + generation, streaming, conversation memory
└── eval/
    ├── gemini_judge.py      # DeepEval judge model (Gemini via Vertex AI)
    └── run_eval.py           # the evaluation script above

app/                        # Flask web UI (server.py, templates/, static/)
scripts/                    # batch scripts (index_all_chapters.py)
data/raw_pdfs/               # source PDFs + chapters.json manifest (PDFs not tracked in git)
data/extracted/              # generated: extracted text/images/captions (not tracked, regenerable)
data/chroma/                  # generated: the vector database (not tracked, regenerable)
```

## Deployment (Render)

The Flask app is a standard WSGI app (`gunicorn app.server:app`, reads
`$PORT`), but two things need real attention before it'll work on Render or
any similar PaaS -- both are about things that are true and necessary on
your own machine but don't exist by default on a fresh container:

**1. Auth.** Locally this project authenticates via
`gcloud auth application-default login` -- your own interactive Google
login. A Render container can't do that. Instead:
1. GCP Console → IAM & Admin → Service Accounts → create one, grant it the
   **Vertex AI User** role (`roles/aiplatform.user`).
2. Create a JSON key for it, download it.
3. In Render: your service → Environment → **Secret Files** → add the key's
   contents as a file (e.g. path `/etc/secrets/gcp-key.json`).
4. Add an env var `GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/gcp-key.json`
   -- `google-genai`'s Vertex AI client picks this up automatically.

**2. Data.** `data/raw_pdfs/*.pdf`, `data/extracted/`, `data/chroma/`, and
`data/record_manager.sqlite3` are all git-ignored on purpose (large and/or
regenerable) -- which also means a fresh `git`-based deploy won't have them,
and a PaaS's container filesystem is otherwise wiped on every redeploy (and,
on a free tier with no persistent disk, on every cold start too). The
chatbot can't answer anything without the already-built index, so this data
has to come from somewhere at startup. Two ways to solve that -- pick one:

- **Paid tier with a persistent disk** (simplest, no re-download ever):
  1. Render → your service → **Disks** → add a disk (1-2 GB is plenty --
     the built index is under 50 MB), mounted at a path of your choice,
     e.g. `/var/data`.
  2. Set the env var `DATA_DIR=/var/data` (see `.env.example`) -- everything
     in `config.py` resolves from this instead of `<repo>/data`.
  3. Get your already-built `data/` folder onto that disk **once**: upload
     it to a GCS bucket you own, then from Render's Shell,
     `gcloud storage cp -r gs://your-bucket/data/* /var/data/`. It persists
     across redeploys after that.

- **Free tier, download at every cold start** (what this repo actually
  implements, via `bootstrap.py`): Render's Free tier has no Disks/Shell at
  all, so there's nowhere to put a one-time copy. Instead, `app/server.py`
  calls `bootstrap.ensure_data_present()` on startup, which downloads the
  built index (`data/chroma/` + `data/extracted/` +
  `record_manager.sqlite3` -- ~30MB, **not** the ~160MB raw PDFs, which are
  only used for decorative thumbnails and skipped gracefully if absent)
  from a GCS bucket, in parallel, taking roughly 30-60s on a cold start.
  1. Upload your local `data/` folder once: `gcloud storage cp -r data
     gs://your-bucket/`.
  2. Grant your service account read access:
     `gcloud storage buckets add-iam-policy-binding gs://your-bucket
     --member="serviceAccount:YOUR_SA@..." --role="roles/storage.objectViewer"`.
  3. Set the env var `GCS_DATA_BUCKET=your-bucket` (see `.env.example`).
     Leave `DATA_DIR` unset -- the default `<repo>/data` inside the
     container is fine since it's ephemeral either way.
  4. If the service ever cold-starts before the disk-based flow above is
     wired up, `ensure_data_present()` is a no-op locally (unset
     `GCS_DATA_BUCKET`) and idempotent per-boot (skips re-downloading if
     `data/chroma/` already has files, e.g. a gunicorn worker restart
     within an already-warm container).

**Then set up the service itself:**
- New → Web Service → connect this GitHub repo
- Environment: Python, with env var `PYTHON_VERSION=3.13.1` (or newer 3.13.x)
- Build command: `pip install uv && uv sync --frozen`
- Start command: `uv run gunicorn --bind 0.0.0.0:$PORT app.server:app`
- Environment variables: everything in `.env.example`
  (`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GEMINI_CAPTIONING_MODEL`,
  `MULTIMODAL_EMBEDDING_MODEL_NAME`, `MULTIMODAL_EMBEDDING_LOCATION`,
  `LLM_PROVIDER`, `LLM_MODEL`) plus `GOOGLE_APPLICATION_CREDENTIALS` from
  the Auth section above, and either `DATA_DIR` or `GCS_DATA_BUCKET`
  depending on which Data option you picked

Running locally is unaffected by any of this -- `DATA_DIR`,
`GCS_DATA_BUCKET`, and `GOOGLE_APPLICATION_CREDENTIALS` are all optional and
only need setting in production.
