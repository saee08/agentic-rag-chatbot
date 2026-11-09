# Agentic RAG Document Assistant

A multi-user web application for asking questions about your own documents. You upload PDF, DOCX, PPTX, CSV, TXT or Markdown files, and a bounded retrieval agent answers questions about them. Every answer cites the passages it is based on. When your documents don't contain the answer, the assistant says so instead of guessing.

- **Backend:** FastAPI, SQLAlchemy 2 and Alembic. Data is stored in PostgreSQL with pgvector in production and SQLite in development.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally on the CPU.
- **LLM:** any OpenAI-compatible chat-completions endpoint. OpenRouter is the default.
- **Frontend:** a framework-free web app served by the backend. It needs no build step.

---

## Features

| Area | What is implemented |
|---|---|
| Documents | Multi-file upload with a progress bar and background ingestion with live status (queued, processing, ready, failed). Failed documents can be retried and documents can be deleted. Re-uploading the same file is detected by content hash and skipped. |
| Parsing | **PDF:** text per page. **DOCX:** paragraphs, tables, headers and footers. **PPTX:** slides, tables, grouped shapes and speaker notes. **CSV:** self-describing row groups. **TXT/MD:** encoding fallback. Files are checked by their actual content, not just the extension (magic bytes, ZIP structure, zip-bomb limits). Size, page-count and text-length limits apply. |
| Retrieval | Sentence-aware chunks of 180 words with 30 words of overlap, each keeping its page number. Search uses cosine similarity with a calibrated relevance floor and is always restricted to the requesting user's documents. |
| Agent | A bounded decide-and-act loop with typed, permissioned tools. It judges whether the evidence is sufficient, rewrites the search query when it isn't, and stops on hard limits for iterations, tool calls, time and tokens (see [Agent](#agent-execution-flow)). |
| Answers | Numbered citations with expandable source previews (document, page, passage). Citations that point to non-existent sources are removed, and answers without any valid citation are flagged in the UI. |
| Chat | Persistent conversations with follow-up context, streamed answers over Server-Sent Events, and a collapsible "How I answered" trace. |
| Security | Accounts protected with scrypt password hashes; bearer tokens stored only as hashes and revocable on logout; every query scoped to its owner; server-generated storage filenames; request size and type limits; rate limiting; a strict CSP and security headers; prompt-injection hardening; a uniform error format that never leaks internal details. |
| Operations | Structured JSON logs with request IDs and per-stage timings (parse, embed, search, LLM, token usage); `/health` and `/ready` endpoints; Alembic migrations; Docker and Compose files. |

---

## Architecture

```mermaid
flowchart LR
    UI["Web UI<br/>(frontend/)"] -- "REST + SSE<br/>Bearer token" --> API

    subgraph Backend["FastAPI app (backend/app)"]
        API["API routes<br/>auth · documents · conversations · health"]
        API --> DS[DocumentService]
        API --> CS[ChatService]
        DS -- enqueue job --> W["IngestionWorker<br/>(thread pool)"]
        W --> IS["IngestionService<br/>parse → chunk → embed → persist"]
        CS --> AG["AgentRunner<br/>bounded loop"]
        AG --> TR["ToolRegistry<br/>search_documents · list_documents"]
        TR --> RT["Retriever<br/>embed query → vector search"]
        AG --> LLM["LLMClient<br/>timeouts · retries · streaming"]
        IS --> EMB[("Embedder<br/>MiniLM, loaded once")]
        RT --> EMB
    end

    IS --> DB
    RT --> DB
    CS --> DB
    DS --> FS[("Upload storage<br/>owner/uuid.ext")]
    IS --> FS
    LLM --> OR["OpenRouter<br/>(OpenAI-compatible)"]
    DB[("PostgreSQL + pgvector<br/>users · tokens · documents · chunks+vectors<br/>jobs · conversations · messages")]
```

**Key design decisions**

- **Vectors live in the same database row as their chunk.** Inserts and deletes happen in one transaction, so the index can never drift out of sync with the metadata (the FAISS-plus-Python-list problem of the original prototype). On PostgreSQL, search is pgvector cosine distance over an HNSW index, filtered by `owner_id` in SQL. On SQLite, which is meant for local development, the same owner-filtered rows are scored with NumPy.
- **The database is the ingestion queue.** Upload writes a `documents` row and an `ingestion_jobs` row, then an in-process worker processes the job. Claiming a job is a single atomic `UPDATE`, so several API replicas never process the same job twice. Jobs left unfinished by a crashed process are picked up again on startup. No Redis or Celery is needed.
- **There are no module-level singletons.** `app/container.py` builds one set of services per app instance, and the embedding model is loaded once per process.
- **The agent is honest about its limits.** If nothing relevant is found, the "not found" answer is produced without calling the LLM at all. The model can also refuse with a sentinel (`INSUFFICIENT_EVIDENCE`), which is turned into a clear "not found" reply.

### Repository layout

```
backend/
  app/
    main.py              application factory (lifespan: migrate, load model, resume jobs)
    container.py         wires settings → services (dependency container)
    api/deps.py          DB session, current user, rate-limit helpers
    api/routes/          auth, documents, conversations, health (thin routes)
    core/                config, logging, security, exceptions, middleware
    db/                  engine/session, portable column types, migration runner
    models/              SQLAlchemy models
    schemas/             Pydantic request/response models (the API contract)
    services/            auth, document, ingestion, chat (business logic)
    rag/                 parser, chunker, embeddings, vector_store, retriever, prompts, generator, llm
    agents/              tools (typed tool interface) and orchestrator (bounded loop)
    workers/             background ingestion worker
  alembic/               database migrations
  tests/                 unit/ and integration/ (including end-to-end, Postgres and real-model suites)
  pyproject.toml         project metadata, ruff and pytest configuration
  requirements.txt       pinned runtime lock (CPU PyTorch)
  requirements-dev.txt   pinned dev lock
frontend/                index.html, app.js, styles.css (served at /)
Dockerfile, docker-compose.yml, .env.example
audit/                   production audit, Phase 0 plan, implementation progress notes
```

---

## Getting started (local)

**Requirements:** Python **3.11–3.13**. Python 3.14 and later aren't supported yet because `tokenizers`, a dependency of sentence-transformers, doesn't publish wheels for them. You need about 1.5 GB of disk for CPU PyTorch and the model.

```bash
# 1. Create a virtual environment and install pinned dependencies
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements-dev.txt     # or requirements.txt for runtime only

# 2. Configure
cp .env.example .env
#   set OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)

# 3. Run (SQLite database and uploads are created under ./data)
cd backend
uvicorn app.main:create_app --factory --reload
```

Open <http://localhost:8000>, create an account, upload a document and ask a question. The first start downloads the embedding model (about 90 MB). Interactive API docs are at <http://localhost:8000/api/docs>; they are disabled when `APP_ENV=production`.

> **Tip:** `uv` can install a matching Python for you: `uv python install 3.12 && uv venv --python 3.12`.

### Database and migrations

- Migrations run automatically on startup when `AUTO_MIGRATE=true`, which is the default.
- To run them by hand: `cd backend && alembic upgrade head`, or `python -m app.db.migrate`.
- To use PostgreSQL locally:
  1. Set `DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/ragbot`.
  2. Make sure the pgvector extension is available. The migration runs `CREATE EXTENSION IF NOT EXISTS vector`, which needs sufficient privileges.
- **Changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` requires a fresh database.** The app records the model that built the vectors and refuses to start on a mismatch, rather than silently mixing vector spaces.

## Running with Docker

```bash
cp .env.example .env        # set OPENROUTER_API_KEY and POSTGRES_PASSWORD
docker compose up --build
```

This starts `pgvector/pgvector:pg16` and the app on <http://localhost:8000>. The image:
- uses CPU-only PyTorch;
- bakes in the embedding model, so containers start without network access;
- runs as a non-root user;
- runs migrations once before starting the server;
- has a `HEALTHCHECK` on `/health`.

Uploads and the database are kept in named volumes.

### Production guidance

- Set `APP_ENV=production`. This requires PostgreSQL, rejects `CORS_ORIGINS=*` and disables the API docs.
- Terminate TLS at a reverse proxy. Set `TRUST_PROXY_HEADERS=true` and `FORWARDED_ALLOW_IPS=<proxy ip>` so rate limiting sees real client IPs.
- Set `ALLOW_REGISTRATION=false` once your users exist, if this is an internal tool.
- Rate limits are kept per process. With several replicas, enforce limits at the proxy as well.
- Back up the PostgreSQL database and the uploads volume together.
- Use `/ready` for readiness probes. It checks the database and the embedding model, and returns 503 until both are ready. Use `/health` for liveness.

---

## Configuration

All settings are environment variables, which can also be placed in `.env`; see [.env.example](.env.example) for the full list with defaults. The most important ones:

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | (none) | API key for the LLM endpoint. Without it, questions return a clear "not configured" error. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | `mistralai/mistral-7b-instruct` | Model id. A stronger instruction-following model gives better planning decisions. |
| `DATABASE_URL` | `sqlite:///./data/app.db` | `postgresql+psycopg://…` in production |
| `RETRIEVAL_MIN_SCORE` | `0.20` | Relevance floor (see the calibration note below) |
| `MAX_UPLOAD_MB` / `MAX_FILES_PER_UPLOAD` | `20` / `10` | Upload limits |
| `AGENT_MAX_ITERATIONS` / `AGENT_MAX_TOOL_CALLS` / `AGENT_TIMEOUT_SECONDS` / `AGENT_TOKEN_BUDGET` | `4` / `4` / `90` / `12000` | Hard agent limits |
| `APP_ENV_FILE` | (unset) | Overrides where `.env` is read from. Set it to an empty string to disable `.env` loading; the test suite does this. |

**Relevance calibration:** with MiniLM and realistic ~180-word multi-topic chunks, relevant questions scored 0.22–0.73 and unrelated questions 0.02–0.13. The default floor of 0.20 sits between the two ranges, and this is checked by `pytest -m slow`. Re-calibrate if you change the model or the chunk size.

---

## API

All endpoints except health and auth require `Authorization: Bearer <token>`. Errors always use this shape:
`{"error": {"code": "...", "message": "...", "request_id": "..."}}`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (database and embedding model) |
| POST | `/api/v1/auth/register` | Register. Body: `{email, password}`; the password needs at least 10 characters. |
| POST | `/api/v1/auth/login` | Log in. Returns `{access_token, token_type, expires_at}`. |
| POST | `/api/v1/auth/logout` | Revoke the current token |
| GET | `/api/v1/auth/me` | The current user |
| POST | `/api/v1/documents` | Multipart upload of one or more `files`. Returns `202 {items: [{document, duplicate}]}`. |
| GET | `/api/v1/documents` | List your documents |
| GET | `/api/v1/documents/{id}` | Document status and metadata |
| DELETE | `/api/v1/documents/{id}` | Delete the document, its chunks, vectors, jobs and stored file |
| POST | `/api/v1/documents/{id}/reprocess` | Retry a failed or finished document |
| POST | `/api/v1/conversations` | Create a conversation |
| GET | `/api/v1/conversations` | List conversations |
| GET | `/api/v1/conversations/{id}` | A conversation with its messages, citations and agent trace |
| DELETE | `/api/v1/conversations/{id}` | Delete a conversation |
| POST | `/api/v1/conversations/{id}/messages` | Ask a question and get the reply as JSON: `{user_message, assistant_message}` |
| POST | `/api/v1/conversations/{id}/messages/stream` | Ask a question and get the reply as SSE: `user_message`, then `step` events, then `token` events, then `final` or `error` |

An assistant message carries `content`, a `status` (`answered`, `no_answer`, `no_documents`, `direct_reply` or `error`), `citations[]` (number, document id and name, page, score, snippet), `agent_trace[]` and `usage` (tokens, LLM calls, elapsed time, grounded flag).

> **Migration note:** the prototype's `POST /upload` and `GET /query` endpoints were unauthenticated, shared one corpus across all users and never worked as committed. They have been replaced by the authenticated, per-user endpoints above.

---

## RAG pipeline

1. **Upload:** the file is streamed to a temporary file while its size and SHA-256 are computed. Its type is validated from the bytes, it's deduplicated per user by hash, and it's stored as `uploads/<owner_id>/<document_id><ext>`. The original filename is kept only as a display label.
2. **Ingest** (background): parse into sections (pages or slides), chunk them, embed in batches with MiniLM (normalised vectors), then in **one transaction** replace the document's chunks and mark it `ready`. Chunk ids are `uuid5(document_id, ordinal)`, so re-ingesting never creates duplicates.
3. **Retrieve:** embed the query, then search only the current user's `ready` documents, keeping the top-k results above `RETRIEVAL_MIN_SCORE`.
4. **Generate:** the evidence goes to the model as numbered `<source>` blocks marked as untrusted data, with instructions to cite `[n]`, to answer only from the sources and to reply `INSUFFICIENT_EVIDENCE` when they don't contain the answer.
5. **Verify:** invalid citation numbers are removed, the cited sources are returned with snippets, and answers without a valid citation are flagged `grounded: false`.

## Agent execution flow

```mermaid
flowchart TD
    Q[Question + recent history] --> G{Any ready documents?}
    G -- no --> ND["Reply: no documents<br/>(no LLM call)"]
    G -- yes --> S1["search_documents(question)"]
    S1 --> L{"Loop: until max iterations"}
    L --> SUF{Strong evidence?}
    SUF -- yes --> GEN
    SUF -- no --> LIM{"Tool-call / time /<br/>token budget left?"}
    LIM -- no --> GEN
    LIM -- yes --> C["Controller LLM → validated JSON action"]
    C -- "search_documents(refined query)" --> T[Run tool] --> L
    C -- list_documents --> T
    C -- answer --> GEN
    C -- cannot_answer --> NA["Reply: not found<br/>(no generation call)"]
    C -- direct_reply --> DR[Short conversational reply]
    C -- "invalid / repeated query" --> FB["Safe fallback:<br/>answer if evidence, else not found"]
    GEN{Evidence gathered?} -- no --> NA
    GEN -- yes --> A[Generate cited answer] --> V[Verify citations]
```

- **Tools** (`app/agents/tools.py`) take Pydantic inputs (extra fields are forbidden) and return Pydantic outputs. Each tool declares a permission, and the registry refuses tools whose permission wasn't granted. The user id always comes from the server-side auth context, never from model output. Both tools are read-only.
- **Limits:** a `for` loop over `AGENT_MAX_ITERATIONS`, a cap on tool calls, a wall-clock deadline that is also passed to each LLM request as its timeout, and a token budget that reserves enough for the final answer. Every stop reason is recorded in the trace.
- **This is not MCP.** The tool interface is internal; the original prototype's custom "MCP" message class has been removed.

---

## Testing

```bash
cd backend
python -m pytest                   # unit + integration + end-to-end (fake embedder, mocked LLM): ~35 s
python -m pytest -m postgres       # same flow on embedded PostgreSQL + pgvector (needs pgserver, in dev deps)
python -m pytest -m slow           # real MiniLM model: threshold calibration + full flow (downloads ~90 MB)
python -m ruff check app tests alembic
```

The tests never call a real LLM and never read your `.env`. A scripted fake OpenRouter (`tests/helpers.py`) emulates JSON and SSE responses, failures and rate limits, so the real HTTP client code runs against it. Coverage includes:
- parsing of all formats, including corrupted files, spoofed types and zip bombs;
- chunking invariants;
- password and token handling and rate limiting;
- prompt injection neutralisation and citation validation;
- LLM retries, `Retry-After`, timeouts, malformed responses and streaming;
- agent bounds and fallbacks;
- authentication;
- upload limits and path traversal;
- idempotent ingestion and deletion cascades;
- isolation between users at both the API and data layers;
- persistence across restarts and recovery of unfinished jobs;
- migrations (up and down);
- streaming;
- the end-to-end flow: upload, wait, ask, check the cited answer, ask something unrelated, delete, and confirm another user can't access anything.

---

## Known limitations

- **No OCR.** Scanned or image-only PDFs are rejected with an explanatory message.
- **Retrieval is dense-vector only,** with no hybrid keyword search or reranker. Exact identifiers (codes, part numbers) may be missed when they aren't semantically distinctive. A reranker or BM25 is the next step if an evaluation set shows retrieval misses.
- **Tabular questions are answered from retrieved rows only.** Questions that need computation over a whole CSV (for example "average of column X") aren't reliably supported.
- **The planner's quality depends on `LLM_MODEL`.** With small models the controller sometimes returns invalid JSON. The agent then falls back to a safe default and records `fallback: true` in the trace.
- **Rate limiting and the ingestion thread pool are per process.** Use a proxy-level limiter with several replicas. Very large deployments may want a dedicated worker process.
- **SQLite vector search is a linear scan,** fine for development and small collections. Use PostgreSQL with pgvector for anything else.
- **Login tokens are bearer tokens in `localStorage`,** mitigated by a strict CSP with no inline scripts and no HTML rendering of model output. A cookie-based session would need CSRF protection; it's a deliberate trade-off.

## License

MIT. See [LICENSE](LICENSE).
