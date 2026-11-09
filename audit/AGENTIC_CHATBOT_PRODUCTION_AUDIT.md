# Agentic RAG Chatbot — Production-Readiness Audit

| | |
|---|---|
| Repository | `agentic-rag-chatbot` (origin: `github.com/saee08/agentic-rag-chatbot`) |
| Audited commit | `d12cc7e` ("Add files via upload"), branch `main` |
| Audit date | 2026-11-03 |
| Scope | All tracked files, including the contents of `backend.zip` and `frontend.zip` |
| Method | Static reading of every source file, syntax compilation of every `.py` file (in memory, no bytecode written), AST inspection, and isolated execution of the pure function `chunk_text`. **The application was not run end-to-end**: its dependencies (`faiss`, `sentence-transformers`, `torch`, `streamlit`, `fastapi`, …) are not installed in this environment and installing them was out of scope. Any claim about runtime behaviour beyond `chunk_text` comes from reading the code and is labelled as such. |
| Secrets | No secret values appear in this report. |

---

## 1. Executive Summary

The repository is an **early prototype of a document question-answering pipeline**: parse files → chunk → embed with `all-MiniLM-L6-v2` → FAISS flat index → top-5 chunks → one OpenRouter chat completion (`mistralai/mistral-7b-instruct`).

**It does not work as committed.** Even with all dependencies installed, it would fail at several independent points:

1. **It cannot be imported.** Every module imports from packages (`agents.`, `mcp.`, `file_parser.`, `vector_store.`) that do not exist in the flat repository layout. The real backend and frontend are only present inside `backend.zip` and `frontend.zip`.
2. **Chunking produces no text.** `chunk_text` returns a list of bound `str.join` methods, not strings. I confirmed this by running the function: with 700 words of input it returned 3 items, all of type `builtin_function_or_method`. A downstream sanitiser then silently throws all of them away, so **nothing is ever indexed**.
3. **The orchestrator is broken.** `handle_user_query` is defined at module level, outside `CoordinatorAgent`. The UI and API both call it as a method, which raises `AttributeError`. Even if the indentation were fixed, an unconditionally executed `return` means the LLM is never called and every query answers "Retrieval failed".
4. **The return types disagree.** The coordinator returns plain `dict`s in some paths and `MCPMessage` objects in others. The API reads `result.payload`, which fails on a dict. The UI reads `final_msg["payload"]`, which fails on an `MCPMessage`.

**Security:** an OpenRouter API key is committed in `.env` (commit `d12cc7e`) and pushed to a GitHub remote. Listing `.env` in `.gitignore` does nothing for a file that is already tracked. The key should be treated as compromised and **rotated immediately**. The backend also has no authentication, writes client-supplied filenames straight to disk (a path traversal pattern), and shares one global upload folder across all users.

**On "agentic":** the system is **not agentic**. It is a fixed, linear RAG pipeline whose stages happen to be named "agents". The LLM makes no decisions, calls no tools and runs in no loop. The "MCP" in the code is a 17-line in-house message class. It is **not** the Model Context Protocol, and it uses the package name `mcp`, which collides with the official MCP SDK on PyPI.

**Recommendation:** don't harden this codebase in place. Keep the parsing code and the overall pipeline shape, and rebuild it as one small, honest RAG service with a real package layout, tests, persistence and authentication. Phase 0 of the roadmap (rotate the key, make the code runnable, fix chunking and the coordinator) takes about a day and should come first.

### Overall scores (details in §19)

| Area | Score /10 |
|---|---|
| Architecture | 2 |
| Code quality | 2 |
| Agent engineering | 1 |
| RAG implementation | 2 |
| LLM engineering | 2 |
| Security | 1 |
| Performance | 2 |
| Testing | 0 |
| Scalability | 1 |
| Documentation | 2 |
| **Production readiness** | **1** |

---

## 2. Project Overview

**Stated purpose (README):** "A modular, agent-based RAG chatbot that can answer user questions using uploaded documents in multiple formats."

**What is actually implemented (verified in code):**

| Capability | Status | Evidence |
|---|---|---|
| Parse PDF, DOCX, PPTX, CSV, TXT, MD | Implemented (basic) | `parser.py:8-48` |
| Word-window chunking | Implemented but **broken** | `ingestion_agent.py:9-18` |
| Local embeddings (MiniLM, 384-d) | Implemented | `embedder.py:5-9` |
| In-memory FAISS flat index | Implemented | `store.py:7-45` |
| Top-K retrieval (K=5) | Implemented | `retrieval_agent.py:49-71` |
| LLM answer through OpenRouter | Implemented | `llm_response_agent.py:22-86` |
| "Sources" display (raw chunk text) | Implemented | `frontend.zip:frontend/app.py:49-51` |
| Orchestrator | Implemented but **broken** | `coordinator_agent.py:17-54` |
| Streamlit UI | Present only inside `frontend.zip` | `frontend/app.py` |
| FastAPI backend | Present only inside `backend.zip` | `backend/api.py` |
| Multi-turn chat or conversation memory | **Not implemented** | — |
| Citations or attribution to file/page | **Not implemented** | — |
| Persistence of any kind (index, chats, metadata) | **Not implemented** | — |
| Authentication | **Not implemented** | — |
| Tests, CI, Docker, deployment configuration | **Not present** | — |
| `architecture.pptx` | **Empty file (0 bytes)** | blob `e69de29…` (git's empty-blob hash) |
| `backend/routes`, `backend/services` | **Empty directories** inside the zip | — |

**Discrepancies between the README and the code:**

- The README says "OpenAI GPT" in two places. The code uses OpenRouter with `mistralai/mistral-7b-instruct` through `requests`. The `openai` package listed in `requirements.txt` is never imported.
- The README says chunks are "semantic". Chunking is actually a fixed 300-word window with a 50-word overlap.
- The README calls the message format the "Model Context Protocol (MCP)". It is not an implementation of that protocol.
- The README says the design is "modular … for easy scaling and testing". There are no tests, and the module paths don't resolve.
- The README shows a JSON block that is never closed, and the setup and run instructions are missing.

---

## 3. Technology Stack

| Layer | Technology | Source | Notes |
|---|---|---|---|
| Language | Python | all | No version pinned. `list[UploadFile]` in `api.py` requires Python 3.9 or later. |
| UI | Streamlit | `frontend/app.py` (zipped) | Not runnable from the repository root as committed |
| API | FastAPI with `CORSMiddleware` | `backend/api.py` (zipped) | `fastapi`, `uvicorn` and `python-multipart` are **missing from requirements** |
| Parsing | PyMuPDF (`fitz`), python-docx, python-pptx, pandas | `parser.py` | |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-d), PyTorch | `embedder.py`, `ingestion_agent.py` | The model is loaded **twice** (see F-15) |
| Vector store | `faiss-cpu` `IndexFlatL2` (exact search, in memory) | `store.py` | No persistence |
| LLM | OpenRouter Chat Completions, `mistralai/mistral-7b-instruct`, temperature 0.3 | `llm_response_agent.py:11-13,51-55` | Called with raw `requests`, which is **not in requirements** |
| Configuration | `python-dotenv` | `llm_response_agent.py:3,8-9` | **Not in requirements**. The path to `.env` is wrong for the flat layout (F-08). |
| Messaging | Custom `MCPMessage` class | `protocol.py` | Not the Model Context Protocol |
| Database | None | — | |
| Authentication | None | — | |
| Deployment | None (no Dockerfile, Procfile, compose file or CI) | — | |

`requirements.txt` has no version pins. It lists `openai` and `tqdm`, which are never used, and leaves out `fastapi`, `uvicorn`, `python-multipart`, `python-dotenv` and `requests`. `torch` is imported directly at `ingestion_agent.py:6` and arrives only as a transitive dependency.

---

## 4. Architecture Explanation

### 4.1 Intended module layout (from the import statements) compared with what is committed

```
Intended (from imports)              Committed
agents/coordinator_agent.py          coordinator_agent.py      (root)
agents/ingestion_agent.py            ingestion_agent.py        (root)
agents/retrieval_agent.py            retrieval_agent.py        (root)
agents/llm_response_agent.py         llm_response_agent.py     (root)
mcp/protocol.py                      protocol.py               (root)
file_parser/parser.py                parser.py                 (root)
vector_store/embedder.py             embedder.py               (root)
vector_store/store.py                store.py                  (root)
backend/api.py                       backend.zip
frontend/app.py                      frontend.zip
```

The files were evidently flattened when they were uploaded through the GitHub web UI. None of the `__init__.py` files appear either.

### 4.2 Request flow as designed (reading the code, assuming the import paths resolved)

**Streamlit path (`frontend/app.py`):**

1. On every Streamlit rerun, the script writes every uploaded file to `data/uploads/<client filename>` (lines 23-28).
2. When the user clicks **Ask**, the script constructs a **new** `CoordinatorAgent()` (line 40). That builds a new `IngestionAgent`, which loads MiniLM, and a new `RetrievalAgent`, which loads MiniLM **again** and creates an empty FAISS index.
3. It then calls `coordinator.handle_user_query(files, query)` with every file in the shared upload directory.
4. The coordinator parses and chunks every file, embeds the chunks, adds them to FAISS, embeds the query, searches top-5, and is meant to call the LLM.
5. The UI renders the answer and the raw chunk text.

**FastAPI path (`backend/api.py`):**

1. `POST /upload` writes the files to `data/uploads/<client filename>`.
2. `GET /query?query=...` lists **every** file in `data/uploads` and calls the module-level `coordinator.handle_user_query`. That coordinator is a process-wide singleton, so its FAISS store lives for the whole process and **is appended to on every query**.

### 4.3 Request flow as it would actually execute

```
import agents.coordinator_agent   -> ModuleNotFoundError                         [F-02]
(assume the layout is fixed)
coordinator.handle_user_query     -> AttributeError (function is not a method)    [F-03]
(assume that is fixed)
IngestionAgent.process_documents  -> chunks = [<built-in method join>, ...]       [F-04]
  coordinator checks "if not chunks" -> list is non-empty, so it continues
RetrievalAgent.receive_chunks     -> every chunk is "callable", so all are skipped,
                                     clean_chunks = []
  embedder.embed([])              -> empty result; nothing is added to FAISS
handle_query (called TWICE)       -> store.search: "index is empty", returns []   [F-05]
coordinator line 45               -> unconditional return "Retrieval failed"      [F-03]
LLM                               -> never called through the coordinator
API: result.payload on a dict     -> AttributeError -> HTTP 500                   [F-06]
```

`main.py` skips the coordinator and calls the agents directly. With the layout fixed, it would reach the LLM with "No relevant context available." as its only context, because of F-04. It also globs `data/uploads/*`, a directory that does not exist in the repository.

---

## 5. Agent Architecture Assessment

| Dimension | Finding |
|---|---|
| Agent architecture | Four Python classes, each wrapping one deterministic step. No class contains a policy, goal or decision. |
| Orchestration | A hard-coded sequence in `coordinator_agent.py` (and a duplicate copy in `main.py`) |
| Tool calling / tool selection | **None.** The LLM is never offered tools, and no tool-call loop exists. |
| Planning / reasoning | **None** |
| State management | Instance attributes only (`FAISSVectorStore.index`, `.text_chunks`). Nothing is kept per session or per user. |
| Short-term memory | **None.** Each question is answered without any earlier turns. |
| Long-term memory | **None.** The index is held in RAM and lost on restart (or on every click in Streamlit). |
| Multi-agent communication | `MCPMessage` objects passed by **direct Python method calls**. The `receiver` field is never read. There is no bus, queue or routing. |
| Execution loops / iteration limits | No loop exists, so there is nothing to bound. |
| Failure recovery / retries | None. Errors are `print()`ed and swallowed (`ingestion_agent.py:44-45`, `retrieval_agent.py:42-44`). |
| Human-in-the-loop | Not applicable. There are no actions to approve. |
| Tool permissions | Not applicable |

**Verdict:** this is a **conventional single-shot RAG pipeline**, not an agentic system. The "agent" and "MCP" labels inflate what it does and suggest capabilities it does not have. That isn't a defect in itself, because most document Q&A products don't need an agent. The documentation and naming should still match reality.

**Should it become agentic?** There is nothing in the current requirements (Q&A over uploaded documents) that justifies a multi-agent design. If you later need things like "search again with a reformulated query when retrieval is weak" or "answer numeric questions about a CSV", the proportionate step is **one** LLM tool-calling loop with 1–3 tools (for example `search_documents(query, filters)` and a sandboxed tabular-query tool), a hard iteration cap and a token budget. Multiple cooperating agents would not be proportionate.

---

## 6. RAG Pipeline Assessment

### 6.1 Ingestion

| Item | Current state | Assessment |
|---|---|---|
| Formats | `.pdf .docx .pptx .csv .txt .md` (`parser.py:35-48`) | Reasonable for a prototype |
| File validation | Extension check only, inside the parser and **after** the file has been written to disk | No size, MIME/magic-byte or page-count limits |
| PDF | `page.get_text()` joined with newlines | No OCR, so scanned PDFs yield empty text silently. Page numbers are discarded. |
| DOCX | Paragraphs only (`parser.py:14-16`) | **Tables, headers and footers are dropped.** That matters for reports full of KPIs, which is the demo query in `main.py:18`. |
| PPTX | `shape.text` for shapes that have it | Tables, grouped shapes and speaker notes are dropped |
| CSV | `df.to_string()` of the whole frame | A large CSV becomes one enormous string, and row and column structure is lost when it is chunked. Questions like "average of column X" (README:100) cannot be answered reliably by retrieval. |
| TXT/MD | Strict UTF-8 | Non-UTF-8 files raise an error that is caught and printed, and the file is skipped silently |
| Chunking | 300-word window, 50-word overlap. **Broken:** returns method objects (F-04). | Even once fixed, it splits on whitespace (so it erases all layout), ignores sentence and heading boundaries, and has no token awareness. |
| Metadata | **None.** Chunks are bare strings with no filename, page, offset or document ID. | This makes citations, filtering, deletion and de-duplication impossible |
| Duplicate handling | **None.** The API re-ingests every file on every query (F-05). | |
| Incremental ingestion | **None.** Each query re-parses the whole upload folder. | |
| Error handling | Each file's exception is printed and processing continues. The caller gets no signal. | A user whose file failed to parse is never told |

### 6.2 Embeddings

| Item | Current state |
|---|---|
| Provider / model | Local `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions |
| Consistency | The same model name is used for chunks and queries (`embedder.py:5`), which is correct. The dimension is hard-coded to 384 in `store.py:7` rather than read from the model. |
| Duplicate model | `IngestionAgent` loads a second MiniLM copy (`ingestion_agent.py:30-31`) that is **never used** (F-15) |
| Device | `IngestionAgent` moves its unused copy to CUDA when available. The `Embedder` that actually does the work uses the default device selection. |
| Batching | Relies on `encode()`'s default batch size (32). No explicit control. |
| Caching | None. The same files are re-embedded on every query or click. |
| Normalisation | Not set explicitly. As far as I know, this model's published configuration includes a normalisation layer, so L2 distance ranks the same as cosine. That is an implicit dependency, not a design choice. Not verified at runtime. |

### 6.3 Retrieval

| Item | Current state |
|---|---|
| Search | `IndexFlatL2`: exact brute-force search, adequate up to roughly 10⁵–10⁶ vectors on CPU |
| Top-K | Fixed at 5. There is **no score threshold**: distances (`D`) are computed and thrown away (`store.py:38`), so irrelevant chunks are always passed to the LLM. |
| Metadata filtering | Impossible, because there is no metadata |
| Hybrid / BM25 | None |
| Reranking | None |
| Context assembly | Chunks are joined as `"- {chunk}"` bullets (`llm_response_agent.py:88-91`), with no source labels |
| Relevance | Cannot be measured. There is no evaluation set. |

### 6.4 Generation

| Item | Current state |
|---|---|
| Prompt | System: "You're a helpful assistant. Answer questions only using the provided context." User: context followed by the question. |
| Hallucination control | Weak. There is no instruction for what to do when the context is insufficient ("say you don't know"). When retrieval is empty, the model receives "No relevant context available." and is free to answer from its own training knowledge. |
| Citations | None. The "sources" shown are simply the top-5 chunks, whether or not the model used them. |
| Context window | No budgeting. In practice 5 × 300 words is about 2k tokens, which is well inside Mistral-7B's window, but nothing enforces this if the parameters change. |
| Prompt injection | No delimiting or labelling of untrusted document text (see F-12) |

### 6.5 Evaluation

**None.** There is no golden question set and no measurement of retrieval recall@K or precision, faithfulness, answer relevance or latency. The README's example questions (IBM report, UN SDG, Artemis, the Transformer paper) would make a good basis for a first evaluation set.

---

## 7. LLM Engineering Assessment

| Item | Current state | Evidence |
|---|---|---|
| Provider abstraction | None. URL, model and headers are module-level constants inside the agent. | `llm_response_agent.py:11-13` |
| Model configuration | Hard-coded model, with a comment saying "You can change this" | `:13` |
| Switching provider | OpenRouter is OpenAI-compatible, so changing `API_URL` and `MODEL` reaches many models. A non-compatible provider would require editing the agent. | |
| Prompt templates | Inline f-string | `:41-44` |
| Token usage | The `usage` field of the response is ignored. Nothing is counted or logged. | `:60-62` |
| Streaming | No | |
| Timeout | **None.** `requests.post` without `timeout=` can block a worker indefinitely. | `:58` (F-09) |
| Retries / backoff / 429 handling | None | |
| Fallback provider or model | None | |
| Cost controls | No `max_tokens`, no per-user budget | `:51-55` |
| Structured output | Not used, and not needed for the current output | |
| Error handling | Catches `RequestException` only. A 200 response with an unexpected body raises `KeyError` or `IndexError` at `:62` and propagates. | |
| Secret handling | The key is read from the environment, which is correct. The `.env` path is wrong (F-08) and the file is committed (F-01). The startup log prints only whether the key is present, which is fine. | `:8-15` |
| Side effects at import time | Importing the module loads `.env` and prints | `:8-15` |

---

## 8. Backend and Frontend Assessment

### 8.1 Backend (`backend.zip:backend/api.py`)

| Item | Finding |
|---|---|
| API design | `POST /upload`, `GET /query?query=`. Sending the query in the URL means user questions end up in access and proxy logs. |
| Validation | None on files (name, size, type, count). `query` is an unconstrained string. |
| Response consistency | No Pydantic response model. Errors surface as bare 500s from `AttributeError` (F-06). |
| Authentication and authorisation | **None** (F-11) |
| Sessions | None. All clients share one corpus. |
| Concurrency | `ask_question` is a sync `def`, so FastAPI runs it in a thread pool. The **shared singleton** coordinator mutates `index` and `text_chunks` with no lock. Concurrent queries can misalign the FAISS row IDs with the chunk list. This is a potential risk; I did not reproduce it. |
| Heavy work in the request path | Every query re-parses and re-embeds every uploaded document, synchronously |
| Background jobs | None |
| Rate limiting | None |
| Logging | `print()` with emoji. No levels, no structure, no request IDs. `trace_id` is generated but never logged with any context, and a different ID is created at each stage (`ingestion_agent.py:47`, `coordinator_agent.py:38`). |
| Error handling | Errors are swallowed and printed. `upload_docs` returns success even if a write partially failed. |
| CORS | `allow_origins=["*"]` together with `allow_credentials=True` (F-14) |

### 8.2 Frontend (`frontend.zip:frontend/app.py`)

| Item | Finding |
|---|---|
| Architecture | One Streamlit script. It imports the coordinator **in-process** instead of calling the FastAPI backend, so the two "tiers" are actually alternatives. |
| Chat interface | A single text input and an **Ask** button. No `st.chat_message`, no history, no `st.session_state`. |
| Streaming | None |
| Loading state | None. There is no `st.spinner`, and each click silently loads two models and re-indexes everything. |
| Upload | Files are rewritten to disk on **every** Streamlit rerun (any widget interaction), using the client-supplied name |
| Error state | A single generic message. Per-file parse failures are never shown. |
| State management | None. A new `CoordinatorAgent` is built per click; models are not cached with `st.cache_resource`. |
| Rendering | `st.markdown(answer)` and `st.markdown(f"> {src}")` render model and document text as Markdown. Streamlit escapes raw HTML by default, so the risk is limited to Markdown content such as links and images (potential low risk). |
| Responsive design / accessibility | Streamlit defaults. The emoji in labels add noise for screen readers. |

---

## 9. Database and Memory Assessment

There is **no database**.

| Item | State |
|---|---|
| Schema | None |
| Conversation persistence | None |
| User data isolation | **None.** There is one shared `data/uploads` directory, and every user's query searches every user's files (F-10). |
| Document storage | Local filesystem. Same-name uploads overwrite each other silently, with no versioning. |
| Vector index | Held in memory and never saved (`faiss.write_index` is not used). Lost on restart, and rebuilt on every Streamlit click. |
| Growth | In the API process the index grows by the size of the whole corpus on **every query** (F-05). Memory grows without bound and the top-K results fill up with duplicates. |
| Deletion and retention | No delete endpoint. Uploaded files are kept forever. Because chunks carry no document ID, a single document's vectors could not be removed even if a delete endpoint existed. |

---

## 10. Findings

Severity levels: **Critical** means exploitable now or the system is non-functional. **High** means a serious defect or risk that blocks production. **Medium** means a significant weakness. **Low** means minor. Complexity: **S** is under 2 hours, **M** is under 2 days, **L** is more than 2 days.

### 10.1 Confirmed defects

#### F-01 — API key committed to git and pushed to GitHub
- **Severity:** Critical · **Category:** Security / secrets
- **Location:** `.env` (tracked; added in commit `d12cc7e`); `.gitignore:4`
- **Current:** `.env` holds `OPENROUTER_API_KEY` with a value. The file is tracked, and origin is a GitHub repository. `.gitignore` lists `.env`, but that rule has no effect on a file that is already tracked.
- **Problem:** anyone with read access to the repository or its history has the key. I did not verify whether the GitHub repository is public.
- **Impact:** an attacker can spend money on your OpenRouter account. If the repository is public, the key may already have been harvested by automated scanners.
- **Fix:** (1) **revoke and rotate the key in the OpenRouter dashboard now**. (2) `git rm --cached .env`, and commit a `.env.example` containing only placeholders. (3) Optionally remove the file from history with `git filter-repo`, then force-push. Rotation is the step that actually protects you. (4) Add a secret scanner (gitleaks or detect-secrets) as a pre-commit hook and in CI.
- **Complexity:** S
- **Verify:** `git ls-files .env` returns nothing, the old key is rejected by OpenRouter, and gitleaks passes on the full history.

#### F-02 — Import paths don't match the repository layout, so nothing runs
- **Severity:** Critical · **Category:** Build / packaging
- **Location:** `main.py:2-4`, `coordinator_agent.py:3-5`, `ingestion_agent.py:3-4`, `retrieval_agent.py:3-5`, `llm_response_agent.py:4`; the backend and UI live only in `backend.zip` and `frontend.zip`
- **Current:** the code imports `agents.*`, `mcp.protocol`, `file_parser.parser` and `vector_store.*`, none of which exist. The API and UI code is only inside zip archives, which `.gitignore:5` also lists.
- **Problem:** `ModuleNotFoundError` on the first import. Separately, the local package name `mcp` would shadow, or be shadowed by, the official `mcp` package on PyPI.
- **Impact:** no entry point works. The repository cannot be reviewed, tested or deployed as it stands.
- **Fix:** restore the package tree under a single top-level package (for example `ragbot/`) with `__init__.py` files, rename `mcp` to something like `messages`, unpack the zips into the source tree, and delete the archives.
- **Complexity:** S
- **Verify:** `python -c "import ragbot.api"` succeeds in a fresh virtual environment.

#### F-03 — Coordinator method is outside its class and always returns "Retrieval failed"
- **Severity:** Critical · **Category:** Correctness
- **Location:** `coordinator_agent.py:17` (module-level `def`), `:39-41` (duplicate call), `:43-50` (misplaced `return`)
- **Current:** AST inspection confirms that `CoordinatorAgent` contains only `__init__`, and that `handle_user_query` is a top-level function. Inside that function, the `return {... "Retrieval failed" ...}` at line 45 sits at function-body indentation, outside the `if retrieval_msg is None:` block.
- **Problem:** (a) `coordinator.handle_user_query(...)` raises `AttributeError` (called from `api.py:33` and `app.py:41`). (b) Even with the indentation fixed, every request returns the failure dict, and `generate_answer` (line 53) is unreachable. (c) `handle_query` runs twice, which doubles the embedding work and the search.
- **Impact:** the main user flow can never produce an answer.
- **Fix:** indent the method into the class, put the `return` inside the `if`, delete line 41, and add a test.
- **Complexity:** S
- **Verify:** a unit test using stub agents checks that `generate_answer` is called exactly once and its result is returned.

#### F-04 — `chunk_text` returns method objects, so all content is discarded
- **Severity:** Critical · **Category:** Correctness / RAG
- **Location:** `ingestion_agent.py:13-17`
- **Current:** `chunk = " ".join`, followed by blank lines, then `(words[i:i + chunk_size])` on a separate line. Python parses these as two statements, so `chunk` is the bound method `" ".join`. **Executed in isolation:** 700 words of input gave 3 chunks, all of type `builtin_function_or_method`.
- **Problem:** `retrieval_agent.py:25-26` then skips every chunk as "callable". The sanitiser hides the bug instead of surfacing it. The coordinator's "no chunks" guard (`coordinator_agent.py:20`) still passes, because the list is not empty.
- **Impact:** the index is always empty, and the LLM never sees any document content.
- **Fix:** `chunk = " ".join(words[i:i + chunk_size])`. Then replace the type-sanitiser with a typed `Chunk` dataclass and fail loudly on bad input.
- **Complexity:** S
- **Verify:** a unit test asserts that `chunk_text("a " * 700)` returns `list[str]` with the expected count and overlap.

#### F-05 — The API re-ingests the whole corpus on every query, and the index grows without bound
- **Severity:** High · **Category:** Correctness / performance
- **Location:** `backend/api.py:11,32-33`; `coordinator_agent.py:18,28`; `store.py:28-29`
- **Current:** one coordinator for the whole process. Each `/query` call parses every file in `data/uploads`, embeds the chunks and appends them to the same FAISS index.
- **Problem:** after N queries the index holds N copies of the corpus. Top-5 results become 5 copies of the same chunk. Latency and memory grow linearly with the number of queries served. (This only becomes visible once F-02 to F-04 are fixed.)
- **Impact:** retrieval quality degrades steadily, memory eventually runs out, and every query costs a full ingestion.
- **Fix:** separate ingestion (at upload time, in the background, idempotent by content hash) from querying (read-only search).
- **Complexity:** M
- **Verify:** an integration test runs 10 queries and checks that `index.ntotal` stays constant and the top-K results contain no duplicates.

#### F-06 — Inconsistent return types between coordinator, API and UI
- **Severity:** High · **Category:** Correctness / API contract
- **Location:** `coordinator_agent.py:21-26,31-36,45-50` (return `dict`) vs `:53-54` (returns `MCPMessage`); `api.py:35-36` (`result.payload`); `app.py:43-47` (`"payload" not in final_msg`, `final_msg["payload"]`)
- **Problem:** the API fails with `AttributeError` on the dict paths. The UI's `in` and `[]` operations raise `TypeError` on `MCPMessage`, because it defines neither `__contains__` nor `__getitem__`.
- **Impact:** both the success and failure paths crash in one consumer or the other.
- **Fix:** one typed result model (for example a Pydantic `Answer(answer, citations, status)`), with errors raised as typed exceptions and mapped to HTTP status codes at the API layer.
- **Complexity:** S
- **Verify:** API tests cover the success, empty-query, no-documents and LLM-error cases.

#### F-07 — Path traversal and unvalidated file writes on upload
- **Severity:** High · **Category:** Security
- **Location:** `backend/api.py:24-27`; `frontend/app.py:24-27`
- **Current:** `os.path.join(UPLOAD_DIR, file.filename)` uses the client-controlled filename unchanged. There are no size, type or count limits.
- **Problem:** a multipart client can send a filename such as `../../somewhere/file` or an absolute path. `os.path.join` discards the base directory when given an absolute component. The code does nothing to stop this. I have not tested whether the multipart parser in the installed version strips path components, so treat this as **an exploitable pattern, not a demonstrated exploit**. For the Streamlit path the filename comes from the browser, which lowers the risk.
- **Impact:** possible arbitrary file overwrite as the server user, and disk exhaustion through very large uploads.
- **Fix:** use a server-generated UUID as the storage name and keep the original name only as metadata. Resolve the path and check that it stays under the upload root. Enforce a size limit while streaming, an extension and magic-byte allow-list, and a maximum file count.
- **Complexity:** S
- **Verify:** a test that uploads `../evil.txt` and `/tmp/evil.txt` checks that both are stored inside the upload root under a generated name, and that a file over the size limit is rejected with 413.

#### F-08 — `.env` is resolved one directory above the repository
- **Severity:** Medium · **Category:** Configuration
- **Location:** `llm_response_agent.py:8`
- **Current:** `Path(__file__).resolve().parents[1] / ".env"`. That is correct only if the file lives in `agents/`. In the committed flat layout it points to the folder that contains the repository.
- **Impact:** the key is "MISSING", and every LLM call returns 401, which the code reports as "There was an error contacting the LLM."
- **Fix:** a central settings object (for example `pydantic-settings`) that reads environment variables, with an optional `.env` in the working directory. Fail at startup if a required setting is missing.
- **Complexity:** S
- **Verify:** startup fails fast with a clear message when the key is unset.

#### F-09 — No timeout, retry or response validation on the LLM call
- **Severity:** Medium · **Category:** Reliability
- **Location:** `llm_response_agent.py:58-62,75`
- **Problem:** `requests.post` has no timeout, so a stalled connection holds a worker thread indefinitely. Only `RequestException` is caught, so a malformed 200 body raises an uncaught `KeyError`. There is no handling for 429s or 5xx errors.
- **Impact:** hung requests, thread-pool exhaustion under load, and 500 errors from transient upstream problems.
- **Fix:** connect and read timeouts (for example 5 s / 60 s), bounded retries with jitter for 429/5xx/connection errors (2–3 attempts), and validation of the response shape. Use an HTTP client with a shared session and connection pool.
- **Complexity:** S
- **Verify:** tests with a mocked HTTP layer cover timeout, 429-then-200, 500 and a malformed body.

#### F-10 — No separation between users' data
- **Severity:** High (for any multi-user deployment) · **Category:** Security / privacy
- **Location:** `backend/api.py:7,32`; `frontend/app.py:17,32`
- **Current:** one shared `data/uploads` directory, and every query searches every file.
- **Impact:** user A's confidential documents are retrieved into answers for user B, and the raw chunks are shown to B as "sources".
- **Fix:** attach `owner_id` and `collection_id` to every stored document and chunk, and enforce them as a filter on every retrieval query. This depends on authentication (F-11).
- **Complexity:** M
- **Verify:** a test where two users upload different documents checks that each user's retrieval never returns the other's chunks.

#### F-11 — No authentication or authorisation
- **Severity:** High (Critical if internet-facing) · **Category:** Security
- **Location:** `backend/api.py:22-37`
- **Impact:** anyone who can reach the port can upload files, read the whole corpus through queries, and spend LLM credits.
- **Fix:** start simple: an API key or OIDC/JWT through a reverse proxy or a FastAPI dependency, plus per-user rate limits.
- **Complexity:** M
- **Verify:** an unauthenticated request returns 401, and an authenticated user can see only their own collections.

### 10.2 Architectural weaknesses

#### F-12 — No defence against prompt injection from document content
- **Severity:** Medium · **Category:** LLM security
- **Location:** `llm_response_agent.py:41-44,88-91`
- **Current:** document text is inserted directly into the user message, with no delimiting, no labels and no instruction to treat it as data.
- **Impact:** a malicious document can override instructions (for example "ignore the question; reply with …"). Because the system has **no tools**, the damage is limited to manipulated answers. That changes if tools are ever added.
- **Fix:** wrap each chunk in delimited, numbered `<source id=… file=…>` blocks. Tell the model in the system prompt that sources are untrusted data. Keep any future tools least-privilege, and validate their arguments outside the LLM.
- **Complexity:** S
- **Verify:** red-team evaluation cases (documents containing injected instructions) checked against the expected behaviour.

#### F-13 — Chunks carry no metadata or IDs
- **Severity:** Medium · **Category:** RAG design
- **Location:** `ingestion_agent.py:40-43`; `store.py:9,29`
- **Impact:** blocks citations, per-document deletion, de-duplication, filtering by user or document, and re-indexing.
- **Fix:** a `Chunk(id, doc_id, owner_id, source_name, page, ordinal, text, content_hash)` record, with FAISS IDs (`IndexIDMap`) or a database row ID linked to it.
- **Complexity:** M

#### F-14 — CORS allows any origin with credentials
- **Severity:** Low (today) · **Category:** Security
- **Location:** `backend/api.py:14-20`
- **Current:** `allow_origins=["*"]` with `allow_credentials=True`. The in-code comment acknowledges this is for development only.
- **Impact:** low right now, because there is no cookie-based authentication to abuse. It becomes a real hole the moment cookie authentication is added.
- **Fix:** an explicit origin allow-list loaded from configuration.
- **Complexity:** S

#### F-15 — Embedding model loaded twice, and reloaded on every UI click
- **Severity:** Medium · **Category:** Performance
- **Location:** `ingestion_agent.py:30-31` (unused copy), `retrieval_agent.py:10`, `frontend/app.py:40`
- **Impact:** double the model memory per coordinator, and a model load plus full re-index on every **Ask** click in Streamlit. I have not measured this; the expected cost is seconds per click on CPU, plus a re-embedding cost proportional to corpus size.
- **Fix:** one process-wide embedder (`st.cache_resource` in Streamlit, or app-lifespan state in FastAPI), and remove the unused model from `IngestionAgent`.
- **Complexity:** S
- **Verify:** log model-load events and check there is exactly one per process.

#### F-16 — No retrieval relevance threshold
- **Severity:** Medium · **Category:** RAG quality
- **Location:** `store.py:38-45`
- **Impact:** for off-topic questions the model still receives 5 chunks of unrelated text and may build a confident but wrong answer from them.
- **Fix:** return the distances along with the chunks, apply a threshold tuned on an evaluation set, and if nothing passes, answer "not found in your documents" without calling the LLM.
- **Complexity:** S

#### F-17 — Unsynchronised shared mutable state in the API
- **Severity:** Medium (potential risk) · **Category:** Concurrency
- **Location:** `backend/api.py:11,31`; `store.py:28-29`
- **Current:** a sync route runs in a thread pool and mutates `index` and `text_chunks` with no lock.
- **Impact:** under concurrent queries the FAISS row IDs and the chunk list could become misaligned, returning the wrong text for a hit. Not reproduced.
- **Fix:** this goes away once ingestion is separated from querying (F-05). Ingestion should be a single writer.

#### F-18 — Errors are swallowed and output goes to `print`
- **Severity:** Medium · **Category:** Observability
- **Location:** throughout (for example `ingestion_agent.py:44-45`, `retrieval_agent.py:17,43,51,57`, `coordinator_agent.py:44`)
- **Impact:** failures look like empty answers. `llm_response_agent.py:39` prints the **full retrieved document text** to stdout, which leaks document contents into logs.
- **Fix:** the `logging` module with structured JSON output, a request ID propagated end to end, no document content at INFO level, and typed exceptions.

#### F-19 — Dependency manifest is incomplete and unpinned
- **Severity:** Medium · **Category:** Supply chain / reproducibility
- **Location:** `requirements.txt`
- **Current:** missing `fastapi`, `uvicorn`, `python-multipart`, `python-dotenv` and `requests`. Includes `openai` and `tqdm`, which are unused. Nothing is pinned.
- **Impact:** installs are not reproducible, and `pip-audit` has nothing to check against. **Dependency CVEs were not assessed**, because nothing is pinned or installed.
- **Fix:** a `pyproject.toml` with a lock file (uv, pip-tools or Poetry), and `pip-audit` in CI.

#### F-20 — Fake "multi-agent" and "MCP" abstraction
- **Severity:** Low · **Category:** Architecture / maintainability
- **Location:** `protocol.py`, all `*_agent.py` files
- **Impact:** extra ceremony with no benefit (the `receiver` field is never read, and `trace_id` is never correlated). Misleading to readers, and the name collides with the official `mcp` package.
- **Fix:** replace with plain typed functions or services (`ingest()`, `retrieve()`, `answer()`) and typed dataclasses or Pydantic models.

### 10.3 Missing capabilities

| ID | Missing | Priority |
|---|---|---|
| M-01 | Persistence of the vector index, document metadata and chats | High |
| M-02 | Citations (numbered sources with file and page) | High |
| M-03 | Multi-turn conversation (history plus rewriting follow-up questions into standalone queries) | Medium |
| M-04 | Background, idempotent ingestion with status reporting | High |
| M-05 | Delete-document / delete-account operations | High (privacy) |
| M-06 | Evaluation harness (golden Q&A set, recall@K, faithfulness) | High |
| M-07 | Automated tests and CI | High |
| M-08 | Streaming responses | Medium |
| M-09 | OCR for scanned PDFs; table extraction for DOCX and PPTX | Medium |
| M-10 | Dockerfile, health and readiness endpoints, configuration from environment | High |
| M-11 | Rate limiting and per-user LLM budget | Medium |
| M-12 | Structured CSV and tabular Q&A (needed for the README's "average of column X" example) | Low (only if required) |

### 10.4 Optional improvements

- Hybrid search (BM25 plus vector) and a cross-encoder reranker. Add these **only** if the evaluation set shows retrieval is the bottleneck.
- A token-aware chunker that respects headings and sentences.
- A stronger embedding model, chosen by evaluation results rather than by default.
- An OpenAI-compatible client abstraction so the model or provider can be changed through configuration.
- Clean up the repository: remove the empty `architecture.pptx` and the zips, and replace them with a Markdown or Mermaid architecture diagram.

---

## 11. Security Findings Summary

| Check | Result |
|---|---|
| Hardcoded or committed credentials | **Confirmed**: F-01 |
| SQL injection | Not applicable (no database) |
| Unsafe file uploads / path traversal | **Confirmed pattern**: F-07 |
| Missing authentication or authorisation | **Confirmed**: F-11 |
| Cross-user data leakage | **Confirmed by design**: F-10 |
| Prompt injection | **Unmitigated**, limited impact today: F-12 |
| Retrieval-based leakage | Yes. Raw chunks from any user's documents are returned as "sources" (F-10). |
| Unsafe tool execution | Not applicable (no tools) |
| Insecure CORS | Low today: F-14 |
| Sensitive data in logs | **Confirmed**: full document chunks printed (`llm_response_agent.py:39`), F-18 |
| Dependency vulnerabilities | **Not assessed.** Unpinned, not installed (F-19). |
| Parser attack surface | PyMuPDF, python-docx, python-pptx and pandas parse untrusted files in the API process. Recommend size and page limits, and eventually an isolated worker. Potential risk. |

---

## 12. Performance Findings

No benchmarks were run. The following are structural observations with expected effects, not measured results.

| Bottleneck | Location | Expected effect | Recommendation |
|---|---|---|---|
| Full re-ingestion on every query | F-05 | Query latency grows with corpus size times query count | Ingest once at upload time |
| Model reload on every click (UI) | F-15 | Seconds of overhead per question | Cache one embedder per process |
| Duplicate model in memory | F-15 | About twice the embedding-model RAM | Remove the unused copy |
| Synchronous LLM call with no timeout | F-09 | Thread-pool exhaustion under load | Timeouts, an async client, a concurrency limit |
| Duplicate `handle_query` | F-03 | Double the query embedding and search | Remove line 41 |
| `IndexFlatL2` | `store.py:8` | Fine below about 10⁵–10⁶ vectors | Keep. Move to pgvector HNSW or FAISS IVF/HNSW only when measurements show the need. |
| Whole-CSV `to_string()` | `parser.py:28-29` | Large memory spike for big CSVs | Stream rows, cap size |

**Instrumentation to add first:** per-stage timing (parse, embed, search, LLM), recorded as p50/p95 in structured logs or a Prometheus histogram, plus token counts per request. Every later performance decision should be based on these numbers.

---

## 13. Testing Assessment

- **Tests: none.** No `tests/` directory, no pytest configuration, no CI. Nothing could be run.
- Every Critical defect (F-02 to F-04, F-06) would have been caught by the most basic smoke test.
- **Critical missing tests, in priority order:**
  1. `chunk_text` returns strings, with the correct counts, overlap and edge cases (empty text, text shorter than one chunk).
  2. Each parser against a small fixture file per format, including a non-UTF-8 text file and an empty PDF.
  3. Store: add, search and ID alignment; an empty index; dimension mismatch.
  4. Coordinator flow with stubbed embedder and LLM: success, empty query, no documents, LLM failure.
  5. API: upload validation (path traversal, size limit, type), authentication required, response schema.
  6. LLM client with mocked HTTP: timeout, 429 retry, malformed body.
  7. Isolation between users.
  8. Evaluation set: 20–50 questions with expected source passages drawn from the README's sample documents; track recall@5 and a faithfulness check (LLM-as-judge or manual) on every change.
- **Mocking strategy:** inject the embedder and LLM client as interfaces. Use a deterministic fake embedder (for example a hash-based one) in unit tests, and the real MiniLM only in a marked, slower integration suite.

---

## 14. Production Readiness Assessment

| Requirement | Status today | Needed |
|---|---|---|
| Multiple users | No | Authentication, an owner ID on all data, filtered retrieval |
| Multiple organisations | No | Only if the product needs it. Then a `tenant_id` column alongside the owner ID, enforced the same way. Don't build this pre-emptively. |
| Large document collections | No | Persistent store, background ingestion, ANN index once the corpus exceeds about 10⁶ chunks |
| Persistent conversations | No | `conversations` and `messages` tables |
| Concurrent executions | Unsafe | Stateless API workers, single-writer ingestion, LLM concurrency limit |
| Production LLM traffic | No | Timeouts, retries, budgets, usage logging, optional fallback model |
| Monitoring | No | Structured logs, request IDs, per-stage latency, error-rate alerts, token and cost metrics |
| Deployment | No | Dockerfile, environment configuration, health endpoint, CI pipeline |
| Recovery from failure | No | Durable storage, so a restart no longer loses all state; ingestion jobs that can be retried |

**Verdict: not production-ready. The current state is a non-functional prototype.**

---

## 15. Critical Technical Debt

1. A live credential in git history (F-01).
2. The source tree isn't importable, and real code is shipped inside zip archives (F-02).
3. Silent-failure culture: sanitisers and `print`-and-continue hide real bugs. F-04 survived *because* the retrieval sanitiser masked it.
4. No data model. Chunks are bare strings, which blocks citations, deletion, isolation and de-duplication (F-13).
5. Ingestion and querying are coupled in a single call (F-05).
6. No tests and no CI.
7. The documentation describes a system that does not exist (OpenAI GPT, semantic chunking, MCP, multi-agent).

---

## 16. Prioritised Implementation Roadmap

Each phase leaves the system working and tested. Estimates assume one engineer who knows Python and FastAPI.

### Phase 0 — Stop the bleeding (about 1 day)
- Rotate the OpenRouter key; untrack `.env`; add `.env.example` and gitleaks (F-01).
- Restore the package layout, unpack the zips, rename `mcp` (F-02).
- Fix `chunk_text`, the coordinator indentation, the misplaced `return` and the duplicate call (F-03, F-04).
- Unify the result type (F-06). Fix the `.env` loading (F-08).
- Complete and pin dependencies (F-19).
- **Exit criteria:** the smoke test passes, meaning a fixture PDF uploaded and queried produces an answer grounded in that PDF.

### Phase 1 — Correctness and safety baseline (about 1 week)
- Separate ingestion (at upload time, idempotent by content hash) from querying (F-05, F-17).
- A `Chunk` model with metadata, plus numbered citations in the prompt and UI (F-13, M-02).
- Upload hardening: generated filenames, size and type limits (F-07).
- LLM client with timeouts, retries, response validation, `max_tokens` and usage logging (F-09).
- Relevance threshold and a "not found" path (F-16). Delimited sources and an injection-aware system prompt (F-12).
- `logging` with request IDs; stop logging document content (F-18).
- Unit and API tests (§13, items 1–6), with CI on every push.

### Phase 2 — Multi-user and persistence (about 1–2 weeks)
- Authentication (F-11); owner-scoped documents and retrieval (F-10); a restricted CORS list (F-14).
- Postgres for documents, chunks and conversations; vectors in pgvector, or FAISS persisted per collection.
- Delete-document and delete-account endpoints, with a retention policy (M-05).
- Background ingestion worker with status (M-04). A simple database-backed job table is enough to begin with.
- Multi-turn chat with history and rewriting of follow-up questions (M-03); streaming (M-08).
- Dockerfile, health endpoint, environment-based configuration (M-10).

### Phase 3 — Quality, guided by measurement (ongoing)
- Evaluation harness and a regression gate in CI (M-06).
- Then, **only where the evaluation shows a gap:** a better chunker, a different embedding model, hybrid search, a reranker, OCR and table extraction (M-09).
- Optional: one bounded tool-calling loop (maximum 3–5 iterations, a token budget, a tool allow-list), **only if** a concrete requirement such as tabular questions (M-12) or multi-hop questions calls for it.

---

## 17. Recommended Target Architecture

The target is intentionally simple: one service, one database, one worker.

```
            ┌──────────────────────────┐
 Browser ──▶│ Streamlit UI (or SPA)    │──HTTP──┐
            └──────────────────────────┘        │  (UI calls the API; it does
                                                ▼   not import pipeline code)
            ┌──────────────────────────────────────────────┐
            │ FastAPI (stateless, N replicas)              │
            │  auth dependency → owner_id                  │
            │  POST /documents   → store file, enqueue job │
            │  GET  /documents   DELETE /documents/{id}    │
            │  POST /chat (SSE stream)                     │
            │   ├ retrieve(owner_id, query, k, threshold)  │
            │   ├ build_prompt(history, cited sources)     │
            │   └ llm_client.complete(... timeout/retry)   │
            └──────────────┬───────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
 Object/file storage   Postgres + pgvector     Ingestion worker
 (raw uploads,         documents, chunks       parse → chunk → embed
  UUID names)          (owner_id, page, hash,  idempotent by hash,
                        embedding), convos,    status written to DB
                        messages, jobs
```

Design rules:
- **Pipeline code is plain, typed functions** (`ingest`, `retrieve`, `answer`) behind small interfaces (`Embedder`, `VectorStore`, `LLMClient`), injected through configuration. Changing the provider or vector store must not touch business logic.
- **Every retrieval is filtered by `owner_id`**, enforced in the data-access layer and not in the prompt.
- **One embedding model per index**, with the model name and dimension stored alongside the index so a mismatch is detected.
- **Observability from the start:** request ID, per-stage latency, token usage.
- **pgvector instead of a separate vector database**, because it gives transactional deletes, owner filters in SQL, and one fewer system to run. Revisit only if measured scale requires it.
- No message bus, no multi-agent framework, no Kubernetes until the load requires it.

---

## 18. Features Worth Preserving and Features to Redesign

### Worth preserving
- **The multi-format parser** (`parser.py`): a simple, readable dispatch by extension. Extend it with tables, page numbers and limits rather than rewriting it.
- **Local open-source embeddings** (MiniLM): zero cost per call, and private. A sensible default until the evaluation says otherwise.
- **FAISS store's defensive checks** (`store.py:15-25`): dimension and count validation. Keep the idea, but raise exceptions instead of printing.
- **An OpenAI-compatible endpoint through OpenRouter**: keeps provider choice open with minimal code.
- **The basic pipeline shape** (ingest → retrieve → generate), and the README's sample documents and questions, which become the first evaluation set.

### Redesign or remove
| Item | Action | Reason |
|---|---|---|
| `MCPMessage` and the "agent" classes | **Remove** in favour of typed functions and dataclasses | Ceremony with no behaviour; misleading name; package collision |
| `CoordinatorAgent` (and the duplicate flow in `main.py`) | **Redesign** as one `answer()` service function | Broken, and duplicated |
| Ingestion on every query | **Redesign** as upload-time background ingestion | Correctness and performance (F-05) |
| Type-sanitiser in `receive_chunks` | **Remove** | It masked F-04; use types and fail loudly |
| Unused embedding model in `IngestionAgent` | **Remove** | Wasted memory |
| Shared `data/uploads` directory | **Redesign** with owner-scoped storage and UUID names | F-07, F-10 |
| Streamlit importing the pipeline directly | **Redesign** so the UI calls the API | One source of truth, and the backend can be secured |
| `backend.zip`, `frontend.zip`, the empty `architecture.pptx` | **Remove** from the repository; unpack the code into the source tree | Code inside archives can't be reviewed or diffed |
| `GET /query?query=` | **Redesign** as `POST /chat` with a JSON body | Keeps questions out of URLs and logs |
| README | **Rewrite** to describe what is actually built, with setup steps | Accuracy |

---

## 19. Scores and Reasoning

| Area | Score | Reasoning |
|---|---|---|
| **Architecture** | 2/10 | The pipeline decomposition (parse, chunk, embed, store, retrieve, generate) is reasonable, which is why the score isn't 1. But ingestion and querying are coupled, there is no data model, the "agents" add indirection without separating concerns, and the UI and API tiers are alternatives rather than layers. The source tree is unimportable. |
| **Code quality** | 2/10 | Small, readable functions and decent parser dispatch. Against that: a method outside its class, an unconditional `return`, a duplicate call, a statement split across lines that breaks chunking, inconsistent return types, defensive code that hides bugs, `print` debugging throughout, and side effects at import time. |
| **Agent engineering** | 1/10 | There is no agent: no tool use, no decision-making, no loop, no memory. The "multi-agent" and "MCP" framing is nominal only. The 1 rather than 0 reflects that the stage boundaries exist and could host real tools later. |
| **RAG implementation** | 2/10 | The minimum components exist (a chunker, a consistent embedding model, a FAISS index, top-K). The chunker is broken, so effective retrieval is zero. There is no metadata, citations, threshold, de-duplication, persistence or evaluation. |
| **LLM engineering** | 2/10 | The key is read from the environment, the endpoint is OpenAI-compatible, the temperature is low, and HTTP errors are caught. But there is no timeout, retry, `max_tokens`, usage tracking, provider abstraction or injection defence, and the `.env` path is wrong. |
| **Security** | 1/10 | A live key is committed to a GitHub remote. No authentication, a path traversal pattern on upload, cross-user data exposure by design, and document contents printed to logs. Only the absence of tools and a database keeps this from being worse. |
| **Performance** | 2/10 | Exact FAISS search and a local embedding model are fine at prototype scale. The architecture multiplies the work on every query: re-ingestion, a duplicate model, reloads on every UI click, a duplicate search, and an LLM call that can block indefinitely. |
| **Testing** | 0/10 | No tests, fixtures, CI or evaluation of any kind. |
| **Scalability** | 1/10 | State lives in process memory, the index grows on every query, no worker separation, unsynchronised shared state, a single shared corpus, and nothing persisted. |
| **Documentation** | 2/10 | The README explains the intent and includes useful sample documents and questions. It misstates the LLM provider, the chunking method and the "MCP" claim, has no setup or run instructions, and the architecture deck is a 0-byte file. |
| **Production readiness** | 1/10 | It doesn't run as committed, has a committed credential, and has no authentication, persistence, tests, deployment configuration or observability. |

---

## Appendix A — Evidence Log

| Check | Method | Result |
|---|---|---|
| Repository file inventory | `git ls-files`, `ls -la` | 17 tracked files, flat layout, no packages |
| Archive contents | Python `zipfile` listing and read | `backend/api.py`, `frontend/app.py`, and empty `routes/` and `services/` |
| `.env` tracked | `git ls-files -s .env`, `git log -- .env` | Tracked since `d12cc7e`. Key **name** recorded; value deliberately not inspected. |
| Remote | `git remote -v` | GitHub origin. Public visibility not verified. |
| `architecture.pptx` | Size and type | 0 bytes (git's empty-blob hash) |
| Syntax | In-memory `compile()` of all 9 `.py` files | All parse. The defects are semantic, not syntax errors. |
| Coordinator structure | AST inspection | Class methods: `['__init__']`. `handle_user_query` is top-level. |
| `chunk_text` | Function extracted with AST and executed on 700 words | 3 items, type `builtin_function_or_method` |
| End-to-end run | **Not performed** | Dependencies not installed; installing them was out of scope |
| Dependency CVE scan | **Not performed** | Unpinned manifest; nothing installed |

*End of report. Waiting for your approval before implementing anything.*
