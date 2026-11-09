# Phase 0 Implementation Plan: Stabilise and Make Runnable

| | |
|---|---|
| Based on | `audit/AGENTIC_CHATBOT_PRODUCTION_AUDIT.md` (findings F-01 to F-04, F-06, F-08, F-19, plus a minimal F-07 guard) |
| Repository state re-inspected | 2026-11-03, `main` at `d12cc7e`. The working tree is clean apart from the untracked `audit/`. |
| Status | **PLAN ONLY. Nothing has been implemented.** Waiting for approval. |
| Secrets | No secret values appear in this document. |

## 0. Scope

**Phase 0 covers:** making the code importable and runnable, fixing the four blocking defects, getting the secret out of tracked files, loading configuration correctly, and producing a reproducible dependency manifest with a first unit-test suite.

**Phase 0 does not cover** (left for Phase 1 and later, as the audit sets out): re-ingestion on every query (F-05), the LLM timeout and retries (F-09), user data isolation and authentication (F-10, F-11), structured logging (F-18), metadata and citations, persistence, and removing the agent/`MCPMessage` abstraction (F-20).

**Two scope additions, with reasons:**
1. **A minimal upload filename guard (from F-07).** Phase 0 turns the dormant `/upload` endpoint into running code. Without a guard, Phase 0 would *activate* the path traversal pattern. The guard is a few lines. Full upload hardening stays in Phase 1.
2. **Removing the unused embedding model from `IngestionAgent` (from F-15).** Without this, every unit test that constructs `IngestionAgent` downloads and loads a PyTorch model. Removing it also removes the only direct `import torch`.

---

## 1. Re-inspection results

### 1.1 Tracked files containing secrets

I scanned every revision with `git grep -l` (output limited to file names) for the case-insensitive pattern `api[_-]?key|secret|token|password|bearer|sk-[a-z0-9]`, and checked the zip members by match count only.

| Revision | File | Contains a secret value? |
|---|---|---|
| `d12cc7e` | `.env` | **Yes.** It defines `OPENROUTER_API_KEY` with a value. The value was not inspected. |
| `d12cc7e` | `llm_response_agent.py` | **No.** The match is the variable name in `API_KEY = os.getenv("OPENROUTER_API_KEY")` (line 11). |
| `300d555` | — | No matches. This commit contains only `LICENSE` and `README.md`. |
| `backend.zip:backend/api.py` | — | 0 matches |
| `frontend.zip:frontend/app.py` | — | 0 matches |

`.env` has existed since `d12cc7e`, which was pushed to `origin` (`github.com/saee08/agentic-rag-chatbot`). **Exactly one secret-bearing file is tracked.**

### 1.2 Archive contents

| Archive | Member | Size | Content |
|---|---|---|---|
| `backend.zip` | `backend/` | dir | |
| | `backend/api.py` | 1182 B | FastAPI app with `POST /upload` and `GET /query` |
| | `backend/routes/` | dir | **empty** |
| | `backend/services/` | dir | **empty** |
| `frontend.zip` | `frontend/` | dir | |
| | `frontend/app.py` | 1724 B | Streamlit UI |

### 1.3 Environment

- The only interpreter installed locally is **Python 3.15** (`py -0`).
- **Risk:** I could not verify whether `torch` and `faiss-cpu` publish wheels for Python 3.15. If they don't, installation fails. Step 0 of the execution order (§5) resolves this before any code changes.

---

## 2. Proposed directory tree

**Decisions:**
- **One top-level package, `ragbot/`, at the repository root.** I chose a flat layout over `src/` so that `streamlit run` and `python -m` work from the repository root with no extra setup.
- **The original sub-package names are kept** so the code stays easy to recognise. The one exception is `mcp`, renamed to `messaging`, because `mcp` collides with the official PyPI package.
- **Everything is namespaced under `ragbot.`**, which also stops `parser` and `store` from shadowing other modules.

```
agentic-rag-chatbot/
├── .env.example                  NEW  placeholder-only configuration template
├── .gitignore                    MOD  add .venv/, data/, .pytest_cache/, *.egg-info/
├── .pre-commit-config.yaml       NEW  gitleaks hook
├── LICENSE                       unchanged
├── README.md                     MOD  setup and run instructions only (full rewrite is Phase 1)
├── pyproject.toml                NEW  project metadata and dependencies (replaces requirements.txt)
├── requirements.lock             NEW  fully pinned, generated (see §4.4)
├── requirements.txt              MOD  becomes "-r requirements.lock" for older tooling (or removed, see §4.4)
├── main.py                       MOD  thin CLI demo, imports from ragbot
├── architecture.pptx             unchanged (0-byte file; removal is a separate decision)
├── backend.zip                   unchanged (kept, as instructed)
├── frontend.zip                  unchanged (kept, as instructed)
├── audit/                        audit documents
├── data/                         runtime only, git-ignored (uploads)
├── ragbot/
│   ├── __init__.py               NEW
│   ├── config.py                 NEW  settings and .env loading
│   ├── agents/
│   │   ├── __init__.py           NEW
│   │   ├── coordinator_agent.py  MOVED + FIXED
│   │   ├── ingestion_agent.py    MOVED + FIXED
│   │   ├── retrieval_agent.py    MOVED + FIXED
│   │   └── llm_response_agent.py MOVED + FIXED
│   ├── messaging/
│   │   ├── __init__.py           NEW
│   │   └── protocol.py           MOVED (MCPMessage unchanged; small helper added)
│   ├── file_parser/
│   │   ├── __init__.py           NEW
│   │   └── parser.py             MOVED, unchanged
│   ├── vector_store/
│   │   ├── __init__.py           NEW
│   │   ├── embedder.py           MOVED, unchanged
│   │   └── store.py              MOVED, unchanged
│   ├── api/
│   │   ├── __init__.py           NEW
│   │   └── app.py                EXTRACTED from backend.zip:backend/api.py + FIXED
│   └── ui/
│       ├── __init__.py           NEW
│       └── streamlit_app.py      EXTRACTED from frontend.zip:frontend/app.py + FIXED
└── tests/
    ├── conftest.py               NEW  stub embedder, stub LLM, fixture files
    ├── test_chunking.py          NEW
    ├── test_parser.py            NEW
    ├── test_store.py             NEW
    ├── test_retrieval_agent.py   NEW
    ├── test_coordinator.py       NEW
    ├── test_llm_response_agent.py NEW
    ├── test_config.py            NEW
    ├── test_api.py               NEW
    └── test_smoke_integration.py NEW  marked @pytest.mark.integration (real model; LLM stubbed)
```

The empty `backend/routes/` and `backend/services/` directories are **not** recreated. They hold nothing, and git does not track empty directories. Nothing is lost by leaving them out.

---

## 3. File mapping

| # | Existing | Destination | Import changes |
|---|---|---|---|
| 1 | `coordinator_agent.py` | `ragbot/agents/coordinator_agent.py` | `agents.*` → `ragbot.agents.*` |
| 2 | `ingestion_agent.py` | `ragbot/agents/ingestion_agent.py` | `file_parser.parser` → `ragbot.file_parser.parser`; `mcp.protocol` → `ragbot.messaging.protocol`; drop `sentence_transformers` and `torch` |
| 3 | `retrieval_agent.py` | `ragbot/agents/retrieval_agent.py` | `mcp.*`, `vector_store.*` → `ragbot.*` |
| 4 | `llm_response_agent.py` | `ragbot/agents/llm_response_agent.py` | `mcp.protocol` → `ragbot.messaging.protocol`; add `ragbot.config` |
| 5 | `protocol.py` | `ragbot/messaging/protocol.py` | none |
| 6 | `parser.py` | `ragbot/file_parser/parser.py` | none |
| 7 | `embedder.py` | `ragbot/vector_store/embedder.py` | none |
| 8 | `store.py` | `ragbot/vector_store/store.py` | none |
| 9 | `backend.zip` → `backend/api.py` | `ragbot/api/app.py` | `agents.coordinator_agent` → `ragbot.agents.coordinator_agent` |
| 10 | `frontend.zip` → `frontend/app.py` | `ragbot/ui/streamlit_app.py` | same; remove the `sys.path.append` hack |
| 11 | `main.py` | `main.py` (stays at the root) | `agents.*` → `ragbot.agents.*` |
| 12 | `requirements.txt` | `pyproject.toml` + `requirements.lock` | — |

Root files 1–8 are moved with `git mv` so that `git log --follow` keeps their history. Files 9–10 are new to git, because their history lives only inside the zips.

**Preserved functionality:** every function, class, method and the `MCPMessage` wire format keep their current behaviour, except for the defect fixes described in §4.3. The only behaviour that disappears is the unused second embedding model.

---

## 4. Change specifications

Each change below gives: the existing file, its destination, the current problem, the proposed modification, what it depends on, its regression risks, the unit tests it needs, and its acceptance criteria.

### 4.1 Security

#### C-01 — Stop tracking `.env` (F-01)
- **Existing file:** `.env` (tracked)
- **Destination:** stays on disk, untracked
- **Problem:** the OpenRouter key is committed and pushed. `.gitignore:4` already lists `.env`, but that has no effect on a file that is already tracked.
- **Modification:** `git rm --cached .env`. This removes the file from the index and **keeps the local file**. Commit it with a message saying the key has been rotated.
- **What it does *not* do:** remove the key from history. Commit `d12cc7e` still contains it, and so do any clones or forks. This is why rotation (C-02) is mandatory. You asked me not to rewrite history, so a history purge (`git filter-repo --invert-paths --path .env` plus a force-push) is **listed only as an optional follow-up that needs your separate approval**. Once the key is rotated, a purge adds little.
- **Depends on:** C-02, which must be completed before or alongside this.
- **Regression risk:** none for the code. Anyone who clones the repository afterwards needs `.env.example` (C-03).
- **Tests:** none (repository hygiene). Verified by checks.
- **Acceptance:** `git ls-files .env` prints nothing. `git status` does not list `.env`. The local `.env` still exists.

#### C-02 — Key rotation procedure (manual, done by you)
I can't do this step: it happens in your OpenRouter account.
1. Sign in to the OpenRouter dashboard and open the **API keys** page.
2. **Create a new key.** Set a credit limit on it if your plan allows.
3. Put the new key into your **local** `.env` only. Never commit it, paste it into chat, or put it in an issue.
4. Check that the new key works. After Phase 0 the smoke command in §8 does this; until then, any authenticated request will do.
5. **Delete or disable the old key.**
6. Look through the account's usage and activity for any unexpected spend since **2025-07-23**, the date of commit `d12cc7e`. Report it to OpenRouter if you find any.
7. If GitHub raised a secret-scanning alert for this repository, close it as "revoked".
8. **Acceptance:** a request made with the old key is rejected, and the app works with the new key.

#### C-03 — Create `.env.example`
- **Existing file:** none
- **Destination:** `.env.example` (tracked)
- **Problem:** there is no documented configuration contract.
- **Proposed contents** (placeholders only):
  ```dotenv
  # Copy to .env and fill in. Never commit .env.
  OPENROUTER_API_KEY=replace-me
  # Optional overrides (defaults shown)
  OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
  LLM_MODEL=mistralai/mistral-7b-instruct
  EMBEDDING_MODEL=all-MiniLM-L6-v2
  UPLOAD_DIR=data/uploads
  ```
  The defaults match the current hard-coded values in `llm_response_agent.py:12-13`, `embedder.py:5` and `api.py:7`, so behaviour does not change.
- **Depends on:** C-08 (`config.py` reads these names).
- **Risk:** none.
- **Tests:** `test_config.py` checks that every key in `.env.example` is a field `Settings` knows about, so the template can't drift from the code.
- **Acceptance:** `.env.example` is tracked, contains no real values, and gitleaks passes on it.

#### C-04 — Secret-scanning guardrail
- **Destination:** `.pre-commit-config.yaml` (gitleaks hook); `.gitignore` adds `.venv/`, `data/`, `.pytest_cache/` and `*.egg-info/`
- **Modification:** add a pre-commit gitleaks hook. Pin the hook revision at implementation time; I won't guess a version number here.
- **Risk:** gitleaks *will* flag `d12cc7e:.env` in a full-history scan. That is expected until the key is rotated, and its baseline entry should note "rotated on <date>".
- **Acceptance:** `pre-commit run gitleaks --all-files` passes on the working tree.

### 4.2 Project structure

#### C-05 — Create the package skeleton and move files (F-02)
- **Existing files:** the 8 root modules in rows 1–8 of §3
- **Destination:** as mapped in §3, with an empty `__init__.py` in each package
- **Problem:** the imports reference packages that don't exist, so `ModuleNotFoundError` is raised on the first import.
- **Modification:** `git mv` each file, then rewrite **only the import lines** listed in §3. Leave all other content unchanged in this commit, so the move diff contains nothing but import changes.
- **Depends on:** C-10 (an installable package) for the import tests to run.
- **Regression risks:**
  - A missed import. Mitigation: `python -c "import ragbot.agents.coordinator_agent, ragbot.api.app"` plus a `compileall` check.
  - `parents[1]` in the `.env` lookup changes meaning after the move. C-08 replaces that lookup anyway.
- **Tests:** `test_imports` (inside `test_smoke_integration.py` or a separate fast test) imports every module.
- **Acceptance:** every `ragbot.*` module imports in a fresh virtual environment, and no root-level `.py` file remains except `main.py`.

#### C-06 — Extract the backend and UI from the zips (F-02)
- **Existing files:** `backend.zip:backend/api.py`, `frontend.zip:frontend/app.py`
- **Destinations:** `ragbot/api/app.py`, `ragbot/ui/streamlit_app.py`
- **Problem:** the code inside the archives can't be imported, reviewed or diffed.
- **Modification:**
  - Extract both files byte-for-byte in their own commit, so the extraction can be audited.
  - In a later commit, change only the imports. In `streamlit_app.py`, also delete `sys.path.append(...)` (line 7), which is no longer needed once the package is installed.
  - **The zips stay in the repository untouched**, as you instructed. Deleting them is a separate decision for later (see §9).
- **Depends on:** C-05
- **Regression risk:** the Streamlit launch command changes to `streamlit run ragbot/ui/streamlit_app.py`. This goes in the README (C-12).
- **Tests:** `test_api.py` uses FastAPI's `TestClient`.
- **Acceptance:** after extraction, a SHA-256 of each extracted file equals the SHA-256 of the corresponding zip member (checked before the import edits). Both apps start.

### 4.3 Correctness

#### C-07a — Fix `chunk_text()` (F-04)
- **Existing:** `ingestion_agent.py:9-18` → `ragbot/agents/ingestion_agent.py`
- **Problem:** `chunk = " ".join`, then blank lines, then `(words[i:i + chunk_size])` on its own line. Python reads this as two separate statements, so each chunk is a bound method instead of a string.
- **Modification:**
  ```python
  def chunk_text(text, chunk_size=300, overlap=50):
      if chunk_size <= 0 or not 0 <= overlap < chunk_size:
          raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")
      words = text.split()
      step = chunk_size - overlap
      chunks = []
      for i in range(0, len(words), step):
          chunks.append(" ".join(words[i:i + chunk_size]))
          if i + chunk_size >= len(words):
              break
      return chunks
  ```
- **Notes on this version:**
  - The parameter check matters because with `overlap >= chunk_size`, the old loop either raised an opaque `range()` error (step 0) or silently returned no chunks (negative step).
  - The `break` stops a redundant tail chunk. Without it, a 300-word text yields `[0:300]` plus `[250:300]`, and the second chunk is entirely contained in the first.
  - **This is a deliberate small change to the intended behaviour.** If you'd rather keep the original loop exactly, drop the `break`; the tests in that case would assert the original chunk counts.
- **Depends on:** none
- **Risk:** chunk counts drop by at most one per document. Nothing persists chunks, so no stored data is affected.
- **Unit tests** (`test_chunking.py`):
  - `""` → `[]`.
  - 10 words → one chunk equal to the input, with whitespace normalised.
  - 300 words → exactly 1 chunk.
  - 700 words → 3 chunks: `[0:300]`, `[250:550]`, `[500:700]`.
  - Every element is a `str`. This is the direct regression test for F-04.
  - Consecutive chunks overlap by exactly `overlap` words.
  - Every input word appears in at least one chunk.
  - `overlap >= chunk_size` and `chunk_size <= 0` raise `ValueError`.
- **Acceptance:** all of the above pass. Running the AST check from the audit again now shows `str` values.

#### C-07b — Remove the unused model from `IngestionAgent` (F-15, test enabler)
- **Existing:** `ingestion_agent.py:5-6,25-31`
- **Problem:** a second MiniLM model is loaded and never used. That wastes memory, forces a model load in every unit test, and is the only direct use of `torch`.
- **Modification:** delete the `SentenceTransformer` and `torch` imports and the model-loading lines. Keep `self.name` and the start-up print.
- **Risk:** none. A search for `self.model` in `ingestion_agent.py` shows it is never read outside `__init__`; this will be re-checked with `grep` at implementation time.
- **Tests:** `IngestionAgent()` constructs without importing `torch`. `process_documents` on fixture files returns an `MCPMessage` whose chunks are strings. A file that fails to parse is skipped, and the remaining files are still processed.
- **Acceptance:** `"torch" not in sys.modules` after importing and constructing the agent in a fresh interpreter.

#### C-07c — Fail loudly on bad chunks; skip embedding when there is nothing to embed (F-04 follow-up)
- **Existing:** `retrieval_agent.py:20-44` → `ragbot/agents/retrieval_agent.py`
- **Problem:** the type sanitiser silently dropped every chunk, which is what hid F-04. An empty list still reaches `embed([])`.
- **Modification:**
  - Replace the sanitiser loop with a check: if any chunk is not a `str`, raise `TypeError("chunks must be str, got …")`.
  - Return early, before calling the embedder, when there are no chunks.
  - Add optional constructor injection: `RetrievalAgent(name=..., embedder=None, store=None)`. When either argument is `None`, the agent uses the current default, so existing callers are unaffected.
- **Risk:** the sanitiser used to convert non-string values (for example numbers) to `str`. After C-07a, chunks are always strings, so that conversion was a no-op and a hard error is the safer behaviour.
- **Tests (stub embedder):**
  - Valid strings are added to the store, with `ntotal` equal to the number of chunks.
  - A non-string chunk raises `TypeError`.
  - An empty list leads to zero embedder calls.
  - `handle_query("")` returns `None`.
  - `handle_query` on an empty index returns a message with `retrieved_context == []`.
  - `handle_query` passes `top_k` through to the store.
- **Acceptance:** all tests pass with no model download.

#### C-07d — Fix `CoordinatorAgent`: method placement, misplaced `return`, duplicate retrieval (F-03)
- **Existing:** `coordinator_agent.py:8-54` → `ragbot/agents/coordinator_agent.py`
- **Problems:**
  - `handle_user_query` is defined at module level (line 17).
  - The `return` at line 45 sits outside its `if` block, so the function always returns the failure result.
  - `handle_query` is called twice (lines 39 and 41).
  - The query is validated only *after* the expensive ingestion step.
- **Modification:** indent the method into the class, delete line 41, and put the `return` inside the `if`. Move query validation to the top of the method. Add optional constructor injection (`ingestion_agent=None, retrieval_agent=None, llm_agent=None`). Pass the ingestion message's `trace_id` through instead of generating a third one.
- **Target flow:**
  ```
  validate query           → error message, status "invalid_query"   (no ingestion)
  ingest                   → if no chunks: error message, status "no_content"
  receive_chunks
  handle_query (once)      → if None: error message, status "retrieval_failed"
  generate_answer          → return its MCPMessage (status "ok" or "llm_error")
  ```
- **Depends on:** C-07e (return type)
- **Risk:** the order of the checks changes. An empty query now returns before the files are touched, which is strictly cheaper and has no other visible effect. The error message strings stay the same.
- **Tests (`test_coordinator.py`, all agents stubbed):**
  - `handle_user_query` exists as a method (`hasattr(CoordinatorAgent, "handle_user_query")`).
  - On the happy path, `handle_query` is called **once** and `generate_answer` is called **once** with its result, and the LLM message is returned.
  - An empty or whitespace-only query returns `invalid_query`, and ingestion is never called.
  - No chunks returns `no_content`, and retrieval and the LLM are never called.
  - Retrieval returning `None` gives `retrieval_failed`, and the LLM is never called.
  - Every path returns an `MCPMessage`.
- **Acceptance:** all tests pass, and the AST check now lists `['__init__', 'handle_user_query']` as the class methods.

#### C-07e — Standardise return types (F-06)
- **Existing:** `coordinator_agent.py` (dict returns), `llm_response_agent.py` (`MCPMessage`), `api.py:35-36`, `app.py:43-51`
- **Problem:** the coordinator returns either a dict or an `MCPMessage`. The API reads `.payload`; the UI reads `["payload"]`. Each crashes on the other type.
- **Decision:** **every path returns an `MCPMessage`**. The payload contract is `{"answer": str, "sources": list[str], "status": str}`, where `status` is one of `ok`, `invalid_query`, `no_content`, `retrieval_failed`, `llm_error`, `invalid_input`. This keeps the existing message type, so the change is minimal; replacing it with typed models belongs to Phase 1 (F-20).
- **Modification:**
  - Add a `MCPMessage.error(sender, receiver, trace_id, answer, status)` helper in `protocol.py`. **The constructor and `to_dict()` stay unchanged.**
  - `LLMResponseAgent` adds a `status` to its three return paths.
  - API: return `result.payload["answer"]` and `result.payload["sources"]`, plus `status`. Respond with HTTP 400 for `invalid_query` and 200 otherwise. This keeps the current response shape and adds one field.
  - UI: `final_msg.payload[...]` replaces subscripting. Show `st.warning` or `st.error` according to `status`.
- **Risk:** API clients now see an extra `status` field (additive) and a 400 for empty queries. Today empty queries crash with a 500, so the 400 is an improvement.
- **Tests:** coordinator tests assert `isinstance(result, MCPMessage)` and the payload keys on every path. API tests cover 200 with `{answer, sources, status}` and 400 for an empty query. LLM agent tests check the `status` on the success, `RequestException` and malformed-input paths.
- **Acceptance:** none of the three consumers (`main.py`, the API, the UI) branches on the result type any more.

#### C-07f — Minimal upload filename guard (activation safety from F-07)
- **Existing:** `backend/api.py:24-27`, `frontend/app.py:24-27` → `ragbot/api/app.py`, `ragbot/ui/streamlit_app.py`
- **Problem:** Phase 0 makes this code live, and it joins a client-supplied filename onto the upload path.
- **Modification:** a shared helper `ragbot.file_parser.safe_upload_path(upload_dir, filename)`:
  - takes `os.path.basename` after normalising `\` to `/`;
  - rejects empty names, `.` and `..` with `ValueError`, which the API maps to HTTP 400;
  - resolves the result and asserts it is inside `upload_dir`.
  Both the API and the UI use it. **Size limits, type allow-lists and UUID storage names remain Phase 1.**
- **Risk:** a file whose name contains path components is now stored under its base name only. That is the intended behaviour.
- **Tests:** `safe_upload_path(d, "../x.txt")`, `"/etc/x"`, `"C:\\x\\y.txt"` and `"a/../../b.txt"` all resolve inside `d`. `""` and `".."` raise. An API upload of `../evil.txt` ends up at `data/uploads/evil.txt`.
- **Acceptance:** all pass. No test run writes any file outside `tmp_path`.

### 4.4 Configuration

#### C-08 — Central configuration and environment loading (F-08)
- **Existing:** `llm_response_agent.py:1-15` (side effects at import time, a wrong `parents[1]` path, module-level `API_KEY`)
- **Destination:** a new `ragbot/config.py`, used by the LLM agent, the API and the UI
- **Problem:** `.env` is looked up relative to the module file, which is wrong both in the flat layout and in the new one. The key is read once, at import time, which makes it impossible to test without import tricks.
- **Modification:**
  ```python
  # ragbot/config.py
  @dataclass(frozen=True)
  class Settings:
      openrouter_api_key: str | None
      openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"
      llm_model: str = "mistralai/mistral-7b-instruct"
      embedding_model: str = "all-MiniLM-L6-v2"
      upload_dir: str = "data/uploads"

  def get_settings() -> Settings:   # cached
      load_dotenv(find_dotenv(usecwd=True), override=False)
      ...read os.environ with the defaults above...
  ```
  - The real environment wins over `.env` (`override=False`), so deployments can set variables directly.
  - `.env` is found by searching up from the **current working directory**, not from the module file.
  - `LLMResponseAgent.generate_answer` reads `get_settings()` at **call time**. If the key is missing, it returns `status="llm_error"` with "OPENROUTER_API_KEY is not set" instead of sending a request that fails with 401. The start-up print still reports only whether the key is present, never its value.
  - `Embedder()` and `FAISSVectorStore()` keep their current defaults. Wiring `embedding_model` through is optional and has the same default, so behaviour is identical.
- **Depends on:** C-10 (`python-dotenv` declared as a dependency)
- **Risk:** someone who previously relied on a `.env` one level above the repository (the accidental current behaviour) now needs it in the repository root. The README documents this.
- **Tests (`test_config.py`, using `monkeypatch` and `tmp_path`):**
  - A `.env` in the current directory is loaded.
  - A real environment variable overrides `.env`.
  - The defaults apply when variables are unset.
  - A missing key makes `generate_answer` return `llm_error` without any HTTP call (asserted with a mocked `requests.post`).
  - `.env.example` keys match the `Settings` fields.
  - **No test reads the developer's real `.env`.** Every test changes into a `tmp_path` directory and clears the relevant variables.
- **Acceptance:** importing `ragbot.agents.llm_response_agent` produces no output and reads no files. The configuration is read when first used.

#### C-09 — Dependency inventory (F-19)
Built from the actual `import` statements, cross-checked against `requirements.txt`:

| Import (file) | Distribution | In `requirements.txt`? | Action |
|---|---|---|---|
| `fitz` (parser) | `PyMuPDF` | yes | keep |
| `docx` (parser) | `python-docx` | yes | keep |
| `pptx` (parser) | `python-pptx` | yes | keep |
| `pandas` (parser) | `pandas` | yes | keep |
| `sentence_transformers` (embedder) | `sentence-transformers` | yes | keep |
| `faiss` (store) | `faiss-cpu` | yes | keep |
| `numpy` (store, retrieval) | `numpy` | **no** (transitive only) | **add as a direct dependency** |
| `requests` (LLM agent) | `requests` | **no** | **add** |
| `dotenv` (LLM agent, config) | `python-dotenv` | **no** | **add** |
| `fastapi` (api) | `fastapi` | **no** | **add** |
| — (to serve the API) | `uvicorn` | **no** | **add** |
| `UploadFile`/`File` form parsing (api) | `python-multipart` | **no** | **add** (FastAPI requires it for `File(...)`) |
| `streamlit` (ui) | `streamlit` | yes | keep |
| `torch` (ingestion, removed by C-07b) | `torch` | no (transitive via sentence-transformers) | leave transitive |
| — | `openai` | yes, **unused** | **remove** |
| — | `tqdm` | yes, **unused directly** | **remove** (still installed transitively) |
| tests | `pytest`, `httpx` (needed by FastAPI `TestClient`) | — | dev extra |
| tooling | `pre-commit`, `pip-audit` | — | dev extra |

#### C-10 — Reproducible manifest
- **Existing:** `requirements.txt` (unpinned)
- **Destination:** `pyproject.toml` with dependency ranges, plus `requirements.lock` with exact pins. The lock is generated, never written by hand.
- **Modification:**
  - `pyproject.toml`: `[project]` with `name = "ragbot"`, `requires-python` (see below), the runtime dependencies from C-09 **listed without guessed version numbers**, `[project.optional-dependencies] dev = [...]`, `[tool.setuptools.packages.find] include = ["ragbot*"]`, and `[tool.pytest.ini_options] markers = ["integration"]` with `addopts = "-m 'not integration'"`.
  - Generate the lock with **uv** (`uv pip compile pyproject.toml --extra dev -o requirements.lock`), or with `pip-tools` if you prefer. Commit the lock. Lower bounds are set from the versions that actually resolve, not from guesses.
  - Replace `requirements.txt` with a single line, `-r requirements.lock`, so existing instructions keep working. Alternatively, delete it; that is your choice, and my recommendation is to keep the one-line version.
  - **Python version:** at Step 0 (§5), check whether `torch` and `faiss-cpu` resolve on Python 3.15. If they don't, install the newest Python for which both publish wheels (for example with `uv python install`) and set `requires-python` to that range. I am **not** claiming either way about 3.15; it has to be checked.
- **Risk:** resolving the dependencies requires network access and an install. **This needs your approval** under the audit's rules; without it, I can only write the manifest and not verify it.
- **Tests:** CI or a local `pip install -r requirements.lock && pip install -e . --no-deps && pytest` in a fresh virtual environment.
- **Acceptance:** a fresh virtual environment built from the lock alone passes the unit suite. `pip-audit -r requirements.lock` runs, and any findings are recorded rather than necessarily fixed in Phase 0.

#### C-11 — Update `main.py`
- **Existing:** `main.py`
- **Modification:** change the imports to `ragbot.agents.*`. Create `data/uploads` if it is missing and print a hint when it is empty. Read the query from `sys.argv[1]`, keeping the current hard-coded question as the default. Run through `CoordinatorAgent` rather than duplicating the pipeline, so there is one code path. Print `status`, `answer` and `sources` from the `MCPMessage`.
- **Risk:** the console output format changes slightly.
- **Acceptance:** `python main.py "question"` with a fixture file in `data/uploads` prints an answer, or an `llm_error` status if no key is set.

#### C-12 — Minimal README update
- Add these sections: Setup (virtual environment, install from the lock, `cp .env.example .env`), Run (CLI, `uvicorn ragbot.api.app:app`, `streamlit run ragbot/ui/streamlit_app.py`) and Test (`pytest`, plus `pytest -m integration`).
- Correct "OpenAI GPT" to "OpenRouter (default: mistralai/mistral-7b-instruct)" and close the unclosed JSON code block.
- The full README rewrite stays in Phase 1.

---

## 5. Execution order

| Step | Change | Why it comes here |
|---|---|---|
| **0** | **You:** rotate the key (C-02). **I, after approval:** check whether the dependencies resolve on Python 3.15 (C-10, dry run) | The security exposure is happening now. The Python version decision affects everything after it. |
| 1 | Create a branch and tag the baseline (§6) | Rollback point |
| 2 | C-01, C-03, C-04: untrack `.env`, add `.env.example`, gitleaks, `.gitignore` | Gets the secret out of tracked files before any further commits |
| 3 | C-09, C-10: `pyproject.toml`, the lock, a working virtual environment | Nothing can be tested until the code installs |
| 4 | C-05: `git mv` plus import rewrites only | Keeps the move diff readable on its own |
| 5 | C-06: byte-for-byte extraction from the zips, then their import fixes | The extraction can be audited by hash |
| 6 | Test scaffolding (`tests/conftest.py` with a stub embedder and a stub LLM) | Each later fix arrives with its tests |
| 7 | C-07a and C-07b: chunking, plus removing the unused model | Pure functions, no dependencies |
| 8 | C-07c: retrieval validation and injection | Depends on step 7 |
| 9 | C-08: configuration and environment loading | Needed by C-07e's `llm_error` path |
| 10 | C-07e, then C-07d: return contract, then the coordinator | The coordinator is built on the contract |
| 11 | C-07f: upload guard (API and UI) | Must land before the API is declared runnable |
| 12 | C-11, C-12: `main.py`, README | Documentation reflects the final commands |
| 13 | Full verification (§8), then open a PR | |

---

## 6. Git branch and commit strategy

- **Branch:** `phase-0/stabilize` from `main` at `d12cc7e`.
- **Baseline tag:** `pre-phase-0`, pointing at `d12cc7e`. It is local, and pushed only with your approval.
- **One commit per logical change.** Each commit leaves the tree importable from step 4 onward and keeps tests green from step 6 onward:
  1. `chore(security): stop tracking .env; add .env.example and gitleaks hook`
  2. `build: add pyproject.toml and pinned requirements.lock`
  3. `refactor: move modules into ragbot package (imports only)`
  4. `chore: extract backend/api.py and frontend/app.py from archives (verbatim)`
  5. `refactor: wire extracted API and UI into ragbot package`
  6. `test: add pytest scaffolding with stub embedder and LLM`
  7. `fix(ingestion): chunk_text returns strings; drop unused embedding model`
  8. `fix(retrieval): reject non-str chunks; allow dependency injection`
  9. `fix(config): load settings from cwd .env at call time`
  10. `fix(coordinator): make handle_user_query a method; fix return; single retrieval; uniform MCPMessage results`
  11. `fix(upload): constrain uploaded filenames to the upload directory`
  12. `docs: setup, run and test instructions`
- **Commit trailer:** every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Nothing is pushed and no PR is opened without your explicit approval.** There is no history rewrite and no force-push.
- **Merge:** a PR into `main`, reviewed by you. Use a merge commit or rebase-merge, **not** squash, so that the `git mv` history and the per-fix commits stay separate.

---

## 7. Rollback strategy

| Scenario | Rollback |
|---|---|
| Work in progress goes wrong | Drop or reset the branch. `main` is untouched until the merge. |
| A single fix causes a regression | `git revert <commit>`. The commits are independent from step 7 onward. |
| The merged Phase 0 needs to be backed out | `git revert -m 1 <merge-commit>` on `main`. This is non-destructive, and the tag `pre-phase-0` marks the old state. |
| The dependency lock breaks someone's environment | Revert commit 2. The old `requirements.txt` comes back with it. |
| **Untracking `.env` (commit 1)** | Reverting it would re-add the file to tracking. **Never revert this commit.** If it has to be undone, restore only `.env.example`, `.gitignore` and the hook manually. |
| **Key rotation** | Not reversible, and not meant to be. The old key stays revoked. |
| The zip archives | Never modified, so nothing to roll back |

---

## 8. Phase 0 completion checklist

**Security**
- [ ] A new OpenRouter key has been issued and the old key revoked. A request with the old key is rejected. (Done by you.)
- [ ] Usage since 2025-07-23 has been reviewed for unexpected spend. (Done by you.)
- [ ] `git ls-files .env` prints nothing, and the local `.env` still exists.
- [ ] `.env.example` is tracked with placeholders only.
- [ ] `pre-commit run gitleaks --all-files` passes on the working tree.
- [ ] No test reads the real `.env`, and no secret value appears in any commit, log or document.

**Structure**
- [ ] All modules live under `ragbot/` with `__init__.py` files. No module is named `mcp`.
- [ ] The SHA-256 of each extracted file matches its zip member.
- [ ] `backend.zip`, `frontend.zip` and `architecture.pptx` are unchanged.
- [ ] `git log --follow ragbot/agents/coordinator_agent.py` shows the original history.

**Correctness**
- [ ] `chunk_text` returns `list[str]` (regression test for F-04).
- [ ] `CoordinatorAgent.handle_user_query` is a method. Retrieval and the LLM each run exactly once on the happy path.
- [ ] Every coordinator path returns an `MCPMessage` with `answer`, `sources` and `status`.
- [ ] The API returns 200 `{answer, sources, status}`, and 400 for an empty query.
- [ ] Uploading `../evil.txt` is stored inside `data/uploads`.

**Configuration and build**
- [ ] A fresh virtual environment installs from `requirements.lock`, and `pip install -e .` succeeds.
- [ ] `pytest` (unit suite) passes, with the output recorded in the PR.
- [ ] `pytest -m integration` passes: real MiniLM model, stubbed LLM, a fixture text file whose distinctive sentence comes back in `sources`.
- [ ] `pip-audit -r requirements.lock` has been run and its results recorded.
- [ ] Manual smoke tests with the **new** key:
  - `python main.py "<question about fixture>"` returns `status ok` and an answer grounded in the fixture.
  - `uvicorn ragbot.api.app:app` serves `/upload` and `/query`.
  - `streamlit run ragbot/ui/streamlit_app.py` answers a question.
- [ ] The README documents setup, running and testing.

**Known limitations remaining after Phase 0** (state these in the PR description): re-ingestion on every query, with the index growing in the API process (F-05); no LLM timeout or retries (F-09); no authentication or user isolation (F-10, F-11); no upload size or type limits (rest of F-07); full document text still printed to stdout (F-18); no citations or persistence. **The API must not be exposed beyond localhost until Phase 2.**

---

## 9. Decisions needed before I implement

1. **Approve the overall plan.**
2. **Network and installs:** may I create a virtual environment and install the dependencies, needed to resolve the lock and run the tests? Without this, I can write the code but can't verify it.
3. **Python version:** if `torch` or `faiss-cpu` don't resolve on 3.15, may I install an older Python through `uv`?
4. **Chunking tail fix (C-07a):** take the `break` that removes the redundant tail chunk (my recommendation), or keep the original loop exactly?
5. **`requirements.txt`:** reduce it to one line, `-r requirements.lock` (my recommendation), or delete it?
6. **Later, separately (not part of Phase 0):** delete the zips and the empty `architecture.pptx` once the extraction is verified, and decide whether to purge `.env` from git history.
