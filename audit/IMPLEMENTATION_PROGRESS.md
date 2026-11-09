# Implementation Progress

Status: **all stages complete (2026-11-09).** Nothing is committed: the changes are in the working tree, with legacy-file deletions staged through `git rm`.

## Environment notes (verified)
- **Python version:** the project targets Python 3.11–3.13. Python 3.15 can't install `tokenizers` (no wheel). Development used a uv-managed CPython 3.12 with `.venv`.
- **Blocked venv launcher:** midway through the work, Windows Application Control began blocking `.venv\Scripts\python.exe`. Commands were then run with the uv base interpreter (`%APPDATA%\uv\python\cpython-3.12-…\python.exe`) and `PYTHONPATH=.venv/Lib/site-packages`.
- **No Docker or Postgres installed:** PostgreSQL with pgvector was tested with an embedded server (`pgserver`, pgvector 0.6.2). The Docker image itself was **not built**; the lock was dry-run resolved for Linux x86_64 / Python 3.12 (71 wheels, `torch==2.15.0+cpu`, no CUDA).

## Verification results (final run)
| Check | Result |
|---|---|
| `pytest` (unit + integration + end-to-end, fake embedder and mocked LLM) | 121 passed |
| `pytest -m postgres` (embedded PG + pgvector, HNSW index) | 1 passed |
| `pytest -m slow` (real all-MiniLM-L6-v2: calibration + full flow) | 14 passed |
| All three together | **136 passed** in 86 s |
| `ruff check app tests alembic` | clean |
| `node --check frontend/app.js` | ok |
| Live smoke test: real uvicorn, real MiniLM, local fake LLM server over HTTP | upload → ready in 0.52 s; streamed, cited answer; refusal on an unrelated question; 404 for another user; 413 / 415 limits; delete 204; `/ready` ok |
| Log leak scan of the live run (document text, password, API key, bearer token) | 0 occurrences |

## Not verified
- A real OpenRouter call. The existing key was treated as compromised and not used.
- `docker build` and `docker compose up` (Docker not installed).
- Browser rendering and interaction of the frontend. Only the HTTP serving of its files and a JS syntax check were verified.

## Outstanding actions for the owner
1. Rotate the OpenRouter key exposed in commit `d12cc7e` (see `PHASE_0_IMPLEMENTATION_PLAN.md` C-02).
2. Review and commit the working tree on a branch.
3. Optional: purge `.env` from git history (needs explicit approval and a force-push).
