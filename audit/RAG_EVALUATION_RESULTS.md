# RAG Evaluation: `rag_test_supply_chain_case_study.pdf`

| | |
|---|---|
| Date | 2026-11-09 |
| Pipeline | The real application: upload → background ingestion → hybrid retrieval → bounded agent → OpenRouter `mistralai/mistral-small-3.2-24b-instruct` → verification |
| Interfaces tested | Streaming API (each question in a fresh conversation), and the browser UI in headless Edge (all six questions in one conversation) |
| Evaluation data | Isolated scratch database. The user's `data/app.db` was not used. |

Nothing in the application is specific to these questions. The prompts contain no facts from this document, and the arithmetic example in the prompt uses unrelated numbers.

## Results

| # | Question | Expected | Runs 1 and 2–3 (before fixes) | Runs 4, 5, 6 + UI (final) |
|---|---|---|---|---|
| 1 | Where is Northstar Home's central distribution centre located? | Pune | ✅ Pune, page 1 | ✅ Pune, page 1, excerpt "Its central distribution centre (DC) is located in Pune." |
| 2 | Supplier and lead time for NH-205? | Bright Appliances, 14 days | ✅ (3 overlapping sources cited) | ✅ Bright Appliances, 14 calendar days, page 1. Overlapping table chunks merged. |
| 3 | Minimum order quantity for NH-205? | Not specified | ❌ Run 1 gave a generic "couldn't find"; the planner gave up after seeing only previews. ✅ in runs 2–3. | ✅ "The documents do not specify the minimum order quantity for NH-205", citing "This document does not specify MOQs…", page 1 |
| 4 | Should the 180-unit mixer delivery be considered confirmed inventory? | No, because the supplier did not confirm the delivery date | ✅ (run 3 showed one irrelevant excerpt) | ✅ "No…", explaining that the supplier did not confirm the final delivery date and that only confirmed quantities count. Page 2. |
| 5 | Exact reorder point for NH-412? | Not specified | ❌ Run 1 gave a generic "couldn't find"; the model ignored the "No reorder point… is defined" passage. ✅ in runs 2–3, but run 2 was terse ("Not specified [1]."). | ✅ "The documents do not specify the exact reorder point for NH-412", citing "No reorder point… is defined by this document", page 2. That passage was found only by **keyword** search (cosine score 0.18). |
| 6 | Target safety stock quantity for NH-205? | 66 units (11 × 6) | ⚠️ Run 1 restated the rule. ❌ **Run 3 stated "72 units — 11 × 6"** (an arithmetic error by the LLM). | ✅ **66 units**, computed by the server from `[[calc: 11 * 6]]` with both inputs cited, page 1. The UI run in a multi-turn conversation also gives 66. |
| control | What is the boiling point of tungsten? | No answer | ✅ | ✅ "I couldn't find information about that in your documents." |

In every final run: no `null`, `undefined`, `NaN` or `[object Object]` in any response field or in the rendered DOM; exactly one reply per question; the streamed answer matches the stored one; reloading reproduces the conversation exactly; and there were zero browser console errors.

## Fixes made in response to these results

| Finding | Root cause | Fix |
|---|---|---|
| Q5's evidence was never retrieved (diagnostic before run 1) | Dense embeddings ranked the only relevant passage last (0.18) | Hybrid retrieval: full-text keyword search plus RRF, with a term-coverage gate so unrelated questions don't match on a single shared word |
| Q3 "couldn't find" (run 1) | The planner judged from 240-character previews and skipped answer generation | The planner may stop searching but can't skip the answer step when evidence exists |
| Q5 "couldn't find" (run 1) | The prompt didn't say that an explicit "not defined" statement is an answer | Answer prompt: "not specified" is a valid answer; `INSUFFICIENT_EVIDENCE` is only for off-topic passages |
| Q6 "72 units" (run 3) | LLM arithmetic | The model writes `[[calc: …]]` and the server evaluates it exactly; explicit wrong equations are recomputed and corrected |
| Q6 restated the rule (run 1 and the first UI run) | No instruction to answer in the unit the question asks for | Prompt rule: answer in the form asked, computing quantities from rules stated in the documents |
| Irrelevant excerpt on one source (run 3, Q4) | Excerpts were scored against the whole answer | Each source's excerpt is scored against the sentences that cite it |
| Overlapping passages shown as separate sources (Q2) | Chunk overlap | Near-identical cited passages from the same page are merged, and citation numbers are updated to match |

## Known observations
- Wording varies between runs, for example "Pune" vs "located in **Pune**", and how much supporting detail is included. The facts and citations were stable across the final runs.
- The PDF's rupee sign extracts as "■" because of a font-encoding issue. It is shown as extracted.
- Token use per run of seven questions, from `metadata.tokens`: about 13k before the fixes (run 1), and 17–23.5k after them (runs 4–6). The increase comes from refined searches and from always generating an answer when evidence exists; that is roughly 2.4–5.8k tokens per question.
