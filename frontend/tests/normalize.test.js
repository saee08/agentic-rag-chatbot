import { test } from "node:test";
import assert from "node:assert/strict";
import {
  INVALID_RESPONSE, normalizeApiError, normalizeDocument, normalizeMessage, normalizeMessages, normalizeSources, sourceSummary,
} from "../js/normalize.js";

const SOURCE = {
  number: 1, document_id: "d1", document_name: "rag_test_supply_chain_case_study.pdf", page_number: 1,
  chunk_id: "c1", excerpt: "Its central distribution centre (DC) is located in Pune.", content: "… full passage …",
  score: 0.63, match: "semantic+keyword",
};
const VALID = {
  id: "m2", role: "assistant", status: "answered", content: "The DC is in **Pune** [1].",
  sources: [SOURCE],
  metadata: { model: "mistral", latency_ms: 1200, llm_calls: 1, tokens: { prompt: 900, completion: 30, total: 930 },
              grounded: true, retrieval: { passages_retrieved: 4, passages_cited: 1, documents: 1, pages: 1 } },
  steps: [{ kind: "search", label: "Searched your documents", detail: "3 relevant passages" }],
  error: null, created_at: "2026-11-09T10:00:00Z",
};

/** Deep-scan a view model for values that must never reach the screen. */
function assertRenderSafe(value, path = "message") {
  if (value === undefined) assert.fail(`${path} is undefined`);
  if (typeof value === "number") assert.ok(Number.isFinite(value), `${path} is not finite`);
  if (typeof value === "string") assert.ok(!/^(null|undefined|NaN|\[object Object\])$/.test(value), `${path} = ${value}`);
  if (value && typeof value === "object") for (const [k, v] of Object.entries(value)) assertRenderSafe(v, `${path}.${k}`);
}

test("valid answer", () => {
  const { ok, message } = normalizeMessage(VALID);
  assert.ok(ok);
  assert.equal(message.content, "The DC is in **Pune** [1].");
  assert.equal(message.sources[0].pageNumber, 1);
  assert.equal(message.sources[0].documentName, "rag_test_supply_chain_case_study.pdf");
  assert.equal(message.metadata.retrieval.passagesRetrieved, 4);
  assert.equal(message.steps.length, 1);
  assertRenderSafe(message);
});

test("missing answer becomes an explicit, retryable error (not an empty bubble)", () => {
  const { content, ...noContent } = VALID;
  const { ok, message } = normalizeMessage(noContent);
  assert.ok(ok);
  assert.equal(message.status, "error");
  assert.equal(message.error.code, "invalid_response");
  assert.equal(message.content, INVALID_RESPONSE);
  assert.equal(message.error.retryable, true);
  assertRenderSafe(message);
});

test("null and blank answers are treated like missing answers", () => {
  for (const content of [null, "", "   ", 42, { text: "x" }]) {
    const { message } = normalizeMessage({ ...VALID, content });
    assert.equal(message.status, "error", `content=${JSON.stringify(content)}`);
    assertRenderSafe(message);
  }
});

test("missing sources", () => {
  const { sources, ...rest } = VALID;
  const { message } = normalizeMessage(rest);
  assert.deepEqual(message.sources, []);
  assertRenderSafe(message);
});

test("empty and null sources", () => {
  for (const sources of [[], null, "nope", {}]) {
    assert.deepEqual(normalizeMessage({ ...VALID, sources }).message.sources, []);
  }
});

test("sources that cannot be identified are dropped; duplicates removed; order by number", () => {
  const sources = normalizeSources([
    { ...SOURCE, number: 2 }, null, "junk", { ...SOURCE, document_name: null }, { ...SOURCE, content: "" },
    { ...SOURCE, number: "1" }, SOURCE, { ...SOURCE, excerpt: "duplicate number" },
  ]);
  assert.deepEqual(sources.map((s) => s.number), [1, 2]);
  assert.equal(sources[0].excerpt, SOURCE.excerpt);
});

test("missing source metadata uses explicit nulls and safe fallbacks", () => {
  const [source] = normalizeSources([{ number: 1, document_name: "a.pdf", content: "Full text." }]);
  assert.equal(source.pageNumber, null);
  assert.equal(source.score, null);
  assert.equal(source.excerpt, "Full text.");
});

test("missing or malformed message metadata", () => {
  for (const metadata of [undefined, null, "x", { retrieval: null, tokens: "many", latency_ms: NaN }]) {
    const { message } = normalizeMessage({ ...VALID, metadata });
    if (metadata && typeof metadata === "object") {
      assert.equal(message.metadata.tokens, null);
      assert.equal(message.metadata.latencyMs, null);
      assert.equal(message.metadata.retrieval.passagesRetrieved, null);
    } else {
      assert.equal(message.metadata, null);
    }
    assertRenderSafe(message);
  }
});

test("server error messages keep their real message and retryability", () => {
  const { message } = normalizeMessage({
    id: "e1", role: "assistant", status: "error", content: "The model 'x' was not found.",
    error: { code: "llm_not_configured", message: "The model 'x' was not found.", retryable: false },
  });
  assert.equal(message.status, "error");
  assert.deepEqual(message.error, { code: "llm_not_configured", message: "The model 'x' was not found.", retryable: false });
});

test("unexpected structures are rejected or converted, never rendered raw", () => {
  for (const raw of [null, undefined, "text", 42, [], {}, { id: "x" }, { id: "x", role: "system" }]) {
    assert.equal(normalizeMessage(raw).ok, false);
  }
  const { message } = normalizeMessage({ id: "m", role: "assistant", status: "weird_status", content: "Hi" });
  assert.equal(message.status, "error");
  const list = normalizeMessages([VALID, null, { foo: 1 }, { id: "u", role: "user", content: "Q?" }]);
  assert.equal(list.messages.length, 2);
  assert.equal(list.skipped, 2);
  assert.deepEqual(normalizeMessages("not a list"), { messages: [], skipped: 0 });
});

test("failed API responses normalise to one shape", () => {
  assert.deepEqual(normalizeApiError(503, { error: { code: "llm_unavailable", message: "Down." } }),
    { status: 503, code: "llm_unavailable", message: "Down.", retryable: true });
  assert.equal(normalizeApiError(0, null).code, "network");
  assert.equal(normalizeApiError(0, null).retryable, true);
  const html = normalizeApiError(502, "<html>Bad gateway</html>");
  assert.equal(html.message, "The server ran into a problem. Please try again.");
  assert.equal(normalizeApiError(422, { error: { message: null } }).message, "Some of the information sent was invalid.");
  assert.equal(normalizeApiError(400, {}).message, "The request failed.");
  assert.equal(normalizeApiError(429, null).retryable, true);
  assert.equal(normalizeApiError(404, null).retryable, false);
  assert.equal(normalizeApiError(undefined, undefined).status, 0);
});

test("source summary distinguishes passages, documents and pages", () => {
  const sources = normalizeSources([
    SOURCE, { ...SOURCE, number: 2, chunk_id: "c2" }, { ...SOURCE, number: 3, chunk_id: "c3", page_number: 2 },
    { ...SOURCE, number: 4, chunk_id: "c4" },
  ]);
  assert.equal(sourceSummary(sources), "4 passages · 1 document · 2 pages");
  assert.equal(sourceSummary(normalizeSources([{ number: 1, document_name: "Library", content: "x" }])), "1 passage");
});

test("documents normalise status and missing fields", () => {
  const doc = normalizeDocument({ id: "d", filename: null, file_type: ".pptx", status: "exploded", page_count: null });
  assert.equal(doc.filename, "Untitled document");
  assert.equal(doc.status, "pending");
  assert.equal(doc.fileType, "PPTX");
  assert.equal(doc.isSlides, true);
  assert.equal(normalizeDocument(null), null);
});
