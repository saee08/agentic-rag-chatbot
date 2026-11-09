// Response normalisation: the ONLY place backend payloads are interpreted.
//
// Every view model produced here is safe to render: text fields are strings (never null,
// undefined, NaN or objects), optional fields are explicitly null, and anything malformed is
// turned into an explicit error state instead of being silently blanked.

const MESSAGE_STATUSES = new Set(["sent", "answered", "no_answer", "no_documents", "direct_reply", "error"]);
const STEP_KINDS = new Set(["search", "library", "assess", "generate", "stop"]);

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const text = (v) => (typeof v === "string" && v.trim() ? v : null);
const int = (v) => (Number.isInteger(v) ? v : null);
const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const bool = (v) => (typeof v === "boolean" ? v : null);

export const INVALID_RESPONSE = "The assistant returned an invalid response.";

export function normalizeSource(raw) {
  if (!isObject(raw)) return null;
  const number = int(raw.number);
  const documentName = text(raw.document_name);
  const content = text(raw.content);
  if (number === null || number < 1 || documentName === null || content === null) return null;
  return {
    number,
    documentId: text(raw.document_id),
    documentName,
    pageNumber: int(raw.page_number),
    chunkId: text(raw.chunk_id),
    excerpt: text(raw.excerpt) ?? content,
    content,
    score: num(raw.score),
    match: text(raw.match),
  };
}

export function normalizeSources(raw) {
  if (!Array.isArray(raw)) return [];
  const byNumber = new Map();
  for (const item of raw) {
    const source = normalizeSource(item);
    if (source && !byNumber.has(source.number)) byNumber.set(source.number, source);
  }
  return [...byNumber.values()].sort((a, b) => a.number - b.number);
}

function normalizeMetadata(raw) {
  if (!isObject(raw)) return null;
  const tokens = isObject(raw.tokens) && int(raw.tokens.total) !== null
    ? { prompt: int(raw.tokens.prompt), completion: int(raw.tokens.completion), total: raw.tokens.total }
    : null;
  const r = isObject(raw.retrieval) ? raw.retrieval : {};
  return {
    model: text(raw.model),
    latencyMs: num(raw.latency_ms),
    llmCalls: int(raw.llm_calls),
    tokens,
    grounded: bool(raw.grounded),
    retrieval: {
      passagesRetrieved: int(r.passages_retrieved),
      passagesCited: int(r.passages_cited),
      documents: int(r.documents),
      pages: int(r.pages),
    },
  };
}

export function normalizeStep(raw) {
  if (!isObject(raw) || !STEP_KINDS.has(raw.kind)) return null;
  const label = text(raw.label);
  return label ? { kind: raw.kind, label, detail: text(raw.detail) } : null;
}

function normalizeSteps(raw) {
  return Array.isArray(raw) ? raw.map(normalizeStep).filter(Boolean) : [];
}

function errorMessage(id, createdAt, code, message, retryable) {
  return {
    id, role: "assistant", status: "error", content: message, sources: [], metadata: null, steps: [],
    error: { code, message, retryable }, createdAt,
  };
}

/**
 * Normalise one message from the API.
 * Returns { ok: true, message } or { ok: false, problem } for payloads that are not messages at all.
 * An assistant message with a missing/blank answer becomes an explicit, retryable error message.
 */
export function normalizeMessage(raw) {
  if (!isObject(raw)) return { ok: false, problem: "not an object" };
  const id = text(raw.id);
  const role = raw.role;
  if (!id || (role !== "user" && role !== "assistant")) return { ok: false, problem: "missing id or role" };
  const createdAt = text(raw.created_at);

  if (role === "user") {
    const content = typeof raw.content === "string" ? raw.content : "";
    return { ok: true, message: { id, role, status: "sent", content, sources: [], metadata: null, steps: [], error: null, createdAt } };
  }

  const status = MESSAGE_STATUSES.has(raw.status) ? raw.status : null;
  if (status === "error" || isObject(raw.error)) {
    const e = isObject(raw.error) ? raw.error : {};
    const message = text(e.message) ?? text(raw.content) ?? "Something went wrong while answering.";
    return { ok: true, message: errorMessage(id, createdAt, text(e.code) ?? "error", message, bool(e.retryable) ?? true) };
  }
  const content = text(raw.content);
  if (status === null || content === null) {
    return { ok: true, message: errorMessage(id, createdAt, "invalid_response", INVALID_RESPONSE, true) };
  }
  return {
    ok: true,
    message: {
      id, role, status, content,
      sources: normalizeSources(raw.sources),
      metadata: normalizeMetadata(raw.metadata),
      steps: normalizeSteps(raw.steps),
      error: null,
      createdAt,
    },
  };
}

/** Normalise a list of messages, skipping (and reporting) entries that are not messages. */
export function normalizeMessages(raw) {
  const messages = [];
  let skipped = 0;
  for (const item of Array.isArray(raw) ? raw : []) {
    const result = normalizeMessage(item);
    if (result.ok) messages.push(result.message);
    else skipped += 1;
  }
  return { messages, skipped };
}

export function normalizeConversation(raw) {
  if (!isObject(raw) || !text(raw.id)) return null;
  return { id: raw.id, title: text(raw.title) ?? "Untitled conversation", updatedAt: text(raw.updated_at) };
}

const DOC_STATUSES = new Set(["pending", "processing", "ready", "failed"]);

export function normalizeDocument(raw) {
  if (!isObject(raw) || !text(raw.id)) return null;
  const fileType = text(raw.file_type) ?? "";
  return {
    id: raw.id,
    filename: text(raw.filename) ?? "Untitled document",
    fileType: fileType.replace(/^\./, "").toUpperCase(),
    sizeBytes: int(raw.size_bytes),
    status: DOC_STATUSES.has(raw.status) ? raw.status : "pending",
    error: text(raw.error),
    pageCount: int(raw.page_count),
    chunkCount: int(raw.chunk_count),
    isSlides: fileType === ".pptx",
  };
}

const STATUS_MESSAGES = {
  0: "Can't reach the server. Check that it is running and try again.",
  401: "Your session has expired. Please sign in again.",
  403: "You don't have permission to do that.",
  404: "That item no longer exists.",
  409: "That action conflicts with the current state. Refresh and try again.",
  413: "The file is too large to upload.",
  415: "That file type isn't supported.",
  422: "Some of the information sent was invalid.",
  429: "You're sending requests too quickly. Please wait a moment.",
  503: "The service is temporarily unavailable. Please try again.",
};

/** Normalise any failed HTTP response (or network failure, status 0) into one shape. */
export function normalizeApiError(status, body) {
  const s = Number.isInteger(status) ? status : 0;
  const e = isObject(body) && isObject(body.error) ? body.error : {};
  const message = text(e.message) ?? STATUS_MESSAGES[s] ?? (s >= 500 ? "The server ran into a problem. Please try again." : "The request failed.");
  return {
    status: s,
    code: text(e.code) ?? (s === 0 ? "network" : "http_" + s),
    message,
    retryable: s === 0 || s === 408 || s === 429 || s >= 500,
  };
}

/** "3 passages · 1 document · 1 page" (only counts that are actually known). */
export function sourceSummary(sources) {
  const docs = new Set(sources.filter((s) => s.chunkId).map((s) => s.documentId ?? s.documentName));
  const pages = new Set(sources.filter((s) => s.chunkId && s.pageNumber !== null)
    .map((s) => `${s.documentId ?? s.documentName}#${s.pageNumber}`));
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const parts = [plural(sources.length, "passage")];
  if (docs.size) parts.push(plural(docs.size, "document"));
  if (pages.size) parts.push(plural(pages.size, "page"));
  return parts.join(" · ");
}

export function formatBytes(bytes) {
  if (bytes === null) return null;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
