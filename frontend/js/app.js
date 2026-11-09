// Lumen document assistant: application controller.
import { ApiError, createClient, uploadFiles } from "./api.js";
import { h, setChildren, renderMarkdown } from "./dom.js";
import { parseBlocks } from "./markdown.js";
import {
  formatBytes, normalizeConversation, normalizeDocument, normalizeMessage, normalizeMessages, sourceSummary,
} from "./normalize.js";

const TOKEN_KEY = "lumen.token";
const THEME_KEY = "lumen.theme";
const POLL_MS = 2000;

const state = {
  token: localStorage.getItem(TOKEN_KEY),
  user: null,
  documents: [],
  conversations: [],
  currentId: null,
  messages: [],
  busy: false, // one question at a time: prevents duplicate submissions
  pollTimer: null,
  authMode: "login",
};

const $ = (id) => document.getElementById(id);
const api = createClient({ getToken: () => state.token, onUnauthorized: () => signOut(false) });

const ICONS = {
  info: "M12 8h.01M11 12h1v4h1M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z",
  warn: "M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  chevron: "M9 6l6 6-6 6",
  copy: "M9 9h10v10H9zM5 15V5h10",
  retry: "M4 4v6h6M20 20v-6h-6M5.5 15a7 7 0 0 0 12.4 2M18.5 9A7 7 0 0 0 6.1 7",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
};
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", ICONS[name]);
  svg.append(path);
  return svg;
}

function toast(message, kind = "info") {
  const node = h("div", { class: `toast${kind === "error" ? " toast-error" : ""}`, role: kind === "error" ? "alert" : "status" }, message);
  $("toasts").append(node);
  setTimeout(() => node.remove(), kind === "error" ? 7000 : 4000);
}

// ---------------------------------------------------------------------------------------------
// Theme

function applyTheme(theme) {
  const value = ["light", "dark", "system"].includes(theme) ? theme : "system";
  document.documentElement.dataset.theme = value;
  $("theme-select").value = value;
  localStorage.setItem(THEME_KEY, value);
}

// ---------------------------------------------------------------------------------------------
// Auth

function setAuthMode(mode) {
  state.authMode = mode;
  const login = mode === "login";
  $("tab-login").setAttribute("aria-selected", String(login));
  $("tab-register").setAttribute("aria-selected", String(!login));
  $("auth-title").textContent = login ? "Welcome back" : "Create your account";
  $("auth-submit").textContent = login ? "Sign in" : "Create account";
  $("auth-password").autocomplete = login ? "current-password" : "new-password";
  $("password-hint").hidden = login;
  $("auth-error").hidden = true;
}

async function onAuthSubmit(event) {
  event.preventDefault();
  const email = $("auth-email").value.trim();
  const password = $("auth-password").value;
  const box = $("auth-error");
  box.hidden = true;
  $("auth-submit").disabled = true;
  try {
    if (state.authMode === "register") await api.request("/auth/register", { method: "POST", body: { email, password } });
    const tokens = await api.request("/auth/login", { method: "POST", body: { email, password } });
    if (typeof tokens?.access_token !== "string") throw new ApiError({ status: 0, code: "invalid_response", message: "Sign-in failed: unexpected server response.", retryable: true });
    state.token = tokens.access_token;
    localStorage.setItem(TOKEN_KEY, state.token);
    $("auth-password").value = "";
    await startApp();
  } catch (err) {
    box.textContent = err.status === 422 && state.authMode === "register"
      ? "Enter a valid email and a password of at least 10 characters."
      : err.message;
    box.hidden = false;
  } finally {
    $("auth-submit").disabled = false;
  }
}

function signOut(callServer = true) {
  if (callServer && state.token) api.request("/auth/logout", { method: "POST" }).catch(() => {});
  Object.assign(state, { token: null, user: null, documents: [], conversations: [], currentId: null, messages: [] });
  localStorage.removeItem(TOKEN_KEY);
  clearInterval(state.pollTimer);
  $("app-view").hidden = true;
  $("auth-view").hidden = false;
  setAuthMode("login");
}

// ---------------------------------------------------------------------------------------------
// Documents

const STATUS_TEXT = { pending: "Queued", processing: "Processing", ready: "Ready", failed: "Failed" };

async function loadDocuments() {
  const data = await api.request("/documents");
  state.documents = (Array.isArray(data?.items) ? data.items : []).map(normalizeDocument).filter(Boolean);
  renderDocuments();
  schedulePolling();
}

function renderDocuments() {
  const docs = state.documents;
  setChildren($("document-list"), docs.map((doc) => {
    const meta = [doc.fileType, formatBytes(doc.sizeBytes)];
    if (doc.pageCount !== null) meta.push(`${doc.pageCount} ${doc.isSlides ? "slides" : doc.pageCount === 1 ? "page" : "pages"}`);
    return h("li", { class: "doc-item" },
      h("span", { class: "doc-icon", "aria-hidden": "true" }, doc.fileType.slice(0, 4) || "FILE"),
      h("div", { class: "doc-main" },
        h("div", { class: "doc-name", title: doc.filename }, doc.filename),
        h("div", { class: "doc-meta" },
          h("span", { class: `badge badge-${doc.status}` }, STATUS_TEXT[doc.status]),
          meta.filter(Boolean).join(" · ")),
        doc.status === "failed" && doc.error ? h("div", { class: "doc-error" }, doc.error) : null),
      doc.status === "failed"
        ? h("button", { type: "button", class: "icon-btn", title: "Try again", "aria-label": `Process ${doc.filename} again`,
                        onclick: () => reprocessDocument(doc) }, icon("retry"))
        : null,
      h("button", { type: "button", class: "icon-btn", title: "Delete", "aria-label": `Delete ${doc.filename}`,
                    onclick: () => deleteDocument(doc) }, icon("trash")));
  }));
  $("document-empty").hidden = docs.length > 0;
  const ready = docs.filter((d) => d.status === "ready").length;
  const busy = docs.filter((d) => d.status === "pending" || d.status === "processing").length;
  $("docs-count").hidden = docs.length === 0;
  $("docs-count").textContent = String(docs.length);
  const pill = $("ready-pill");
  pill.hidden = docs.length === 0;
  pill.textContent = busy ? `${ready} ready · ${busy} processing` : `${ready} document${ready === 1 ? "" : "s"} ready`;
  $("welcome-upload").hidden = docs.length > 0;
}

function schedulePolling() {
  clearInterval(state.pollTimer);
  if (!state.documents.some((d) => d.status === "pending" || d.status === "processing")) return;
  state.pollTimer = setInterval(async () => {
    const before = new Map(state.documents.map((d) => [d.id, d.status]));
    try { await loadDocuments(); } catch { return; }
    for (const doc of state.documents) {
      const prev = before.get(doc.id);
      if (!prev || prev === doc.status) continue;
      if (doc.status === "ready") toast(`“${doc.filename}” is ready to search.`);
      if (doc.status === "failed") toast(`“${doc.filename}” could not be processed.`, "error");
    }
  }, POLL_MS);
}

async function handleFiles(files) {
  if (!files.length) return;
  const progress = $("upload-progress");
  progress.hidden = false;
  $("upload-bar").style.width = "0%";
  $("upload-label").textContent = `Uploading ${files.length} file${files.length === 1 ? "" : "s"}…`;
  try {
    const data = await uploadFiles(files, {
      getToken: () => state.token,
      onProgress: (fraction) => {
        $("upload-bar").style.width = `${Math.round(fraction * 100)}%`;
        if (fraction >= 1) $("upload-label").textContent = "Processing…";
      },
    });
    for (const item of Array.isArray(data?.items) ? data.items : []) {
      const doc = normalizeDocument(item?.document);
      if (doc && item.duplicate) toast(`“${doc.filename}” was already uploaded.`);
    }
  } catch (err) {
    if (err.status === 401) return signOut(false);
    toast(`Upload failed: ${err.message}`, "error");
  } finally {
    progress.hidden = true;
  }
  await loadDocuments().catch((err) => toast(err.message, "error"));
}

async function deleteDocument(doc) {
  if (!window.confirm(`Delete “${doc.filename}”? Everything indexed from it will be removed.`)) return;
  try {
    await api.request(`/documents/${doc.id}`, { method: "DELETE" });
    toast(`Deleted “${doc.filename}”.`);
  } catch (err) {
    toast(err.message, "error");
  }
  await loadDocuments().catch(() => {});
}

async function reprocessDocument(doc) {
  try {
    await api.request(`/documents/${doc.id}/reprocess`, { method: "POST" });
  } catch (err) {
    toast(err.message, "error");
  }
  await loadDocuments().catch(() => {});
}

// ---------------------------------------------------------------------------------------------
// Conversations

async function loadConversations() {
  const data = await api.request("/conversations");
  state.conversations = (Array.isArray(data?.items) ? data.items : []).map(normalizeConversation).filter(Boolean);
  renderConversations();
}

function renderConversations() {
  setChildren($("conversation-list"), state.conversations.map((c) =>
    h("li", { class: "nav-item", "aria-current": c.id === state.currentId ? "true" : null },
      h("button", { type: "button", class: "nav-link", title: c.title, onclick: () => openConversation(c.id) }, c.title),
      h("button", { type: "button", class: "icon-btn", title: "Delete conversation", "aria-label": `Delete conversation “${c.title}”`,
                    onclick: () => deleteConversation(c) }, icon("trash")))));
  $("conversation-empty").hidden = state.conversations.length > 0;
}

function showWelcome() {
  state.currentId = null;
  state.messages = [];
  $("chat-title").textContent = "New chat";
  setChildren($("thread"), welcomeNode);
  renderConversations();
}

async function openConversation(id) {
  closeSidebar();
  try {
    const data = await api.request(`/conversations/${id}`);
    const { messages, skipped } = normalizeMessages(data?.messages);
    if (skipped) console.warn(`Skipped ${skipped} malformed message(s)`);
    state.currentId = typeof data?.id === "string" ? data.id : id;
    state.messages = messages;
    $("chat-title").textContent = normalizeConversation(data)?.title ?? "Conversation";
    renderThread();
    renderConversations();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function deleteConversation(conv) {
  if (!window.confirm(`Delete the conversation “${conv.title}”?`)) return;
  try {
    await api.request(`/conversations/${conv.id}`, { method: "DELETE" });
    if (state.currentId === conv.id) showWelcome();
  } catch (err) {
    toast(err.message, "error");
  }
  await loadConversations().catch(() => {});
}

// ---------------------------------------------------------------------------------------------
// Rendering

function renderThread() {
  const thread = $("thread");
  if (!state.messages.length) return setChildren(thread, welcomeNode);
  setChildren(thread, state.messages.map((m, i) => renderMessage(m, i)));
  thread.scrollTop = thread.scrollHeight;
}

function lastUserMessageId(beforeIndex = state.messages.length) {
  for (let i = beforeIndex - 1; i >= 0; i--) if (state.messages[i].role === "user") return state.messages[i].id;
  return null;
}

function renderMessage(message, index) {
  if (message.role === "user") return h("article", { class: "msg msg-user", "aria-label": "You" }, h("div", { class: "bubble" }, message.content));
  const isLast = index === state.messages.length - 1;
  return assistantShell(message.status === "error"
    ? renderError(message.error, isLast ? () => retryFrom(index) : null)
    : renderAnswer(message));
}

function assistantShell(...body) {
  return h("article", { class: "msg msg-assistant", "aria-label": "Assistant" },
    h("span", { class: "logo msg-avatar", "aria-hidden": "true" }),
    h("div", { class: "msg-body" }, body));
}

const ERROR_TITLES = {
  llm_not_configured: "The language model isn't configured",
  llm_rate_limited: "Rate limit reached",
  llm_unavailable: "The language model is unavailable",
  llm_bad_response: "The language model returned an invalid response",
  invalid_response: "Invalid response",
  timeout: "The response timed out",
  interrupted: "The connection was interrupted",
  network: "Can't reach the server",
};

function renderError(error, onRetry) {
  const title = ERROR_TITLES[error.code] ?? "Something went wrong";
  return h("div", { class: "error-card", role: "alert" },
    h("h3", {}, icon("warn"), title),
    error.message !== title ? h("p", {}, error.message) : null,
    error.retryable && onRetry
      ? h("button", { type: "button", class: "btn btn-secondary btn-sm", onclick: onRetry }, icon("retry"), "Try again")
      : null);
}

function renderAnswer(message) {
  const valid = new Set(message.sources.map((s) => s.number));
  const sourcesNode = message.sources.length ? renderSources(message.sources) : null;
  const onCite = sourcesNode ? (n) => highlightSource(sourcesNode, n) : null;
  const prose = h("div", { class: "prose" }, renderMarkdown(parseBlocks(message.content), { onCite, validCites: valid }));

  let notice = null;
  if (message.status === "no_answer") {
    notice = h("div", { class: "notice notice-info" }, icon("info"), "Not found in your documents. No answer was generated without evidence.");
  } else if (message.status === "no_documents") {
    notice = h("div", { class: "notice notice-info" }, icon("info"),
      h("span", {}, "Upload a document and wait until it shows ", h("strong", {}, "Ready"), "."));
  } else if (message.status === "answered" && message.metadata?.grounded === false) {
    notice = h("div", { class: "notice notice-warn" }, icon("warn"), "This answer has no source references. Verify it against your documents.");
  }
  return [prose, notice, sourcesNode, renderDetails(message), renderActions(message)];
}

function renderSources(sources) {
  return h("section", { class: "sources", "aria-label": "Sources" },
    h("div", { class: "sources-head" }, h("strong", {}, `Sources (${sources.length})`), h("span", {}, sourceSummary(sources))),
    h("ol", { class: "source-list" }, sources.map(renderSource)));
}

function renderSource(source) {
  const full = h("p", { class: "source-full", hidden: true }, source.content);
  const excerpt = h("p", { class: "source-excerpt" }, `“${source.excerpt.replace(/^…\s*|\s*…$/g, "")}”`);
  const expandable = source.content.trim() !== source.excerpt.replace(/^…\s*|\s*…$/g, "").trim();
  const toggle = expandable
    ? h("button", { type: "button", class: "link-btn", "aria-expanded": "false" }, "View full passage")
    : null;
  toggle?.addEventListener("click", () => {
    const open = full.hidden;
    full.hidden = !open;
    excerpt.hidden = open;
    toggle.textContent = open ? "Show less" : "View full passage";
    toggle.setAttribute("aria-expanded", String(open));
  });
  const where = source.pageNumber !== null ? `Page ${source.pageNumber}` : null;
  return h("li", { class: "source", dataset: { num: String(source.number) } },
    h("div", { class: "source-top" },
      h("span", { class: "source-num", "aria-label": `Source ${source.number}` }, String(source.number)),
      h("span", { class: "source-name", title: source.documentName }, source.documentName),
      where ? h("span", { class: "source-page" }, `· ${where}`) : null),
    excerpt, full, toggle);
}

function highlightSource(container, number) {
  const node = container.querySelector(`[data-num="${number}"]`);
  if (!node) return;
  node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  node.classList.add("is-highlighted");
  setTimeout(() => node.classList.remove("is-highlighted"), 1600);
}

function renderDetails(message) {
  const meta = message.metadata;
  if (!message.steps.length && !meta) return null;
  const facts = [];
  const r = meta?.retrieval;
  if (r?.passagesRetrieved !== null && r?.passagesRetrieved !== undefined) facts.push(`${r.passagesRetrieved} passage${r.passagesRetrieved === 1 ? "" : "s"} retrieved`);
  if (meta?.model) facts.push(`Model: ${meta.model}`);
  if (meta?.latencyMs !== null && meta?.latencyMs !== undefined) facts.push(`${(meta.latencyMs / 1000).toFixed(1)} s`);
  if (meta?.tokens) facts.push(`${meta.tokens.total.toLocaleString()} tokens`);
  return h("details", { class: "details" },
    h("summary", {}, icon("chevron"), "Answer details"),
    h("div", { class: "details-body" },
      message.steps.length
        ? h("ol", { class: "steps" }, message.steps.map((s) => h("li", {}, s.label, s.detail ? h("span", { class: "step-detail" }, ` · ${s.detail}`) : null)))
        : null,
      facts.length ? h("div", { class: "meta" }, facts.map((f) => h("span", {}, f))) : null));
}

function renderActions(message) {
  if (message.status !== "answered" && message.status !== "direct_reply") return null;
  const copy = h("button", { type: "button", class: "btn btn-ghost btn-sm" }, icon("copy"), "Copy");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(message.content.replace(/\s*\[\d+\]/g, ""));
      copy.lastChild.textContent = "Copied";
      setTimeout(() => { copy.lastChild.textContent = "Copy"; }, 1500);
    } catch {
      toast("Couldn't copy to the clipboard.", "error");
    }
  });
  return h("div", { class: "msg-actions" }, copy);
}

// Pending reply: progress → streaming text → final message (or error).
function createPending() {
  const status = h("span", { class: "thinking-status" }, "Searching your documents…");
  const thinking = h("div", { class: "thinking", role: "status" }, status, h("div", { class: "progress-track" }));
  const body = h("div", { class: "msg-body" }, thinking);
  const node = h("article", { class: "msg msg-assistant", "aria-label": "Assistant", "aria-busy": "true" },
    h("span", { class: "logo msg-avatar", "aria-hidden": "true" }), body);
  let text = "";
  let prose = null;
  return {
    node,
    setStatus(label) { status.textContent = `${label}…`; },
    appendToken(delta) {
      if (typeof delta !== "string" || !delta) return;
      text += delta;
      if (!prose) {
        prose = h("div", { class: "prose streaming-caret" });
        setChildren(body, prose);
      }
      // Citation markers are renumbered and [[calc: …]] expressions are evaluated on the server when
      // the answer completes, so neither is shown raw while streaming.
      const visible = text.replace(/\[\[[^\]]*\]\]/g, "…").replace(/\[\[[^\]]*$/, "");
      setChildren(prose, renderMarkdown(parseBlocks(visible), { hideUnknownCites: true }));
    },
  };
}

// ---------------------------------------------------------------------------------------------
// Asking questions

function setBusy(busy) {
  state.busy = busy;
  $("send").classList.toggle("is-busy", busy);
  $("send").setAttribute("aria-label", busy ? "Answering…" : "Send message");
  updateSendState();
}

function updateSendState() {
  $("send").disabled = state.busy || !$("prompt").value.trim();
}

function scrollToBottom() {
  const thread = $("thread");
  thread.scrollTop = thread.scrollHeight;
}

async function ask(content) {
  if (state.busy || !content.trim()) return;
  setBusy(true);
  try {
    if (!state.currentId) {
      const conv = normalizeConversation(await api.request("/conversations", { method: "POST", body: {} }));
      if (!conv) throw new ApiError({ status: 0, code: "invalid_response", message: "Could not start a conversation.", retryable: true });
      state.currentId = conv.id;
      state.messages = [];
      $("chat-title").textContent = "New chat";
    }
    const optimistic = { id: `local-${Date.now()}`, role: "user", status: "sent", content, sources: [], metadata: null, steps: [], error: null, createdAt: null };
    state.messages.push(optimistic);
    await runStream(`/conversations/${state.currentId}/messages/stream`, { content }, optimistic);
  } catch (err) {
    toast(err.message, "error");
  } finally {
    setBusy(false);
  }
}

/** Retry the question before a failed reply: regenerate it on the server if it was stored,
 *  otherwise (it never reached the server) send it again. */
function retryFrom(errorIndex) {
  const userId = lastUserMessageId(errorIndex);
  const userMessage = state.messages.find((m) => m.id === userId);
  if (!userMessage) return;
  if (!userId.startsWith("local-")) return regenerate(userId);
  state.messages = state.messages.slice(0, state.messages.indexOf(userMessage));
  renderThread();
  return ask(userMessage.content);
}

async function regenerate(userMessageId) {
  if (state.busy || !userMessageId || userMessageId.startsWith("local-")) return;
  setBusy(true);
  try {
    // Drop the failed reply locally; the server replaces it too.
    const index = state.messages.findIndex((m) => m.id === userMessageId);
    state.messages = state.messages.slice(0, index + 1);
    await runStream(`/conversations/${state.currentId}/messages/${userMessageId}/regenerate/stream`, undefined, null);
  } finally {
    setBusy(false);
  }
}

/**
 * Stream one answer into the current conversation. Rendering is bound to the conversation that
 * started the request: if the user switches away, events are no longer drawn (the stored answer
 * appears when they come back), so a stale response can never overwrite another conversation.
 */
async function runStream(path, body, optimisticUser) {
  const conversationId = state.currentId;
  const isCurrent = () => state.currentId === conversationId;
  renderThread();
  const pending = createPending();
  $("thread").append(pending.node);
  scrollToBottom();

  let finished = false;
  const finish = (message) => {
    finished = true;
    if (!isCurrent()) return;
    state.messages.push(message);
    renderThread();
  };
  const fail = (error) => finish({
    id: `local-error-${Date.now()}`, role: "assistant", status: "error", content: error.message, sources: [],
    metadata: null, steps: [], error: { code: error.code ?? "error", message: error.message, retryable: error.retryable ?? true }, createdAt: null,
  });

  try {
    await api.stream(path, {
      body,
      onEvent(event, data) {
        if (finished) return;
        if (event === "user_message") {
          const result = normalizeMessage(data);
          if (result.ok && optimisticUser) optimisticUser.id = result.message.id; // enables retry
        } else if (event === "step") {
          if (isCurrent() && typeof data?.label === "string") pending.setStatus(data.label);
        } else if (event === "token") {
          if (isCurrent()) { pending.appendToken(data?.delta); scrollToBottom(); }
        } else if (event === "final") {
          const result = normalizeMessage(data);
          if (result.ok) finish(result.message);
          else fail({ code: "invalid_response", message: "The assistant returned an invalid response.", retryable: true });
        } else if (event === "error") {
          fail({ code: typeof data?.code === "string" ? data.code : "error",
                 message: typeof data?.message === "string" ? data.message : "Something went wrong while answering.",
                 retryable: data?.retryable !== false });
        } else if (event === "invalid") {
          fail({ code: "invalid_response", message: "The assistant returned an invalid response.", retryable: true });
        }
      },
    });
    if (!finished) fail({ code: "interrupted", message: "The connection closed before the answer finished.", retryable: true });
  } catch (err) {
    if (optimisticUser && optimisticUser.id.startsWith("local-") && err.status && err.status !== 0) {
      // The server rejected the question before storing it (e.g. rate limit): take it back.
      state.messages = state.messages.filter((m) => m !== optimisticUser);
      if (isCurrent()) { $("prompt").value = optimisticUser.content; autoGrow(); }
    }
    fail(err instanceof ApiError ? err : { code: "error", message: "Something went wrong while answering.", retryable: true });
  } finally {
    loadConversations().then(() => {
      const conv = state.conversations.find((c) => c.id === conversationId);
      if (conv && isCurrent()) $("chat-title").textContent = conv.title;
    }).catch(() => {});
  }
}

// ---------------------------------------------------------------------------------------------
// Layout & boot

function openSidebar() { $("sidebar").classList.add("is-open"); $("scrim").hidden = false; }
function closeSidebar() { $("sidebar").classList.remove("is-open"); $("scrim").hidden = true; }

function autoGrow() {
  const prompt = $("prompt");
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight, 220)}px`;
  updateSendState();
}

let welcomeNode = null;

async function startApp() {
  try {
    const user = await api.request("/auth/me");
    if (typeof user?.email !== "string") throw new Error("invalid user");
    state.user = user;
  } catch {
    signOut(false);
    return;
  }
  $("auth-view").hidden = true;
  $("app-view").hidden = false;
  $("user-email").textContent = state.user.email;
  $("user-initial").textContent = state.user.email.charAt(0).toUpperCase();
  showWelcome();
  const results = await Promise.allSettled([loadDocuments(), loadConversations()]);
  const failed = results.find((r) => r.status === "rejected");
  if (failed) toast(failed.reason?.message ?? "Couldn't load your data.", "error");
  $("prompt").focus();
}

function bind() {
  welcomeNode = $("welcome");
  applyTheme(localStorage.getItem(THEME_KEY) ?? "system");
  $("theme-select").addEventListener("change", (e) => applyTheme(e.target.value));
  $("tab-login").addEventListener("click", () => setAuthMode("login"));
  $("tab-register").addEventListener("click", () => setAuthMode("register"));
  $("auth-form").addEventListener("submit", onAuthSubmit);
  $("logout").addEventListener("click", () => signOut(true));
  $("new-chat").addEventListener("click", () => { showWelcome(); closeSidebar(); $("prompt").focus(); });
  $("open-sidebar").addEventListener("click", openSidebar);
  $("close-sidebar").addEventListener("click", closeSidebar);
  $("scrim").addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSidebar(); });

  const input = $("file-input");
  input.addEventListener("change", () => { handleFiles([...input.files]); input.value = ""; });
  $("welcome-upload-btn").addEventListener("click", () => input.click());
  const zone = $("dropzone");
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("is-dragging"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-dragging"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("is-dragging"); handleFiles([...e.dataTransfer.files]); });

  const prompt = $("prompt");
  prompt.addEventListener("input", autoGrow);
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const content = prompt.value.trim();
    if (!content || state.busy) return;
    prompt.value = "";
    autoGrow();
    ask(content);
  });
  for (const chip of document.querySelectorAll(".suggestion")) chip.addEventListener("click", () => ask(chip.textContent));
}

bind();
if (state.token) startApp(); else signOut(false);
