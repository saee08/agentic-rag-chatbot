"use strict";

// Document Assistant front end. No framework, no build step.
// Security: all server/model/document text is rendered with textContent / DOM nodes, never innerHTML.

const API = "/api/v1";
const TOKEN_KEY = "docassist.token";
const POLL_MS = 2000;

const state = {
  token: localStorage.getItem(TOKEN_KEY),
  user: null,
  conversations: [],
  documents: [],
  currentId: null,
  sending: false,
  pollTimer: null,
  authMode: "login",
};

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// ---------------------------------------------------------------------------------------------
// API

class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function api(path, { method = "GET", body, raw = false } = {}) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(API + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new ApiError(0, "network", "Cannot reach the server. Check your connection.");
  }
  if (raw && response.ok) return response;
  if (response.status === 204) return null;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const err = data.error || {};
    if (response.status === 401 && state.token) signOut(false);
    throw new ApiError(response.status, err.code || "error", err.message || `Request failed (${response.status})`);
  }
  return data;
}

function toast(message, kind = "info") {
  const node = el("div", { class: `toast ${kind}` }, message);
  $("toasts").append(node);
  setTimeout(() => node.remove(), kind === "error" ? 7000 : 4000);
}

function confirmDialog(title, body, confirmLabel) {
  return new Promise((resolve) => {
    const fragment = $("tpl-dialog").content.cloneNode(true);
    const backdrop = fragment.querySelector(".backdrop");
    const dialog = fragment.querySelector(".dialog");
    const close = (value) => { backdrop.remove(); resolve(value); };
    const cancel = el("button", { type: "button", class: "btn", onclick: () => close(false) }, "Cancel");
    dialog.setAttribute("aria-label", title);
    dialog.append(
      el("h2", {}, title),
      el("p", { class: "muted" }, body),
      el("div", { class: "actions" }, cancel,
        el("button", { type: "button", class: "btn danger", onclick: () => close(true) }, confirmLabel)),
    );
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(false); });
    backdrop.addEventListener("keydown", (e) => { if (e.key === "Escape") close(false); });
    document.body.append(backdrop);
    cancel.focus();
  });
}

// ---------------------------------------------------------------------------------------------
// Auth

function setAuthMode(mode) {
  state.authMode = mode;
  const login = mode === "login";
  $("tab-login").classList.toggle("active", login);
  $("tab-register").classList.toggle("active", !login);
  $("tab-login").setAttribute("aria-selected", String(login));
  $("tab-register").setAttribute("aria-selected", String(!login));
  $("auth-title").textContent = login ? "Sign in" : "Create account";
  $("auth-submit").textContent = login ? "Sign in" : "Create account";
  $("auth-password").autocomplete = login ? "current-password" : "new-password";
  $("password-hint").hidden = login;
  $("auth-error").hidden = true;
}

async function onAuthSubmit(event) {
  event.preventDefault();
  const email = $("auth-email").value.trim();
  const password = $("auth-password").value;
  const errorBox = $("auth-error");
  errorBox.hidden = true;
  $("auth-submit").disabled = true;
  try {
    if (state.authMode === "register") {
      await api("/auth/register", { method: "POST", body: { email, password } });
    }
    const tokens = await api("/auth/login", { method: "POST", body: { email, password } });
    state.token = tokens.access_token;
    localStorage.setItem(TOKEN_KEY, state.token);
    $("auth-password").value = "";
    await startApp();
  } catch (err) {
    errorBox.textContent = err.status === 422 && state.authMode === "register"
      ? "Please enter a valid e-mail and a password of at least 10 characters."
      : err.message;
    errorBox.hidden = false;
  } finally {
    $("auth-submit").disabled = false;
  }
}

function signOut(callServer = true) {
  if (callServer && state.token) api("/auth/logout", { method: "POST" }).catch(() => {});
  state.token = null;
  state.user = null;
  localStorage.removeItem(TOKEN_KEY);
  clearInterval(state.pollTimer);
  $("app-view").hidden = true;
  $("auth-view").hidden = false;
  setAuthMode("login");
}

// ---------------------------------------------------------------------------------------------
// Documents

const STATUS_LABEL = { pending: "queued", processing: "processing", ready: "ready", failed: "failed" };

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

async function loadDocuments() {
  state.documents = (await api("/documents")).items;
  renderDocuments();
  schedulePolling();
}

function renderDocuments() {
  const list = $("document-list");
  list.replaceChildren(...state.documents.map((doc) => {
    const meta = [doc.file_type.replace(".", "").toUpperCase(), formatSize(doc.size_bytes)];
    if (doc.page_count) meta.push(`${doc.page_count} ${doc.file_type === ".pptx" ? "slides" : "pages"}`);
    if (doc.status === "ready") meta.push(`${doc.chunk_count} passages`);
    return el("li", {},
      el("div", { class: "doc-info" },
        el("div", { class: "doc-name", title: doc.filename }, doc.filename),
        el("div", { class: "doc-meta" },
          el("span", { class: `badge ${doc.status}` }, STATUS_LABEL[doc.status] || doc.status), " ", meta.join(" · ")),
        doc.status === "failed" && doc.error ? el("div", { class: "doc-error" }, doc.error) : null),
      doc.status === "failed"
        ? el("button", { type: "button", class: "icon-btn", title: "Retry", "aria-label": `Retry ${doc.filename}`,
                         onclick: () => reprocessDocument(doc) }, "↻")
        : null,
      el("button", { type: "button", class: "icon-btn", title: "Delete", "aria-label": `Delete ${doc.filename}`,
                     onclick: () => deleteDocument(doc) }, "🗑"));
  }));
  $("document-empty").hidden = state.documents.length > 0;
}

function schedulePolling() {
  clearInterval(state.pollTimer);
  if (state.documents.some((d) => d.status === "pending" || d.status === "processing")) {
    state.pollTimer = setInterval(async () => {
      const before = new Map(state.documents.map((d) => [d.id, d.status]));
      try {
        await loadDocuments();
      } catch { /* transient; try again on the next tick */ }
      for (const doc of state.documents) {
        const prev = before.get(doc.id);
        if (prev && prev !== doc.status && doc.status === "ready") toast(`“${doc.filename}” is ready.`);
        if (prev && prev !== doc.status && doc.status === "failed") toast(`“${doc.filename}” could not be processed.`, "error");
      }
    }, POLL_MS);
  }
}

function uploadFiles(files) {
  if (!files.length) return;
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  const progress = $("upload-progress");
  const bar = $("upload-progress-bar");
  const label = $("upload-progress-text");
  progress.hidden = false;
  bar.style.width = "0%";
  label.textContent = `Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`;

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${API}/documents`);
  xhr.setRequestHeader("Authorization", `Bearer ${state.token}`);
  xhr.upload.addEventListener("progress", (e) => {
    if (e.lengthComputable) bar.style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
  });
  xhr.addEventListener("load", async () => {
    progress.hidden = true;
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
    if (xhr.status === 401) return signOut(false);
    if (xhr.status >= 400) {
      toast(data.error?.message || `Upload failed (${xhr.status}).`, "error");
      return;
    }
    for (const item of data.items || []) {
      if (item.duplicate) toast(`“${item.document.filename}” was already uploaded.`);
    }
    await loadDocuments();
  });
  xhr.addEventListener("error", () => {
    progress.hidden = true;
    toast("Upload failed: network error.", "error");
  });
  xhr.send(form);
}

async function deleteDocument(doc) {
  const ok = await confirmDialog("Delete document?",
    `“${doc.filename}” and everything indexed from it will be permanently removed.`, "Delete");
  if (!ok) return;
  try {
    await api(`/documents/${doc.id}`, { method: "DELETE" });
    toast(`Deleted “${doc.filename}”.`);
    await loadDocuments();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function reprocessDocument(doc) {
  try {
    await api(`/documents/${doc.id}/reprocess`, { method: "POST" });
    await loadDocuments();
  } catch (err) {
    toast(err.message, "error");
  }
}

// ---------------------------------------------------------------------------------------------
// Conversations

async function loadConversations() {
  state.conversations = (await api("/conversations")).items;
  renderConversations();
}

function renderConversations() {
  $("conversation-list").replaceChildren(...state.conversations.map((c) =>
    el("li", { class: c.id === state.currentId ? "active" : "" },
      el("button", { type: "button", class: "item-main", onclick: () => openConversation(c.id),
                     "aria-current": c.id === state.currentId ? "true" : null },
        el("span", {}, c.title)),
      el("button", { type: "button", class: "icon-btn", title: "Delete conversation",
                     "aria-label": `Delete conversation ${c.title}`, onclick: () => deleteConversation(c) }, "🗑"))));
  $("conversation-empty").hidden = state.conversations.length > 0;
}

function showWelcome() {
  state.currentId = null;
  $("chat-title").textContent = "New conversation";
  $("messages").replaceChildren(welcomeNode);
  renderConversations();
}

let welcomeNode = null; // captured at startup so the welcome screen can be restored

async function openConversation(id) {
  closeSidebar();
  try {
    const conv = await api(`/conversations/${id}`);
    state.currentId = conv.id;
    $("chat-title").textContent = conv.title;
    const container = $("messages");
    container.replaceChildren(...conv.messages.map(renderMessage));
    if (!conv.messages.length) container.replaceChildren(welcomeNode);
    renderConversations();
    container.scrollTop = container.scrollHeight;
  } catch (err) {
    toast(err.message, "error");
  }
}

async function deleteConversation(conv) {
  const ok = await confirmDialog("Delete conversation?", `“${conv.title}” will be permanently removed.`, "Delete");
  if (!ok) return;
  try {
    await api(`/conversations/${conv.id}`, { method: "DELETE" });
    if (state.currentId === conv.id) showWelcome();
    await loadConversations();
  } catch (err) {
    toast(err.message, "error");
  }
}

// ---------------------------------------------------------------------------------------------
// Message rendering

// Minimal, safe Markdown subset: paragraphs, bullet/numbered lists, **bold**, `code`, [n] citations.
function renderRichText(text, onCite) {
  const fragment = document.createDocumentFragment();
  const blocks = text.split(/\n{2,}/);
  for (const block of blocks) {
    const lines = block.split("\n").filter((l) => l.trim());
    if (!lines.length) continue;
    const bullet = lines.every((l) => /^\s*[-*•]\s+/.test(l));
    const numbered = lines.every((l) => /^\s*\d+[.)]\s+/.test(l));
    if (bullet || numbered) {
      const list = el(numbered ? "ol" : "ul");
      for (const line of lines) list.append(el("li", {}, inline(line.replace(/^\s*([-*•]|\d+[.)])\s+/, ""), onCite)));
      fragment.append(list);
    } else {
      const p = el("p");
      lines.forEach((line, i) => {
        if (i) p.append(el("br"));
        p.append(inline(line, onCite));
      });
      fragment.append(p);
    }
  }
  return fragment;
}

function inline(text, onCite) {
  const fragment = document.createDocumentFragment();
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[\d{1,3}\])/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    fragment.append(text.slice(last, match.index));
    const token = match[0];
    if (token.startsWith("**")) fragment.append(el("strong", {}, token.slice(2, -2)));
    else if (token.startsWith("`")) fragment.append(el("code", {}, token.slice(1, -1)));
    else {
      const n = Number(token.slice(1, -1));
      fragment.append(onCite
        ? el("button", { type: "button", class: "cite", "aria-label": `Show source ${n}`, onclick: () => onCite(n) }, n)
        : token);
    }
    last = match.index + token.length;
  }
  fragment.append(text.slice(last));
  return fragment;
}

function describeStep(step) {
  const d = step.detail || {};
  if (step.kind === "tool" && step.name === "search_documents") {
    return `Searched “${d.args?.query ?? ""}” → ${d.hits ?? 0} relevant passage${d.hits === 1 ? "" : "s"}`;
  }
  if (step.kind === "tool" && step.name === "list_documents") return `Listed your documents (${d.documents ?? 0})`;
  if (step.kind === "decision") return d.query ? `Decided to search again: “${d.query}”` : `Decided: ${String(d.action).replace(/_/g, " ")}`;
  if (step.kind === "generation") return `Wrote the answer from ${d.sources ?? 0} source${d.sources === 1 ? "" : "s"}`;
  const reasons = {
    sufficient_evidence: "Found enough evidence",
    insufficient_evidence: "Not enough evidence in your documents",
    no_ready_documents: "No processed documents available",
    max_tool_calls: "Reached the search limit",
    max_iterations: "Reached the step limit",
    time_budget: "Reached the time limit",
    token_budget: "Reached the token budget",
  };
  return reasons[step.name] || step.name;
}

function renderSteps(steps) {
  if (!steps?.length) return null;
  return el("details", { class: "steps" },
    el("summary", {}, `How I answered (${steps.length} step${steps.length === 1 ? "" : "s"})`),
    el("ol", {}, steps.map((s) => el("li", {}, describeStep(s)))));
}

function renderSources(citations) {
  if (!citations?.length) return null;
  return el("div", { class: "sources" }, citations.map((c) => {
    const where = c.page ? `page ${c.page}` : "";
    return el("details", { class: "source", "data-num": c.number },
      el("summary", {},
        el("span", { class: "num" }, `[${c.number}]`),
        el("span", { class: "name", title: c.document_name }, c.document_name),
        el("span", { class: "where" }, where)),
      el("blockquote", {}, c.snippet));
  }));
}

function renderMessage(message) {
  if (message.role === "user") {
    return el("article", { class: "msg user", "aria-label": "You" }, el("div", { class: "bubble" }, message.content));
  }
  const classes = ["msg", "assistant"];
  if (message.status === "error") classes.push("error");
  if (message.status === "no_answer" || message.status === "no_documents") classes.push("no-answer");
  const node = el("article", { class: classes.join(" "), "aria-label": "Assistant" });
  fillAssistant(node, message);
  return node;
}

function fillAssistant(node, message) {
  const sources = renderSources(message.citations);
  const onCite = sources ? (n) => {
    const target = sources.querySelector(`[data-num="${n}"]`);
    if (!target) return;
    target.open = true;
    target.classList.remove("flash");
    void target.offsetWidth;
    target.classList.add("flash");
    target.scrollIntoView({ block: "nearest", behavior: "smooth" });
  } : null;
  const bubble = el("div", { class: "bubble" }, renderRichText(message.content, onCite));
  const ungrounded = message.status === "answered" && message.usage && message.usage.grounded === false;
  node.replaceChildren(
    bubble,
    ungrounded ? el("div", { class: "notice" }, "This answer has no source references; verify it against your documents.") : null,
    sources,
    renderSteps(message.agent_trace),
  );
}

// ---------------------------------------------------------------------------------------------
// Sending (streaming)

function parseSse(buffer) {
  const events = [];
  let index;
  while ((index = buffer.indexOf("\n\n")) !== -1) {
    const block = buffer.slice(0, index);
    buffer = buffer.slice(index + 2);
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (data) events.push({ event, data: JSON.parse(data) });
  }
  return { events, rest: buffer };
}

async function sendMessage(content) {
  if (state.sending || !content.trim()) return;
  state.sending = true;
  $("send").disabled = true;
  const container = $("messages");
  try {
    if (!state.currentId) {
      const conv = await api("/conversations", { method: "POST", body: {} });
      state.currentId = conv.id;
      container.replaceChildren();
    }
    $("welcome")?.remove();
    container.append(renderMessage({ role: "user", content }));
    const pending = el("article", { class: "msg assistant", "aria-label": "Assistant", "aria-busy": "true" });
    const status = el("span", {}, "Searching your documents");
    pending.append(el("div", { class: "bubble" }, el("span", { class: "thinking" },
      el("span", { class: "dots", "aria-hidden": "true" }, el("i"), el("i"), el("i")), status)));
    container.append(pending);
    container.scrollTop = container.scrollHeight;

    const response = await api(`/conversations/${state.currentId}/messages/stream`,
      { method: "POST", body: { content }, raw: true });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";
    let answerBubble = null;
    let finished = false;
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      const parsed = parseSse(buffer + decoder.decode(value, { stream: true }));
      buffer = parsed.rest;
      for (const { event, data } of parsed.events) {
        if (event === "step") {
          status.textContent = describeStep(data);
        } else if (event === "token") {
          answer += data.delta;
          if (!answerBubble) {
            answerBubble = el("div", { class: "bubble" });
            pending.replaceChildren(answerBubble);
          }
          answerBubble.replaceChildren(renderRichText(answer, null));
          container.scrollTop = container.scrollHeight;
        } else if (event === "final") {
          pending.removeAttribute("aria-busy");
          pending.replaceWith(renderMessage(data));
          finished = true;
        } else if (event === "error") {
          pending.replaceWith(renderMessage({ role: "assistant", status: "error", content: data.message }));
          finished = true;
        }
      }
    }
    if (!finished) {
      pending.replaceWith(renderMessage({ role: "assistant", status: "error", content: "The response was interrupted." }));
    }
    container.scrollTop = container.scrollHeight;
    await loadConversations();
    $("chat-title").textContent = state.conversations.find((c) => c.id === state.currentId)?.title || "Conversation";
  } catch (err) {
    container.querySelector('[aria-busy="true"]')?.remove();
    if (err.status === 429) toast("You're sending messages too quickly. Please wait a moment.", "error");
    else toast(err.message, "error");
  } finally {
    state.sending = false;
    $("send").disabled = false;
    $("prompt").focus();
  }
}

// ---------------------------------------------------------------------------------------------
// Layout helpers & boot

function openSidebar() { $("sidebar").classList.add("open"); }
function closeSidebar() { $("sidebar").classList.remove("open"); }

function autoGrow() {
  const prompt = $("prompt");
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight, 200)}px`;
}

async function startApp() {
  try {
    state.user = await api("/auth/me");
  } catch {
    signOut(false);
    return;
  }
  $("auth-view").hidden = true;
  $("app-view").hidden = false;
  $("user-email").textContent = state.user.email;
  showWelcome();
  try {
    await Promise.all([loadDocuments(), loadConversations()]);
  } catch (err) {
    toast(err.message, "error");
  }
  $("prompt").focus();
}

function bind() {
  welcomeNode = $("welcome");
  $("tab-login").addEventListener("click", () => setAuthMode("login"));
  $("tab-register").addEventListener("click", () => setAuthMode("register"));
  $("auth-form").addEventListener("submit", onAuthSubmit);
  $("logout").addEventListener("click", () => signOut(true));
  $("new-chat").addEventListener("click", () => { showWelcome(); closeSidebar(); $("prompt").focus(); });
  $("open-sidebar").addEventListener("click", openSidebar);
  $("close-sidebar").addEventListener("click", closeSidebar);

  const input = $("file-input");
  input.addEventListener("change", () => { uploadFiles([...input.files]); input.value = ""; });
  const zone = $("dropzone");
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    uploadFiles([...e.dataTransfer.files]);
  });

  const prompt = $("prompt");
  prompt.addEventListener("input", autoGrow);
  prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const content = prompt.value.trim();
    if (!content) return;
    prompt.value = "";
    autoGrow();
    sendMessage(content);
  });
  for (const chip of document.querySelectorAll(".suggestion")) {
    chip.addEventListener("click", () => sendMessage(chip.textContent));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  bind();
  if (state.token) startApp();
  else signOut(false);
});
