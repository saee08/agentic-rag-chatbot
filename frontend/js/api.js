// HTTP + SSE client. Every failure is converted to an ApiError via normalizeApiError().
import { normalizeApiError } from "./normalize.js";

export const API_BASE = "/api/v1";

export class ApiError extends Error {
  constructor(info) {
    super(info.message);
    Object.assign(this, info);
  }
}

export function createClient({ getToken, onUnauthorized }) {
  async function request(path, { method = "GET", body, signal } = {}) {
    const headers = {};
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let response;
    try {
      response = await fetch(API_BASE + path, { method, headers, signal, body: body === undefined ? undefined : JSON.stringify(body) });
    } catch (err) {
      if (err?.name === "AbortError") throw err;
      throw new ApiError(normalizeApiError(0, null));
    }
    if (response.status === 204) return null;
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      if (response.status === 401 && token) onUnauthorized();
      throw new ApiError(normalizeApiError(response.status, data));
    }
    if (data === null) throw new ApiError({ status: response.status, code: "invalid_response", message: "The server returned an unreadable response.", retryable: true });
    return data;
  }

  /**
   * POST and consume a Server-Sent Events stream. Calls onEvent(name, data) per event.
   * Fails with a retryable "timeout" error if no event arrives for idleTimeoutMs.
   */
  async function stream(path, { body, onEvent, idleTimeoutMs = 120_000 } = {}) {
    const controller = new AbortController();
    let timedOut = false;
    let timer = null;
    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(() => { timedOut = true; controller.abort(); }, idleTimeoutMs);
    };
    const headers = { Accept: "text/event-stream" };
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    arm();
    try {
      let response;
      try {
        response = await fetch(API_BASE + path, { method: "POST", headers, signal: controller.signal, body: body === undefined ? undefined : JSON.stringify(body) });
      } catch (err) {
        if (timedOut) throw new ApiError(TIMEOUT);
        if (err?.name === "AbortError") throw err;
        throw new ApiError(normalizeApiError(0, null));
      }
      if (!response.ok) {
        const data = await response.json().catch(() => null);
        if (response.status === 401 && token) onUnauthorized();
        throw new ApiError(normalizeApiError(response.status, data));
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        let chunk;
        try {
          chunk = await reader.read();
        } catch {
          throw new ApiError(timedOut ? TIMEOUT : INTERRUPTED);
        }
        if (chunk.done) break;
        arm();
        buffer += decoder.decode(chunk.value, { stream: true });
        let index;
        while ((index = buffer.indexOf("\n\n")) !== -1) {
          const block = buffer.slice(0, index);
          buffer = buffer.slice(index + 2);
          const parsed = parseSseBlock(block);
          if (parsed) onEvent(parsed.event, parsed.data);
        }
      }
    } finally {
      clearTimeout(timer);
    }
  }

  return { request, stream };
}

const TIMEOUT = { status: 0, code: "timeout", message: "The assistant took too long to respond.", retryable: true };
const INTERRUPTED = { status: 0, code: "interrupted", message: "The connection was interrupted before the answer finished.", retryable: true };

export function parseSseBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (!data.length) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return { event: "invalid", data: null };
  }
}

/** Multipart upload with progress (fetch has no upload progress events). */
export function uploadFiles(files, { getToken, onProgress }) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/documents`);
    const token = getToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.addEventListener("progress", (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); });
    xhr.addEventListener("load", () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* handled below */ }
      if (xhr.status >= 200 && xhr.status < 300 && data) resolve(data);
      else reject(new ApiError(normalizeApiError(xhr.status, data)));
    });
    xhr.addEventListener("error", () => reject(new ApiError(normalizeApiError(0, null))));
    xhr.send(form);
  });
}
