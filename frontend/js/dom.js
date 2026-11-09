// DOM helpers. All text goes through text nodes; no HTML strings are ever parsed.

/**
 * Create an element. Children that are null/undefined/false/"" are skipped; numbers become text;
 * plain objects are rejected (they would render as "[object Object]").
 */
export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  append(node, children);
  return node;
}

export function append(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false || child === "") continue;
    if (child instanceof Node) parent.append(child);
    else if (typeof child === "string") parent.append(document.createTextNode(child));
    else if (typeof child === "number" && Number.isFinite(child)) parent.append(document.createTextNode(String(child)));
    else console.warn("Refusing to render a non-text value", child);
  }
  return parent;
}

/** Replace all children (same null-safety as h()). */
export function setChildren(parent, ...children) {
  parent.replaceChildren();
  return append(parent, children);
}

const SAFE_HREF = /^https?:\/\//i;

function renderInline(nodes, ctx) {
  return nodes.map((node) => {
    switch (node.type) {
      case "text": return node.value;
      case "br": return h("br");
      case "strong": return h("strong", {}, renderInline(node.children, ctx));
      case "em": return h("em", {}, renderInline(node.children, ctx));
      case "code": return h("code", {}, node.value);
      case "link":
        return SAFE_HREF.test(node.href)
          ? h("a", { href: node.href, target: "_blank", rel: "noopener noreferrer nofollow" }, renderInline(node.children, ctx))
          : renderInline(node.children, ctx);
      case "cite":
        // Only numbers that match a displayed source become interactive; others stay plain text.
        return ctx.onCite && ctx.validCites.has(node.number)
          ? h("button", { type: "button", class: "cite", "aria-label": `Show source ${node.number}`,
                          onclick: () => ctx.onCite(node.number) }, String(node.number))
          : (ctx.hideUnknownCites ? null : `[${node.number}]`);
      default: return null;
    }
  });
}

/** Render a parsed Markdown AST to a DocumentFragment. */
export function renderMarkdown(blocks, { onCite = null, validCites = new Set(), hideUnknownCites = false } = {}) {
  const ctx = { onCite, validCites, hideUnknownCites };
  const fragment = document.createDocumentFragment();
  for (const block of blocks) {
    switch (block.type) {
      case "heading": fragment.append(h(`h${Math.min(block.level + 2, 6)}`, { class: "md-heading" }, renderInline(block.children, ctx))); break;
      case "paragraph": fragment.append(h("p", {}, renderInline(block.children, ctx))); break;
      case "list":
        fragment.append(h(block.ordered ? "ol" : "ul", { start: block.ordered && block.start !== 1 ? block.start : null },
          block.items.map((item) => h("li", {}, renderInline(item, ctx)))));
        break;
      case "table":
        fragment.append(h("div", { class: "table-wrap" }, h("table", {},
          h("thead", {}, h("tr", {}, block.header.map((cell) => h("th", { scope: "col" }, renderInline(cell, ctx))))),
          h("tbody", {}, block.rows.map((row) => h("tr", {}, row.map((cell) => h("td", {}, renderInline(cell, ctx)))))))));
        break;
      case "code": fragment.append(h("pre", {}, h("code", {}, block.text))); break;
      case "quote": fragment.append(h("blockquote", {}, renderMarkdown(block.children, ctx))); break;
      case "hr": fragment.append(h("hr")); break;
      default: break;
    }
  }
  return fragment;
}
