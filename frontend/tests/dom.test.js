// Tests the DOM layer against a minimal fake DOM (no browser or dependencies needed).
import { test, before } from "node:test";
import assert from "node:assert/strict";

class FakeNode {
  constructor() { this.childNodes = []; }
  append(...nodes) { this.childNodes.push(...nodes); }
  replaceChildren(...nodes) { this.childNodes = [...nodes]; }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
}
class FakeText extends FakeNode {
  constructor(value) { super(); this.value = value; }
  get textContent() { return this.value; }
}
class FakeElement extends FakeNode {
  constructor(tag) { super(); this.tagName = tag.toUpperCase(); this.attributes = {}; this.dataset = {}; this.listeners = {}; }
  setAttribute(k, v) { this.attributes[k] = v; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  set className(v) { this.attributes.class = v; }
}
class FakeFragment extends FakeNode {}

let dom;
let md;
before(async () => {
  globalThis.Node = FakeNode;
  globalThis.document = {
    createElement: (tag) => new FakeElement(tag),
    createTextNode: (v) => new FakeText(v),
    createDocumentFragment: () => new FakeFragment(),
  };
  globalThis.console = { ...console, warn: () => {} };
  dom = await import("../js/dom.js");
  md = await import("../js/markdown.js");
});

const find = (node, tag) => node.childNodes.flatMap((c) => [...(c.tagName === tag ? [c] : []), ...(c.childNodes ? find(c, tag) : [])]);

test("h() never renders null, undefined, false, NaN or objects", () => {
  const el = dom.h("div", { class: "x", title: null, hidden: false }, "Pune", null, undefined, false, "", NaN, { a: 1 }, [null, ["!"]], 3);
  assert.equal(el.textContent, "Pune!3");
  assert.ok(!("title" in el.attributes) && !("hidden" in el.attributes));
});

test("setChildren() is null-safe (the original nullnull bug)", () => {
  const el = dom.h("div");
  dom.setChildren(el, dom.h("p", {}, "answer"), null, null, null);
  assert.equal(el.textContent, "answer");
  assert.equal(el.childNodes.length, 1);
});

test("markdown renders citations as buttons only for known sources", () => {
  const blocks = md.parseBlocks("In **Pune** [1], also [7].");
  const fragment = dom.renderMarkdown(blocks, { onCite: () => {}, validCites: new Set([1]) });
  const buttons = find(fragment, "BUTTON");
  assert.equal(buttons.length, 1);
  assert.equal(buttons[0].textContent, "1");
  assert.equal(fragment.textContent, "In Pune 1, also [7].");
});

test("streaming mode hides citation markers", () => {
  const fragment = dom.renderMarkdown(md.parseBlocks("Pune [3]."), { hideUnknownCites: true });
  assert.equal(fragment.textContent, "Pune .");
});

test("unsafe link schemes are rendered as text", () => {
  const fragment = dom.renderMarkdown([{ type: "paragraph", children: [{ type: "link", href: "javascript:alert(1)", children: [{ type: "text", value: "x" }] }] }]);
  assert.equal(find(fragment, "A").length, 0);
  assert.equal(fragment.textContent, "x");
});

test("tables render header and body cells", () => {
  const blocks = md.parseBlocks("| A | B |\n|---|---|\n| 1 | 2 |");
  const fragment = dom.renderMarkdown(blocks);
  assert.equal(find(fragment, "TH").length, 2);
  assert.equal(find(fragment, "TD").map((c) => c.textContent).join(","), "1,2");
});
