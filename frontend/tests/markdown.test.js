import { test } from "node:test";
import assert from "node:assert/strict";
import { citedNumbers, parseBlocks, parseInline } from "../js/markdown.js";

test("paragraph with bold, italic, code and citations", () => {
  const [p] = parseBlocks("The DC is in **Pune** [1], see `NH-205` and *notes* [2][3].");
  assert.equal(p.type, "paragraph");
  assert.deepEqual(p.children.map((n) => n.type),
    ["text", "strong", "text", "cite", "text", "code", "text", "em", "text", "cite", "cite", "text"]);
  assert.deepEqual(p.children.filter((n) => n.type === "cite").map((n) => n.number), [1, 2, 3]);
});

test("headings, lists, numbered lists with start", () => {
  const blocks = parseBlocks("## Summary\n- one\n- two **bold**\n\n3. third\n4. fourth");
  assert.deepEqual(blocks.map((b) => b.type), ["heading", "list", "list"]);
  assert.equal(blocks[0].level, 2);
  assert.equal(blocks[1].ordered, false);
  assert.equal(blocks[1].items.length, 2);
  assert.equal(blocks[2].ordered, true);
  assert.equal(blocks[2].start, 3);
});

test("pipe tables", () => {
  const [table] = parseBlocks("| SKU | Supplier | Lead time |\n|---|---|---|\n| NH-205 | Bright Appliances | 14 days |");
  assert.equal(table.type, "table");
  assert.equal(table.header.length, 3);
  assert.equal(table.rows[0][1][0].value, "Bright Appliances");
});

test("fenced code keeps formatting and is not parsed", () => {
  const [code] = parseBlocks("```python\nx = **not bold** [1]\n```");
  assert.deepEqual(code, { type: "code", lang: "python", text: "x = **not bold** [1]" });
});

test("blockquote and horizontal rule", () => {
  const blocks = parseBlocks("> quoted **text**\n\n---\n\nafter");
  assert.deepEqual(blocks.map((b) => b.type), ["quote", "hr", "paragraph"]);
});

test("links: only http(s) URLs become links", () => {
  const nodes = parseInline("[site](https://example.com) and [bad](javascript:alert(1))");
  assert.equal(nodes[0].type, "link");
  assert.equal(nodes[0].href, "https://example.com");
  assert.ok(!nodes.some((n) => n.type === "link" && n.href.startsWith("javascript")));
});

test("HTML is plain text, never markup", () => {
  const [p] = parseBlocks("<img src=x onerror=alert(1)> <script>alert(1)</script>");
  assert.deepEqual(p.children, [{ type: "text", value: "<img src=x onerror=alert(1)> <script>alert(1)</script>" }]);
});

test("line breaks inside a paragraph become br nodes", () => {
  const [p] = parseBlocks("line one\nline two");
  assert.deepEqual(p.children.map((n) => n.type), ["text", "br", "text"]);
});

test("empty, null and undefined input produce no blocks", () => {
  for (const input of ["", "   \n\n ", null, undefined]) assert.deepEqual(parseBlocks(input), []);
});

test("snake_case identifiers are not italicised", () => {
  const nodes = parseInline("use file_name_here please");
  assert.deepEqual(nodes, [{ type: "text", value: "use file_name_here please" }]);
});

test("citedNumbers", () => {
  assert.deepEqual(citedNumbers("a [1] b [3][1]"), [1, 3, 1]);
  assert.deepEqual(citedNumbers(null), []);
});
