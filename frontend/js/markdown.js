// A small, safe Markdown parser for assistant answers -> a plain-data AST (no HTML is ever produced).
//
// Blocks: headings (#..####), paragraphs, bullet/numbered lists, pipe tables, fenced code, block
// quotes, horizontal rules. Inline: **bold**, *italic*/_italic_, `code`, [n] citations, and
// [text](http(s) URL) links. Anything else is plain text.

const FENCE = /^\s*```\s*([\w+-]*)\s*$/;
const HEADING = /^(#{1,4})\s+(.+?)\s*#*\s*$/;
const BULLET = /^\s*[-*•+]\s+(.*)$/;
const NUMBERED = /^\s*(\d{1,3})[.)]\s+(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
const HR = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/;
const QUOTE = /^\s*>\s?(.*)$/;

function splitRow(line) {
  let row = line.trim();
  if (row.startsWith("|")) row = row.slice(1);
  if (row.endsWith("|")) row = row.slice(0, -1);
  return row.split("|").map((cell) => cell.trim());
}

const isTableRow = (line) => line.includes("|") && splitRow(line).length >= 2;

export function parseBlocks(source) {
  const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let paragraph = [];
  const flush = () => {
    if (paragraph.length) blocks.push({ type: "paragraph", children: parseInline(paragraph.join("\n")) });
    paragraph = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) { flush(); continue; }

    const fence = line.match(FENCE);
    if (fence) {
      flush();
      const code = [];
      for (i += 1; i < lines.length && !FENCE.test(lines[i]); i++) code.push(lines[i]);
      blocks.push({ type: "code", lang: fence[1] || null, text: code.join("\n") });
      continue;
    }
    const heading = line.match(HEADING);
    if (heading) {
      flush();
      blocks.push({ type: "heading", level: heading[1].length, children: parseInline(heading[2]) });
      continue;
    }
    if (HR.test(line)) { flush(); blocks.push({ type: "hr" }); continue; }
    if (isTableRow(line) && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      flush();
      const header = splitRow(line);
      const rows = [];
      for (i += 2; i < lines.length && lines[i].trim() && isTableRow(lines[i]); i++) {
        const cells = splitRow(lines[i]);
        rows.push(header.map((_, c) => parseInline(cells[c] ?? "")));
      }
      i -= 1;
      blocks.push({ type: "table", header: header.map((h) => parseInline(h)), rows });
      continue;
    }
    const bullet = line.match(BULLET);
    const numbered = line.match(NUMBERED);
    if (bullet || numbered) {
      flush();
      const ordered = Boolean(numbered && !bullet);
      const start = ordered ? Number(numbered[1]) : 1;
      const items = [];
      for (; i < lines.length; i++) {
        const m = ordered ? lines[i].match(NUMBERED) : lines[i].match(BULLET);
        if (m) { items.push(ordered ? m[2] : m[1]); continue; }
        // Indented continuation lines belong to the previous item.
        if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
          items[items.length - 1] += " " + lines[i].trim();
          continue;
        }
        break;
      }
      i -= 1;
      blocks.push({ type: "list", ordered, start, items: items.map((t) => parseInline(t)) });
      continue;
    }
    const quote = line.match(QUOTE);
    if (quote) {
      flush();
      const quoted = [];
      for (; i < lines.length && QUOTE.test(lines[i]); i++) quoted.push(lines[i].match(QUOTE)[1]);
      i -= 1;
      blocks.push({ type: "quote", children: parseBlocks(quoted.join("\n")) });
      continue;
    }
    paragraph.push(line.trim());
  }
  flush();
  return blocks;
}

const INLINE = /(\*\*([^*]+?)\*\*|__([^_]+?)__|`([^`]+)`|\[(\d{1,3})\]|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|\*([^*\s][^*]*?)\*|(?<![\w])_([^_\s][^_]*?)_(?![\w]))/g;

export function parseInline(source) {
  const input = String(source ?? "");
  const nodes = [];
  let last = 0;
  const pushText = (value) => {
    if (!value) return;
    const prev = nodes[nodes.length - 1];
    if (prev && prev.type === "text") prev.value += value;
    else nodes.push({ type: "text", value });
  };
  for (const m of input.matchAll(INLINE)) {
    pushText(input.slice(last, m.index));
    if (m[2] !== undefined || m[3] !== undefined) nodes.push({ type: "strong", children: parseInline(m[2] ?? m[3]) });
    else if (m[4] !== undefined) nodes.push({ type: "code", value: m[4] });
    else if (m[5] !== undefined) nodes.push({ type: "cite", number: Number(m[5]) });
    else if (m[6] !== undefined) nodes.push({ type: "link", href: m[7], children: parseInline(m[6]) });
    else nodes.push({ type: "em", children: parseInline(m[8] ?? m[9]) });
    last = m.index + m[0].length;
  }
  pushText(input.slice(last));
  // Line breaks inside a paragraph are kept as explicit break nodes.
  return nodes.flatMap((node) => {
    if (node.type !== "text" || !node.value.includes("\n")) return [node];
    return node.value.split("\n").flatMap((part, i) => (i ? [{ type: "br" }, { type: "text", value: part }] : [{ type: "text", value: part }]))
      .filter((n) => n.type === "br" || n.value);
  });
}

/** Citation numbers referenced anywhere in the text (used to cross-check sources). */
export function citedNumbers(source) {
  return [...String(source ?? "").matchAll(/\[(\d{1,3})\]/g)].map((m) => Number(m[1]));
}
