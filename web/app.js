/* Agentic RAG front end. Plain JS, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined) node.append(kid);
    return node;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v) => (typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : v ?? "");

  let state = null;
  const turns = [];
  let selected = null;

  // ------------------------------------------------------------------ api
  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
      throw new Error(msg);
    }
    return res.json();
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

  function toast(message, bad = false) {
    const t = $("#toast");
    t.textContent = message;
    t.className = "toast" + (bad ? " bad" : "");
    t.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => (t.hidden = true), bad ? 6000 : 3000);
  }

  // ------------------------------------------------------------------ state + rail
  async function refresh() {
    state = await api("/api/state");
    renderRail();
    renderAskHead();
  }

  function activeDataset() { return state.datasets.find((d) => d.active); }

  function renderRail() {
    const list = $("#dataset-list");
    list.replaceChildren(...state.datasets.map((d) => el("li", {},
      el("button", { class: "dataset-btn", "aria-pressed": String(d.active), onclick: () => activate(d.name) },
        el("span", { class: "ds-name", text: d.name }),
        el("span", { class: "ds-meta", text: `${d.rows.toLocaleString()} rows, ${d.columns} columns` })))));
    $("#count-docs").textContent = state.counts.documents || "";
    $("#count-mem").textContent = state.counts.memories || "";

    const p = state.providers.find((x) => x.id === state.provider);
    const chip = $("#model-chip");
    chip.replaceChildren(el("strong", { text: providerName(state.provider) }), el("span", { text: state.model }));
    const sel = $("#provider-select");
    sel.replaceChildren(...state.providers.map((x) => el("option", { value: x.id, selected: x.id === state.provider, text: providerName(x.id) })));
    $("#model-input").value = state.model;
    syncKeyRow(p);
    $("#memory-toggle").checked = state.use_memory;
    $("#embed-hint").textContent = state.embeddings === "hashing"
      ? "Retrieval uses keyword hashing (offline). Install fastembed for meaning-based search."
      : `Retrieval uses the ${state.embeddings} embedding model, running locally.`;
  }

  function providerName(id) {
    return { anthropic: "Claude (Anthropic)", groq: "Groq", ollama: "Ollama (local)", openai: "OpenAI", offline: "Offline rules" }[id] || id;
  }

  function syncKeyRow(p) {
    const needs = p && p.needs_key;
    $("#key-row").hidden = !needs;
    $("#key-hint").textContent = !p ? "" : p.id === "offline"
      ? "Offline mode routes questions with simple rules. It's for trying the app without a key."
      : p.id === "ollama" ? "Runs models on this computer. Start Ollama first."
      : p.key_set ? "A key is set for this provider. Paste a new one to replace it." : "The key stays in this app's memory and is never saved to disk.";
  }

  $("#model-chip").addEventListener("click", () => {
    const panel = $("#model-panel");
    panel.hidden = !panel.hidden;
    $("#model-chip").setAttribute("aria-expanded", String(!panel.hidden));
  });
  $("#provider-select").addEventListener("change", (e) => {
    const p = state.providers.find((x) => x.id === e.target.value);
    $("#model-input").value = p.default_model;
    syncKeyRow(p);
  });
  $("#model-panel").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      state = await post("/api/settings", {
        provider: $("#provider-select").value, model: $("#model-input").value,
        api_key: $("#key-input").value || null, use_memory: $("#memory-toggle").checked,
      });
      $("#key-input").value = "";
      $("#model-panel").hidden = true;
      $("#model-chip").setAttribute("aria-expanded", "false");
      renderRail();
      toast(`Model settings saved: ${providerName(state.provider)}, ${state.model}`);
    } catch (err) { toast(err.message, true); }
  });
  $("#menu-toggle").addEventListener("click", (e) => {
    const rail = $(".rail");
    rail.classList.toggle("show-settings");
    e.currentTarget.setAttribute("aria-expanded", String(rail.classList.contains("show-settings")));
  });

  async function activate(name) {
    try {
      state = await post(`/api/datasets/${encodeURIComponent(name)}/activate`);
      renderRail();
      renderAskHead();
      clearThread();
      if (currentView === "data") renderData();
    } catch (err) { toast(err.message, true); }
  }

  $("#dataset-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      toast(`Loading ${file.name}`);
      state = await api("/api/datasets/upload", { method: "POST", body: form });
      renderRail(); renderAskHead(); clearThread();
      toast(`Loaded ${file.name}. Its column cards were added to the knowledge base.`);
    } catch (err) { toast(err.message, true); }
    e.target.value = "";
  });

  // ------------------------------------------------------------------ views
  let currentView = "ask";
  document.querySelectorAll(".view-btn").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));

  function showView(name) {
    currentView = name;
    document.querySelectorAll(".view-btn").forEach((b) => b.toggleAttribute("aria-current", b.dataset.view === name));
    document.querySelectorAll(".view-btn[aria-current]").forEach((b) => b.setAttribute("aria-current", "page"));
    for (const v of ["ask", "data", "knowledge", "memory"]) $(`#view-${v}`).hidden = v !== name;
    $("#evidence").hidden = name !== "ask";
    $(".app").classList.toggle("no-evidence", name !== "ask");
    if (name === "data") renderData();
    if (name === "knowledge") renderKnowledge();
    if (name === "memory") renderMemory();
  }

  // ------------------------------------------------------------------ ask
  const SUGGESTIONS = {
    subscriptions: ["What does churn rate mean here?", "Which plan has the highest churn, and is the difference significant?",
      "What drives NPS?", "Are there any unusual monthly revenue values?", "Active customers excludes cancelled accounts. Remember that.",
      "How many active customers per region?"],
    retail_sales: ["What's the revenue trend by quarter?", "What counts as the festive season?",
      "Which store has the highest average order value?", "Chart revenue by category"],
  };

  function renderAskHead() {
    const ds = activeDataset();
    $("#ask-title").textContent = ds ? `Ask about ${ds.name}` : "Load a dataset to begin";
    const kinds = state.schema.reduce((acc, c) => ((acc[c.kind] = (acc[c.kind] || 0) + 1), acc), {});
    $("#schema-line").textContent = ds
      ? `${ds.rows.toLocaleString()} rows. ` + Object.entries(kinds).map(([k, n]) => `${n} ${k}`).join(", ") + " columns."
      : "Add a CSV or Excel file from the left panel.";
    const qs = SUGGESTIONS[ds?.name] || genericSuggestions();
    $("#suggestions").replaceChildren(...qs.map((q) => el("button", { class: "suggestion", type: "button", text: q, onclick: () => ask(q) })));
  }

  function genericSuggestions() {
    const by = (k) => state.schema.filter((c) => c.kind === k).map((c) => c.column);
    const num = by("numeric"), cat = by("categorical"), date = by("datetime");
    const out = ["Give me an overview of this data"];
    if (num[0] && cat[0]) out.push(`Average ${num[0]} by ${cat[0]}`);
    if (date[0]) out.push(`Show the trend over time by month`);
    if (num[0]) out.push(`Any unusual ${num[0]} values?`);
    return out;
  }

  function clearThread() {
    turns.length = 0;
    selected = null;
    $("#thread").replaceChildren($("#empty") || buildEmpty());
    $("#empty").hidden = false;
    renderEvidence(null);
  }
  function buildEmpty() { return document.getElementById("empty"); }

  $("#new-chat").addEventListener("click", async () => {
    await post("/api/reset").catch(() => {});
    clearThread();
  });

  const qBox = $("#question");
  qBox.addEventListener("input", () => { qBox.style.height = "auto"; qBox.style.height = Math.min(qBox.scrollHeight, 180) + "px"; });
  qBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
  });
  $("#ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = qBox.value.trim();
    if (q) ask(q);
  });

  let busy = false;
  async function ask(question) {
    if (busy) return;
    if (!activeDataset()) { toast("Add a dataset first.", true); return; }
    busy = true;
    $("#ask-btn").disabled = true;
    qBox.value = ""; qBox.style.height = "auto";
    $("#empty").hidden = true;

    const turn = { question, events: [], passages: [], memories: [], stale: [], trace: [], answer: null };
    turns.push(turn);
    const steps = el("ul", { class: "steps" });
    const card = el("div", { class: "a", tabindex: "0", role: "button", "aria-label": "Answer. Select to show its evidence." }, steps);
    const node = el("article", { class: "turn" }, el("div", { class: "q", text: question }), card);
    turn.node = node; turn.card = card;
    $("#thread").append(node);
    node.scrollIntoView({ block: "end", behavior: "smooth" });
    const addStep = (text, cls = "running") => { const li = el("li", { class: cls, html: text }); steps.append(li); return li; };
    let pending = addStep("Searching the knowledge base and memory");
    const running = new Map();

    try {
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
      if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail; } catch (_) {} throw new Error(m); }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line);
          turn.events.push(ev);
          if (ev.type === "retrieval") {
            turn.passages = ev.passages; turn.memories = ev.memories; turn.stale = ev.stale;
            const docs = ev.passages.filter((p) => p.kind === "document").length;
            const cards = ev.passages.length - docs;
            pending.className = "done";
            pending.innerHTML = `Retrieved ${plural(docs, "document passage")}, ${plural(cards, "column card")} and ${plural(ev.memories.length, "memory", "memories")}`;
            pending = null;
          } else if (ev.type === "tool_start") {
            running.set(ev.name + JSON.stringify(ev.args), addStep(`Running <code>${esc(ev.name)}</code>`));
          } else if (ev.type === "tool_end") {
            const li = running.get(ev.name + JSON.stringify(ev.args)) || addStep("");
            const failed = ev.result && ev.result.error;
            li.className = failed ? "failed" : "done";
            li.innerHTML = `<code>${esc(ev.name)}</code> ${failed ? "returned an error, retrying" : "finished in " + ev.seconds + "s"}`;
            turn.trace.push(ev);
          } else if (ev.type === "answer") {
            turn.answer = ev;
            renderAnswer(turn, steps);
          } else if (ev.type === "error") {
            throw new Error(ev.message);
          }
        }
      }
      if (!turn.answer) throw new Error("The answer stream ended early. Try asking again.");
    } catch (err) {
      card.classList.add("error");
      card.removeAttribute("role"); card.removeAttribute("tabindex");
      card.replaceChildren(el("div", { class: "prose", text: err.message }));
    } finally {
      busy = false;
      $("#ask-btn").disabled = false;
      qBox.focus();
      refreshCounts();
    }
  }

  const plural = (n, one, many) => `${n} ${n === 1 ? one : many || one + "s"}`;

  async function refreshCounts() {
    try { const s = await api("/api/state"); state.counts = s.counts; $("#count-mem").textContent = s.counts.memories || ""; } catch (_) {}
  }

  function renderAnswer(turn, steps) {
    const a = turn.answer;
    const prose = el("div", { class: "prose", html: markdown(a.text, a.sources.map((s) => s.label)) });
    const kids = [prose];
    a.charts.forEach((c) => kids.push(chart(c)));
    const applied = turn.memories.filter((m) => a.applied_memory_ids.includes(m.id));
    const foot = el("div", { class: "a-foot" },
      el("span", { html: `<b>${a.trace.length}</b> tool ${a.trace.length === 1 ? "call" : "calls"}` }),
      el("span", { html: `<b>${a.cited.length}</b> cited ${a.cited.length === 1 ? "source" : "sources"}` }),
      applied.length ? el("span", { html: `Applied memory: <b>${esc(applied.map((m) => m.key).join(", "))}</b>` }) : null,
      el("span", { text: `${a.seconds}s` }));
    kids.push(foot);
    steps.hidden = true;
    turn.card.replaceChildren(steps, ...kids);
    const select = () => selectTurn(turn);
    turn.card.addEventListener("click", (e) => {
      const mark = e.target.closest("mark.cite");
      select();
      if (mark) lightSource(mark.dataset.label);
    });
    turn.card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); } });
    selectTurn(turn);
    turn.node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  function selectTurn(turn) {
    selected = turn;
    turns.forEach((t) => t.card.classList.toggle("selected", t === turn));
    renderEvidence(turn);
    if (window.matchMedia("(max-width: 1180px)").matches) $("#evidence").classList.add("open");
  }
  $("#ev-close").addEventListener("click", () => $("#evidence").classList.remove("open"));

  // ------------------------------------------------------------------ evidence
  function renderEvidence(turn) {
    const body = $("#ev-body");
    if (!turn || !turn.answer) {
      $("#ev-title").textContent = "Evidence";
      body.replaceChildren(el("p", { class: "ev-empty", text: "The sources, memories and tool steps behind an answer show up here." }));
      return;
    }
    const a = turn.answer;
    $("#ev-title").textContent = "Evidence for this answer";
    const cited = new Set(a.cited);
    const sources = [...a.sources].sort((x, y) => (cited.has(y.label) - cited.has(x.label)) || (y.score - x.score));
    const groups = [];
    groups.push(el("section", { class: "ev-group" },
      el("h3", { text: sources.length ? `Retrieved passages (${cited.size} cited)` : "Retrieved passages" }),
      sources.length ? sources.map((s) => sourceCard(s, cited.has(s.label))) : el("p", { class: "ev-empty", text: "Nothing in the knowledge base cleared the relevance bar for this question." })));
    if (turn.memories.length || turn.stale.length) {
      groups.push(el("section", { class: "ev-group" }, el("h3", { text: "Memory" }),
        turn.memories.map((m) => el("div", { class: "source" + (a.applied_memory_ids.includes(m.id) ? " cited" : "") },
          el("div", { class: "source-top" }, el("span", { class: "label", text: m.label }),
            el("span", { class: "source-title", text: m.key }),
            el("span", { class: "score", text: a.applied_memory_ids.includes(m.id) ? "applied" : m.kind })),
          el("div", { class: "source-text", text: m.text }))),
        turn.stale.map((s) => el("p", { class: "hint", text: `Not used: ${s.key}, because ${s.why}.` }))));
    }
    groups.push(el("section", { class: "ev-group" }, el("h3", { text: `Tool steps (${a.trace.length})` }),
      a.trace.length ? a.trace.map(traceStep) : el("p", { class: "ev-empty", text: "Answered from the retrieved context without running analysis tools." })));
    body.replaceChildren(...groups);
  }

  function sourceCard(s, isCited) {
    const card = el("div", { class: "source" + (isCited ? " cited" : ""), "data-label": s.label },
      el("div", { class: "source-top" }, el("span", { class: "label", text: s.label }),
        el("span", { class: "source-title", text: s.title }), el("span", { class: "score", text: `match ${s.score.toFixed(2)}` })),
      s.location ? el("div", { class: "source-where", text: s.location }) : null,
      el("div", { class: "source-text", text: s.text.replace(/^#{1,6}\s*/gm, "") }));
    if (s.text.length > 260) {
      const more = el("button", { class: "source-more", type: "button", text: "Show all" });
      more.addEventListener("click", () => { card.classList.toggle("open"); more.textContent = card.classList.contains("open") ? "Show less" : "Show all"; });
      card.append(more);
    }
    return card;
  }

  function lightSource(label) {
    document.querySelectorAll(".source.lit").forEach((n) => n.classList.remove("lit"));
    const card = document.querySelector(`.source[data-label="${CSS.escape(label)}"]`);
    if (card) { card.classList.add("lit"); card.scrollIntoView({ block: "center", behavior: "smooth" }); }
  }

  function traceStep(t) {
    const failed = t.result && t.result.error;
    return el("details", { class: "trace-step" },
      el("summary", {}, el("span", { class: "t-name", text: t.name }), failed ? el("span", { class: "t-err", text: "error" }) : null,
        el("span", { class: "t-time", text: `${t.seconds}s` })),
      el("pre", { text: "arguments\n" + JSON.stringify(t.args, null, 2) }),
      el("pre", { text: "result\n" + JSON.stringify(t.result, null, 2) }));
  }

  // ------------------------------------------------------------------ markdown (small, safe)
  function inline(text, labels) {
    const code = [];
    let s = esc(text).replace(/`([^`]+)`/g, (_, c) => `\u0000${code.push(c) - 1}\u0000`);
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
    s = s.replace(/(^|[\s(])_([^_\s][^_]*)_(?=[\s.,;:)]|$)/g, "$1<em>$2</em>");
    s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${code[+i]}</code>`);
    s = s.replace(/\[(K\d+)\]/g, (m, label) => labels.includes(label)
      ? `<mark class="cite" data-label="${label}" tabindex="0" title="Show source ${label}">${label}</mark>` : m);
    return s;
  }

  function markdown(src, labels = []) {
    const lines = String(src || "").replace(/\r/g, "").split("\n");
    const out = [];
    let i = 0;
    const isNum = (v) => /^-?[\d,]+(\.\d+)?%?$|^<0\.\d+$/.test(v.trim());
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const head = cells(line);
        const rows = [];
        i += 2;
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
        const numeric = head.map((_, c) => rows.length && rows.every((r) => !r[c] || isNum(r[c]) || r[c] === "None"));
        out.push("<table><thead><tr>" + head.map((h, c) => `<th class="${numeric[c] ? "num" : ""}">${inline(h, labels)}</th>`).join("") +
          "</tr></thead><tbody>" + rows.map((r) => "<tr>" + r.map((v, c) => `<td class="${numeric[c] ? "num" : ""}">${inline(v, labels)}</td>`).join("") + "</tr>").join("") +
          "</tbody></table>");
        continue;
      }
      if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
        const ordered = /^\s*\d+\./.test(line);
        const items = [];
        while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, ""));
        out.push(`<${ordered ? "ol" : "ul"}>` + items.map((t) => `<li>${inline(t, labels)}</li>`).join("") + `</${ordered ? "ol" : "ul"}>`);
        continue;
      }
      const h = line.match(/^(#{1,4})\s+(.*)/);
      if (h) { out.push(`<h${Math.min(h[1].length + 2, 4)}>${inline(h[2], labels)}</h${Math.min(h[1].length + 2, 4)}>`); i++; continue; }
      if (!line.trim()) { i++; continue; }
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*\|/.test(lines[i]) && !/^\s*([-*]|\d+\.)\s+/.test(lines[i]) && !/^#{1,4}\s/.test(lines[i])) para.push(lines[i++]);
      out.push(`<p>${inline(para.join(" "), labels)}</p>`);
    }
    return out.join("");
  }

  // ------------------------------------------------------------------ charts (flat SVG)
  const NS = "http://www.w3.org/2000/svg";
  const svg = (tag, attrs = {}, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const short = (v) => {
    if (typeof v !== "number") return String(v);
    const a = Math.abs(v);
    if (a >= 1e7) return (v / 1e6).toFixed(0) + "M";
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
    if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
    return Number.isInteger(v) ? String(v) : v.toFixed(2);
  };

  function chart(spec) {
    const fig = el("figure", { class: "chart" }, el("figcaption", { text: spec.title }));
    const pts = spec.points || [];
    if (!pts.length) { fig.append(el("p", { class: "hint", text: "No data to draw." })); return fig; }
    // Draw at roughly the on-screen width so text stays readable on phones.
    const W = Math.round(Math.max(300, Math.min(640, ($("#thread").clientWidth || 640) - 90)));
    let node;
    if (spec.kind === "bar") node = barsH(pts, W);
    else if (spec.kind === "pie") node = shareBar(pts, W);
    else if (spec.kind === "line") node = line(pts, spec, W);
    else if (spec.kind === "hist") node = columns(pts, spec, W);
    else node = scatter(pts, spec, W);
    fig.append(node);
    fig.setAttribute("aria-label", `${spec.kind} chart: ${spec.title}`);
    return fig;
  }

  function barsH(pts, W) {
    const row = 26, labelW = Math.round(Math.min(140, W * 0.3)), H = pts.length * row + 4;
    const max = Math.max(...pts.map((p) => p.value || 0), 1);
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    pts.forEach((p, i) => {
      const y = i * row + 2;
      const w = Math.max(1, ((p.value || 0) / max) * (W - labelW - 70));
      s.append(svg("text", { x: labelW - 8, y: y + 16, "text-anchor": "end" }, String(p.label).slice(0, 22)));
      s.append(svg("rect", { x: labelW, y: y + 4, width: w, height: row - 10, fill: "var(--chart-1)", rx: 2 }));
      s.append(svg("text", { x: labelW + w + 6, y: y + 16, class: "val" }, short(p.value)));
    });
    return s;
  }

  function shareBar(pts, W) {
    const H = 70;
    const total = pts.reduce((a, p) => a + Math.max(0, p.value || 0), 0) || 1;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    let x = 0;
    const colors = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];
    pts.forEach((p, i) => {
      const w = (Math.max(0, p.value || 0) / total) * W;
      s.append(svg("rect", { x, y: 0, width: Math.max(w - 2, 0.5), height: 28, fill: colors[i % colors.length] }));
      if (w > 60) s.append(svg("text", { x: x + 4, y: 46 }, String(p.label).slice(0, 14)));
      if (w > 60) s.append(svg("text", { x: x + 4, y: 62, class: "val" }, `${((p.value / total) * 100).toFixed(1)}%`));
      x += w;
    });
    return s;
  }

  function frame(spec, W, H, pad) {
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    s.append(svg("line", { x1: pad.l, y1: H - pad.b, x2: W - pad.r, y2: H - pad.b, class: "axis" }));
    s.append(svg("text", { x: pad.l, y: 10 }, spec.y_label || ""));
    return s;
  }

  function line(pts, spec, W) {
    const H = 240, pad = { l: 48, r: 12, t: 20, b: 36 };
    const s = frame(spec, W, H, pad);
    const vals = pts.map((p) => p.value ?? 0);
    const max = Math.max(...vals), min = Math.min(0, ...vals);
    const X = (i) => pad.l + (pts.length === 1 ? 0 : (i / (pts.length - 1)) * (W - pad.l - pad.r));
    const Y = (v) => H - pad.b - ((v - min) / (max - min || 1)) * (H - pad.t - pad.b);
    [min, (min + max) / 2, max].forEach((v) => {
      s.append(svg("line", { x1: pad.l, x2: W - pad.r, y1: Y(v), y2: Y(v), class: "axis", "stroke-dasharray": "2 4" }));
      s.append(svg("text", { x: pad.l - 6, y: Y(v) + 4, "text-anchor": "end" }, short(v)));
    });
    s.append(svg("polyline", { points: vals.map((v, i) => `${X(i)},${Y(v)}`).join(" "), fill: "none", stroke: "var(--chart-1)", "stroke-width": 2, "stroke-linejoin": "round" }));
    const peak = vals.indexOf(max);
    s.append(svg("circle", { cx: X(peak), cy: Y(max), r: 4, fill: "var(--hi)", stroke: "var(--ink)", "stroke-width": 1 }));
    const every = Math.ceil(pts.length / 6);
    pts.forEach((p, i) => { if (i % every === 0 || i === pts.length - 1) s.append(svg("text", { x: X(i), y: H - pad.b + 16, "text-anchor": "middle" }, String(p.label).slice(0, 10))); });
    return s;
  }

  function columns(pts, spec, W) {
    const H = 220, pad = { l: 40, r: 8, t: 20, b: 36 };
    const s = frame(spec, W, H, pad);
    const max = Math.max(...pts.map((p) => p.value || 0), 1);
    const bw = (W - pad.l - pad.r) / pts.length;
    pts.forEach((p, i) => {
      const h = ((p.value || 0) / max) * (H - pad.t - pad.b);
      s.append(svg("rect", { x: pad.l + i * bw + 1, y: H - pad.b - h, width: Math.max(bw - 2, 1), height: h, fill: "var(--chart-1)" }));
    });
    s.append(svg("text", { x: pad.l - 6, y: pad.t + 4, "text-anchor": "end" }, short(max)));
    s.append(svg("text", { x: pad.l, y: H - pad.b + 16 }, String(pts[0].label).split(" to ")[0]));
    s.append(svg("text", { x: W - pad.r, y: H - pad.b + 16, "text-anchor": "end" }, String(pts[pts.length - 1].label).split(" to ")[1] || ""));
    s.append(svg("text", { x: (W + pad.l) / 2, y: H - 4, "text-anchor": "middle" }, spec.x_label));
    return s;
  }

  function scatter(pts, spec, W) {
    const H = 260, pad = { l: 48, r: 12, t: 20, b: 40 };
    const s = frame(spec, W, H, pad);
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const X = (v) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r);
    const Y = (v) => H - pad.b - ((v - y0) / (y1 - y0 || 1)) * (H - pad.t - pad.b);
    pts.forEach((p) => s.append(svg("circle", { cx: X(p.x), cy: Y(p.y), r: 2.6, fill: "var(--chart-1)", "fill-opacity": 0.45 })));
    s.append(svg("text", { x: pad.l - 6, y: Y(y1) + 4, "text-anchor": "end" }, short(y1)));
    s.append(svg("text", { x: pad.l - 6, y: Y(y0) + 4, "text-anchor": "end" }, short(y0)));
    s.append(svg("text", { x: pad.l, y: H - pad.b + 16 }, short(x0)));
    s.append(svg("text", { x: W - pad.r, y: H - pad.b + 16, "text-anchor": "end" }, short(x1)));
    s.append(svg("text", { x: (W + pad.l) / 2, y: H - 4, "text-anchor": "middle" }, spec.x_label));
    return s;
  }

  // ------------------------------------------------------------------ data view
  async function renderData() {
    const ds = activeDataset();
    if (!ds) return;
    $("#data-title").textContent = ds.name;
    $("#data-sub").textContent = `${ds.rows.toLocaleString()} rows and ${ds.columns} columns, from ${ds.source}.`;
    try {
      const p = await api(`/api/datasets/${encodeURIComponent(ds.name)}/preview`);
      $("#columns-grid").replaceChildren(...p.columns.map((c) => el("div", { class: `col-card kind-${c.kind}` },
        el("div", { class: "c-name", text: c.column }), el("span", { class: "c-kind", text: c.kind }),
        el("div", { class: "c-desc", text: c.description.replace(/^.*?\]\s*/, "") }))));
      const cols = p.columns.map((c) => c.column);
      const numeric = new Set(p.columns.filter((c) => c.kind === "numeric").map((c) => c.column));
      const table = el("table", {},
        el("thead", {}, el("tr", {}, cols.map((c) => el("th", { class: numeric.has(c) ? "num" : "", text: c })))),
        el("tbody", {}, p.rows.map((r) => el("tr", {}, cols.map((c) => el("td", { class: numeric.has(c) ? "num" : "", text: fmt(r[c]) }))))));
      $("#preview-table").replaceChildren(table);
    } catch (err) { toast(err.message, true); }
  }

  // ------------------------------------------------------------------ knowledge view
  async function renderKnowledge() {
    try {
      const { documents } = await api("/api/knowledge");
      const docs = documents.filter((d) => d.kind === "document");
      const cards = documents.filter((d) => d.kind === "dataset");
      const items = [...docs, ...cards].map((d) => {
        const chunksBox = el("div", { class: "chunks", hidden: true });
        const toggle = el("button", { class: "btn ghost small", type: "button", text: "Show chunks" });
        toggle.addEventListener("click", async () => {
          if (chunksBox.hidden && !chunksBox.childElementCount) {
            const r = await api(`/api/knowledge/${d.id}`);
            chunksBox.replaceChildren(...r.chunks.map((c) => el("div", { class: "chunk" },
              el("b", { text: c.section || (c.page ? `Page ${c.page}` : c.column ? `Column ${c.column}` : `Chunk ${c.position + 1}`) }), c.text)));
          }
          chunksBox.hidden = !chunksBox.hidden;
          toggle.textContent = chunksBox.hidden ? "Show chunks" : "Hide chunks";
        });
        const del = d.kind === "document" ? el("button", { class: "btn ghost small danger", type: "button", text: "Remove", onclick: async () => {
          if (!confirm(`Remove ${d.title} from the knowledge base?`)) return;
          try { await api(`/api/knowledge/${d.id}`, { method: "DELETE" }); toast(`Removed ${d.title}`); renderKnowledge(); refresh(); } catch (err) { toast(err.message, true); }
        } }) : null;
        return el("li", { class: "doc-item" },
          el("div", { class: "doc-row" },
            el("div", {}, el("div", { class: "doc-title", text: d.title }),
              el("div", { class: "doc-meta", text: d.kind === "document"
                ? `${plural(d.chunks, "chunk")} from ${d.chars.toLocaleString()} characters, added ${d.created_at.slice(0, 10)}`
                : `Column cards built from the dataset, ${plural(d.chunks, "card")}` })),
            el("div", { class: "doc-actions" }, toggle, del)),
          chunksBox);
      });
      $("#doc-list").replaceChildren(...(items.length ? items : [el("li", { class: "ev-empty", text: "No documents yet. Upload a glossary or data dictionary so the agent can look up what your terms mean." })]));
    } catch (err) { toast(err.message, true); }
  }

  $("#doc-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      toast(`Reading ${file.name}`);
      const r = await api("/api/knowledge/upload", { method: "POST", body: form });
      toast(`Added ${r.document.title} as ${plural(r.document.chunks, "chunk")}.`);
      renderKnowledge(); refresh();
    } catch (err) { toast(err.message, true); }
    e.target.value = "";
  });

  $("#probe-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#probe-input").value.trim();
    if (!q) return;
    try {
      const r = await api(`/api/knowledge/search?q=${encodeURIComponent(q)}&k=5`);
      const box = $("#probe-results");
      box.replaceChildren(
        el("p", { class: "hint", text: `Passages scoring below ${r.floor} aren't given to the agent.` }),
        ...r.hits.map((h, i) => {
          const card = sourceCard({ ...h, label: `#${i + 1}` }, h.score >= r.floor);
          return card;
        }));
    } catch (err) { toast(err.message, true); }
  });

  // ------------------------------------------------------------------ memory view
  async function renderMemory() {
    try {
      const { memories, kinds } = await api("/api/memory");
      const kindSel = $("#mem-kind");
      if (!kindSel.childElementCount) kindSel.replaceChildren(...Object.keys(kinds).filter((k) => k !== "named_filter").map((k) => el("option", { value: k, text: k.replace("_", " ") })));
      $("#mem-list").replaceChildren(...(memories.length ? memories.map(memoryItem)
        : [el("li", { class: "ev-empty", text: "Nothing remembered yet. Teach the agent a definition in the Ask view (for example \"Active customers excludes cancelled accounts. Remember that.\") or add one here." })]));
      $("#count-mem").textContent = memories.length || "";
    } catch (err) { toast(err.message, true); }
  }

  function memoryItem(m) {
    const contentNode = el("div", { class: "mem-content", text: m.content });
    const extra = el("div", {});
    const edit = el("button", { class: "btn ghost small", type: "button", text: "Edit" });
    edit.addEventListener("click", () => {
      const input = el("input", { value: m.content, "aria-label": `New value for ${m.key}` });
      const save = el("button", { class: "btn primary small", type: "submit", text: "Save" });
      const form = el("form", { class: "mem-edit" }, input, save, el("button", { class: "btn ghost small", type: "button", text: "Cancel", onclick: () => extra.replaceChildren() }));
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        try {
          await api(`/api/memory/${m.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content: input.value }) });
          toast(`Saved the new value for ${m.key}. The old one is kept in its history.`);
          renderMemory();
        } catch (err) { toast(err.message, true); }
      });
      extra.replaceChildren(form);
      input.focus();
    });
    const hist = el("button", { class: "btn ghost small", type: "button", text: "History", onclick: async () => {
      const { history } = await api(`/api/memory/${m.id}/history`);
      extra.replaceChildren(el("div", { class: "history" }, history.map((h) =>
        el("div", {}, el("b", { text: `#${h.memory.id} ${h.memory.status}: ` }), h.memory.content,
          ...h.events.map((e) => el("div", { text: `${e.at} ${e.action}${e.detail ? ", " + e.detail : ""}` }))))));
    } });
    const del = el("button", { class: "btn ghost small danger", type: "button", text: "Delete", onclick: async () => {
      try { await api(`/api/memory/${m.id}`, { method: "DELETE" }); toast(`Deleted ${m.key}`); renderMemory(); } catch (err) { toast(err.message, true); }
    } });
    const scope = m.scope === "global" ? "all datasets" : m.dataset_name || "one dataset";
    return el("li", { class: "mem-item" },
      el("div", { class: "mem-row" },
        el("div", {}, el("span", { class: "mem-key", text: m.key }), " ",
          el("span", { class: "badge" + (m.source === "user" ? " user" : ""), text: m.kind.replace("_", " ") }),
          contentNode,
          m.filter ? el("div", { class: "mem-meta", text: "Filter: " + m.filter.map((f) => `${f.column} ${f.op} ${JSON.stringify(f.value ?? "")}`).join(" and ") }) : null,
          el("div", { class: "mem-meta", text: `${m.source === "user" ? "Stated by you" : "Inferred by the agent"}, applies to ${scope}, used ${plural(m.use_count, "time")}` })),
        el("div", { class: "mem-actions" }, edit, hist, del)),
      extra);
  }

  $("#mem-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const r = await post("/api/memory", { kind: $("#mem-kind").value, key: $("#mem-key").value, content: $("#mem-content").value, scope: "global" });
      toast(r.result.status === "updated" ? "Replaced the earlier value." : "Remembered.");
      $("#mem-key").value = ""; $("#mem-content").value = "";
      renderMemory();
    } catch (err) { toast(err.message, true); }
  });
  $("#clear-mem").addEventListener("click", async () => {
    if (!confirm("Forget everything in memory? History is kept, but nothing will be applied any more.")) return;
    await api("/api/memory", { method: "DELETE" });
    toast("Memory cleared.");
    renderMemory();
  });

  // ------------------------------------------------------------------ start
  refresh().catch((err) => toast(`Couldn't reach the server: ${err.message}`, true));
})();
