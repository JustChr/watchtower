// Watchtower's web UI: one module, no dependencies.
//
// Safety: everything from outside (strangers' text, model output, even our own
// Telegram HTML) is put on the page as text nodes or through `telegram()`, a
// whitelist rebuild. Nothing here assigns HTML strings.

const NS = "http://www.w3.org/2000/svg";
const view = document.getElementById("view");
const tip = document.getElementById("tip");
const drawer = document.getElementById("drawer");
const drawerBody = document.getElementById("drawer-body");

// -- building elements ------------------------------------------------------------

function put(el, attrs) {
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("--")) el.style.setProperty(key, value);
    else el.setAttribute(key, value === true ? "" : value);
  }
}

function add(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

// Replace an element's children: arrays are flattened, null and false left out.
const fill = (el, ...kids) => {
  el.replaceChildren();
  return add(el, kids);
};

const h = (tag, attrs, ...kids) => {
  const el = document.createElement(tag);
  put(el, attrs);
  return add(el, kids);
};
const s = (tag, attrs, ...kids) => {
  const el = document.createElementNS(NS, tag);
  put(el, attrs);
  return add(el, kids);
};

// -- talking to the service ----------------------------------------------------------

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

async function act(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Watchtower": "1" },
    body: JSON.stringify(body),
  });
  const answer = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(answer.error || `HTTP ${response.status}`);
  return answer.message;
}

// -- words and numbers ---------------------------------------------------------------

const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const dayName = (t) =>
  new Date(t * 1000).toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" });
const dateTime = (t) => `${dayName(t)}, ${clock(t)}`;
const isoDay = (d) =>
  new Date(`${d}T12:00:00Z`).toLocaleDateString([], { day: "numeric", month: "short" });

function ago(seconds) {
  if (seconds < 90) return `${Math.max(0, Math.round(seconds))} s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 172800) return `${(seconds / 3600).toFixed(1)} h ago`;
  return `${Math.round(seconds / 86400)} days ago`;
}

function took(seconds) {
  if (seconds == null) return "";
  if (seconds < 90) return `${Math.round(seconds)} s`;
  return `${(seconds / 60).toFixed(1)} min`;
}

const count = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const chars = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

const KINDS = ["summary", "findings", "investigate", "assess", "reply", "brief"];
const KIND_LABELS = {
  summary: "Summaries",
  findings: "Reading big files",
  investigate: "Investigating",
  assess: "Assessing",
  reply: "Writing replies",
  brief: "Repo briefs",
};
const TOPICS = { triage: "🩺 Triage", reviews: "🔍 Reviews", replies: "💬 Replies", system: "⚙️ System" };
const STATUS_WORDS = {
  prep: "Fetching the thread",
  queued: "Waiting for the worker",
  drafting: "Being written",
  ready: "Waiting for you",
  posting: "Posting",
  posted: "Posted",
  rejected: "Rejected",
  superseded: "Replaced by a newer draft",
  failed: "Failed",
};

// -- Telegram's HTML, rebuilt from a whitelist --------------------------------------

const ALLOWED = new Set(["B", "STRONG", "I", "EM", "U", "S", "PRE", "CODE", "BLOCKQUOTE", "A"]);

function telegram(html) {
  const doc = new DOMParser().parseFromString(`<div>${html}</div>`, "text/html");
  const rebuild = (node) => {
    const out = document.createDocumentFragment();
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        out.append(document.createTextNode(child.textContent));
      } else if (child.nodeType === Node.ELEMENT_NODE && ALLOWED.has(child.tagName)) {
        const el = document.createElement(child.tagName.toLowerCase());
        if (child.tagName === "A") {
          const href = child.getAttribute("href") || "";
          if (/^https:\/\//i.test(href)) put(el, { href, target: "_blank", rel: "noopener noreferrer" });
        }
        el.append(rebuild(child));
        out.append(el);
      } else if (child.nodeType === Node.ELEMENT_NODE) {
        out.append(rebuild(child));
      }
    }
    return out;
  };
  return rebuild(doc.body.firstChild);
}

const external = (href, label) =>
  /^https:\/\//i.test(href || "") ? h("a", { href, target: "_blank", rel: "noopener noreferrer" }, label) : null;

// -- what each station is (the panel's drawer, and "How it works") -------------------

function stations(o) {
  const poll = o ? `every ${o.poll_seconds} s` : "every couple of minutes";
  return {
    github: {
      title: "GitHub",
      outside: true,
      does: "Where the watched repos live. Watchtower reads issues, pull requests, discussions and comments here, and posts the replies you approved.",
    },
    watcher: {
      title: "Watcher",
      does: `Polls each repo ${poll} (an unchanged repo costs nothing), hands every new item to the worker, and keeps a searchable copy of every thread and a snapshot of the code. When the worker wants to draft a reply, it fetches what the worker can't: the fresh thread, the attached files, the code at the author's version.`,
      holds: "The GitHub read-only token.",
      network: "Internet, for GitHub.",
      never: "Talks to the model or writes to GitHub.",
    },
    worker: {
      title: "Worker",
      does: "All model work, one job at a time: a one-line summary of every new item, reply drafts in passes (read big files, investigate with read-only tools, assess, write), repo briefs, replays. Summaries go first, even between a draft's passes.",
      holds: "Nothing.",
      network: "Only the internal network to Ollama: no internet.",
      never: "Reaches GitHub or Telegram, or holds a secret. It's the one place strangers' text meets a model.",
    },
    ollama: {
      title: "Ollama",
      outside: true,
      does: o
        ? `The local models on this box: ${o.models.summary || "none"} for summaries, ${o.models.agent || "none"} for drafts and briefs. Prompts never leave the box: cloud models are refused.`
        : "The local models on this box. Prompts never leave it.",
    },
    gateway: {
      title: "Gateway",
      does: "The only Telegram client. Sends the outbox to your group, one topic per kind of work, and records your button presses and your replies to drafts. It answers only you, only in the group.",
      holds: "The Telegram bot token.",
      network: "Internet, for Telegram.",
      never: "Applies a decision itself: it only records it.",
    },
    telegram: {
      title: "Telegram",
      outside: true,
      does: "Your group, with the topics Triage, Reviews, Replies and System. Where you approve, edit and reject.",
    },
    poster: {
      title: "Poster",
      does: "The only GitHub writer. Applies decisions: posts exactly the version you approved, as the GitHub App; turns an edit into a new version; rejects. A post needs your tap in Telegram: it refuses one from anywhere else.",
      holds: "The GitHub App's private key.",
      network: "Internet, for GitHub.",
      never: "Talks to the model, or posts a version you didn't approve.",
    },
    web: {
      title: "Web UI",
      does: "This page. It shows everything, and can edit or reject a draft, or send a version to Telegram for your tap. It can't post.",
      holds: "Nothing.",
      network: "Your LAN.",
      never: "Posts to GitHub.",
    },
  };
}

// -- the panel -----------------------------------------------------------------------

const BOXES = {
  github: [30, 60], watcher: [290, 60], worker: [610, 60], ollama: [990, 60],
  gateway: [610, 230], telegram: [990, 230], poster: [290, 380], web: [990, 380],
};
const BOX_W = 150;
const BOX_H = 64;
const STALE = { watcher: 900, gateway: 120, worker: 1200, poster: 120, web: 120 };

function railStates(o) {
  const out = o.outbox.pending;
  const jobs = Object.values(o.queued_jobs || {}).reduce((a, b) => a + b, 0);
  const waiting = (o.in_flight || []).filter((d) => d.status === "prep" || d.status === "queued").length;
  const worker = o.heartbeats.worker;
  const thinking = worker && worker.detail && worker.detail !== "idle";
  // "owner/repo#12: investigating, step 3" -> "investigating, step 3"
  const step = thinking ? worker.detail.replace(/^[^:]*#\d+: /, "") : "";
  const posting = (o.in_flight || []).some((d) => d.status === "posting");
  const alive = (name) => o.heartbeats[name] && o.now - o.heartbeats[name].at < STALE[name];
  return [
    { id: "poll", d: "M180 92 H290", busy: false, set: alive("watcher"), label: "polls", at: [235, 80] },
    { id: "jobs", d: "M440 92 H610", busy: jobs + waiting > 0, set: true, label: jobs + waiting ? count(jobs + waiting, "job") : "jobs", at: [525, 80] },
    { id: "model", d: "M760 92 H990", busy: thinking, set: alive("worker"), label: thinking ? (step.length > 34 ? `${step.slice(0, 33)}…` : step) : "idle", at: [875, 80] },
    { id: "report", d: "M685 124 V230", busy: out > 0, set: true, label: "", at: [0, 0] },
    { id: "bots", d: "M365 124 V222 L405 262 H610", busy: out > 0, set: true, label: "", at: [0, 0] },
    { id: "send", d: "M760 262 H990", busy: out > 0, set: alive("gateway"), label: out ? `${out} to send` : "messages", at: [875, 250] },
    { id: "decide", d: "M685 294 V412", busy: o.open_decisions > 0, set: true, label: "", at: [0, 0] },
    { id: "decisions", d: "M990 412 H440", busy: o.open_decisions > 0, set: alive("poster"), label: o.open_decisions ? count(o.open_decisions, "decision") : "decisions", at: [820, 400] },
    { id: "post", d: "M290 412 H145 L105 372 V124", busy: posting, set: alive("poster"), label: "posts", at: [200, 400] },
  ];
}

function panel(o, { boot = false, onPick = openStation } = {}) {
  const info = stations(o);
  const root = s("svg", {
    class: `panel${boot ? " boot" : ""}`,
    viewBox: "0 0 1180 470",
    role: "img",
    "aria-label": "How Watchtower's services connect, and where work is right now",
  });
  // The panel's tile grid.
  for (let x = 20; x < 1180; x += 40) root.append(s("line", { class: "grid", x1: x, y1: 0, x2: x, y2: 470 }));
  for (let y = 20; y < 470; y += 40) root.append(s("line", { class: "grid", x1: 0, y1: y, x2: 1180, y2: y }));
  const rails = railStates(o);
  rails.forEach((rail, i) => {
    const cls = `rail${rail.busy ? " busy" : rail.set ? " set" : ""}`;
    root.append(s("path", { class: cls, d: rail.d, pathLength: 400, "--i": i }));
    if (rail.busy) root.append(s("path", { class: "train", d: rail.d, pathLength: 400 }));
    if (rail.label) {
      root.append(s("text", { class: `rail-label${rail.busy ? " busy" : ""}`, x: rail.at[0], y: rail.at[1], "text-anchor": "middle" }, rail.label));
    }
  });
  Object.entries(BOXES).forEach(([name, [x, y]], i) => {
    const station = info[name];
    const beat = o.heartbeats[name];
    const stale = beat && o.now - beat.at >= STALE[name];
    const lampClass = station.outside ? "lamp" : !beat ? "lamp" : stale ? "lamp bad" : "lamp good";
    const g = s("g", {
      class: `station${station.outside ? " outside" : ""}`,
      tabindex: 0,
      role: "button",
      "aria-label": `${station.title}: what it does`,
      onclick: () => onPick(name, o),
      onkeydown: (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onPick(name, o)),
    });
    g.append(s("rect", { class: "body", x, y, width: BOX_W, height: BOX_H, rx: 4 }));
    if (station.outside) {
      g.append(s("text", { class: "name", x: x + 14, y: y + 40 }, station.title));
    } else {
      g.append(s("text", { class: "name", x: x + 14, y: y + 30 }, station.title));
      g.append(s("circle", { class: lampClass, cx: x + BOX_W - 20, cy: y + 20, r: 8, "--i": i }));
      const detail = beat ? (stale ? `silent for ${ago(o.now - beat.at)}` : beat.detail || ago(o.now - beat.at)) : "not started";
      const busy = name === "worker" && detail !== "idle" ? "busy" : detail;
      const short = busy.length > 22 ? `${busy.slice(0, 21)}…` : busy;
      g.append(s("text", { class: "detail", x: x + 14, y: y + 52 }, short));
      g.append(s("title", {}, `${station.title}: ${detail}`));
    }
    root.append(g);
  });
  return h("div", { class: "panel-wrap" }, root);
}

function railLegend() {
  return h(
    "p",
    { class: "legend-rail" },
    h("span", {}, "rail: idle"),
    h("span", { class: "l-set" }, "ready"),
    h("span", { class: "l-busy" }, "work in transit"),
    "Click a station for what it does and what it holds.",
  );
}

function openStation(name, o) {
  const station = stations(o)[name];
  const beat = o && o.heartbeats[name];
  fill(drawerBody, 
    h("h2", {}, station.title),
    h("p", {}, station.does),
    station.outside
      ? h("p", { class: "muted" }, "Outside Watchtower.")
      : h(
          "dl",
          {},
          h("dt", {}, "Holds"), h("dd", {}, station.holds),
          h("dt", {}, "Network"), h("dd", {}, station.network),
          h("dt", {}, "Never"), h("dd", {}, station.never),
          h("dt", {}, "Last heartbeat"),
          h("dd", {}, beat ? `${ago(o.now - beat.at)}${beat.detail ? `: ${beat.detail}` : ""}` : "none yet"),
        ),
  );
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  document.getElementById("drawer-close").focus();
}

function closeDrawer() {
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
}
document.getElementById("drawer-close").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => e.key === "Escape" && closeDrawer());

// -- charts ----------------------------------------------------------------------------

function lastDays(n) {
  const days = [];
  const today = new Date();
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate() - i));
    days.push(d.toISOString().slice(0, 10));
  }
  return days;
}

function showTip(event, text) {
  tip.textContent = text;
  tip.hidden = false;
  const x = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${event.clientY + 14}px`;
}
const hideTip = () => (tip.hidden = true);

// Columns per day; `stack` = [{key, cls, label, value(day)}], one entry for a single series.
function columns(days, stack, { unit, height = 170 } = {}) {
  const W = 460;
  const pad = { l: 40, r: 8, t: 10, b: 24 };
  const totals = days.map((d) => stack.reduce((sum, series) => sum + series.value(d), 0));
  const max = Math.max(1, ...totals);
  const step = (W - pad.l - pad.r) / days.length;
  const bar = Math.max(3, step - 3);
  const y = (v) => pad.t + (height - pad.t - pad.b) * (1 - v / max);
  const svgEl = s("svg", { class: "chart", viewBox: `0 0 ${W} ${height}`, role: "img" });
  svgEl.append(s("line", { class: "axis", x1: pad.l, x2: W - pad.r, y1: y(0), y2: y(0) }));
  svgEl.append(s("text", { class: "tick", x: pad.l - 6, y: y(max) + 4, "text-anchor": "end" }, fmtValue(max, unit)));
  svgEl.append(s("text", { class: "tick", x: pad.l - 6, y: y(0), "text-anchor": "end" }, "0"));
  days.forEach((day, i) => {
    const x = pad.l + i * step + 1.5;
    let base = 0;
    stack.forEach((series) => {
      const v = series.value(day);
      if (v <= 0) return;
      const top = y(base + v);
      // A 2px surface gap between stacked segments.
      const hgt = Math.max(1, y(base) - top - (base > 0 ? 2 : 0));
      svgEl.append(s("rect", { class: series.cls, x, y: top, width: bar, height: hgt, rx: base === 0 ? 2 : 0 }));
      base += v;
    });
    const lines = [dayName(Date.parse(`${day}T12:00:00Z`) / 1000)];
    stack.forEach((series) => {
      const v = series.value(day);
      if (stack.length === 1 || v > 0) lines.push(`${stack.length > 1 ? `${series.label}: ` : ""}${fmtValue(v, unit)}`);
    });
    svgEl.append(
      s("rect", {
        class: "hit", x: pad.l + i * step, y: pad.t, width: step, height: height - pad.t - pad.b,
        onpointermove: (e) => showTip(e, lines.join("\n")), onpointerleave: hideTip,
      }),
    );
    if (i % 7 === days.length % 7 || i === days.length - 1) {
      const last = i === days.length - 1;
      svgEl.append(s("text", { class: "tick", x: last ? x + bar : x + bar / 2, y: height - 6, "text-anchor": last ? "end" : "middle" }, isoDay(day)));
    }
  });
  const legend =
    stack.length > 1
      ? h("ul", { class: "legend" }, stack.map((series) => h("li", {}, h("i", { class: series.cls }), series.label)))
      : null;
  const table = h(
    "details",
    {},
    h("summary", { class: "small" }, "Show as a table"),
    h(
      "div",
      { class: "table-scroll" },
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, h("th", {}, "Day"), stack.map((series) => h("th", { class: "num" }, series.label)))),
        h(
          "tbody",
          {},
          days
            .filter((d, i) => totals[i] > 0)
            .map((d) => h("tr", {}, h("td", {}, isoDay(d)), stack.map((series) => h("td", { class: "num" }, fmtValue(series.value(d), unit))))),
        ),
      ),
    ),
  );
  return h("div", {}, legend, svgEl, table);
}

function fmtValue(v, unit) {
  if (unit === "min") return v >= 10 ? `${Math.round(v)} min` : `${v.toFixed(1)} min`;
  return String(Math.round(v));
}

function statsCharts(stats) {
  const days = lastDays(stats.days);
  const messages = Object.fromEntries(stats.messages.map((r) => [r.day, r.count]));
  const minutes = {};
  for (const row of stats.calls) minutes[`${row.day}|${row.kind}`] = row.seconds / 60;
  return [
    h("h3", {}, "Messages to Telegram per day"),
    columns(days, [{ key: "m", cls: "bar-one", label: "Messages", value: (d) => messages[d] || 0 }]),
    h("h3", {}, "Model time per day"),
    stats.calls.length
      ? columns(
          days,
          KINDS.map((k) => ({ key: k, cls: `k-${k}`, label: KIND_LABELS[k], value: (d) => minutes[`${d}|${k}`] || 0 })),
          { unit: "min" },
        )
      : h("p", { class: "muted small" }, "No model calls recorded yet. The worker records every call from now on."),
  ];
}

// -- messages -------------------------------------------------------------------------

function message(m) {
  const buttons = (m.buttons || []).map(([label]) => h("span", { class: "chip" }, label));
  return h(
    "li",
    { class: `msg${m.silent ? " silent" : ""}` },
    h("time", { datetime: new Date(m.created * 1000).toISOString(), title: dateTime(m.created) }, clock(m.created)),
    h(
      "div",
      {},
      h("div", { class: "topic" }, TOPICS[m.topic] || m.topic, m.sent ? "" : m.error ? " · not delivered" : " · waiting to be sent"),
      h("div", { class: "body" }, telegram(m.text)),
      buttons.length || m.url ? h("div", { class: "chips" }, buttons, m.url ? external(m.url, "Open on GitHub") : null) : null,
      m.ref && m.ref.startsWith("draft:") ? h("a", { class: "small", href: `#/drafts/${m.ref.slice(6)}` }, "See this draft in full") : null,
    ),
  );
}

// -- views ------------------------------------------------------------------------------

let timers = [];
let overview = null;
function every(ms, fn) {
  fn();
  timers.push(setInterval(fn, ms));
}

async function refreshPulse() {
  const pulse = document.getElementById("pulse");
  try {
    overview = await api("/api/overview");
    const w = overview.heartbeats.worker;
    pulse.textContent = w && w.detail && w.detail !== "idle" ? `Worker: ${w.detail}` : `Updated ${clock(overview.now)}`;
    pulse.classList.remove("stale");
  } catch {
    pulse.textContent = "No answer from the web service";
    pulse.classList.add("stale");
  }
  return overview;
}

let booted = false;

async function viewNow() {
  const top = h("div", {});
  const panelSlot = h("div", {});
  const feed = h("ul", { class: "feed" });
  const side = h("div", {});
  fill(view, 
    h("h1", {}, "Watchtower"),
    top,
    panelSlot,
    railLegend(),
    h(
      "div",
      { class: "cols" },
      h("section", {}, h("h2", {}, "Latest messages"), h("div", { class: "sheet" }, feed), h("p", {}, h("a", { href: "#/history" }, "All messages and jobs"))),
      h("section", {}, h("h2", {}, "Drafts and model time"), side),
    ),
  );
  every(5000, async () => {
    const o = await refreshPulse();
    if (!o) return;
    const draft = o.in_flight.find((d) => d.status === "drafting");
    fill(top, 
      h(
        "p",
        { class: "lead" },
        `Watching ${o.repos.join(" and ")}. `,
        o.drafting ? `Every stranger's new issue or discussion gets a reply draft for you to approve in Telegram.` : "Reply drafts are off.",
      ),
      h(
        "div",
        { class: "now-line" },
        draft
          ? h("span", {}, h("strong", {}, "Drafting now: "), h("a", { href: `#/drafts/${draft.id}` }, `${draft.repo}#${draft.number} ${draft.title}`))
          : h("span", {}, h("strong", {}, "The worker is idle.")),
        o.last_call ? h("span", {}, `Last model call ${ago(o.now - o.last_call.at)} (${KIND_LABELS[o.last_call.kind] || o.last_call.kind}, ${took(o.last_call.seconds)})`) : null,
        h("span", {}, `${o.outbox.sent} messages in 24 h`),
      ),
    );
    fill(panelSlot, panel(o, { boot: !booted }));
    booted = true;
  });
  every(20000, async () => {
    const [messages, stats] = await Promise.all([api("/api/activity?limit=12"), api("/api/stats?days=30")]);
    fill(feed, ...messages.slice(0, 12).map(message));
    if (!messages.length) fill(feed, h("li", { class: "muted" }, "Nothing sent yet. New activity on the watched repos shows up here and in Telegram."));
    const d = (overview && overview.drafts) || {};
    fill(side, 
      h(
        "div",
        { class: "tiles" },
        [["ready", "waiting for you"], ["posted", "posted"], ["rejected", "rejected"], ["failed", "failed"]].map(([k, label]) =>
          h("div", { class: "tile" }, h("b", {}, d[k] || 0), h("span", {}, `drafts ${label}`)),
        ),
      ),
      ...statsCharts(stats),
    );
  });
}

async function viewHistory(params) {
  let topic = params.get("topic") || "";
  let tab = params.get("tab") || "messages";
  const list = h("div", {});
  const more = h("button", { type: "button" }, "Show older");
  let oldest = null;
  let lastDay = "";
  const load = async () => {
    const rows = await api(`/api/activity?limit=100${oldest ? `&before=${oldest}` : ""}`);
    if (!rows.length) more.disabled = true;
    oldest = rows.length ? rows[rows.length - 1].id : oldest;
    let feed = list.lastElementChild && list.lastElementChild.tagName === "UL" ? list.lastElementChild : null;
    for (const row of rows.filter((r) => !topic || r.topic === topic)) {
      const day = dayName(row.created);
      if (day !== lastDay) {
        lastDay = day;
        list.append(h("h3", { class: "day" }, day));
        feed = h("ul", { class: "feed" });
        list.append(feed);
      }
      feed.append(message(row));
    }
    if (!list.children.length) list.append(h("p", { class: "muted" }, "Nothing here yet."));
  };
  more.addEventListener("click", load);
  const filter = (value, label) =>
    h("button", { type: "button", "aria-pressed": String(topic === value), onclick: () => (location.hash = `#/history?tab=messages&topic=${value}`) }, label);
  const tabs = h(
    "div",
    { class: "filters" },
    h("button", { type: "button", "aria-pressed": String(tab === "messages"), onclick: () => (location.hash = "#/history") }, "Messages"),
    h("button", { type: "button", "aria-pressed": String(tab === "jobs"), onclick: () => (location.hash = "#/history?tab=jobs") }, "Work queue"),
  );
  if (tab === "jobs") {
    const jobs = await api("/api/jobs");
    fill(view, 
      h("h1", {}, "History"),
      tabs,
      h("p", { class: "lead" }, "What the watcher handed the worker: a summary for every new item, a brief for every release, replays you started."),
      h(
        "div",
        { class: "table-scroll" },
        h(
          "table",
          {},
          h("thead", {}, h("tr", {}, ["Queued", "Kind", "About", "Status", "Took"].map((t) => h("th", {}, t)))),
          h(
            "tbody",
            {},
            jobs.map((j) =>
              h("tr", {}, h("td", {}, dateTime(j.created)), h("td", {}, j.kind), h("td", {}, j.key), h("td", {}, h("span", { class: `status s-${j.status}` }, j.status)), h("td", {}, j.status === "done" || j.status === "failed" ? took(j.updated - j.created) : "")),
            ),
          ),
        ),
      ),
    );
    return;
  }
  fill(view, 
    h("h1", {}, "History"),
    tabs,
    h("p", { class: "lead" }, "Everything Watchtower told you in Telegram, newest first: its own record of what it did."),
    h("div", { class: "filters" }, filter("", "All topics"), Object.entries(TOPICS).map(([k, v]) => filter(k, v))),
    list,
    more,
  );
  await load();
}

async function viewDrafts() {
  const rows = await api("/api/drafts");
  fill(view, 
    h("h1", {}, "Drafts"),
    h("p", { class: "lead" }, "Every reply Watchtower drafted, from the stranger's message to what you did with it."),
    rows.length
      ? h(
          "div",
          { class: "table-scroll" },
          h(
            "table",
            {},
            h("thead", {}, h("tr", {}, ["", "Thread", "Assessment", "Status", "Versions", "Started"].map((t) => h("th", {}, t)))),
            h(
              "tbody",
              {},
              rows.map((d) =>
                h(
                  "tr",
                  { class: "link", tabindex: 0, onclick: () => (location.hash = `#/drafts/${d.id}`), onkeydown: (e) => e.key === "Enter" && (location.hash = `#/drafts/${d.id}`) },
                  h("td", { class: "muted" }, d.id),
                  h("td", {}, h("a", { href: `#/drafts/${d.id}` }, `${d.repo.split("/")[1]}#${d.number}`), " ", d.title),
                  h("td", {}, d.category ? `${d.category.replace("_", " ")} (${d.confidence})` : ""),
                  h("td", {}, h("span", { class: `status s-${d.status}` }, STATUS_WORDS[d.status] || d.status)),
                  h("td", { class: "num" }, d.versions),
                  h("td", {}, dateTime(d.created)),
                ),
              ),
            ),
          ),
        )
      : h("p", { class: "notice" }, "No drafts yet. A stranger's new issue or discussion on a watched repo starts one."),
  );
}

function journey(d) {
  const stage = Object.fromEntries(d.stages.map((st) => [st.name, st]));
  const posted = d.status === "posted";
  const endName = { posted: "Posted", rejected: "Rejected", failed: "Failed", superseded: "Replaced" }[d.status] || "Your call";
  const stops = [
    { name: "Asked", done: true, time: clock(d.created) },
    { name: "Fetched", done: d.status !== "prep" },
    { name: "Files", done: !!stage.files, time: stage.files && took(stage.files.data.seconds) },
    { name: "Investigated", done: !!stage.investigation, time: stage.investigation && took(stage.investigation.data.seconds) },
    { name: "Assessed", done: !!stage.assessment, time: stage.assessment && took(stage.assessment.data.seconds) },
    { name: "Written", done: d.versions.length > 0, time: d.versions.length ? `${count(d.versions.length, "version")}` : "" },
    { name: endName, done: ["posted", "rejected", "superseded"].includes(d.status), time: posted ? "" : "" },
  ];
  const active = ["prep", "queued", "drafting", "ready", "posting"].includes(d.status);
  const here = active ? stops.findIndex((st) => !st.done) : -1;
  const W = 900;
  const gap = (W - 80) / (stops.length - 1);
  const svgEl = s("svg", { viewBox: `0 0 ${W} 96`, role: "img", "aria-label": "This draft's way so far" });
  stops.forEach((stop, i) => {
    const x = 40 + i * gap;
    if (i < stops.length - 1) {
      const next = stops[i + 1];
      const cls = next.done ? "rail set" : i + 1 === here ? "rail busy" : "rail";
      svgEl.append(s("path", { class: cls, d: `M${x + 12} 34 H${x + gap - 12}` }));
    }
    svgEl.append(s("circle", { class: `stop${i === here ? " here" : stop.done ? " done" : ""}`, cx: x, cy: 34, r: 11 }));
    svgEl.append(s("text", { class: "name", x, y: 70, "text-anchor": "middle" }, stop.name));
    if (stop.time) svgEl.append(s("text", { class: "time", x, y: 88, "text-anchor": "middle" }, stop.time));
  });
  return h("div", { class: "journey" }, svgEl);
}

function passBar(calls) {
  const seconds = {};
  for (const c of calls) seconds[c.kind] = (seconds[c.kind] || 0) + c.seconds;
  const total = Object.values(seconds).reduce((a, b) => a + b, 0);
  if (!total) return h("p", { class: "muted small" }, "No model calls recorded for this draft (drafts from before the trace existed have none).");
  const kinds = KINDS.filter((k) => seconds[k]);
  return h(
    "div",
    {},
    h(
      "div",
      { class: "passbar", role: "img", "aria-label": kinds.map((k) => `${KIND_LABELS[k]} ${took(seconds[k])}`).join(", ") },
      kinds.map((k) => {
        const bar = h("span", { class: `k-${k}`, onpointermove: (e) => showTip(e, `${KIND_LABELS[k]}: ${took(seconds[k])}`), onpointerleave: hideTip });
        bar.style.flexGrow = String(seconds[k]);
        return bar;
      }),
    ),
    h("ul", { class: "legend" }, kinds.map((k) => h("li", {}, h("i", { class: `k-${k}` }), `${KIND_LABELS[k]} ${took(seconds[k])}`))),
    h("p", { class: "small muted" }, `${count(calls.length, "model call")}, ${took(total)} of model time in all.`),
  );
}

// A pull request's review: what it does, what code checked, and each finding.
function reviewSheet(v) {
  return h(
    "div",
    { class: "sheet" },
    h("p", {}, h("strong", {}, v.label), ` · confidence ${v.confidence} · ${count(v.attempts, "attempt")}`),
    h("p", {}, v.summary),
    v.facts.length ? [h("h3", {}, "Checked by code"), h("ul", {}, v.facts.map((f) => h("li", {}, f)))] : null,
    v.decision ? [h("h3", {}, "Yours to decide"), h("p", {}, v.decision)] : null,
    v.findings.length
      ? [
          h("h3", {}, "Findings"),
          h(
            "ul",
            { class: "evidence" },
            v.findings.map((f) =>
              h(
                "li",
                {},
                `${f.mark} `,
                f.verified ? h("span", { class: "ok" }, "✓ checked ") : h("span", { class: "warn" }, "⚠ quote not found "),
                h("span", { class: "muted" }, f.where, ": "),
                h("q", {}, f.quote),
                h("div", {}, f.point),
                f.fix ? h("div", { class: "muted" }, `Fix: ${f.fix}`) : null,
              ),
            ),
          ),
        ]
      : null,
    v.missing.length ? [h("h3", {}, "Still missing"), h("ul", {}, v.missing.map((m) => h("li", {}, m)))] : null,
    v.judged_at ? h("p", { class: "small muted" }, `Judged against ${v.judged_at}.`) : null,
    v.looked_at.length ? [h("h3", {}, `Looked up (${v.looked_at.length})`), h("ol", {}, v.looked_at.map((step) => h("li", {}, step)))] : null,
  );
}

function assessment(v) {
  if (!v) return h("p", { class: "muted" }, "Not assessed yet.");
  if (v.review) return reviewSheet(v);
  return h(
    "div",
    { class: "sheet" },
    h("p", {}, h("strong", {}, v.label), ` · confidence ${v.confidence} · ${count(v.attempts, "attempt")}`),
    v.asks.length ? [h("h3", {}, "What the message asks"), h("ul", {}, v.asks.map((a) => h("li", {}, h("q", {}, a.quote), a.verified ? "" : h("span", { class: "warn" }, " not found in the message"))))] : null,
    v.decision ? [h("h3", {}, "Yours to decide"), h("p", {}, v.decision)] : null,
    v.evidence.length
      ? [
          h("h3", {}, "Evidence"),
          h(
            "ul",
            { class: "evidence" },
            v.evidence.map((e) =>
              h(
                "li",
                {},
                e.verified ? h("span", { class: "ok" }, "✓ checked ") : h("span", { class: "warn" }, "⚠ not found in its source "),
                h("span", { class: "muted" }, e.source, ": "),
                h("q", {}, e.quote),
                h("div", {}, e.point),
              ),
            ),
          ),
        ]
      : null,
    v.missing.length ? [h("h3", {}, "Still missing"), h("ul", {}, v.missing.map((m) => h("li", {}, m)))] : null,
    v.code ? [h("h3", {}, "Where"), h("p", {}, v.code)] : null,
    v.fix ? [h("h3", {}, "Fix"), h("p", {}, v.fix)] : null,
    v.unknown_paths.length ? h("p", { class: "warn" }, `No such file in the code: ${v.unknown_paths.join(", ")}`) : null,
    v.judged_at ? h("p", { class: "small muted" }, `Judged against ${v.judged_at}.`) : null,
    v.looked_at.length ? [h("h3", {}, `Looked up (${v.looked_at.length})`), h("ol", {}, v.looked_at.map((step) => h("li", {}, step)))] : null,
  );
}

// The page is served over plain http on the LAN, where the clipboard API may be
// missing: then the text is selected for Ctrl+C.
async function copyText(area) {
  try {
    await navigator.clipboard.writeText(area.value);
    return "Copied.";
  } catch {
    area.focus();
    area.select();
    return document.execCommand("copy") ? "Copied." : "Selected: press Ctrl+C.";
  }
}

function handoffBlock(x) {
  const area = h("textarea", { readonly: "", rows: "16", spellcheck: "false" });
  area.value = x.prompt;
  const said = h("span", { class: "small muted", role: "status" });
  const copy = h("button", { type: "button", class: "primary" }, "Copy prompt for Claude");
  copy.addEventListener("click", async () => (said.textContent = await copyText(area)));
  return h(
    "div",
    { class: "sheet" },
    x.confirmed
      ? h("p", {}, h("span", { class: "ok" }, "✓ Confirmed"), " by Watchtower's checks. Claude still double-checks it.")
      : [h("p", {}, h("span", { class: "warn" }, "Suspected"), ", not confirmed. The checks it failed:"), h("ul", {}, x.doubts.map((t) => h("li", {}, t)))],
    x.label ? h("p", { class: "small muted" }, `Posting the reply from Telegram also labels the issue "${x.label}" (unless you choose "Post only").`) : null,
    area,
    h("div", { class: "chips" }, copy, said),
  );
}

function versionsBlock(d, rerender) {
  const latest = d.versions[d.versions.length - 1];
  const notice = h("div", {});
  const say = (text, error = false) => fill(notice, h("p", { class: `notice${error ? " error" : ""}` }, text));
  const run = async (button, path, body) => {
    button.disabled = true;
    try {
      say(await act(path, body));
      setTimeout(rerender, 4000);
    } catch (err) {
      say(err.message, true);
      button.disabled = false;
    }
  };
  const blocks = d.versions.map((v, i) =>
    h(
      "div",
      {},
      h("h3", {}, `Version ${i + 1}`, h("span", { class: "muted small" }, ` · ${{ model: "by the model", revised: "revised as you asked", user: "your edit" }[v.author] || v.author}, ${dateTime(v.created)}`)),
      h("pre", { class: "text" }, v.text),
    ),
  );
  if (d.status !== "ready" || !latest) return h("div", {}, blocks, notice);
  const editor = h("div", { hidden: true });
  const area = h("textarea", { "aria-label": "Your version of the reply", maxlength: d.max_text });
  area.value = latest.text;
  const save = h("button", { type: "button", class: "primary" }, "Send my version");
  save.addEventListener("click", () => run(save, `/api/drafts/${d.id}/edit`, { text: area.value }));
  editor.append(area, h("div", { class: "chips" }, save, h("button", { type: "button", onclick: () => (editor.hidden = true) }, "Cancel")));
  const instruction = h("textarea", { class: "short", "aria-label": "What to change", rows: "3", maxlength: "2000", placeholder: "What to change, e.g. \"Go with option B, and ask for the log.\"" });
  const revise = h("button", { type: "button", class: "primary" }, "Revise it");
  revise.addEventListener("click", () => run(revise, `/api/drafts/${d.id}/revise`, { text: instruction.value }));
  const offer = h("button", { type: "button", class: "primary" }, "Post via Telegram");
  offer.addEventListener("click", () => run(offer, `/api/versions/${latest.id}/offer`));
  // A decision left open: choose first (each choice is an instruction to the model), then post.
  const choices = (d.choices || []).map((c) => {
    const button = h("button", { type: "button", class: "primary" }, c.label);
    button.addEventListener("click", () => run(button, `/api/drafts/${d.id}/revise`, { text: c.instruction }));
    return button;
  });
  const reject = h("button", { type: "button" }, "Reject");
  reject.addEventListener("click", () => confirm("Reject this draft?") && run(reject, `/api/versions/${latest.id}/reject`));
  return h(
    "div",
    {},
    blocks,
    d.open_decision
      ? [
          h("h3", {}, "A decision is left open"),
          h("p", { class: "small muted" }, "This text can't be posted until it is settled. Choose an option (the model writes it in), or say what to do below."),
          choices.length ? h("div", { class: "chips" }, choices) : null,
        ]
      : null,
    h("h3", {}, "Tell the model what to change"),
    instruction,
    h("div", { class: "chips" }, revise),
    h(
      "div",
      { class: "chips" },
      d.open_decision ? null : offer,
      h("button", { type: "button", onclick: () => ((editor.hidden = false), area.focus()) }, "Edit the text myself"),
      reject,
    ),
    h("p", { class: "small muted" }, "Post via Telegram sends this version to your group again; it's posted only when you tap ✅ Post there. A revision or an edit comes back as a new version, here and in Telegram."),
    editor,
    notice,
  );
}

function callRow(c, withSubject = false) {
  const detail = h("tr", { hidden: true }, h("td", { colspan: withSubject ? 8 : 7 }));
  const row = h(
    "tr",
    {
      class: "link",
      tabindex: 0,
      "aria-expanded": "false",
      onclick: () => toggleCall(c.id, row, detail),
      onkeydown: (e) => e.key === "Enter" && toggleCall(c.id, row, detail),
    },
    h("td", {}, dateTime(c.at)),
    withSubject ? h("td", {}, subjectLink(c.subject)) : null,
    h("td", {}, c.step || ""),
    h("td", {}, h("i", { class: `dot k-${c.kind}` }), " ", KIND_LABELS[c.kind] || c.kind),
    h("td", { class: "num" }, took(c.seconds)),
    h("td", { class: "num" }, chars(c.prompt_chars)),
    h("td", { class: "num" }, chars(c.answer_chars)),
    h("td", {}, c.error ? h("span", { class: "warn" }, c.error) : c.tool_calls ? count(c.tool_calls, "tool call") : ""),
  );
  return [row, detail];
}

function subjectLink(subject) {
  const m = /^draft:(\d+)$/.exec(subject || "");
  if (m) return h("a", { href: `#/drafts/${m[1]}` }, `Draft ${m[1]}`);
  return (subject || "").replace(/^event:/, "Summary of ");
}

async function toggleCall(id, row, detail) {
  const open = detail.hidden;
  detail.hidden = !open;
  row.setAttribute("aria-expanded", String(open));
  if (!open || detail.dataset.loaded) return;
  detail.dataset.loaded = "1";
  const c = await api(`/api/calls/${id}`);
  const part = (m) =>
    h(
      "div",
      {},
      h("div", { class: "role" }, m.role === "tool" ? `tool result${m.tool_name ? `: ${m.tool_name}` : ""}` : m.role),
      m.thinking ? h("details", {}, h("summary", { class: "small" }, "Its thinking"), h("pre", { class: "text" }, m.thinking)) : null,
      m.content ? h("pre", { class: "text" }, m.content) : null,
      m.tool_calls ? h("pre", { class: "text" }, JSON.stringify(m.tool_calls, null, 2)) : null,
    );
  fill(detail.firstChild, 
    h("p", { class: "small muted" }, `${c.model}, context ${c.num_ctx} tokens. `, c.request.tools.length ? `Tools offered: ${c.request.tools.join(", ")}.` : "No tools offered."),
    h("details", { open: true }, h("summary", {}, `What the model was given (${count(c.request.messages.length, "message")}, ${chars(c.prompt_chars)} characters)`), c.request.messages.map(part)),
    h("h3", {}, "What it answered"),
    part({ role: "assistant", ...c.answer }),
  );
}

function callsTable(calls, withSubject) {
  return h(
    "div",
    { class: "table-scroll" },
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, ["When", withSubject ? "For" : null, "Step", "Pass", "Took", "Prompt", "Answer", ""].filter((t) => t !== null).map((t) => h("th", { class: ["Took", "Prompt", "Answer"].includes(t) ? "num" : null }, t)))),
      h("tbody", {}, calls.map((c) => callRow(c, withSubject))),
    ),
  );
}

async function viewDraft(id) {
  const d = await api(`/api/drafts/${id}`);
  const rerender = () => route();
  const thread = d.thread;
  fill(view, 
    h("p", {}, h("a", { href: "#/drafts" }, "All drafts")),
    h("h1", {}, d.title),
    h(
      "p",
      { class: "lead" },
      `${d.repo}#${d.number} · `,
      h("span", { class: `status s-${d.status}` }, STATUS_WORDS[d.status] || d.status),
      " · ",
      external(d.url, "the message it answers"),
      d.posted_url ? [" · ", external(d.posted_url, "the posted reply")] : null,
    ),
    journey(d),
    h("h2", {}, "Where the time went"),
    passBar(d.calls),
    h("h2", {}, "The assessment"),
    assessment(d.verdict),
    d.handoff ? [h("h2", {}, "For Claude"), h("p", { class: "muted small" }, "A bug is Watchtower's to report, not to fix: paste this into Claude Code in the project."), handoffBlock(d.handoff)] : null,
    d.stages.find((st) => st.name === "investigation")
      ? [
          h("h2", {}, "The investigation"),
          h("p", { class: "muted small" }, "The model's own lookups before it judged: its notes, each tool call (>>>) and what came back."),
          h("pre", { class: "text" }, d.stages.find((st) => st.name === "investigation").data.transcript || "(no lookups)"),
        ]
      : null,
    h("h2", {}, "The reply"),
    d.note ? h("p", { class: "notice" }, d.note) : null,
    d.attachments ? h("p", { class: "small muted" }, `Attached: ${d.attachments}`) : null,
    versionsBlock(d, rerender),
    d.reason ? [h("h3", {}, "Why you rejected it"), h("p", {}, d.reason)] : null,
    h("h2", {}, "Decisions"),
    d.decisions.length
      ? h("div", { class: "table-scroll" }, h(
          "table",
          {},
          h("thead", {}, h("tr", {}, ["When", "What", "From", "Applied"].map((t) => h("th", {}, t)))),
          h("tbody", {}, d.decisions.map((x) => h("tr", {}, h("td", {}, dateTime(x.at)), h("td", {}, x.action === "reply" ? "edit or reason" : x.action === "offer" ? "send to Telegram" : x.action), h("td", {}, x.origin === "web" ? "web UI" : "Telegram"), h("td", {}, x.applied ? clock(x.applied) : "waiting")))),
        ))
      : h("p", { class: "muted" }, "None yet."),
    h("h2", {}, `Model calls (${d.calls.length})`),
    d.calls.length ? callsTable(d.calls, false) : h("p", { class: "muted" }, "None recorded."),
    thread
      ? [
          h("h2", {}, "The thread"),
          h("p", { class: "muted small" }, "As Watchtower last fetched it. Written by other people: shown as text."),
          [thread, ...thread.comments].map((entry) =>
            h(
              "div",
              {},
              h("div", { class: "role" }, `${entry.author} (${entry.maintainer || ["OWNER", "MEMBER", "COLLABORATOR"].includes(entry.association) ? "maintainer" : "user"})`),
              h("pre", { class: "text" }, entry.body || ""),
            ),
          ),
        ]
      : null,
  );
  if (["prep", "queued", "drafting", "posting"].includes(d.status)) timers.push(setTimeout(rerender, 10000));
}

async function viewCalls(params) {
  const rows = await api("/api/calls");
  const table = callsTable(rows, true);
  const more = h("button", { type: "button" }, "Show older");
  let oldest = rows.length ? rows[rows.length - 1].id : null;
  more.addEventListener("click", async () => {
    const older = await api(`/api/calls?before=${oldest}`);
    if (!older.length) return (more.disabled = true);
    oldest = older[older.length - 1].id;
    table.querySelector("tbody").append(...older.flatMap((c) => callRow(c, true)));
  });
  fill(view, 
    h("h1", {}, "Model calls"),
    h("p", { class: "lead" }, "Every call to the local models: what it was for, how long it took, and (click a row) the full prompt and the full answer."),
    rows.length ? [table, more] : h("p", { class: "notice" }, "No calls recorded yet. The worker records every call from now on; they're kept for 90 days."),
  );
}

async function viewKnowledge() {
  const repos = await api("/api/knowledge");
  fill(view, 
    h("h1", {}, "Knowledge"),
    h("p", { class: "lead" }, "What the model learns each repo from. Your notes rank first; the brief only counts once you approved it."),
    repos.map((r) =>
      h(
        "section",
        {},
        h("h2", {}, r.repo),
        h(
          "div",
          { class: "tiles" },
          [["issue", "issues"], ["pr", "pull requests"], ["discussion", "discussions"], ["comment", "comments"]].map(([k, label]) =>
            h("div", { class: "tile" }, h("b", {}, r.counts[k] || 0), h("span", {}, label)),
          ),
        ),
        h("p", { class: "small muted" }, r.snapshot ? `Code snapshot at ${r.snapshot.slice(0, 7)}.` : "No code snapshot yet."),
        h("h3", {}, "Your notes"),
        r.notes.length
          ? h("ul", {}, r.notes.map((n) => h("li", {}, n)))
          : h("p", { class: "muted" }, "None yet. Add Markdown files under docs/knowledge/ in the repo, one topic per file: facts the code doesn't state, like how the service behaves and its limits."),
        h("details", {}, h("summary", {}, `The repo's docs (${r.docs.length})`), h("ul", {}, r.docs.map((n) => h("li", {}, n)))),
        h("h3", {}, "Briefs"),
        r.briefs.length
          ? r.briefs.map((b) =>
              h("details", {}, h("summary", {}, `${b.label}: ${b.status}, ${dateTime(b.created)}`), h("pre", { class: "text" }, b.text || "(failed)")),
            )
          : h("p", { class: "muted" }, "No brief yet. One is written for every new release."),
        h(
          "details",
          {},
          h("summary", {}, `Releases (${r.releases.length})`),
          h("table", {}, h("tbody", {}, r.releases.map((x) => h("tr", {}, h("td", {}, x.tag), h("td", {}, x.prerelease ? "beta" : "stable"), h("td", {}, x.published.slice(0, 10)))))),
        ),
      ),
    ),
  );
}

async function viewHow() {
  const o = overview || (await refreshPulse());
  const info = stations(o);
  fill(view, 
    h("h1", {}, "How Watchtower works"),
    h(
      "p",
      { class: "lead" },
      "Watchtower watches your GitHub repos, tells you in Telegram what's new, and drafts replies with a model that runs on this box. Nothing reaches GitHub until you approve it. Each part holds only the one key it needs, and the part that reads strangers' text holds none.",
    ),
    o ? panel(o) : null,
    railLegend(),
    h("h2", {}, "The life of an issue"),
    h(
      "ol",
      { class: "steps" },
      [
        ["Someone opens an issue", `The watcher sees it on its next poll (every ${o ? o.poll_seconds : "…"} s) and hands it to the worker.`],
        ["The worker summarises it", "The small model writes one line: what the author reports or wants. The gateway posts it into the matching Telegram topic."],
        ["A draft is asked for", "Every stranger's new issue or discussion gets one, and a comment does when it needs a reply. The watcher fetches the fresh thread, the attached files and the code at the author's version: the worker has no internet."],
        ["The model reads and investigates", "Big attached files are read in parts. Then the model looks things up with read-only tools: the code, the files, earlier threads, your notes and the docs."],
        ["It assesses the thread", "Category, evidence as exact quotes, what's missing, where the fault is, what the message asks, and a decision it leaves to you. Code checks every quote against its source. The assessment reaches Telegram as soon as it's done."],
        ["It writes the reply", "From the assessment, in the author's language, without links outside the repo or @mentions."],
        ["You decide in Telegram", "✅ Post, 🗑 Reject, or reply with your own version. The web UI can edit and reject too, and send a version to Telegram, but only your tap there posts."],
        ["The poster posts it", "Exactly the approved text, as the GitHub App. If it's unclear whether GitHub got it, you're told to check instead of it being retried."],
      ].map(([title, text]) => h("li", {}, h("div", {}, h("h3", {}, title), h("p", {}, text)))),
    ),
    h("h2", {}, "Who holds what"),
    h(
      "div",
      { class: "table-scroll" },
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, ["Service", "Holds", "Network", "Never"].map((t) => h("th", {}, t)))),
        h(
          "tbody",
          {},
          ["watcher", "worker", "gateway", "poster", "web"].map((name) =>
            h("tr", {}, h("td", {}, h("strong", {}, info[name].title)), h("td", {}, info[name].holds), h("td", {}, info[name].network), h("td", {}, info[name].never)),
          ),
        ),
      ),
    ),
    h("h2", {}, "Rules it keeps"),
    h(
      "ul",
      {},
      h("li", {}, "Strangers' text and the model's answers are untrusted: they never reach a shell, a GitHub write, or Telegram unescaped."),
      h("li", {}, "The models run on this box. Cloud models are refused, so prompts never leave it."),
      h("li", {}, "The model answers what was asked, and leaves your decisions to you: a draft that still holds a [YOUR DECISION: …] line can't be posted."),
      h("li", {}, "Quality is measured by replaying closed issues, not by feel."),
    ),
  );
}

// -- routing ------------------------------------------------------------------------------

async function route() {
  timers.forEach((t) => (clearInterval(t), clearTimeout(t)));
  timers = [];
  hideTip();
  closeDrawer();
  const [path, query = ""] = location.hash.replace(/^#/, "").split("?");
  const params = new URLSearchParams(query);
  const parts = path.split("/").filter(Boolean);
  const name = parts[0] || "now";
  document.querySelectorAll(".top nav a").forEach((a) => {
    if (a.dataset.route === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  try {
    if (name === "now") await viewNow();
    else if (name === "history") await viewHistory(params);
    else if (name === "drafts" && parts[1]) await viewDraft(Number(parts[1]));
    else if (name === "drafts") await viewDrafts();
    else if (name === "calls") await viewCalls(params);
    else if (name === "knowledge") await viewKnowledge();
    else if (name === "how") await viewHow();
    else fill(view, h("h1", {}, "Not here"), h("p", {}, h("a", { href: "#/" }, "Back to now")));
  } catch (err) {
    fill(view, h("h1", {}, "Couldn't load this"), h("p", { class: "notice error" }, `${err.message}. Is the web service running?`));
  }
  if (name !== "now") refreshPulse();
}

window.addEventListener("hashchange", () => (route(), view.focus({ preventScroll: true })));
route();
setInterval(() => location.hash.replace(/^#\/?/, "") !== "" && refreshPulse(), 15000);
