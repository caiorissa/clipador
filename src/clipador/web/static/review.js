/* Clipador — /revisao: keyboard shortcuts, live (approximate) 9:16 preview drawn on a canvas,
   HTML captions synced to currentTime, waveform timeline with draggable edges, faithful preview.
   No build step, no external libraries. */
(function () {
  "use strict";
  const root = document.getElementById("review");
  const help = document.getElementById("help");
  const undoForm = document.getElementById("undo-form");

  function toggleHelp() { if (help) help.hidden = !help.hidden; }
  document.querySelectorAll("[data-help-toggle]").forEach((b) => b.addEventListener("click", toggleHelp));

  function typing(el) {
    if (!el) return false;
    const tag = el.tagName;
    return el.isContentEditable || tag === "TEXTAREA" || tag === "SELECT" ||
      (tag === "INPUT" && !["checkbox", "radio", "button", "submit"].includes(el.type));
  }

  if (!root) {  // empty queue: only undo / help work
    document.addEventListener("keydown", (ev) => {
      if (typing(ev.target) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (ev.key === "?") toggleHelp();
      else if (ev.key.toLowerCase() === "z" && undoForm) undoForm.submit();
    });
    return;
  }

  const cfg = JSON.parse(document.getElementById("review-data").textContent);
  const video = document.getElementById("src-video");
  const canvas = document.getElementById("stage-canvas");
  const ctx = canvas.getContext("2d");
  const stage = document.getElementById("stage");
  const capEl = document.getElementById("stage-caption");
  const openEl = document.getElementById("stage-opening");
  const creditEl = document.getElementById("stage-credit");
  const fStart = document.getElementById("f-start"), fEnd = document.getElementById("f-end");
  const snapBox = document.getElementById("snap");
  const rangeStatus = document.getElementById("range-status");
  const rangeWarn = document.getElementById("range-warnings");
  const resetBtn = document.getElementById("reset-range");
  const form = document.getElementById("decision-form");
  const reasons = document.getElementById("reasons");
  const st = { start: cfg.start, end: cfg.end, layout: cfg.layout, words: [], events: [], ev: -1 };
  const style = cfg.style;

  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const fmt = (t) => { t = Math.max(0, t); const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60;
    return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + s.toFixed(2).padStart(5, "0"); };

  // --- geometry (mirrors clipador.render.filters) ---------------------------------------------
  function cropAspect(sw, sh, ow, oh, zoom, fx) {
    const z = Math.max(1, zoom || 1);
    let ch = sh / z, cw = ch * ow / oh;
    if (cw > sw) { cw = sw / z; ch = cw * oh / ow; }
    return [(sw - cw) * clamp(fx, 0, 1), (sh - ch) / 2, cw, ch];
  }
  function cover(sw, sh, ow, oh) {  // scale to cover, then center crop (in source pixels)
    const s = Math.max(ow / sw, oh / sh), cw = ow / s, ch = oh / s;
    return [(sw - cw) / 2, (sh - ch) / 2, cw, ch];
  }
  function draw() {
    const W = canvas.width, H = canvas.height, sw = video.videoWidth, sh = video.videoHeight;
    ctx.filter = "none";
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
    if (!sw || !sh) return;
    if (st.layout === "jogo_cheio") {
      const [x, y, w, h] = cropAspect(sw, sh, W, H, cfg.zoom, cfg.foco_x);
      ctx.drawImage(video, x, y, w, h, 0, 0, W, H);
    } else if (st.layout === "facecam_embaixo" && cfg.rect) {
      const gh = Math.round(H * cfg.game_h), ch = H - gh;
      const [x, y, w, h] = cropAspect(sw, sh, W, gh, cfg.zoom, cfg.foco_x);
      ctx.drawImage(video, x, y, w, h, 0, 0, W, gh);
      const [rx, ry, rw, rh] = cfg.rect.map((v, i) => v * (i % 2 ? sh : sw));
      const [cx, cy, cw, chh] = cover(rw, rh, W, ch);
      ctx.drawImage(video, rx + cx, ry + cy, cw, chh, 0, gh, W, ch);
    } else {  // blur
      const [bx, by, bw, bh] = cover(sw, sh, W, H);
      ctx.filter = "blur(14px) brightness(0.9) saturate(1.15)";
      ctx.drawImage(video, bx, by, bw, bh, -20, -20, W + 40, H + 40);
      ctx.filter = "none";
      let fw = W, fh = W * sh / sw;
      if (fh > H) { fh = H; fw = H * sw / sh; }
      ctx.drawImage(video, 0, 0, sw, sh, (W - fw) / 2, (H - fh) / 2, fw, fh);
    }
  }

  // --- captions (mirrors clipador.render.subtitles.chunk_words / highlight_intervals) -------
  function clipWords() {  // words of the current range, relative to the clip start
    const dur = st.end - st.start, out = [];
    st.words.forEach((w) => {
      if (w.e <= st.start || w.s >= st.end) return;
      const s = Math.max(0, w.s - st.start), e = Math.min(dur, w.e - st.start);
      if (e - s > 0.01 && w.t) out.push({ s, e, t: w.t });
    });
    return out;
  }
  function chunkWords(words) {
    const chunks = []; let cur = [];
    const maxW = style.max_words, maxC = style.max_line_chars;
    words.forEach((w) => {
      if (cur.length) {
        const gap = w.s - cur[cur.length - 1].e;
        const chars = cur.reduce((n, x) => n + x.t.length, 0) + cur.length + w.t.length;
        const endsSentence = /[.!?…]$/.test(cur[cur.length - 1].t);
        if (cur.length >= maxW || chars > 2 * maxC || gap > 0.6 || endsSentence) { chunks.push(cur); cur = []; }
      }
      cur.push(w);
    });
    if (cur.length) chunks.push(cur);
    return chunks;
  }
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function buildEvents() {
    const dur = st.end - st.start, minEv = 0.25, chunks = chunkWords(clipWords()), events = [];
    chunks.forEach((chunk, ci) => {
      const next = ci + 1 < chunks.length ? chunks[ci + 1][0].s : dur;
      let cend = Math.min(next, chunk[chunk.length - 1].e + 0.35, dur);
      if (cend - chunk[0].s < minEv) cend = Math.min(next, dur, chunk[0].s + minEv);
      const texts = chunk.map((w) => (style.uppercase ? w.t.toUpperCase() : w.t));
      if (style.mode === "classico") { events.push({ s: chunk[0].s, e: cend, html: esc(texts.join(" ")) }); return; }
      const merged = [];
      chunk.forEach((w, i) => {
        const e = i + 1 < chunk.length ? chunk[i + 1].s : cend;
        if (merged.length && merged[merged.length - 1].e - merged[merged.length - 1].s < minEv) {
          merged[merged.length - 1].e = e; merged[merged.length - 1].pos.push(i);
        } else merged.push({ s: w.s, e, pos: [i] });
      });
      if (merged.length > 1 && merged[merged.length - 1].e - merged[merged.length - 1].s < minEv) {
        const last = merged.pop(); merged[merged.length - 1].e = last.e; merged[merged.length - 1].pos.push(...last.pos);
      }
      merged.forEach((m) => events.push({ s: m.s, e: m.e,
        html: texts.map((t, i) => (m.pos.includes(i) ? '<span class="hl">' + esc(t) + "</span>" : esc(t))).join(" ") }));
    });
    st.events = events; st.ev = -2;
  }
  function hexA(hex, alpha) {
    const n = parseInt(hex.slice(1), 16);
    return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + alpha + ")";
  }
  function applyStyle() {
    const k = stage.clientHeight / 1920;
    capEl.style.fontSize = Math.round(style.font_size * k) + "px";
    capEl.style.color = style.color;
    capEl.style.setProperty("--hl", style.highlight);
    capEl.style.setProperty("--pop", String((style.pop || 100) / 100));
    capEl.classList.toggle("mode-pop", style.mode === "pop");
    capEl.style.bottom = (100 * (cfg.safe.bottom + cfg.safe.lift)) + "%";
    if (style.box) {
      capEl.style.webkitTextStroke = "0";
      capEl.style.background = hexA(style.box_color, 1 - style.box_alpha / 255);
      capEl.style.padding = "0 .25em";
    } else {
      capEl.style.webkitTextStroke = Math.max(1, style.outline * k * 2) + "px " + style.outline_color;
      capEl.style.background = "none";
    }
    openEl.style.top = (100 * cfg.safe.top) + 1 + "%";
    openEl.style.fontSize = Math.round(70 * k) + "px";
    creditEl.style.top = "calc(" + (100 * cfg.safe.top) + "% + " + Math.round(190 * k) + "px)";
    creditEl.style.fontSize = Math.round(46 * k) + "px";
    creditEl.textContent = cfg.credit || "";
  }
  function updateOverlays() {
    const t = video.currentTime - st.start;
    let idx = -1;
    for (let i = 0; i < st.events.length; i++) { if (t >= st.events[i].s && t < st.events[i].e) { idx = i; break; } }
    if (idx !== st.ev) {
      st.ev = idx;
      capEl.innerHTML = idx >= 0 ? st.events[idx].html : "";
      capEl.hidden = idx < 0;
    }
    const showOpen = cfg.opening && (cfg.opening_seconds <= 0 || (t >= 0 && t < cfg.opening_seconds));
    openEl.textContent = showOpen ? cfg.opening : "";
    openEl.hidden = !showOpen;
  }

  // --- player limited to [start, end] ------------------------------------------------------------
  const playBtn = document.getElementById("play-btn"), readout = document.getElementById("time-readout");
  function frame() {
    // stop at the end (only while playing: re-seeking a paused video would loop via 'seeked')
    if (!video.paused && video.currentTime >= st.end) { video.pause(); video.currentTime = st.end; }
    draw(); updateOverlays();
    if (editor) editor.tick();
    readout.textContent = fmt(video.currentTime - st.start) + " / " + fmt(st.end - st.start);
    drawTimeline();
    if (!video.paused) requestAnimationFrame(frame);
  }
  video.addEventListener("play", () => {
    if (video.currentTime < st.start - 0.3 || video.currentTime >= st.end - 0.05) video.currentTime = st.start;
    playBtn.textContent = "⏸ Pause (Espaço)"; requestAnimationFrame(frame);
  });
  video.addEventListener("pause", () => { playBtn.textContent = "▶ Play (Espaço)"; frame(); });
  ["seeked", "loadeddata"].forEach((n) => video.addEventListener(n, frame));
  video.addEventListener("loadedmetadata", () => { if (video.currentTime < st.start - 0.5) video.currentTime = st.start; });
  function togglePlay() { if (video.paused) video.play().catch(() => {}); else video.pause(); }
  playBtn.addEventListener("click", togglePlay);
  function seekBy(d) { video.currentTime = clamp(video.currentTime + d, st.start, st.end); }

  // --- range editing --------------------------------------------------------------------------------
  let saveTimer = null, savePending = null, saveSnap = false;
  function setRange(s, e, opts) {
    opts = opts || {};
    const maxEnd = cfg.duration || Infinity, minLen = 0.5;
    s = clamp(s, 0, maxEnd - minLen); e = clamp(e, s + minLen, maxEnd);
    st.start = Math.round(s * 100) / 100; st.end = Math.round(e * 100) / 100;
    fStart.value = st.start.toFixed(2); fEnd.value = st.end.toFixed(2);
    root.dataset.start = st.start; root.dataset.end = st.end;
    const len = st.end - st.start;
    document.getElementById("dur-readout").textContent = len.toFixed(1) + " s";
    rangeStatus.textContent = len > cfg.max_s ? "longo demais (máx. " + cfg.max_s + " s)" : (len < cfg.min_s ? "curto demais" : "");
    buildEvents();
    if (editor && !editorEl.querySelector(".cap-input")) editor.render();  // list follows the cut
    if (opts.seek !== false && (video.currentTime < st.start || video.currentTime > st.end)) video.currentTime = st.start;
    else frame();
    ensureWaveWindow();
    if (opts.save !== false) scheduleSave(!!opts.snap);
  }
  function scheduleSave(snap) {
    saveSnap = saveSnap || snap;
    clearTimeout(saveTimer);
    rangeStatus.textContent = "salvando…";
    saveTimer = setTimeout(flushSave, 500);
  }
  function flushSave() {
    clearTimeout(saveTimer); saveTimer = null;
    const body = new FormData();
    body.append("start", st.start); body.append("end", st.end); body.append("snap", saveSnap ? "1" : "0");
    saveSnap = false;
    savePending = fetch("/revisao/" + cfg.candidate + "/intervalo", { method: "POST", body })
      .then((r) => r.json()).then((d) => {
        if (!d.ok) { rangeStatus.textContent = "⚠ " + d.error; rangeStatus.classList.add("error-text"); return; }
        rangeStatus.classList.remove("error-text"); rangeStatus.textContent = "salvo";
        rangeWarn.textContent = (d.warnings || []).join(" ");
        resetBtn.hidden = !d.adjusted;
        if (Math.abs(d.start - st.start) > 0.005 || Math.abs(d.end - st.end) > 0.005) setRange(d.start, d.end, { save: false, seek: false });
        refreshFaithful();
      }).catch(() => { rangeStatus.textContent = "⚠ não salvou (sem conexão?)"; })
      .finally(() => { savePending = null; });
    return savePending;
  }
  function flushIfNeeded() { return saveTimer ? flushSave() : (savePending || Promise.resolve()); }
  function nudge(which, d) {
    if (which === "start") setRange(st.start + d, st.end, { seek: true });
    else { setRange(st.start, st.end + d, { seek: false }); video.currentTime = Math.max(st.start, st.end - 2); }
  }
  [fStart, fEnd].forEach((inp) => inp.addEventListener("change", () => {
    const s = Number(fStart.value.replace(",", ".")), e = Number(fEnd.value.replace(",", "."));
    if (isFinite(s) && isFinite(e) && e > s) setRange(s, e);
  }));
  document.querySelectorAll("[data-context]").forEach((b) => b.addEventListener("click", () =>
    b.dataset.context === "start" ? nudge("start", -5) : nudge("end", 5)));
  resetBtn.addEventListener("click", () => {
    const body = new FormData(); body.append("reset", "1");
    fetch("/revisao/" + cfg.candidate + "/intervalo", { method: "POST", body }).then((r) => r.json()).then((d) => {
      if (d.ok) { setRange(d.start, d.end, { save: false }); resetBtn.hidden = true; rangeWarn.textContent = ""; refreshFaithful(); }
    });
  });

  // --- snapping (mirrors services.review.snap_edge) --------------------------------------------
  function snapEdge(t, side) {
    const ws = st.words.slice().sort((a, b) => a.s - b.s), edges = ws.map((w) => (side === "start" ? w.s : w.e));
    for (let i = 1; i < ws.length; i++) if (ws[i].s - ws[i - 1].e >= 0.3) edges.push((ws[i - 1].e + ws[i].s) / 2);
    let best = null;
    edges.forEach((x) => { if (best === null || Math.abs(x - t) < Math.abs(best - t)) best = x; });
    return best !== null && Math.abs(best - t) <= 0.35 ? best : t;
  }

  // --- timeline: waveform + words + handles ------------------------------------------------------
  const tl = document.getElementById("timeline-canvas"), tctx = tl.getContext("2d");
  const wave = { t0: 0, step: 0.05, peaks: [], w0: 0, w1: 0, loading: false, scale: 255 };
  function viewRange() { return [wave.w0, wave.w1]; }
  function loadWave(center0, center1) {
    wave.loading = true;
    const a = Math.max(0, center0), b = center1;
    fetch("/fontes/" + cfg.source + "/forma-de-onda.json?start=" + a.toFixed(2) + "&end=" + b.toFixed(2))
      .then((r) => r.json()).then((d) => {
        wave.t0 = d.t0; wave.step = d.step || 0.05; wave.peaks = d.peaks || []; wave.scale = d.scale || 255;
        wave.w0 = Math.max(0, a - 60); wave.w1 = b + 60;
        if (cfg.duration) wave.w1 = Math.min(wave.w1, cfg.duration);
        if (d.error) rangeStatus.textContent = d.error;
        drawTimeline();
      }).finally(() => { wave.loading = false; });
  }
  function ensureWaveWindow() {
    if (wave.loading) return;
    const empty = wave.w1 <= wave.w0;
    const nearStart = st.start < wave.w0 + 5 && wave.w0 > 0;
    const nearEnd = st.end > wave.w1 - 5 && (!cfg.duration || wave.w1 < cfg.duration);
    if (empty || nearStart || nearEnd) loadWave(st.start, st.end);
  }
  // caption editor (static/caption_editor.js): it loads palavras.json and hands the words to
  // the live preview; every saved correction re-renders the captions at once
  const editorEl = document.getElementById("caption-editor");
  const editor = editorEl && window.CaptionEditor ? window.CaptionEditor.mount(editorEl, {
    candidate: cfg.candidate, pad: 90,
    getTime: () => video.currentTime, getRange: () => [st.start, st.end],
    onData: (d) => { st.words = (d.words || []).map((w) => ({ s: w.s, e: w.e, t: w.t, i: w.i })); buildEvents(); frame(); },
    onChange: () => refreshFaithful(),
  }) : null;
  function loadWords() {
    if (editor) { editor.reload(); return; }
    fetch("/candidatos/" + cfg.candidate + "/palavras.json?pad=90").then((r) => r.json()).then((d) => {
      st.words = (d.words || []).map((w) => ({ s: w.s, e: w.e, t: w.t, i: w.i }));
      buildEvents(); frame();
    });
  }
  function xOf(t) { const [a, b] = viewRange(); return (t - a) / (b - a) * tl.width; }
  function tOf(x) { const [a, b] = viewRange(); return a + x / tl.width * (b - a); }
  function drawTimeline() {
    const dpr = window.devicePixelRatio || 1, cw = Math.round(tl.clientWidth * dpr), ch = Math.round(110 * dpr);
    if (tl.width !== cw || tl.height !== ch) { tl.width = cw; tl.height = ch; }
    const W = tl.width, H = tl.height, waveH = H * 0.7;
    tctx.fillStyle = "#12151b"; tctx.fillRect(0, 0, W, H);
    if (wave.w1 <= wave.w0) return;
    // waveform bars (max per pixel column)
    tctx.fillStyle = "#4fb6ff";
    const n = wave.peaks.length;
    for (let x = 0; x < W; x++) {
      const i0 = Math.floor((tOf(x) - wave.t0) / wave.step), i1 = Math.max(i0 + 1, Math.floor((tOf(x + 1) - wave.t0) / wave.step));
      let m = 0;
      for (let i = Math.max(0, i0); i < Math.min(n, i1); i++) m = Math.max(m, wave.peaks[i]);
      if (m) { const h = m / wave.scale * waveH; tctx.fillRect(x, (waveH - h) / 2, 1, Math.max(1, h)); }
    }
    // words
    tctx.fillStyle = "rgba(255,224,0,.55)";
    st.words.forEach((w) => { const a = xOf(w.s), b = xOf(w.e); if (b >= 0 && a <= W) tctx.fillRect(a, waveH + 6 * dpr, Math.max(1, b - a - 1), 8 * dpr); });
    // outside the cut: dimmed
    const xs = xOf(st.start), xe = xOf(st.end);
    tctx.fillStyle = "rgba(0,0,0,.55)";
    tctx.fillRect(0, 0, Math.max(0, xs), H); tctx.fillRect(xe, 0, W - xe, H);
    // handles
    [[xs, "#2fbf71"], [xe, "#ef4b4b"]].forEach(([x, c]) => {
      tctx.fillStyle = c; tctx.fillRect(x - 1.5 * dpr, 0, 3 * dpr, H);
      tctx.fillRect(x - 6 * dpr, H - 18 * dpr, 12 * dpr, 18 * dpr);
    });
    // playhead + scale
    const xp = xOf(video.currentTime);
    tctx.fillStyle = "#fff"; tctx.fillRect(xp, 0, 1 * dpr, H);
    tctx.fillStyle = "#8b93a3"; tctx.font = (10 * dpr) + "px sans-serif";
    const [a, b] = viewRange(), stepS = (b - a) > 200 ? 30 : 10;
    for (let t = Math.ceil(a / stepS) * stepS; t < b; t += stepS) tctx.fillText(fmt(t).replace(/\.\d+$/, ""), xOf(t) + 2, H - 4 * dpr);
  }
  let drag = null;
  function evX(ev) { const r = tl.getBoundingClientRect(); return (ev.clientX - r.left) * (tl.width / r.width); }
  tl.addEventListener("pointerdown", (ev) => {
    const x = evX(ev), tol = 10 * (window.devicePixelRatio || 1);
    if (Math.abs(x - xOf(st.start)) < tol) drag = "start";
    else if (Math.abs(x - xOf(st.end)) < tol) drag = "end";
    else { const t = tOf(x); if (t >= st.start && t <= st.end) video.currentTime = t; return; }
    tl.setPointerCapture(ev.pointerId); ev.preventDefault();
  });
  tl.addEventListener("pointermove", (ev) => {
    const x = evX(ev), tol = 10 * (window.devicePixelRatio || 1);
    tl.style.cursor = drag || Math.abs(x - xOf(st.start)) < tol || Math.abs(x - xOf(st.end)) < tol ? "ew-resize" : "pointer";
    if (!drag) return;
    let t = tOf(x);
    if (snapBox.checked && !ev.altKey) t = snapEdge(t, drag);
    if (drag === "start") { setRange(t, st.end, { save: false, seek: false }); video.currentTime = st.start; }
    else { setRange(st.start, t, { save: false, seek: false }); video.currentTime = Math.max(st.start, st.end - 0.05); }
  });
  tl.addEventListener("pointerup", () => { if (drag) { drag = null; scheduleSave(snapBox.checked); } });
  new ResizeObserver(() => { drawTimeline(); applyStyle(); }).observe(tl);

  // --- layout (L) ----------------------------------------------------------------------------------
  function cycleLayout() {
    const body = new FormData(); body.append("layout", "proximo");
    fetch("/revisao/" + cfg.candidate + "/layout", { method: "POST", body }).then((r) => r.json()).then((d) => {
      st.layout = d.layout; stage.dataset.layout = d.layout;
      document.getElementById("layout-label").textContent = d.label;
      frame(); refreshFaithful();
    });
  }
  document.getElementById("layout-btn").addEventListener("click", cycleLayout);

  // --- faithful preview (P) ----------------------------------------------------------------------
  const fStatus = document.getElementById("faithful-status"), fVideo = document.getElementById("faithful-video");
  let pollTimer = null;
  const LABELS = { queued: "na fila…", running: "renderizando…", done: "pronta", failed: "falhou", canceled: "cancelada",
    nenhuma: "ainda não gerada para este corte", indisponivel: "indisponível" };
  function showFaithful(d) {
    clearTimeout(pollTimer);
    let txt = LABELS[d.status] || d.status;
    if (d.status === "running" && d.progress != null) txt = "renderizando… " + Math.round(d.progress) + "%";
    if (d.error) txt += " — " + d.error;
    fStatus.textContent = txt;
    if (d.url) { if (fVideo.dataset.src !== d.url) { fVideo.src = d.url; fVideo.dataset.src = d.url; } fVideo.hidden = false; }
    else fVideo.hidden = true;
    if (d.status === "queued" || d.status === "running") pollTimer = setTimeout(refreshFaithful, 2000);
  }
  function refreshFaithful() {
    fetch("/revisao/" + cfg.candidate + "/previa.json").then((r) => r.json()).then(showFaithful).catch(() => {});
  }
  function requestFaithful() {
    flushIfNeeded().then(() => fetch("/revisao/" + cfg.candidate + "/previa", { method: "POST" }))
      .then((r) => r.json()).then(showFaithful);
  }
  document.getElementById("faithful-btn").addEventListener("click", requestFaithful);

  // --- pre-publish checks ------------------------------------------------------------------------
  const checksStatus = document.getElementById("checks-status"), readyBadge = document.getElementById("ready-badge");
  document.querySelectorAll("[data-check]").forEach((box) => box.addEventListener("change", () => {
    const body = new FormData(); body.append(box.dataset.check, box.checked ? "1" : "0");
    checksStatus.textContent = "salvando…";
    fetch("/revisao/" + cfg.candidate + "/verificacoes", { method: "POST", body }).then((r) => r.json()).then((d) => {
      if (!d.ok) { checksStatus.textContent = "⚠ " + d.error; box.checked = !box.checked; return; }
      checksStatus.textContent = "salvo";
      if (readyBadge) readyBadge.hidden = !d.ready;
    }).catch(() => { checksStatus.textContent = "⚠ não salvou (sem conexão?)"; box.checked = !box.checked; });
  }));

  // --- decisions ------------------------------------------------------------------------------------
  function decide(action, reason) {
    form.action.value = action; form.reason_tag.value = reason || "";
    video.pause();
    flushIfNeeded().then(() => form.submit());
  }
  function openReasons(open) { reasons.hidden = !open; }
  // a clicked button must not keep the focus: Space/Enter would "click" it again
  document.addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (b) b.blur(); });
  form.querySelectorAll("[data-action]").forEach((b) => b.addEventListener("click", () =>
    b.dataset.action === "reject" ? openReasons(reasons.hidden) : decide(b.dataset.action)));
  reasons.querySelectorAll("[data-reason]").forEach((b) => b.addEventListener("click", () => decide("reject", b.dataset.reason)));

  // --- keyboard ---------------------------------------------------------------------------------------
  const KEYS = {
    "a": () => decide("approve"), "r": () => openReasons(true), "s": () => decide("skip"),
    "z": () => flushIfNeeded().then(() => undoForm.submit()),
    " ": togglePlay, "arrowleft": () => seekBy(-2), "arrowright": () => seekBy(2),
    "[": () => nudge("start", -0.5), "{": () => nudge("start", -2),
    "]": () => nudge("start", 0.5), "}": () => nudge("start", 2),
    ",": () => nudge("end", -0.5), "<": () => nudge("end", -2),
    ".": () => nudge("end", 0.5), ">": () => nudge("end", 2),
    "l": cycleLayout, "p": requestFaithful, "?": toggleHelp,
    "e": () => { if (editor) editor.focus(); },
    "backspace": () => { const a = document.querySelector('[data-shortcut="anterior"]'); if (a) location.href = a.href; },
  };
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      if (typing(ev.target)) ev.target.blur();
      if (!reasons.hidden) openReasons(false);
      if (help && !help.hidden) help.hidden = true;
      return;
    }
    if (typing(ev.target) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (!reasons.hidden && /^[1-9]$/.test(ev.key)) {
      const k = cfg.reasons[Number(ev.key) - 1];
      if (k) { ev.preventDefault(); decide("reject", k); }
      return;
    }
    const fn = KEYS[ev.key.toLowerCase()];
    if (!fn) return;
    ev.preventDefault();
    if (ev.repeat && ["a", "r", "s", "z", "p", "l"].includes(ev.key.toLowerCase())) return;
    fn();
  });

  // --- start -------------------------------------------------------------------------------------------
  applyStyle();
  setRange(st.start, st.end, { save: false });  // also loads the waveform window
  loadWords();
  refreshFaithful();
})();
