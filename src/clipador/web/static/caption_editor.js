/* Clipador — caption editor (word by word). Reusable: /revisao and the source page.
   CaptionEditor.mount(el, {candidate, pad, getTime, getRange, onData, onChange}) -> api
     getTime(): current ABSOLUTE time of the player (highlights the word playing now)
     getRange(): [start, end] of the clip (absolute), words outside are not listed
     onData(d): called with the palavras.json payload after each load (live preview)
     onChange(): called after a correction was saved
   api: reload(), render(), tick(), focus()
   No build step, no external libraries. */
(function () {
  "use strict";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (t) => { t = Math.max(0, t); const m = Math.floor(t / 60), s = t % 60; return m + ":" + s.toFixed(1).padStart(4, "0"); };

  function mount(el, opts) {
    const o = Object.assign({ pad: 0, getTime: () => -1, getRange: null, onData: null, onChange: null }, opts);
    let data = { editor: [], stale: [], streamer: null };
    let editing = null, nowIdx = null;

    el.classList.add("cap-editor");
    el.innerHTML =
      '<div class="cap-head"><h3>Legenda <span class="small muted">palavra a palavra</span></h3>' +
      '<label class="check small"><input type="checkbox" data-glossary> sempre corrigir assim neste streamer' +
      ' <span data-streamer></span></label><span class="small muted" data-status></span></div>' +
      '<div class="flash warn small" data-stale hidden></div>' +
      '<div class="cap-words" data-words><span class="small muted">carregando…</span></div>' +
      '<p class="small muted">Clique numa palavra para corrigir (o tempo da palavra não muda): <kbd>Enter</kbd> salva, ' +
      "<kbd>Esc</kbd> cancela. ⦸ oculta, ↶ desfaz a correção. Na revisão, <kbd>E</kbd> traz o foco para cá.</p>";
    const wordsEl = el.querySelector("[data-words]"), statusEl = el.querySelector("[data-status]");
    const staleEl = el.querySelector("[data-stale]"), glossBox = el.querySelector("[data-glossary]");

    function status(msg, isError) { statusEl.textContent = msg || ""; statusEl.classList.toggle("error-text", !!isError); }

    function rows() {
      const r = o.getRange ? o.getRange() : null;
      return data.editor.filter((w) => !r || (w.e > r[0] && w.s < r[1]));
    }

    function render() {
      el.querySelector("[data-streamer]").textContent = data.streamer ? "(" + data.streamer + ")" : "";
      if (data.stale.length) {
        staleEl.hidden = false;
        staleEl.innerHTML = "⚠ " + data.stale.length + " correção(ões) obsoleta(s): a transcrição mudou e a palavra " +
          "nessa posição não é mais a que foi corrigida, então a correção <strong>não está sendo aplicada</strong>. " +
          data.stale.map((s) => "“" + esc(s.original) + "” → “" + esc(s.text || "(oculta)") + "” (agora: “" + esc(s.current) + "”)").join("; ") +
          ' <button type="button" class="btn small" data-remove-stale>Remover obsoletas</button>';
      } else { staleEl.hidden = true; staleEl.innerHTML = ""; }
      const list = rows();
      if (!list.length) { wordsEl.innerHTML = '<span class="small muted">Sem fala transcrita neste trecho.</span>'; return; }
      wordsEl.innerHTML = list.map((w) => {
        const cls = ["cap-w"];
        if (w.ov !== null) cls.push("edited");
        if (w.hidden) cls.push("hidden-word");
        if (w.gloss !== null && w.ov === null) cls.push("gloss");
        if (w.stale) cls.push("stale");
        if (w.i === nowIdx) cls.push("now");
        let title = fmt(w.s) + " · transcrição: " + w.raw;
        if (w.gloss !== null) title += " · glossário: " + w.gloss;
        if (editing === w.i) {
          return '<span class="' + cls.join(" ") + ' editing" data-i="' + w.i + '"><input class="cap-input" value="' +
            esc(w.hidden ? w.raw : w.t) + '" aria-label="corrigir palavra"><button type="button" class="btn small" data-save>✓</button></span>';
        }
        return '<span class="' + cls.join(" ") + '" data-i="' + w.i + '" data-s="' + w.s + '" data-e="' + w.e + '">' +
          '<button type="button" class="cap-t" title="' + esc(title) + '">' + esc(w.hidden ? w.raw : w.t) + "</button>" +
          '<button type="button" class="cap-mini" data-hide title="' + (w.hidden ? "mostrar" : "ocultar") + ' esta palavra">⦸</button>' +
          (w.ov !== null ? '<button type="button" class="cap-mini" data-undo title="desfazer a correção">↶</button>' : "") +
          "</span>";
      }).join(" ");
      const inp = wordsEl.querySelector(".cap-input");
      if (inp) { inp.focus(); inp.select(); }
    }

    function reload() {
      let url = "/candidatos/" + o.candidate + "/palavras.json?pad=" + o.pad;
      return fetch(url).then((r) => r.json()).then((d) => {
        data = { editor: d.editor || [], stale: d.stale || [], streamer: d.streamer };
        if (o.onData) o.onData(d);
        render();
        return d;
      }).catch(() => status("⚠ não carregou as palavras", true));
    }

    function post(fields) {
      const body = new FormData();
      Object.keys(fields).forEach((k) => body.append(k, fields[k]));
      status("salvando…");
      return fetch("/candidatos/" + o.candidate + "/legenda/palavra", { method: "POST", body })
        .then((r) => r.json()).then((d) => {
          if (!d.ok) { status("⚠ " + d.error, true); return; }
          status(d.message || "salvo");
          editing = null;
          return reload().then(() => { if (o.onChange) o.onChange(); });
        }).catch(() => status("⚠ não salvou (sem conexão?)", true));
    }

    function save() {
      const inp = wordsEl.querySelector(".cap-input");
      if (!inp || editing === null) return;
      const w = data.editor.find((x) => x.i === editing);
      const text = inp.value.trim();
      if (w && text === w.t && !glossBox.checked) { editing = null; render(); return; }
      post({ i: editing, text, hide: text ? "0" : "1", glossary: glossBox.checked ? "1" : "0" });
    }

    wordsEl.addEventListener("click", (ev) => {
      const chip = ev.target.closest("[data-i]");
      if (!chip) return;
      const i = Number(chip.dataset.i), w = data.editor.find((x) => x.i === i);
      if (ev.target.closest("[data-save]")) { save(); return; }
      if (ev.target.closest("[data-undo]")) { post({ i, undo: "1" }); return; }
      if (ev.target.closest("[data-hide]")) { post(w && w.hidden ? { i, undo: "1" } : { i, hide: "1" }); return; }
      if (ev.target.closest(".cap-t")) { editing = i; render(); }
    });
    wordsEl.addEventListener("keydown", (ev) => {
      if (!ev.target.classList.contains("cap-input")) return;
      if (ev.key === "Enter") { ev.preventDefault(); save(); }
      else if (ev.key === "Escape") { editing = null; render(); focus(); }
      ev.stopPropagation();  // letters typed here are text, not shortcuts
    });
    staleEl.addEventListener("click", (ev) => {
      if (!ev.target.closest("[data-remove-stale]")) return;
      fetch("/candidatos/" + o.candidate + "/legenda/obsoletas/remover", { method: "POST" })
        .then((r) => r.json()).then((d) => { status(d.removed + " removida(s)"); return reload(); })
        .then(() => { if (o.onChange) o.onChange(); });
    });

    function tick() {  // highlight the word playing now (cheap: only touches 2 nodes)
      const t = o.getTime();
      const w = data.editor.find((x) => t >= x.s && t < x.e);
      const idx = w ? w.i : null;
      if (idx === nowIdx) return;
      const prev = wordsEl.querySelector(".cap-w.now");
      if (prev) prev.classList.remove("now");
      nowIdx = idx;
      if (idx !== null) {
        const cur = wordsEl.querySelector('.cap-w[data-i="' + idx + '"]');
        if (cur) cur.classList.add("now");
      }
    }

    function focus() {  // the word playing now, else the first one
      const b = wordsEl.querySelector(".cap-w.now .cap-t") || wordsEl.querySelector(".cap-t");
      if (b) { b.focus(); b.scrollIntoView({ block: "nearest" }); }
      else el.scrollIntoView({ block: "nearest" });
    }

    return { reload, render, tick, focus, get data() { return data; } };
  }

  window.CaptionEditor = { mount };
})();
