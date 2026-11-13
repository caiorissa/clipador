"""`clipador dump` / `clipador eval`: inspect candidates and measure detection quality."""

from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session

from clipador.config import Settings
from clipador.detection.evaluate import EvalResult, RankedCandidate, evaluate, parse_golden
from clipador.detection.params import SIGNALS
from clipador.errors import ClipadorError
from clipador.models import AnalysisRun, Source
from clipador.services import analysis
from clipador.services.enrichment import enrichments_for
from clipador.utils import format_hms

CSV_COLUMNS = [
    "rank",
    "inicio",
    "fim",
    "duracao_s",
    "nota",
    *SIGNALS,
    "status",
    "feedback",
    "trecho",
]


def _run(session: Session, source_id: int, run_id: int | None) -> AnalysisRun:
    if session.get(Source, source_id) is None:
        raise ClipadorError(f"Fonte #{source_id} não existe.")
    runs = analysis.runs_for(session, source_id)
    if not runs:
        raise ClipadorError(f"A fonte #{source_id} ainda não foi analisada.")
    if run_id is None:
        return runs[0]
    for r in runs:
        if r.id == run_id:
            return r
    raise ClipadorError(f"Análise #{run_id} não pertence à fonte #{source_id}.")


def dump(engine: Engine, source_id: int, top: int, out_csv: Path, run_id: int | None = None) -> str:
    with Session(engine) as s:
        run = _run(s, source_id, run_id)
        rows = analysis.candidates_for(s, run.id)[:top]
    lines = [f"Fonte #{source_id} · análise #{run.id} · {len(rows)} candidato(s)"]
    lines.append(
        f"{'#':>3} {'início':>8} {'dur':>5} {'nota':>5}  "
        + " ".join(f"{k[:5]:>5}" for k in SIGNALS)
        + "  trecho"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:  # BOM: opens fine in Excel
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for c, fb in rows:
            s0, e0 = c.eff_start, c.eff_end
            bd = [c.breakdown.get(k) for k in SIGNALS]
            w.writerow([c.rank, format_hms(s0), format_hms(e0), round(e0 - s0, 1), c.score,
                        *["" if v is None else v for v in bd], c.status,
                        fb.verdict if fb else "", c.transcript_snippet])  # fmt: skip
            bd_txt = " ".join(f"{'—' if v is None else round(v):>5}" for v in bd)
            snippet = c.transcript_snippet[:70].replace("\n", " ")
            lines.append(
                f"{c.rank:>3} {format_hms(s0):>8} {e0 - s0:>5.0f} {c.score:>5.0f}  "
                f"{bd_txt}  {snippet}"
            )
    lines.append(f"CSV salvo em {out_csv}")
    return "\n".join(lines)


def run_eval(
    engine: Engine,
    settings: Settings,
    source_id: int,
    golden_path: Path,
    run_id: int | None = None,
    ns: tuple[int, ...] = (5, 10, 20),
    with_llm: bool = False,
) -> tuple[EvalResult, str]:
    try:
        golden = parse_golden(golden_path.read_text("utf-8-sig"))
    except FileNotFoundError:
        raise ClipadorError(f"Arquivo não encontrado: {golden_path}") from None
    except ValueError as e:
        raise ClipadorError(
            f"golden.txt inválido ({e}). Use um horário HH:MM:SS por linha."
        ) from None
    if not golden:
        raise ClipadorError("O golden.txt está vazio: coloque um horário (HH:MM:SS) por linha.")
    with Session(engine) as s:
        run = _run(s, source_id, run_id)
        rows = analysis.candidates_for(s, run.id)
    ranked = [
        RankedCandidate(c.rank, c.eff_start, c.eff_end, c.score, fb.verdict if fb else None)
        for c, fb in rows
    ]
    feats = analysis.load_features(engine, settings, source_id)
    weights = run.params.get("candidates", {}).get("weights", {})

    def curve_at(t: float):
        return analysis.curve_value_at(feats, weights, t) if feats is not None else None

    res = evaluate(
        golden, ranked, ns=ns, tolerance=settings.eval_tolerance_seconds, curve_at=curve_at
    )
    text = format_eval(res, source_id, run.id, len(ranked), settings.eval_tolerance_seconds)
    if with_llm:
        text += "\n\n" + _llm_section(
            engine, golden, rows, ns, settings.eval_tolerance_seconds, res
        )
    return res, text


def pairwise_agreement(pairs: list[tuple[float, str]]) -> tuple[float | None, int]:
    """Share of (bom, ruim) pairs where the "bom" candidate has the higher score (ties = 0.5).
    0.5 = no better than chance; 1.0 = perfect. Returns (value, number of pairs)."""
    good = [s for s, v in pairs if v == "bom"]
    bad = [s for s, v in pairs if v == "ruim"]
    if not good or not bad:
        return None, 0
    total = sum(1.0 if g > b else 0.5 if g == b else 0.0 for g in good for b in bad)
    return total / (len(good) * len(bad)), len(good) * len(bad)


def _llm_section(engine, golden, rows, ns, tol, base: EvalResult) -> str:
    with Session(engine) as s:
        enr = enrichments_for(s, [c.id for c, _ in rows])
    enriched = [c for c, _ in rows if c.id in enr and enr[c.id].nota_geral is not None]
    out = [f"Com a IA ({len(enriched)} de {len(rows)} candidatos avaliados pela IA):"]
    if not enriched:
        return "\n".join([*out, "  nenhum candidato foi enriquecido; rode Enriquecer com IA."])
    by_final = sorted(rows, key=lambda r: (r[0].final_rank or 10**6, r[0].rank))
    ranked_llm = [
        RankedCandidate(i, c.eff_start, c.eff_end, c.final_score or c.score,
                        fb.verdict if fb else None)
        for i, (c, fb) in enumerate(by_final, 1)
    ]  # fmt: skip
    llm_res = evaluate(golden, ranked_llm, ns=ns, tolerance=tol)
    for n in ns:
        h0, total = base.recall[n]
        h1, _ = llm_res.recall[n]
        delta = h1 - h0
        tag = "melhorou" if delta > 0 else ("PIOROU" if delta < 0 else "igual")
        out.append(f"  recall@{n:<3} detecção {h0}/{total} → com IA {h1}/{total} ({tag})")
        p0, p1 = base.precision.get(n), llm_res.precision.get(n)
        if p0 or p1:
            fmt = lambda p: f"{p[0]}/{p[1]}" if p else "—"  # noqa: E731
            out.append(f"  precisão@{n:<2} detecção {fmt(p0)} → com IA {fmt(p1)}")
    judged = [
        (c, fb.verdict) for c, fb in rows if fb and c.id in enr and enr[c.id].nota_geral is not None
    ]
    a_llm, npairs = pairwise_agreement([(enr[c.id].nota_geral, v) for c, v in judged])
    a_det, _ = pairwise_agreement([(c.score, v) for c, v in judged])
    out.append("")
    if a_llm is None:
        out.append(
            "  Acordo com seus vereditos: marque candidatos como Bom e Ruim (os dois) para medir."
        )
    else:
        out.append(
            f"  Acordo com seus vereditos ({npairs} pares bom×ruim; 50% = acaso, 100% = perfeito): "
            f"IA {100 * a_llm:.0f}% · detecção {100 * a_det:.0f}%"
        )
        for verdict in ("bom", "ruim"):
            vals = [enr[c.id].nota_geral for c, v in judged if v == verdict]
            if vals:
                avg = sum(vals) / len(vals)
                out.append(f"  nota média da IA nos marcados como {verdict}: {avg:.1f}/10")
    return "\n".join(out)


def format_eval(res: EvalResult, source_id: int, run_id: int, n_cands: int, tol: float) -> str:
    out = [
        f"Fonte #{source_id} · análise #{run_id} · {n_cands} candidatos · "
        f"{res.total} momentos no golden (tolerância ±{tol:g}s)",
        "",
    ]
    for n, (hits, total) in res.recall.items():
        pct = 100 * hits / total if total else 0
        prec = res.precision.get(n)
        prec_txt = (
            f"precisão@{n}: {prec[0]}/{prec[1]} avaliados ({100 * prec[0] / prec[1]:.0f}%)"
            if prec
            else f"precisão@{n}: — (marque candidatos como Bom/Ruim)"
        )
        out.append(f"recall@{n:<3} {hits}/{total} ({pct:.0f}%)   {prec_txt}")
    if res.missed:
        out += ["", f"Perdidos (fora do top {max(res.recall)}):"]
        for m in res.missed:
            nota = "—" if m.curve_score is None else f"{m.curve_score:.0f}"
            near = (
                f"candidato mais próximo: #{m.nearest_rank} a {m.nearest_distance:.0f}s"
                if m.nearest_rank is not None
                else "nenhum candidato perto"
            )
            out.append(f"  {format_hms(m.t)}  interesse nesse instante: {nota}/100 · {near}")
    else:
        out += ["", "Nenhum momento do golden ficou de fora. 🎯"]
    return "\n".join(out)
