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
    return res, format_eval(res, source_id, run.id, len(ranked), settings.eval_tolerance_seconds)


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
