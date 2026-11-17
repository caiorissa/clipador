"""Detection presets: the built-in ones plus calibrated presets saved in
`<data_dir>/presets_calibrados.json` (5B). The file also keeps, per source, the analysis params
in use before a calibrated preset was applied, so "Reverter" can go back to them.

Applying never happens on its own: only from the "Aplicar como novo preset" button.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session

from clipador.config import Settings
from clipador.detection.calibrate import Calibration, preset_weights
from clipador.detection.params import PRESET_LABELS, PRESETS, SIGNALS, DetectParams
from clipador.errors import ValidationError
from clipador.llm.scoring import RankingParams
from clipador.models import Job
from clipador.services import analysis

FILE_NAME = "presets_calibrados.json"


def store_path(settings: Settings) -> Path:
    return settings.data_dir / FILE_NAME


def load_store(settings: Settings) -> dict[str, Any]:
    path = store_path(settings)
    try:
        data = json.loads(path.read_text("utf-8"))
    except (FileNotFoundError, ValueError):
        data = {}
    data.setdefault("presets", {})
    data.setdefault("applied", [])
    return data


def save_store(settings: Settings, data: dict[str, Any]) -> None:
    path = store_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    os.replace(tmp, path)


def _with_defaults(weights: dict) -> dict[str, float]:
    """Calibrated presets saved before a signal existed get the default weight for it."""
    base = PRESETS["equilibrado"]
    return {k: float(weights[k]) if k in weights else base[k] for k in SIGNALS}


def all_presets(settings: Settings) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """(labels, weights) of the built-in + calibrated presets, built-ins first."""
    labels, weights = dict(PRESET_LABELS), {k: dict(v) for k, v in PRESETS.items()}
    for key, p in load_store(settings)["presets"].items():
        labels[key] = p.get("label", key)
        weights[key] = _with_defaults(p["weights"])
    return labels, weights


def params_for_preset(settings: Settings, key: str) -> DetectParams:
    p = DetectParams.from_preset(key)
    custom = load_store(settings)["presets"].get(key)
    if custom:
        p.candidates.preset = key
        p.candidates.weights = _with_defaults(custom["weights"])
        p.ranking = custom.get("ranking")
    return p


def _run_params(run) -> dict[str, Any]:
    return {k: v for k, v in run.params.items() if k != "effective_weights"}


def apply_calibration(
    session: Session, settings: Settings, source_id: int, cal: Calibration
) -> tuple[str, Job]:
    """Save the suggestion as a new preset and queue the cheap re-analysis (detect only, no
    re-transcription) of the source with it. The caller commits."""
    if not cal.can_apply or cal.weights is None:
        raise ValidationError(
            "Ainda não dá para aplicar: poucos dados ou nenhuma sugestão de pesos."
        )
    runs = analysis.runs_for(session, source_id)
    if not runs:
        raise ValidationError("Analise a fonte primeiro (botão Analisar).")
    run = runs[0]
    previous = _run_params(run)
    now = datetime.now().astimezone()
    store = load_store(settings)
    key = f"calibrado_{now:%Y%m%d_%H%M%S}"
    n = 1
    while key in store["presets"] or key in PRESETS:
        n += 1
        key = f"calibrado_{now:%Y%m%d_%H%M%S}_{n}"
    weights = preset_weights(cal.weights)
    ranking = None
    if cal.ranking:
        base = RankingParams.from_run(run.params, settings).model_dump()
        ranking = {**base, **cal.ranking}
    params = DetectParams.model_validate(previous)
    params.candidates.preset = key
    params.candidates.weights = weights
    params.ranking = ranking
    job = analysis.reanalyze(session, source_id, params)
    store["presets"][key] = {
        "label": f"Calibrado {now:%d/%m %H:%M}", "weights": weights, "ranking": ranking,
        "created_at": now.isoformat(timespec="seconds"),
        "stats": {"n": cal.n, "aprovados": cal.n_pos, "rejeitados": cal.n_neg,
                  "auc_antes": cal.curve.auc_before, "auc_depois": cal.curve.auc_after},
    }  # fmt: skip
    store["applied"].append({
        "source_id": source_id, "preset": key, "previous_run_id": run.id,
        "previous_preset": previous.get("candidates", {}).get("preset"),
        "previous_params": previous, "job_id": job.id, "at": now.isoformat(timespec="seconds"),
    })  # fmt: skip
    save_store(settings, store)
    return key, job


def last_applied(settings: Settings, source_id: int) -> dict[str, Any] | None:
    for entry in reversed(load_store(settings)["applied"]):
        if entry.get("source_id") == source_id:
            return entry
    return None


def revert(session: Session, settings: Settings, source_id: int) -> tuple[dict[str, Any], Job]:
    """Re-analyze the source with the params it had before the last calibrated preset was
    applied (the preset itself stays available in the list). The caller commits."""
    store = load_store(settings)
    pos = next((i for i in range(len(store["applied"]) - 1, -1, -1)
                if store["applied"][i].get("source_id") == source_id), None)  # fmt: skip
    if pos is None:
        raise ValidationError("Nenhum preset calibrado foi aplicado nesta fonte.")
    entry = store["applied"][pos]
    params = DetectParams.model_validate(entry["previous_params"])
    job = analysis.reanalyze(session, source_id, params)
    del store["applied"][pos]
    save_store(settings, store)
    return entry, job
