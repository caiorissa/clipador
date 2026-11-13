"""Calibration from my own review decisions (pure: numpy/scipy, no DB).

Approved = positive, rejected = negative. Two small, explainable models:

1. curve weights: logistic regression (L2, standardized features) on the per-signal breakdown of
   each candidate (0-100). Coefficients are mapped back to the raw scale, negatives are clipped
   to 0 and the result is normalized (sum 1): the suggested interest-curve weights.
2. detection x AI: the same on [detection score, AI nota x 10] for candidates the AI scored:
   suggested w_detect / w_llm.

The detection score of a candidate is approximated as the weighted mean of its breakdown with
the weights renormalized over the signals it has (the real score uses peak/mean of the combined
curve: close, and the same approximation is used before and after, so the comparison is fair).
"Before" uses the weights each candidate was found with; "after" is measured with stratified
k-fold cross-validation (fixed seed): weights fitted without the fold are used to score it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

MIN_TOTAL = 50
MIN_CLASS = 10


@dataclass(frozen=True)
class Sample:
    breakdown: dict[str, float | None]  # per-signal 0-100 (None = signal missing, e.g. no chat)
    label: int  # 1 = approved, 0 = rejected
    weights: dict[str, float]  # detection weights the candidate was found with
    llm: float | None = None  # AI nota_geral 0-10
    w_detect: float = 0.5  # ranking weights in use
    w_llm: float = 0.5


# --- small numeric helpers -----------------------------------------------------------------------


def linear_score(bd: dict[str, float | None], weights: dict[str, float]) -> float:
    """Weighted mean of the available signals (weights renormalized like the detector does)."""
    avail = {k: float(v) for k, v in bd.items() if v is not None}
    if not avail:
        return 0.0
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in avail}
    total = sum(w.values())
    if total <= 0:
        return sum(avail.values()) / len(avail)
    return sum(w[k] * avail[k] for k in avail) / total


def blend(det: float, llm10: float, w_detect: float, w_llm: float) -> float:
    total = max(0.0, w_detect) + max(0.0, w_llm)
    if total <= 0:
        return det
    return (max(0.0, w_detect) * det + max(0.0, w_llm) * llm10) / total


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (X - mu) / sd, mu, sd


def fit_logistic(Z: np.ndarray, y: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, float]:
    """Logistic regression, L2 on the coefficients (not the intercept). Deterministic."""
    d = Z.shape[1]
    s = 2.0 * y - 1.0  # labels as -1/+1

    def loss(theta):
        w, b = theta[:d], theta[d]
        m = s * (Z @ w + b)
        val = np.logaddexp(0.0, -m).sum() + 0.5 * l2 * float(w @ w)
        g = -s / (1.0 + np.exp(np.clip(m, -500, 500)))  # d loss / d (Zw+b)
        grad = np.concatenate([Z.T @ g + l2 * w, [g.sum()]])
        return val, grad

    res = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return res.x[:d], float(res.x[d])


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Test indexes of k folds with the class balance kept (fixed seed -> same folds)."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    offset = 0
    for cls in (1, 0):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[(j + offset) % k].append(int(i))
        offset += len(idx)
    return [np.array(sorted(f), dtype=int) for f in folds if f]


def auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """ROC AUC (Mann-Whitney, ties count half). None when a class is missing."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    r = rankdata(scores)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def precision_at(scores: np.ndarray, y: np.ndarray, n: int) -> float | None:
    """Share of approved among the top-n by score (ties broken by order, stable)."""
    if len(y) == 0:
        return None
    top = np.argsort(-scores, kind="stable")[: min(n, len(y))]
    return float(y[top].mean())


def _design(samples: list[Sample], signals: list[str], fill: np.ndarray | None = None):
    X = np.array(
        [[np.nan if s.breakdown.get(k) is None else float(s.breakdown[k]) for k in signals]
         for s in samples],
        dtype=float,
    ).reshape(len(samples), len(signals))  # fmt: skip
    if fill is None:
        with np.errstate(all="ignore"):
            fill = np.nanmean(np.where(np.isnan(X).all(axis=0), 0.0, X), axis=0)
        fill = np.nan_to_num(fill)
    return np.where(np.isnan(X), fill, X), fill


def suggest_weights(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray | None:
    """Non-negative, normalized (sum 1) weights on the raw feature scale; None if the model
    finds no positive association at all."""
    Z, _mu, sd = standardize(X)
    coef, _b = fit_logistic(Z, y, l2)
    raw = np.clip(coef / sd, 0.0, None)
    if raw.sum() <= 1e-12:
        return None
    return raw / raw.sum()


# --- results -------------------------------------------------------------------------------------


@dataclass
class Metrics:
    auc_before: float | None = None
    auc_after: float | None = None  # cross-validated
    precision_before: dict[int, float | None] = field(default_factory=dict)
    precision_after: dict[int, float | None] = field(default_factory=dict)


@dataclass
class Calibration:
    n: int
    n_pos: int
    n_neg: int
    enough_data: bool
    warnings: list[str]
    signals: list[str]
    weights: dict[str, float] | None  # suggested curve weights, sum 1 (None = no suggestion)
    current_weights: dict[str, float]  # mean of the (normalized) weights in use, for reference
    importance: dict[str, float]  # standardized coefficients (sign = direction)
    curve: Metrics
    n_llm: int = 0
    ranking: dict[str, float] | None = None  # {"w_detect", "w_llm"} sum 1
    current_ranking: dict[str, float] | None = None
    blend: Metrics | None = None
    folds: int = 0
    seed: int = 0

    @property
    def can_apply(self) -> bool:
        return self.enough_data and self.weights is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["can_apply"] = self.can_apply
        return d


def _norm(w: dict[str, float], keys: list[str]) -> dict[str, float]:
    v = {k: max(0.0, float(w.get(k, 0.0))) for k in keys}
    t = sum(v.values())
    return {k: (x / t if t > 0 else 1.0 / len(keys)) for k, x in v.items()}


def _cv_scores(samples, signals, y, folds, l2, fallback_scores) -> np.ndarray:
    out = np.array(fallback_scores, dtype=float)
    for test in folds:
        train = np.setdiff1d(np.arange(len(samples)), test)
        if len(set(y[train].tolist())) < 2:
            continue
        Xtr, _fill = _design([samples[i] for i in train], signals)
        w = suggest_weights(Xtr, y[train], l2)
        if w is None:
            continue
        wd = dict(zip(signals, w.tolist(), strict=True))
        for i in test:
            out[i] = linear_score(samples[i].breakdown, wd)
    return out


def calibrate(
    samples: list[Sample],
    signals: tuple[str, ...] | list[str],
    *,
    k: int = 5,
    seed: int = 0,
    l2: float = 1.0,
    ns: tuple[int, ...] = (5, 10),
    min_total: int = MIN_TOTAL,
    min_class: int = MIN_CLASS,
) -> Calibration:
    y = np.array([s.label for s in samples], dtype=int)
    n, n_pos = len(samples), int(y.sum())
    n_neg = n - n_pos
    used = [k_ for k_ in signals if any(s.breakdown.get(k_) is not None for s in samples)]
    warnings: list[str] = []
    enough = n >= min_total and min(n_pos, n_neg) >= min_class
    if not enough:
        warnings.append(
            f"Poucos dados: {n} revisados ({n_pos} aprovados, {n_neg} rejeitados). Para sugerir "
            f"pesos, revise pelo menos ~{min_total} candidatos com ~{min_class} de cada classe. "
            "A análise abaixo é só indicativa: NÃO aplique estes pesos ainda."
        )
    current = (
        {k_: float(np.mean([_norm(s.weights, list(signals))[k_] for s in samples]))
         for k_ in signals} if samples else {k_: 0.0 for k_ in signals}
    )  # fmt: skip
    cal = Calibration(n, n_pos, n_neg, enough, warnings, list(signals), None, current, {},
                      Metrics(), seed=seed)  # fmt: skip
    if n_pos == 0 or n_neg == 0 or not used:
        warnings.append("É preciso ter aprovados E rejeitados (com detalhamento por sinal).")
        return cal

    X, _fill = _design(samples, used)
    Z, _mu, _sd = standardize(X)
    coef, _b = fit_logistic(Z, y, l2)
    cal.importance = {k_: round(float(c), 3) for k_, c in zip(used, coef, strict=True)}
    w = suggest_weights(X, y, l2)
    if w is None:
        warnings.append("Nenhum sinal ajudou a separar aprovados de rejeitados: sem sugestão.")
    else:
        sw = dict(zip(used, w.tolist(), strict=True))
        cal.weights = {k_: round(sw.get(k_, 0.0), 4) for k_ in signals}

    before = np.array([linear_score(s.breakdown, s.weights) for s in samples])
    cal.curve.auc_before = auc(before, y)
    cal.curve.precision_before = {m: precision_at(before, y, m) for m in ns if m <= n}
    kf = min(k, n_pos, n_neg)
    if kf >= 2:
        folds = stratified_folds(y, kf, seed)
        cal.folds = kf
        after = _cv_scores(samples, used, y, folds, l2, before)
        cal.curve.auc_after = auc(after, y)
        cal.curve.precision_after = {m: precision_at(after, y, m) for m in ns if m <= n}
    else:
        warnings.append("Poucos exemplos de uma das classes para validação cruzada.")

    # --- detection x AI ---
    idx = [i for i, s in enumerate(samples) if s.llm is not None]
    cal.n_llm = len(idx)
    if idx:
        cal.current_ranking = {
            "w_detect": float(np.mean([samples[i].w_detect for i in idx])),
            "w_llm": float(np.mean([samples[i].w_llm for i in idx])),
        }
    yl = y[idx] if idx else np.array([], dtype=int)
    if len(idx) >= 4 and 0 < int(yl.sum()) < len(idx):
        det = before[idx]
        llm10 = np.array([10.0 * float(samples[i].llm) for i in idx])
        XL = np.column_stack([det, llm10])
        wl = suggest_weights(XL, yl, l2)
        if wl is not None:
            cal.ranking = {"w_detect": round(float(wl[0]), 3), "w_llm": round(float(wl[1]), 3)}
        b_before = np.array([blend(det[j], llm10[j], samples[i].w_detect, samples[i].w_llm)
                             for j, i in enumerate(idx)])  # fmt: skip
        m = Metrics(auc(b_before, yl))
        m.precision_before = {q: precision_at(b_before, yl, q) for q in ns if q <= len(idx)}
        kl = min(k, int(yl.sum()), len(idx) - int(yl.sum()))
        if kl >= 2:
            b_after = b_before.copy()
            for test in stratified_folds(yl, kl, seed):
                train = np.setdiff1d(np.arange(len(idx)), test)
                if len(set(yl[train].tolist())) < 2:
                    continue
                wf = suggest_weights(XL[train], yl[train], l2)
                if wf is not None:
                    b_after[test] = [blend(det[j], llm10[j], wf[0], wf[1]) for j in test]
            m.auc_after = auc(b_after, yl)
            m.precision_after = {q: precision_at(b_after, yl, q) for q in ns if q <= len(idx)}
        cal.blend = m
    elif idx:
        warnings.append("Poucos candidatos com nota da IA (e das duas classes) para o peso da IA.")
    return cal


def preset_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalized weights -> preset scale (largest = 2.0, steps of 0.05, like the built-ins).
    The detector renormalizes, so only the proportions matter."""
    top = max(weights.values(), default=0.0)
    if top <= 0:
        return dict.fromkeys(weights, 1.0)
    return {k: round(round(v / top * 2.0 * 20) / 20, 2) for k, v in weights.items()}


# --- text output ---------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def _auc(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def format_text(cal: Calibration, labels: dict[str, str] | None = None, scope: str = "") -> str:
    lab = labels or {}
    out = [f"Calibração a partir das suas decisões{(' · ' + scope) if scope else ''}",
           f"{cal.n} revisados: {cal.n_pos} aprovados, {cal.n_neg} rejeitados "
           f"(pulados e sugeridos não entram).", ""]  # fmt: skip
    for w in cal.warnings:
        out.append(f"AVISO: {w}")
    if cal.warnings:
        out.append("")
    out.append("Pesos da curva de interesse (normalizados, soma 1):")
    out.append(f"  {'sinal':<10} {'atual':>7} {'sugerido':>9} {'efeito (coef. padronizado)':>28}")
    for k in cal.signals:
        cur = cal.current_weights.get(k, 0.0)
        sug = "—" if cal.weights is None else f"{cal.weights.get(k, 0.0):.2f}"
        imp = cal.importance.get(k)
        out.append(f"  {lab.get(k, k):<10} {cur:>7.2f} {sug:>9} "
                   f"{'sem dados' if imp is None else f'{imp:+.2f}':>28}")  # fmt: skip
    c = cal.curve
    out.append("")
    out.append(f"Seus vereditos vs. nota de detecção (validação cruzada {cal.folds}-fold, "
               f"semente {cal.seed}):")  # fmt: skip
    out.append(f"  AUC        atual {_auc(c.auc_before)} → sugerido {_auc(c.auc_after)}")
    for m in sorted(c.precision_before):
        out.append(f"  precisão@{m:<2} atual {_pct(c.precision_before[m])} → sugerido "
                   f"{_pct(c.precision_after.get(m))}")  # fmt: skip
    out.append("")
    if cal.blend is not None:
        b, cr, r = cal.blend, cal.current_ranking or {}, cal.ranking
        out.append(f"Detecção × IA ({cal.n_llm} revisados com nota da IA):")
        now = f"detecção {cr.get('w_detect', 0):.2f} / IA {cr.get('w_llm', 0):.2f}"
        sug = f"detecção {r['w_detect']:.2f} / IA {r['w_llm']:.2f}" if r else "sem sugestão"
        out.append(f"  pesos atuais {now} → sugeridos {sug}")
        out.append(f"  AUC        atual {_auc(b.auc_before)} → sugerido {_auc(b.auc_after)}")
        for m in sorted(b.precision_before):
            out.append(f"  precisão@{m:<2} atual {_pct(b.precision_before[m])} → sugerido "
                       f"{_pct(b.precision_after.get(m))}")  # fmt: skip
    else:
        out.append(f"Detecção × IA: {cal.n_llm} revisados com nota da IA (poucos para ajustar).")
    out.append("")
    if cal.can_apply:
        out.append("Dados suficientes. Para aplicar, use o card “Pesos sugeridos pela calibração” "
                   "na página da fonte (cria um preset novo e reanalisa; dá para reverter). "
                   "Nada é aplicado automaticamente.")  # fmt: skip
    else:
        out.append("Não sugiro aplicar estes pesos agora (dados insuficientes ou sem sugestão).")
    return "\n".join(out)
