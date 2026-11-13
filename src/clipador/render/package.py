"""Post package text file written next to each rendered clip (pure)."""

from __future__ import annotations

from clipador.llm.drafts import DRAFT_LABELS, DRAFT_PLATFORMS, DraftData


def package_text(drafts: dict[str, DraftData], clip_name: str, alerts: list[dict] | None) -> str:
    lines = [f"Pacote de postagem · {clip_name}", ""]
    if alerts:
        lines.append("⚠ ALERTAS DA IA (revise antes de postar):")
        lines += [f"  - {a.get('tipo')}: {a.get('trecho', '')}" for a in alerts]
        lines.append("")
    lines.append("⚠ Confira se não há música com direitos autorais tocando no fundo.")
    lines.append("")
    for p in DRAFT_PLATFORMS:
        d = drafts.get(p)
        if d is None:
            continue
        lines.append(f"===== {DRAFT_LABELS[p]} =====")
        if d.title:
            lines.append(f"Título: {d.title}")
        lines.append("Descrição:" if p == "youtube" else "Legenda:")
        lines.append(d.description)
        lines.append("Hashtags: " + " ".join(d.hashtags))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
