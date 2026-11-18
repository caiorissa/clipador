"""/publicar: connected accounts (several YouTube channels, TikTok), their auto-post grid,
the posting queue (scheduled / held / sending), history with numbers, and "Publicar" by hand
for one approved clip (steps 6 and 6B)."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ClipadorError, ValidationError
from clipador.models import (
    Candidate,
    Publication,
    PublicationStatus,
    SocialAccount,
    Source,
    Streamer,
)
from clipador.render import variants
from clipador.services import accounts, publishing, tiktok, youtube
from clipador.utils import utcnow
from clipador.web.deps import get_session, get_settings_dep, redirect, render
from clipador.worker import queue

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
# state -> (platform, verifier, redirect uri, Google project client id or "")
_PENDING_AUTH: dict[str, tuple[str, str, str, str]] = {}
QUEUE_STATUSES = (
    PublicationStatus.HELD,
    PublicationStatus.SCHEDULED,
    PublicationStatus.QUEUED,
    PublicationStatus.UPLOADING,
)


def _setup_error(fn, settings) -> str | None:
    try:
        fn(settings)
        return None
    except ClipadorError as e:
        return e.message


def _projects(session: Session, settings: Settings, accs: list[SocialAccount]) -> list[dict]:
    """Google Cloud projects: which channels use each one and today's upload quota."""
    try:
        clients = youtube.load_clients(settings)
    except ClipadorError:
        return []
    now = utcnow()
    return [{"client": c, "label": c.name or c.client_id[:12],
             "accounts": [a for a in accs if a.project_id == c.client_id and a.active],
             "used": accounts.project_uploads_today(session, c.client_id, now),
             "cap": settings.youtube_uploads_per_project_day} for c in clients]  # fmt: skip


def _ctx(session: Session, settings: Settings, cand_id: int | None) -> dict:
    accounts.import_legacy_youtube(session, settings)
    session.commit()
    accs = accounts.accounts(session)
    pubs = session.exec(select(Publication).order_by(col(Publication.id).desc()).limit(300)).all()
    queue_rows = sorted(
        [p for p in pubs if p.status in QUEUE_STATUSES],
        key=lambda p: (
            p.status != PublicationStatus.HELD,
            accounts.aware(p.scheduled_for) or utcnow(),
        ),
    )
    history = [p for p in pubs if p.status not in QUEUE_STATUSES][:100]
    cands = {
        c.id: c
        for c in session.exec(
            select(Candidate).where(
                col(Candidate.id).in_(
                    {p.candidate_id for p in pubs} | ({cand_id} if cand_id else set())
                )
            )
        ).all()
    }
    streamers = session.exec(select(Streamer).order_by(Streamer.display_name)).all()
    form = None
    if cand_id and cand_id in cands:
        c = cands[cand_id]
        src = session.get(Source, c.source_id)
        form = {
            "candidate": c,
            "source": src,
            "draft": publishing.youtube_draft(session, c.id),
            "tiktok_draft": accounts.draft_for(session, c.id, "tiktok"),
            "render": publishing.final_render(session, settings, c.id),
        }
    totals = {a.id: sum(p.views or 0 for p in pubs if p.account_id == a.id) for a in accs}
    return {
        "accs": accs,
        "acc_by_id": {a.id: a for a in accs},
        "queue_rows": queue_rows,
        "history": history,
        "cands": cands,
        "streamers": streamers,
        "streamer_names": {s.id: s.display_name for s in streamers},
        "form": form,
        "totals": totals,
        "yt_error": _setup_error(youtube.load_client, settings),
        "projects": _projects(session, settings, accs),
        "tt_error": _setup_error(tiktok.load_client, settings),
        "privacy_labels": youtube.PRIVACY_LABELS,
        "platform_labels": accounts.PLATFORM_LABELS,
        "status_labels": {
            "scheduled": "agendado",
            "held": "segurado",
            "queued": "na fila",
            "uploading": "enviando",
            "published": "publicado",
            "failed": "falhou",
            "canceled": "cancelado",
        },
        "tiktok_redirect": None,
        "variants": variants,
    }


def _page(
    request: Request,
    session: Session,
    settings: Settings,
    cand_id=None,
    status_code: int = 200,
    **extra,
):
    ctx = _ctx(session, settings, cand_id)
    ctx["tiktok_redirect"] = _redirect_uri(request, settings, "tiktok")
    ctx.update(extra)
    return render(request, "publish/index.html", status_code=status_code, **ctx)


def _redirect_uri(request: Request, settings: Settings, platform: str) -> str:
    """Where the platform sends the user back. TikTok requires https, so a relay page can be
    configured (CLIPADOR_TIKTOK_REDIRECT_URI); it must be identical in the authorize and the
    token requests (the pending state keeps it)."""
    if platform == "tiktok" and settings.tiktok_redirect_uri.strip():
        return settings.tiktok_redirect_uri.strip()
    return str(request.url_for("oauth_callback", platform=platform))


@router.get("/publicar")
def publish_page(
    request: Request, session: SessionDep, settings: SettingsDep, candidato: int | None = None
):
    return _page(request, session, settings, candidato)


# --- connecting accounts ----------------------------------------------------------------------


@router.get("/publicar/{platform}/conectar")
def connect(request: Request, platform: str, settings: SettingsDep, projeto: str = ""):
    mod = {"youtube": youtube, "tiktok": tiktok}.get(platform)
    if mod is None:
        return redirect("/publicar", "Plataforma desconhecida.")
    try:
        if platform == "youtube":
            client = youtube.load_client(settings, projeto or None)
            projeto = client.client_id
        else:
            client = tiktok.load_client(settings)
    except ClipadorError as e:
        return redirect("/publicar", e.message)
    verifier, challenge = mod.new_pkce()
    state = secrets.token_urlsafe(24)
    redirect_uri = _redirect_uri(request, settings, platform)
    _PENDING_AUTH[state] = (platform, verifier, redirect_uri, projeto)
    if platform == "tiktok":
        url = tiktok.auth_url(client, redirect_uri, state, challenge, settings.tiktok_mode)
    else:
        url = youtube.auth_url(client, redirect_uri, state, challenge)
    return RedirectResponse(url, 303)


@router.get("/publicar/{platform}/callback", name="oauth_callback")
def oauth_callback(
    platform: str,
    session: SessionDep,
    settings: SettingsDep,
    state: str = "",
    code: str = "",
    error: str = "",
):
    pending = _PENDING_AUTH.pop(state, None)
    if error or pending is None or not code or pending[0] != platform:
        why = f" ({error})" if error else ""
        return redirect("/publicar", f"Conexão cancelada{why}.")
    _p, verifier, redirect_uri, project = pending
    try:
        if platform == "tiktok":
            tok = tiktok.exchange_code(settings, code, redirect_uri, verifier)
            who = tok.pop("user")
        else:
            tok = youtube.exchange_code(
                settings, code, redirect_uri, verifier, client_id=project or None
            )
            who = tok.pop("channel")
        acc = accounts.upsert_account(session, platform, who["id"], who["title"], tok)
        session.commit()
    except ClipadorError as e:
        session.rollback()
        return redirect("/publicar", e.message)
    return redirect(
        "/publicar",
        f"{accounts.PLATFORM_LABELS[platform]} conectado: {acc.name}. "
        "Configure a postagem automática da conta abaixo.",
    )


# --- account settings -------------------------------------------------------------------------


def _account(session: Session, acc_id: int) -> SocialAccount:
    acc = session.get(SocialAccount, acc_id)
    if acc is None:
        raise ValidationError("Conta não encontrada.")
    return acc


@router.post("/publicar/contas/{acc_id}")
async def save_account(request: Request, acc_id: int, session: SessionDep, settings: SettingsDep):
    form = await request.form()
    try:
        acc = _account(session, acc_id)
        autopost = form.get("autopost") == "on"
        if autopost and not acc.autopost:
            acc.autopost_since = utcnow()  # only clips approved from now on
        acc.autopost = autopost
        acc.paused = form.get("paused") == "on"
        acc.post_now = form.get("post_now") == "on"
        acc.hold_on_alerts = form.get("hold_on_alerts") == "on"
        acc.slots = accounts.parse_slots(str(form.get("slots", "")))
        cap = accounts.TIKTOK_MAX_PER_DAY if acc.platform == "tiktok" else 50
        acc.max_per_day = max(1, min(cap, int(form.get("max_per_day") or 3)))
        acc.delay_minutes = max(
            0,
            min(
                7 * 24 * 60, round(float(str(form.get("delay_hours") or 0).replace(",", ".")) * 60)
            ),
        )
        acc.min_gap_minutes = max(0, min(1440, int(form.get("min_gap_minutes") or 0)))
        privacy = str(form.get("privacy", acc.privacy))
        if privacy not in youtube.PRIVACY:
            raise ValidationError("Visibilidade inválida.")
        acc.privacy = privacy
        acc.streamer_ids = sorted(int(v) for v in form.getlist("streamer_ids") if str(v).isdigit())
        if form.get("style_form"):
            acc.render_style = variants.normalize(
                {
                    "layout": form.get("st_layout", ""),
                    "caption_style": form.get("st_caption", ""),
                    "grade": form.get("st_grade", "nenhum"),
                    "speed": float(str(form.get("st_speed") or 1).replace(",", ".")),
                    "zoom": float(str(form.get("st_zoom") or 1).replace(",", ".")),
                    "foco_x": float(str(form.get("st_foco") or 0.5).replace(",", ".")),
                    "trim_start": float(str(form.get("st_trim") or 0).replace(",", ".")),
                    "opening": form.get("st_opening", "auto"),
                    "credit": form.get("st_credit") == "on",
                    "peak_zoom": form.get("st_peak") == "on",
                }
            )
        if not acc.slots and not acc.post_now:
            raise ValidationError(
                "Defina pelo menos um horário, ou marque 'postar assim que aprovar'."
            )
        acc.updated_at = utcnow()
        session.add(acc)
        session.commit()
    except (ValidationError, ValueError) as e:
        session.rollback()
        return _page(
            request,
            session,
            settings,
            status_code=400,
            error=getattr(e, "message", "Valor inválido."),
        )
    state = "ligada" if acc.autopost else "desligada"
    return redirect("/publicar", f"{acc.name}: configuração salva (postagem automática {state}).")


@router.post("/publicar/contas/{acc_id}/sortear-estilo")
def randomize_style(acc_id: int, session: SessionDep):
    acc = _account(session, acc_id)
    others = [a.render_style for a in accounts.accounts(session) if a.id != acc.id]
    acc.render_style = variants.suggest([*others, acc.render_style, {}])
    acc.updated_at = utcnow()
    session.add(acc)
    session.commit()
    return redirect(
        "/publicar", f"{acc.name}: novo estilo de edição — {variants.describe(acc.render_style)}."
    )


@router.post("/publicar/contas/{acc_id}/desconectar")
def disconnect_account(acc_id: int, session: SessionDep):
    acc = _account(session, acc_id)
    acc.active, acc.autopost, acc.credentials = False, False, {}
    session.add(acc)
    for p in session.exec(
        select(Publication).where(
            Publication.account_id == acc_id,
            col(Publication.status).in_([PublicationStatus.SCHEDULED, PublicationStatus.HELD]),
        )
    ):
        p.status = PublicationStatus.CANCELED
        session.add(p)
    session.commit()
    return redirect("/publicar", f"{acc.name} desconectada (token apagado; agendados cancelados).")


@router.post("/publicar/pausar-tudo")
def pause_all(session: SessionDep, pausar: Annotated[str, Form()] = "1"):
    paused = pausar == "1"
    for acc in accounts.accounts(session):
        acc.paused = paused
        session.add(acc)
    session.commit()
    return redirect("/publicar", "Todas as contas pausadas." if paused else "Contas retomadas.")


# --- the queue --------------------------------------------------------------------------------


@router.post("/publicar/fila/{pub_id}/{action}")
def queue_action(
    request: Request, pub_id: int, action: str, session: SessionDep, settings: SettingsDep
):
    try:
        if action == "liberar":
            p = accounts.release(session, settings, pub_id)
            msg = f"Liberado: vai em {p.scheduled_for.astimezone():%d/%m %H:%M}."
        elif action == "agora":
            accounts.post_now(session, pub_id)
            msg = "Vai ser enviado no próximo minuto (o worker precisa estar rodando)."
        elif action == "cancelar":
            accounts.cancel(session, pub_id)
            msg = "Postagem cancelada."
        else:
            raise ValidationError("Ação desconhecida.")
        session.commit()
    except ClipadorError as e:
        session.rollback()
        return _page(request, session, settings, status_code=400, error=e.message)
    return redirect("/publicar", msg)


@router.post("/publicar/youtube/numeros")
def refresh_numbers(session: SessionDep):
    queue.enqueue(session, "youtube_metrics", None, {})
    session.commit()
    return redirect("/publicar", "Atualização dos números na fila.")


# --- publishing one clip by hand --------------------------------------------------------------


@router.post("/candidatos/{cand_id}/publicar")
def publish_candidate(
    request: Request,
    cand_id: int,
    session: SessionDep,
    settings: SettingsDep,
    account_id: Annotated[int, Form()] = 0,
    privacy: Annotated[str, Form()] = "private",
    publish_at: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "",
    confirm: Annotated[str | None, Form()] = None,
):
    if confirm is None:
        return _page(
            request,
            session,
            settings,
            cand_id,
            status_code=400,
            error="Confirme que revisou o clip (direitos, música e crédito).",
        )
    try:
        when = publishing.parse_publish_at(publish_at)
        pub = publishing.request_publication(
            session,
            settings,
            cand_id,
            account_id,
            privacy=privacy,
            publish_at=when,
            title=title or None,
        )
        session.commit()
    except ClipadorError as e:
        session.rollback()
        return _page(request, session, settings, cand_id, status_code=400, error=e.message)
    what = "agendado" if pub.status == PublicationStatus.SCHEDULED else "na fila de envio"
    return redirect("/publicar", f"'{pub.title}' {what}.")


@router.post("/candidatos/{cand_id}/publicar/youtube")
def publish_candidate_legacy(
    request: Request,
    cand_id: int,
    session: SessionDep,
    settings: SettingsDep,
    privacy: Annotated[str, Form()] = "private",
    publish_at: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "",
    confirm: Annotated[str | None, Form()] = None,
):
    yt = next(
        (a for a in accounts.accounts(session, active_only=True) if a.platform == "youtube"), None
    )
    return publish_candidate(
        request, cand_id, session, settings, yt.id if yt else 0, privacy, publish_at, title, confirm
    )
