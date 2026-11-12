"""Business rules for streamers and sources (used by the web routes and tests)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlmodel import Session, select

from clipador.errors import ValidationError
from clipador.models import PLATFORM_LABELS, Platform, Source, Streamer, StreamerChannel
from clipador.platforms.base import ParsedUrl, PlatformAdapter, normalize_handle
from clipador.platforms.registry import parse_url
from clipador.utils import utcnow

PERMISSION_REQUIRED_MESSAGE = (
    "{name} ainda não tem autorização para clipes confirmada. O Clipador só aceita VODs de "
    "streamers que liberaram clipes: abra o cadastro do streamer, marque 'Autorização "
    "confirmada' e registre a data e onde ele liberou (ex.: 'liberou no Discord em 10/11')."
)


def suggest_credit(streamer: Streamer) -> str:
    """Default credit line for descriptions, based on the first registered channel."""
    if not streamer.channels:
        return f"Clip de {streamer.display_name}"
    ch = streamer.channels[0]
    label = PLATFORM_LABELS.get(Platform(ch.platform), ch.platform)
    prefix = (
        "@" if ch.platform != Platform.YOUTUBE or not ch.channel_handle.startswith("UC") else ""
    )
    return f"Clip de {prefix}{ch.channel_handle} na {label}"


def add_channel(
    session: Session, streamer: Streamer, platform: str, handle: str
) -> StreamerChannel:
    platform = Platform(platform)
    normalized = normalize_handle(handle)
    if not normalized:
        raise ValidationError("Informe o nome do canal (ex.: fulano ou https://twitch.tv/fulano).")
    existing = session.exec(
        select(StreamerChannel).where(
            StreamerChannel.platform == platform, StreamerChannel.channel_handle == normalized
        )
    ).first()
    if existing:
        if existing.streamer_id == streamer.id:
            return existing
        owner = session.get(Streamer, existing.streamer_id)
        raise ValidationError(
            f"O canal {normalized} ({PLATFORM_LABELS[platform]}) já está cadastrado para "
            f"{owner.display_name if owner else 'outro streamer'}."
        )
    ch = StreamerChannel(streamer_id=streamer.id, platform=platform, channel_handle=normalized)
    streamer.channels.append(ch)
    return ch


@dataclass
class SourceDraft:
    url: str
    streamer_id: int
    start_offset: float | None = None
    end_offset: float | None = None


def channel_warning(streamer: Streamer, platform: str, candidates: list[str]) -> str | None:
    """Non-blocking notice when the video's channel isn't one of the streamer's channels."""
    handles = [c.channel_handle for c in streamer.channels if c.platform == platform]
    label = PLATFORM_LABELS.get(Platform(platform), platform)
    if not handles:
        return (
            f"{streamer.display_name} não tem canal do {label} cadastrado; confira se este "
            "vídeo é mesmo dele."
        )
    if candidates and not PlatformAdapter.channel_matches(candidates, handles):
        return (
            f"O canal deste vídeo ({', '.join(candidates[:3])}) não bate com os canais do "
            f"{label} cadastrados para {streamer.display_name} ({', '.join(handles)}). "
            "Confira se é o streamer certo."
        )
    return None


def create_source(session: Session, draft: SourceDraft) -> tuple[Source, ParsedUrl]:
    """Validate and insert a Source. Raises ValidationError (PT-BR) when a rule is violated."""
    parsed = parse_url(draft.url)

    streamer = session.get(Streamer, draft.streamer_id)
    if streamer is None:
        raise ValidationError("Escolha um streamer.")
    if not streamer.active:
        raise ValidationError(f"{streamer.display_name} está desativado. Reative-o antes.")
    if not streamer.permission_confirmed:
        raise ValidationError(PERMISSION_REQUIRED_MESSAGE.format(name=streamer.display_name))

    start, end = draft.start_offset, draft.end_offset
    if start is not None and end is not None and end <= start:
        raise ValidationError("O fim do trecho precisa ser depois do início.")

    dup = session.exec(
        select(Source).where(
            Source.platform == parsed.platform,
            Source.kind == parsed.kind,
            Source.external_id == parsed.external_id,
        )
    ).first()
    if dup:
        raise ValidationError(
            f"Esta fonte já foi adicionada (fonte #{dup.id}). Use 'tentar de novo' no job se o "
            "download falhou."
        )

    warning = None
    if parsed.channel_handle:
        warning = channel_warning(streamer, parsed.platform, [parsed.channel_handle])

    source = Source(
        streamer_id=streamer.id,
        platform=parsed.platform,
        kind=parsed.kind,
        external_id=parsed.external_id,
        url=parsed.url,
        start_offset=start,
        end_offset=end,
        warning=warning,
    )
    session.add(source)
    session.flush()
    return source, parsed


def touch(obj) -> None:
    obj.updated_at = utcnow()
