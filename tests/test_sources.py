from datetime import date

import pytest

from clipador.errors import ValidationError
from clipador.models import SourceStatus, Streamer
from clipador.services.sources import (
    SourceDraft,
    add_channel,
    channel_warning,
    create_source,
    suggest_credit,
)

VOD_URL = "https://www.twitch.tv/videos/2283746281"


def _streamer(session, *, confirmed: bool, name="Fulano") -> Streamer:
    s = Streamer(
        display_name=name,
        permission_confirmed=confirmed,
        permission_date=date(2026, 11, 10) if confirmed else None,
        permission_note="liberou no Discord em 10/11" if confirmed else "",
    )
    session.add(s)
    session.flush()
    add_channel(session, s, "twitch", "https://twitch.tv/Fulano")
    add_channel(session, s, "kick", "fulano")
    session.commit()
    return s


def test_source_blocked_without_permission(session):
    s = _streamer(session, confirmed=False)
    with pytest.raises(ValidationError) as exc:
        create_source(session, SourceDraft(url=VOD_URL, streamer_id=s.id))
    assert "autorização" in exc.value.message
    assert "Fulano" in exc.value.message


def test_source_created_with_permission(session):
    s = _streamer(session, confirmed=True)
    src, _parsed = create_source(
        session, SourceDraft(url=VOD_URL, streamer_id=s.id, start_offset=60, end_offset=600)
    )
    session.commit()
    assert src.id and src.status == SourceStatus.PENDING
    assert (src.platform, src.kind, src.external_id) == ("twitch", "vod", "2283746281")
    assert (src.start_offset, src.end_offset) == (60, 600)


def test_duplicate_source_blocked(session):
    s = _streamer(session, confirmed=True)
    create_source(session, SourceDraft(url=VOD_URL, streamer_id=s.id))
    session.commit()
    with pytest.raises(ValidationError, match="já foi adicionada"):
        create_source(session, SourceDraft(url=VOD_URL + "?t=10s", streamer_id=s.id))


def test_invalid_range_blocked(session):
    s = _streamer(session, confirmed=True)
    with pytest.raises(ValidationError, match="fim do trecho"):
        create_source(
            session, SourceDraft(url=VOD_URL, streamer_id=s.id, start_offset=100, end_offset=50)
        )


def test_inactive_streamer_blocked(session):
    s = _streamer(session, confirmed=True)
    s.active = False
    session.commit()
    with pytest.raises(ValidationError, match="desativado"):
        create_source(session, SourceDraft(url=VOD_URL, streamer_id=s.id))


def test_channel_mismatch_warns_but_does_not_block(session):
    s = _streamer(session, confirmed=True)
    src, _ = create_source(
        session,
        SourceDraft(url="https://kick.com/outra-pessoa/clips/clip_01ABC", streamer_id=s.id),
    )
    assert src.id
    assert src.warning and "não bate" in src.warning


def test_channel_match_has_no_warning(session):
    s = _streamer(session, confirmed=True)
    src, _ = create_source(
        session, SourceDraft(url="https://kick.com/Fulano/clips/clip_01ABC", streamer_id=s.id)
    )
    assert src.warning is None
    assert channel_warning(s, "youtube", ["x"]) is not None  # no YouTube channel registered


def test_channel_unique_across_streamers(session):
    _streamer(session, confirmed=True)
    other = Streamer(display_name="Outro")
    session.add(other)
    session.flush()
    with pytest.raises(ValidationError, match="já está cadastrado"):
        add_channel(session, other, "twitch", "@fulano")


def test_suggest_credit(session):
    s = _streamer(session, confirmed=True)
    assert suggest_credit(s) == "Clip de @fulano na Twitch"
