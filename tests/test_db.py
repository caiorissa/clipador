from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlmodel import SQLModel, select

from clipador.config import Settings
from clipador.models import Job, JobStatus, Source, Streamer, StreamerChannel
from clipador.utils import format_duration, parse_timecode


def test_migrations_match_models(engine):
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"render_as_batch": True})
        diff = compare_metadata(ctx, SQLModel.metadata)
    assert diff == []


def test_settings_paths_are_absolute_and_relative_paths_portable(settings: Settings):
    assert settings.data_dir.is_absolute()
    f = settings.vods_dir / "1" / "video.mp4"
    rel = settings.rel_path(f)
    assert rel == "vods/1/video.mp4"
    assert settings.abs_path(rel) == f


def test_streamer_channels_and_source(session):
    s = Streamer(display_name="Fulano", permission_confirmed=True)
    s.channels.append(StreamerChannel(platform="twitch", channel_handle="fulano"))
    s.channels.append(StreamerChannel(platform="kick", channel_handle="fulano"))
    session.add(s)
    session.commit()
    src = Source(streamer_id=s.id, platform="twitch", kind="vod", external_id="1", url="u")
    session.add(src)
    session.commit()
    job = Job(type="ingest", source_id=src.id, payload={"a": 1})
    session.add(job)
    session.commit()

    loaded = session.exec(select(Streamer)).one()
    assert [c.platform for c in loaded.channels] == ["twitch", "kick"]
    assert session.get(Job, job.id).payload == {"a": 1}
    assert session.get(Job, job.id).status == JobStatus.QUEUED
    assert loaded.facecam_rect is None


def test_parse_timecode():
    assert parse_timecode("") is None
    assert parse_timecode("90") == 90
    assert parse_timecode("1:30") == 90
    assert parse_timecode("1:02:03") == 3723
    assert parse_timecode("1h2m3s") == 3723
    assert parse_timecode("15m") == 900
    for bad in ("abc", "-5", "1:2:3:4"):
        try:
            parse_timecode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    assert format_duration(3723) == "1:02:03"
    assert format_duration(65) == "1:05"
