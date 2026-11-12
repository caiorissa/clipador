import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from clipador.config import Settings
from clipador.errors import JobCanceled
from clipador.services.downloader import (
    DownloadErrorKind,
    ProgressTracker,
    YtDlp,
    classify_ytdlp_error,
    parse_progress_line,
)

# --- progress parsing -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("[download]   0.0% of  250.00MiB at  Unknown B/s ETA Unknown", 0.0),
        ("[download]  42.7% of ~  1.23GiB at    5.10MiB/s ETA 03:21 (frag 52/120)", 42.7),
        ("[download] 100% of  250.00MiB in 00:00:41 at 6.04MiB/s", 100.0),
        ("[download] 100.0% of   10.00MiB", 100.0),
        ("[download] Destination: data\\vods\\1\\video.f137.mp4", None),
        ("[youtube] abc: Downloading webpage", None),
        ("[hlsnative] Total fragments: 120", None),
        ("", None),
    ],
)
def test_parse_progress_line(line, expected):
    assert parse_progress_line(line) == expected


def test_progress_tracker_video_plus_audio_is_monotonic():
    t = ProgressTracker()
    lines = [
        "[download] Destination: video.f137.mp4",
        "[download]  50.0% of 100MiB",
        "[download] 100% of 100MiB in 00:10",
        "[download] Destination: video.f140.m4a",
        "[download]   0.0% of 5MiB",
        "[download] 100% of 5MiB in 00:01",
        '[Merger] Merging formats into "video.mp4"',
    ]
    values = [u[0] for u in (t.feed(ln) for ln in lines) if u is not None]
    assert values == sorted(values)
    assert values[0] == pytest.approx(42.5)
    assert values[1] == pytest.approx(85.0)
    assert max(values) <= 99.0


def test_progress_tracker_ignores_thumbnail_destination():
    t = ProgressTracker()
    t.feed("[info] Writing video thumbnail original to: thumb.webp")
    t.feed("[download] Destination: thumb.jpg")
    t.feed("[download] Destination: video.mp4")
    assert t.feed("[download]  10.0% of 1GiB") == (pytest.approx(8.5), None)


# --- error classification -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "kind", "retryable"),
    [
        (
            "ERROR: [youtube] abc: Sign in to confirm you’re not a bot. Use --cookies-from-browser",
            DownloadErrorKind.ANTI_BOT,
            True,
        ),
        (
            "ERROR: [kick] xyz: Unable to download JSON metadata: HTTP Error 403: Forbidden",
            DownloadErrorKind.ANTI_BOT,
            True,
        ),
        ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", DownloadErrorKind.PRIVATE, False),
        (
            "ERROR: [twitch:vod] 123: This video is only available for subscribers",
            DownloadErrorKind.PRIVATE,
            False,
        ),
        (
            "ERROR: [youtube] abc: Join this channel to get access to members-only content",
            DownloadErrorKind.PRIVATE,
            False,
        ),
        (
            "ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate. "
            "Use --cookies-from-browser or --cookies for the authentication.",
            DownloadErrorKind.PRIVATE,
            False,
        ),
        ("ERROR: [youtube] abc: Video unavailable", DownloadErrorKind.NOT_FOUND, False),
        ("ERROR: [twitch:vod] 1: Unable to download JSON metadata: HTTP Error 404: Not Found", DownloadErrorKind.NOT_FOUND, False),
        ("ERROR: Unsupported URL: https://example.com/x", DownloadErrorKind.INVALID_URL, False),
        (
            "ERROR: [twitch:vod] 1: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>",
            DownloadErrorKind.NETWORK,
            True,
        ),
        ("ERROR: unable to download video data: The read operation timed out", DownloadErrorKind.NETWORK, True),
        (
            "ERROR: [kick] abc: Unable to extract video; please report this issue on https://github.com/yt-dlp/yt-dlp/issues",
            DownloadErrorKind.EXTRACTOR,
            False,
        ),
        ("ERROR: [youtube] abc: This live event will begin in 2 hours.", DownloadErrorKind.LIVE, False),
        ("ERROR: unable to write data: [Errno 28] No space left on device", DownloadErrorKind.DISK, False),
        ("something totally unexpected", DownloadErrorKind.UNKNOWN, True),
    ],
)  # fmt: skip
def test_classify_ytdlp_error(output, kind, retryable):
    err = classify_ytdlp_error(output)
    assert err.kind == kind
    assert err.retryable is retryable
    assert err.message  # PT-BR message present
    assert err.detail


# --- command building / download with a fake yt-dlp ---------------------------------------


def test_download_args_include_cookies_and_height(tmp_path):
    s = Settings(data_dir=tmp_path, cookies_from_browser="firefox", _env_file=None)
    args = YtDlp(s).download_args("https://twitch.tv/videos/1", tmp_path)
    assert args[:3] == [sys.executable, "-m", "yt_dlp"]
    assert "--cookies-from-browser" in args and "firefox" in args
    assert "res:1080,vcodec:h264,acodec:m4a" in args
    assert args[-2:] == ["--", "https://twitch.tv/videos/1"]


def test_cookies_file_takes_precedence(tmp_path):
    s = Settings(
        data_dir=tmp_path, cookies_from_browser="chrome", cookies_file=tmp_path / "c.txt",
        _env_file=None,
    )  # fmt: skip
    args = YtDlp(s).common_args()
    assert "--cookies" in args and "--cookies-from-browser" not in args


FAKE_YTDLP = """
import sys, time, pathlib
out = None
for i, a in enumerate(sys.argv):
    if a == "-o" and not sys.argv[i + 1].startswith("thumbnail:"):
        out = sys.argv[i + 1]
mode = pathlib.Path(out).parent.joinpath("mode.txt").read_text()
if mode == "fail":
    print("ERROR: [youtube] abc: Private video")
    sys.exit(1)
print("[download] Destination: video.mp4", flush=True)
for p in (10, 50, 100):
    print(f"[download] {p}.0% of 10MiB", flush=True)
    if mode == "slow":
        time.sleep(0.5)
pathlib.Path(out.replace("%(ext)s", "mp4")).write_bytes(b"x")
"""


def _fake_settings(tmp_path: Path) -> Settings:
    script = tmp_path / "fake_ytdlp.py"
    script.write_text(FAKE_YTDLP, encoding="utf-8")
    return Settings(data_dir=tmp_path / "data", _env_file=None)


def _patched(yt: YtDlp, tmp_path: Path):
    return mock.patch.object(
        YtDlp, "base_cmd", return_value=[sys.executable, str(tmp_path / "fake_ytdlp.py")]
    )


def test_download_reports_progress(tmp_path):
    yt = YtDlp(_fake_settings(tmp_path))
    dest = tmp_path / "dl"
    dest.mkdir()
    (dest / "mode.txt").write_text("ok")
    seen = []
    with _patched(yt, tmp_path):
        res = yt.download("https://youtu.be/abcdefghijk", dest, lambda p, m: seen.append(p))
    assert res.video_path == dest / "video.mp4"
    assert res.thumbnail_path is None
    assert seen == sorted(seen) and seen[-1] == pytest.approx(85.0)


def test_download_failure_is_classified(tmp_path):
    yt = YtDlp(_fake_settings(tmp_path))
    dest = tmp_path / "dl"
    dest.mkdir()
    (dest / "mode.txt").write_text("fail")
    with _patched(yt, tmp_path), pytest.raises(Exception) as exc:
        yt.download("https://youtu.be/abcdefghijk", dest, lambda p, m: None)
    assert exc.value.kind == DownloadErrorKind.PRIVATE


def test_download_cancel_terminates_process(tmp_path):
    yt = YtDlp(_fake_settings(tmp_path))
    dest = tmp_path / "dl"
    dest.mkdir()
    (dest / "mode.txt").write_text("slow")

    def cb(p, m):
        if p > 0:
            raise JobCanceled()

    with (
        _patched(yt, tmp_path),
        mock.patch("clipador.services.downloader._terminate", wraps=None) as term,
        pytest.raises(JobCanceled),
    ):
        yt.download("https://youtu.be/abcdefghijk", dest, cb)
    proc = term.call_args.args[0]
    assert isinstance(proc, subprocess.Popen)
    proc.kill()
    proc.wait()
