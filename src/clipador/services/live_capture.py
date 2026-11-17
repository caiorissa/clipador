"""The live capture process (`clipador live-capture <id>`, step 5E).

Runs OUTSIDE the worker (a live lasts hours; it must never hold a worker slot). It:
1. asks yt-dlp for the stream URL (or plays a local file for `clipador live --simular`);
2. records it with ffmpeg in N-second segments (`-c copy -f segment`, no re-encoding);
3. every closed segment becomes a LiveSegment on the session's GLOBAL timeline (sum of the
   captured segments, so reconnecting never resets the clock) and a `live_analyze` job;
4. reconnects when the stream drops; stops when the live ends, the user clicks Parar, or free
   disk space falls under `live_min_free_gb`;
5. optionally reads the Twitch chat (anonymous, read-only IRC) into chat.jsonl.
A heartbeat in the DB lets the worker detect (and restart) a dead capture process.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, select

from clipador.config import Settings
from clipador.models import (
    LiveSegment,
    LiveSession,
    LiveStatus,
    SegmentStatus,
    Source,
)
from clipador.services import live
from clipador.utils import utcnow
from clipador.worker import queue

log = logging.getLogger(__name__)

SEGMENT_LIST = "segments.csv"
CHAT_FILE = "chat.jsonl"


# --- ffmpeg command ----------------------------------------------------------------------------


def segment_args(
    input_args: list[str], out_dir: Path, segment_seconds: int, start_number: int
) -> list[str]:
    """ffmpeg arguments that copy the input into numbered MPEG-TS segments and append each
    finished one to segments.csv (name,start,end). MPEG-TS survives an abrupt stop."""
    return [
        "-hide_banner", "-loglevel", "warning", "-y", *input_args,
        "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
        "-f", "segment", "-segment_time", str(segment_seconds), "-reset_timestamps", "1",
        "-segment_start_number", str(start_number),
        "-segment_list", str(out_dir / SEGMENT_LIST), "-segment_list_type", "csv",
        "-segment_list_flags", "+live",
        str(out_dir / live.SEGMENT_PATTERN),
    ]  # fmt: skip


def stream_input_args(url: str) -> list[str]:
    """HLS from Twitch/Kick/YouTube: let ffmpeg ride over short network hiccups by itself
    (the reconnect options only exist for http(s) inputs)."""
    if not url.lower().startswith(("http://", "https://")):
        return ["-i", url]
    return [
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_on_network_error", "1",
        "-reconnect_delay_max", "10", "-rw_timeout", "30000000", "-i", url,
    ]  # fmt: skip


def sim_input_args(path: Path, speed: float | None) -> list[str]:
    """A local file played at `speed`x real time (1 = like a real live)."""
    return ["-readrate", f"{max(0.1, float(speed or 1.0)):g}", "-i", str(path)]


@dataclass
class ClosedSegment:
    index: int
    name: str
    duration: float


def read_segment_list(path: Path) -> list[ClosedSegment]:
    """segments.csv rows -> closed segments (ffmpeg writes a row when a segment is complete)."""
    if not path.is_file():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            name = row[0].strip()
            try:
                index = int(Path(name).stem.split("_")[-1])
                duration = float(row[2]) - float(row[1])
            except ValueError:
                continue
            out.append(ClosedSegment(index, name, max(0.0, duration)))
    return out


# --- Twitch chat (anonymous IRC, read-only) ------------------------------------------------------


class TwitchChatReader(threading.Thread):
    """Reads #channel as an anonymous `justinfan` user (no login, cannot write) and appends
    {"t": global_seconds, "author", "text"} lines to chat.jsonl. `clock()` maps now to the
    session's global timeline."""

    def __init__(
        self,
        channel: str,
        out: Path,
        clock: Callable[[], float | None],
        stop: threading.Event,
        host: str = "irc.chat.twitch.tv",
        port: int = 6667,
    ):
        super().__init__(daemon=True, name="twitch-chat")
        self.channel = channel.lower().lstrip("#")
        self.out, self.clock, self.stop_event = out, clock, stop
        self.host, self.port = host, port
        self.count = 0

    @staticmethod
    def parse(line: str) -> tuple[str, str] | None:
        """':nick!nick@nick.tmi.twitch.tv PRIVMSG #chan :text' -> (nick, text)."""
        if " PRIVMSG " not in line:
            return None
        if line.startswith("@"):  # tags (not requested, but harmless)
            line = line.split(" ", 1)[1] if " " in line else ""
        prefix, _, rest = line.partition(" PRIVMSG ")
        _chan, _, text = rest.partition(" :")
        nick = prefix.lstrip(":").split("!", 1)[0]
        return (nick, text.strip()) if nick and text.strip() else None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._session()
            except OSError as e:
                log.info("Chat da Twitch caiu (%s); reconectando em 10 s.", e)
            self.stop_event.wait(10)

    def _session(self) -> None:
        with socket.create_connection((self.host, self.port), timeout=30) as sock:
            sock.settimeout(5)
            nick = f"justinfan{int(time.time()) % 100000}"
            sock.sendall(f"NICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
            buf = b""
            with self.out.open("a", encoding="utf-8") as f:
                while not self.stop_event.is_set():
                    try:
                        data = sock.recv(65536)
                    except TimeoutError:
                        continue
                    if not data:
                        return
                    buf += data
                    *lines, buf = buf.split(b"\r\n")
                    for raw in lines:
                        line = raw.decode("utf-8", errors="replace")
                        if line.startswith("PING"):
                            sock.sendall(line.replace("PING", "PONG", 1).encode() + b"\r\n")
                            continue
                        msg = self.parse(line)
                        t = self.clock()
                        if msg and t is not None:
                            f.write(
                                json.dumps(
                                    {"t": round(t, 2), "author": msg[0], "text": msg[1][:500]},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            self.count += 1
                    f.flush()


# --- the capture loop --------------------------------------------------------------------------


@dataclass
class CaptureState:
    next_index: int = 0  # number of the next segment ffmpeg will write
    global_end: float = 0.0  # end of the last closed segment on the global timeline
    known: set[int] = field(default_factory=set)
    run_base_global: float = 0.0  # global time at which the current ffmpeg run started
    run_base_wall: float | None = None  # monotonic time of that start (chat clock)
    seen_rows: int = 0  # rows of segments.csv already handled


class Capture:
    """One capture process. `popen` / `resolve_url` / `is_live` are injectable for tests."""

    def __init__(self, engine: Engine, settings: Settings, live_id: int, *,
                 popen=subprocess.Popen, resolve_url=None, is_live=None,
                 sleep: Callable[[float], None] = time.sleep):  # fmt: skip
        self.engine, self.settings, self.live_id = engine, settings, live_id
        self.popen, self.sleep = popen, sleep
        self._resolve_url, self._is_live = resolve_url, is_live
        self.dir = live.session_dir(settings, live_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = CaptureState()
        self.stop_event = threading.Event()
        self.chat: TwitchChatReader | None = None
        self.proc: subprocess.Popen | None = None
        self.feeder: subprocess.Popen | None = None  # yt-dlp piping into ffmpeg (from start)
        with Session(engine) as s:
            ls = s.get(LiveSession, live_id)
            if ls is None:
                raise SystemExit(f"Sessão de live #{live_id} não existe.")
            src = s.get(Source, ls.source_id)
            self.source_id, self.url = ls.source_id, src.url if src else ""
            self.platform = src.platform if src else ""
            self.sim_file = Path(ls.sim_file) if ls.sim_file else None
            self.sim_speed, self.from_start = ls.sim_speed, ls.live_from_start
            self.segment_seconds = ls.segment_seconds
            # resume after a restart: continue numbering and the global clock
            segs = live.segments(s, live_id)
        done = [g for g in segs if g.status != SegmentStatus.RECORDING]
        self.state.known = {g.index for g in done}
        self.state.next_index = max((g.index for g in segs), default=-1) + 1
        self.state.global_end = max((g.end_offset_s for g in done), default=0.0)

    # -- helpers --
    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def chat_clock(self) -> float | None:
        st = self.state
        if st.run_base_wall is None:
            return None
        speed = float(self.sim_speed or 1.0) if self.sim_file else 1.0
        return st.run_base_global + (time.monotonic() - st.run_base_wall) * speed

    def stop_requested(self) -> bool:
        with Session(self.engine) as s:
            ls = s.get(LiveSession, self.live_id)
            return ls is None or ls.stop_requested or not ls.active

    def _resolve(self) -> str:
        if self._resolve_url:
            return self._resolve_url(self.url)
        from clipador.platforms.live import resolve_stream_url
        from clipador.services.downloader import YtDlp

        return resolve_stream_url(YtDlp(self.settings), self.url,
                                  self.settings.max_download_height)  # fmt: skip

    def _check_live(self) -> bool:
        if self._is_live:
            return self._is_live(self.url)
        from clipador.platforms.live import check_live
        from clipador.services.downloader import DownloadError, YtDlp

        try:
            return check_live(YtDlp(self.settings), self.url).live
        except DownloadError:
            return False

    def _start_ffmpeg(self) -> None:
        st = self.state
        out_args = lambda inp: segment_args(inp, self.dir, self.segment_seconds, st.next_index)  # noqa: E731
        stdin = subprocess.PIPE
        if self.sim_file is not None:
            cmd = [
                self.settings.ffmpeg_path,
                *out_args(sim_input_args(self.sim_file, self.sim_speed)),
            ]
        elif self.from_start and not st.known:
            from clipador.platforms.live import pipe_download_args
            from clipador.services.downloader import YtDlp

            yt = YtDlp(self.settings)
            self.feeder = self.popen(
                pipe_download_args(yt, self.url, self.settings.max_download_height, True),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=yt._env(),
            )  # fmt: skip
            stdin = self.feeder.stdout
            cmd = [self.settings.ffmpeg_path, *out_args(["-i", "pipe:0"])]
        else:
            cmd = [self.settings.ffmpeg_path, *out_args(stream_input_args(self._resolve()))]
        self.proc = self.popen(cmd, stdin=stdin, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=False)  # fmt: skip
        st.run_base_global, st.run_base_wall = st.global_end, time.monotonic()
        st.seen_rows = 0  # ffmpeg rewrites segments.csv on every start
        self._set_recording_row()

    def _stop_ffmpeg(self, timeout: float = 15) -> None:
        """Ask ffmpeg to finish cleanly ('q' on stdin, or end of the yt-dlp pipe), so the
        segment being written is closed and listed."""
        proc = self.proc
        if proc is None:
            return
        if self.feeder is not None:
            with contextlib.suppress(OSError):
                self.feeder.terminate()
        elif proc.stdin:
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.write(b"q")
                proc.stdin.flush()
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.send_signal(signal.SIGINT if sys.platform != "win32" else signal.SIGTERM)
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _set_recording_row(self) -> None:
        """A RECORDING row for the segment being written (UI: "gravando o segmento N")."""
        st = self.state
        with Session(self.engine) as s:
            row = s.exec(
                select(LiveSegment).where(
                    LiveSegment.session_id == self.live_id, LiveSegment.index == st.next_index
                )
            ).first()
            if row is None:
                row = LiveSegment(session_id=self.live_id, index=st.next_index)
            row.status, row.start_offset_s, row.updated_at = (
                SegmentStatus.RECORDING,
                st.global_end,
                utcnow(),
            )
            s.add(row)
            s.commit()

    def collect(self) -> list[LiveSegment]:
        """Register segments ffmpeg finished since the last call; queue their analysis."""
        st = self.state
        rows = read_segment_list(self.dir / SEGMENT_LIST)
        new = [r for r in rows[st.seen_rows :] if r.index not in st.known]
        st.seen_rows = len(rows)
        added = []
        if not new:
            return added
        with Session(self.engine, expire_on_commit=False) as s:
            for r in sorted(new, key=lambda r: r.index):
                path = self.dir / r.name
                if r.duration < 0.5 or not path.is_file():  # empty tail after a drop
                    with contextlib.suppress(OSError):
                        path.unlink()
                    st.known.add(r.index)
                    continue
                row = s.exec(
                    select(LiveSegment).where(
                        LiveSegment.session_id == self.live_id, LiveSegment.index == r.index
                    )
                ).first()
                row = row or LiveSegment(session_id=self.live_id, index=r.index)
                row.path = self.settings.rel_path(path)
                row.start_offset_s, row.duration_s = st.global_end, r.duration
                row.status, row.updated_at = SegmentStatus.CLOSED, utcnow()
                s.add(row)
                s.flush()
                st.global_end += r.duration
                st.known.add(r.index)
                st.next_index = max(st.next_index, r.index + 1)
                queue.enqueue(s, "live_analyze", self.source_id,
                              {"session_id": self.live_id, "segment_id": row.id},
                              dedupe=False, priority=live.ANALYZE_PRIORITY)  # fmt: skip
                added.append(row)
            self._update_source_media(s, added)
            s.commit()
        for g in added:
            self.log(f"Segmento {g.index} fechado ({g.duration_s:.0f}s, até "
                     f"{g.end_offset_s / 60:.1f} min): análise na fila.")  # fmt: skip
        if self.proc is not None and self.proc.poll() is None:
            self._set_recording_row()
        return added

    def _update_source_media(self, s: Session, added: list[LiveSegment]) -> None:
        src = s.get(Source, self.source_id)
        if src is None or not added:
            return
        if not src.media_info:
            from clipador.errors import ToolError
            from clipador.services import ffmpeg

            with contextlib.suppress(ToolError):
                info = ffmpeg.probe(self.settings.abs_path(added[0].path),
                                    self.settings.ffprobe_path)  # fmt: skip
                src.media_info = info.to_dict()
        src.duration_seconds = self.state.global_end
        src.updated_at = utcnow()
        s.add(src)

    def _drop_recording_row(self) -> None:
        with Session(self.engine) as s:
            for g in s.exec(select(LiveSegment).where(
                    LiveSegment.session_id == self.live_id,
                    LiveSegment.status == SegmentStatus.RECORDING)).all():  # fmt: skip
                s.delete(g)
            s.commit()

    def _start_chat(self) -> None:
        if not self.settings.live_twitch_chat or self.platform != "twitch" or self.sim_file:
            return
        from clipador.platforms.live import parse_live_url

        try:
            handle = parse_live_url(self.url).channel_handle
        except Exception:
            handle = None
        if handle:
            self.chat = TwitchChatReader(handle, self.dir / CHAT_FILE, self.chat_clock,
                                         self.stop_event)  # fmt: skip
            self.chat.start()
            self.log(f"Lendo o chat de #{handle} (anônimo, só leitura).")

    def _finish(self, status: str = LiveStatus.STOPPED, error: str | None = None,
                warning: str | None = None) -> None:  # fmt: skip
        self.stop_event.set()
        self._drop_recording_row()
        with Session(self.engine, expire_on_commit=False) as s:
            ls = s.get(LiveSession, self.live_id)
            if ls is not None and ls.active:
                if warning:
                    ls.warning = f"{ls.warning} · {warning}" if ls.warning else warning
                live.finish_session(s, ls, status, error)
                s.commit()
        self.log("Captura encerrada." + (f" {error or warning}" if error or warning else ""))

    # -- main --
    def run(self) -> int:
        st = self.state
        settings = self.settings
        live.mark_heartbeat(self.engine, self.live_id, LiveStatus.WATCHING)
        what = f"simulação de {self.sim_file}" if self.sim_file else self.url
        self.log(f"Live #{self.live_id}: {what}")
        failures = 0
        self._start_chat()
        try:
            while True:
                if self.stop_requested():
                    self._finish()
                    return 0
                try:
                    self._start_ffmpeg()
                except Exception as e:  # yt-dlp: offline / blocked / network
                    failures += 1
                    msg = getattr(e, "message", None) or str(e)
                    self.log(f"Não consegui abrir o stream ({msg}); tentativa {failures}.")
                    if self.sim_file or failures >= settings.live_max_reconnect_failures:
                        self._finish(LiveStatus.FAILED if not st.known else LiveStatus.STOPPED,
                                     error=None if st.known else msg)  # fmt: skip
                        return 0 if st.known else 1
                    live.mark_heartbeat(self.engine, self.live_id)
                    self.sleep(settings.live_reconnect_wait_seconds)
                    continue
                live.mark_heartbeat(self.engine, self.live_id, LiveStatus.RECORDING)
                self.log(f"Gravando a partir do segmento {st.next_index} "
                         f"(segmentos de {self.segment_seconds}s).")  # fmt: skip
                stop_reason = self._watch_ffmpeg()
                self.collect()
                if stop_reason == "user":
                    self._finish()
                    return 0
                if stop_reason == "disk":
                    self._finish(warning=f"Parei: menos de {settings.live_min_free_gb:.0f} GB "
                                         "livres no disco.")  # fmt: skip
                    return 0
                if self.sim_file is not None:  # the file ended = the "live" ended
                    self._finish()
                    return 0
                # The stream dropped or the live ended: is it still on?
                self.log("O stream caiu; verificando se a live continua…")
                self.sleep(settings.live_reconnect_wait_seconds)
                if self._check_live():
                    failures = 0
                    with Session(self.engine) as s:
                        ls = s.get(LiveSession, self.live_id)
                        ls.reconnects += 1
                        s.add(ls)
                        s.commit()
                    continue
                failures += 1
                if failures >= settings.live_max_reconnect_failures:
                    self._finish(warning="A live terminou.")
                    return 0
        except KeyboardInterrupt:
            self._stop_ffmpeg()
            self.collect()
            self._finish()
            return 0
        finally:
            self.stop_event.set()
            if self.proc is not None and self.proc.poll() is None:
                self.proc.kill()

    def _watch_ffmpeg(self) -> str:
        """Heartbeat + collect closed segments until ffmpeg exits ('ended') or we stop it
        ('user' / 'disk')."""
        settings = self.settings
        assert self.proc is not None
        while self.proc.poll() is None:
            self.sleep(settings.live_heartbeat_seconds)
            live.mark_heartbeat(self.engine, self.live_id)
            self.collect()
            if self.stop_requested():
                self._stop_ffmpeg()
                return "user"
            if live.disk_free_gb(settings.data_dir) < settings.live_min_free_gb:
                self._stop_ffmpeg()
                return "disk"
        if self.proc.returncode not in (0, None) and self.proc.stderr is not None:
            with contextlib.suppress(OSError, ValueError):
                tail = self.proc.stderr.read()[-600:].decode("utf-8", "replace").strip()
                if tail:
                    self.log(f"ffmpeg: {tail}")
        return "ended"


def run_capture(engine: Engine, settings: Settings, live_id: int) -> int:
    return Capture(engine, settings, live_id).run()
