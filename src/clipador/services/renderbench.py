"""`clipador render-bench`: how fast preview/final renders are on this PC (per encoder)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from clipador.config import Settings
from clipador.render import audio as raudio
from clipador.render import filters
from clipador.render.styles import get_style
from clipador.render.subtitles import IWord, Overlays, build_ass, safe_zone
from clipador.services import ffmpeg, rendering
from clipador.services.rendering import RenderPlan


@dataclass
class BenchRow:
    encoder: str
    kind: str
    seconds: float
    duration: float
    size_bytes: int
    ok: bool
    error: str = ""

    @property
    def speed(self) -> float:
        return self.duration / self.seconds if self.seconds else 0.0


def synthetic_plan(work: Path, seconds: float = 40.0) -> RenderPlan:
    """1920x1080 60 fps test video with tone + speech-like words, like a GTA stream VOD."""
    video = work / "bench_src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=60",
         "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000", "-t", str(seconds + 1),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(video)],
        check=True,
    )  # fmt: skip
    words = [IWord(i, 0.5 + i * 0.4, 0.8 + i * 0.4, w)
             for i, w in enumerate(("olha a polícia corre capotou o carro meu deus " * 8).split())
             if 0.8 + i * 0.4 < seconds]  # fmt: skip
    return RenderPlan(
        candidate_id=0, source_id=0, kind="final", video=video, start=0.5, end=0.5 + seconds,
        src_w=1920, src_h=1080, fps=60, layout="blur", style_key="destaque_amarelo", words=words,
        rect=None, opts=filters.LayoutOptions(), zoom_peak=seconds / 2,
        overlays=Overlays("TESTE DE RENDER", 2.0, "via @teste"), clip_name="bench",
    )  # fmt: skip


def bench_one(settings: Settings, plan: RenderPlan, encoder: str, work: Path) -> BenchRow:
    out = work / f"out_{encoder}_{plan.kind}.mp4"
    t0 = time.monotonic()
    try:
        gain = None
        if plan.kind == "final":
            gain = raudio.final_gain(
                plan.video, plan.start, plan.duration, settings.loudness_target,
                settings.loudness_true_peak, settings.ffmpeg_path,
            )  # fmt: skip
        af = raudio.loudnorm_filter(settings.loudness_target, settings.loudness_true_peak, gain,
                                    plan.duration, settings.audio_fade_seconds)  # fmt: skip
        zone = safe_zone(1080, 1920, settings.safe_bottom, settings.safe_top, settings.caption_lift)
        style = get_style(plan.style_key, settings.caption_styles_path)
        wd = work / f"wd_{encoder}_{plan.kind}"
        rendering.prepare_workdir(wd, settings, build_ass(plan.words, plan.duration, style, zone,
                                                          plan.overlays))  # fmt: skip
        ffmpeg.run_with_progress(
            rendering.build_args(plan, settings, encoder, af, out), duration=plan.duration,
            progress_cb=lambda f: None, cwd=wd, ffmpeg=settings.ffmpeg_path,
        )  # fmt: skip
        return BenchRow(encoder, plan.kind, time.monotonic() - t0, plan.duration,
                        out.stat().st_size, True)  # fmt: skip
    except Exception as e:  # report, don't stop the bench
        return BenchRow(encoder, plan.kind, time.monotonic() - t0, plan.duration, 0, False, str(e))


def run_bench(
    settings: Settings, plan: RenderPlan, encoders: list[str], log=print
) -> list[BenchRow]:
    rows = []
    with tempfile.TemporaryDirectory(dir=settings.tmp_dir) as tmp:
        work = Path(tmp)
        for enc in encoders:
            if (
                enc == ffmpeg.NVENC
                and not ffmpeg.try_encoder(ffmpeg.NVENC, settings.ffmpeg_path)[0]
            ):
                rows.append(BenchRow(enc, "-", 0, plan.duration, 0, False, "NVENC indisponível"))
                continue
            for kind in ("preview", "final"):
                r = bench_one(settings, replace(plan, kind=kind), enc, work)
                log(f"{enc:<11} {kind:<8} {'ok' if r.ok else 'FALHOU'} {r.seconds:5.1f}s")
                rows.append(r)
        shutil.rmtree(work, ignore_errors=True)
    return rows


def format_rows(rows: list[BenchRow], duration: float) -> str:
    lines = [f"Clip de {duration:.0f} s.", f"{'encoder':<11} {'tipo':<8} {'tempo':>7} "
             f"{'velocidade':>11} {'tamanho':>9}"]  # fmt: skip
    for r in rows:
        if not r.ok:
            lines.append(f"{r.encoder:<11} {r.kind:<8} falhou: {r.error[:80]}")
            continue
        lines.append(f"{r.encoder:<11} {r.kind:<8} {r.seconds:6.1f}s {r.speed:9.1f}x "
                     f"{r.size_bytes / 1e6:7.1f} MB")  # fmt: skip
    lines.append(
        "velocidade = quantas vezes mais rápido que a duração do clip (inclui medir o áudio)."
    )
    return "\n".join(lines)
