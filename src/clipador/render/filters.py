"""ffmpeg filter graphs for the layouts (pure string builders).

Inputs are cut with -ss/-t before -i, so filter time `t` starts at 0 = clip start.
Subtitles are read from `subs.ass` in ffmpeg's working directory: using a relative name avoids
escaping Windows paths (C:\\...) inside the filter graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from clipador.render.subtitles import Layout

OUT_W, OUT_H = 1080, 1920
FACECAM_REF_W, FACECAM_REF_H = 1920, 1080  # facecam_rect is given in a 1920x1080 frame
GAME_H = 1280  # facecam layout: game on top, cam below
CAM_H = OUT_H - GAME_H
SUBS_FILE = "subs.ass"

LAYOUT_BLUR = Layout(OUT_W, OUT_H, sub_margin_v=470, title_margin_v=170)
LAYOUT_FACECAM = Layout(OUT_W, OUT_H, sub_margin_v=CAM_H + 40, title_margin_v=110)
LAYOUT_VERTICAL = Layout(OUT_W, OUT_H, sub_margin_v=420, title_margin_v=170)
LAYOUT_LONG = Layout(1920, 1080, sub_margin_v=70, title_margin_v=50, font_scale=0.75)


@dataclass(frozen=True)
class Zoom:
    peak: float | None  # seconds from clip start; None = no zoom
    amount: float = 0.12  # +12% at the peak
    ramp: float = 0.25  # seconds to zoom in
    hold: float = 1.0  # seconds to come back

    def expr(self) -> str:
        """Scale factor as a function of t: quick punch-in at the peak, slower release."""
        if self.peak is None or self.amount <= 0:
            return "1"
        p, a, r, h = self.peak, self.amount, self.ramp, self.hold
        # rising edge (t in [p-r, p]) and falling edge (t in [p, p+h]), 0 elsewhere
        return (
            f"1+{a}*(between(t,{p - r:.3f},{p:.3f})*(t-{p - r:.3f})/{r}"
            f"+between(t,{p:.3f},{p + h:.3f})*(1-(t-{p:.3f})/{h}))"
        )


def _zoomed(label_in: str, label_out: str, w: int, h: int, zoom: Zoom) -> str:
    z = zoom.expr()
    if z == "1":
        return f"[{label_in}]null[{label_out}]"
    return (
        f"[{label_in}]scale=w='trunc({w}*({z})/2)*2':h='trunc({h}*({z})/2)*2':eval=frame,"
        f"crop={w}:{h}:'(iw-{w})/2':'(ih-{h})/2'[{label_out}]"
    )


def scale_rect(
    rect: tuple[int, int, int, int], src_w: int, src_h: int
) -> tuple[int, int, int, int]:
    """facecam_rect (1920x1080 reference) -> source pixels, clamped, even sizes."""
    sx, sy = src_w / FACECAM_REF_W, src_h / FACECAM_REF_H
    x, y, w, h = (
        round(rect[0] * sx),
        round(rect[1] * sy),
        round(rect[2] * sx),
        round(rect[3] * sy),
    )
    x, y = max(0, min(x, src_w - 2)), max(0, min(y, src_h - 2))
    w, h = max(2, min(w, src_w - x)), max(2, min(h, src_h - y))
    return x, y, w - w % 2, h - h % 2


def vertical_filter(
    layout: str,
    src_w: int,
    src_h: int,
    duration: float,
    zoom: Zoom,
    fps: int = 30,
    facecam_rect: tuple[int, int, int, int] | None = None,
    subtitles: bool = True,
) -> tuple[str, Layout]:
    """Returns (filter_complex, subtitle layout). Output labels: [v] and [a]."""
    parts = [f"[0:v]fps={fps},setsar=1,split=2[s1][s2]"]
    if src_h > src_w:  # source already vertical (Shorts, vertical clip): just fill the frame
        parts = [f"[0:v]fps={fps},setsar=1,scale={OUT_W}:{OUT_H}:force_original_aspect_ratio="
                 f"increase,crop={OUT_W}:{OUT_H}[base]"]  # fmt: skip
        parts.append(_zoomed("base", "v0", OUT_W, OUT_H, zoom))
        sub_layout = LAYOUT_VERTICAL
    elif layout == "facecam" and facecam_rect:
        x, y, w, h = scale_rect(facecam_rect, src_w, src_h)
        game_w = min(src_w, round(src_h * OUT_W / GAME_H))
        game_w -= game_w % 2
        parts.append(
            f"[s1]crop={game_w}:{src_h}:{(src_w - game_w) // 2}:0,scale={OUT_W}:{GAME_H}[g]"
        )
        parts.append(_zoomed("g", "gz", OUT_W, GAME_H, zoom))
        parts.append(
            f"[s2]crop={w}:{h}:{x}:{y},scale={OUT_W}:{CAM_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{CAM_H}[cam]"
        )
        parts.append("[gz][cam]vstack=inputs=2[v0]")
        sub_layout = LAYOUT_FACECAM
    else:  # blur (default; also the fallback when there is no facecam rect)
        fg_h = round(OUT_W * src_h / src_w)
        fg_h -= fg_h % 2
        parts.append(
            f"[s1]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},boxblur=24:2,eq=brightness=-0.10:saturation=1.15[bg]"
        )
        parts.append(f"[s2]scale={OUT_W}:{fg_h}[fg]")
        parts.append(_zoomed("fg", "fgz", OUT_W, fg_h, zoom))
        parts.append("[bg][fgz]overlay=(W-w)/2:(H-h)/2[v0]")
        sub_layout = LAYOUT_BLUR
    fade_out = max(0.0, duration - 0.25)
    tail = f"[v0]fade=t=in:d=0.15,fade=t=out:st={fade_out:.3f}:d=0.25"
    tail += f",subtitles={SUBS_FILE}[v]" if subtitles else "[v]"
    parts.append(tail)
    parts.append(audio_chain(duration))
    return ";".join(parts), sub_layout


def horizontal_filter(src_w: int, src_h: int, duration: float, fps: int = 30,
                      subtitles: bool = True) -> str:  # fmt: skip
    """16:9 1920x1080 segment for the long compilation (letterbox if needed)."""
    fade_out = max(0.0, duration - 0.4)
    v = (
        f"[0:v]fps={fps},setsar=1,scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fade=t=in:d=0.3,fade=t=out:st={fade_out:.3f}:d=0.4"
    )
    v += f",subtitles={SUBS_FILE}[v]" if subtitles else "[v]"
    return f"{v};{audio_chain(duration, fade=0.4)}"


def audio_chain(duration: float, fade: float = 0.2) -> str:
    """Loudness to -14 LUFS (Shorts/TikTok/Reels target), short fades, 48 kHz."""
    st = max(0.0, duration - fade)
    return (
        "[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,"
        f"afade=t=in:d=0.08,afade=t=out:st={st:.3f}:d={fade}[a]"
    )


def encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        v = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "21",
             "-b:v", "0", "-maxrate", "14M", "-bufsize", "28M", "-profile:v", "high"]  # fmt: skip
    else:
        v = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high"]
    return [*v, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart"]  # fmt: skip
