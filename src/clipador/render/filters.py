"""ffmpeg filter graphs (pure string builders): layouts, peak zoom, captions, preview scaling.

Inputs are cut with -ss/-t before -i, so filter time `t` starts at 0 = clip start.
Captions are read from `subs.ass` with fonts from `fonts/`, both inside ffmpeg's working
directory: relative names avoid escaping Windows paths (C:\\..., spaces, accents) in the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

OUT_W, OUT_H = 1080, 1920
GAME_H = 1280  # facecam_embaixo: game on top, cam below
CAM_H = OUT_H - GAME_H
SUBS_FILE = "subs.ass"
FONTS_SUBDIR = "fonts"

LAYOUTS = ("blur", "facecam_embaixo", "jogo_cheio")
LAYOUT_LABELS = {
    "blur": "Fundo borrado",
    "facecam_embaixo": "Facecam embaixo",
    "jogo_cheio": "Jogo cheio (recorte 9:16)",
}
_ALIASES = {"facecam": "facecam_embaixo"}
NEEDS_FACECAM = {"facecam_embaixo"}


def normalize_layout(value: str | None) -> str:
    v = _ALIASES.get(value or "", value or "")
    return v if v in LAYOUTS else "blur"


def resolve_layout(wanted: str | None, rect) -> tuple[str, str | None]:
    """Layout actually used + a PT-BR warning when it had to fall back."""
    layout = normalize_layout(wanted)
    if layout in NEEDS_FACECAM and not rect:
        return "blur", (
            "Este streamer não tem retângulo de facecam definido: usei 'Fundo borrado'. "
            "Defina a facecam na página do streamer."
        )
    return layout, None


@dataclass(frozen=True)
class Zoom:
    peak: float | None  # seconds from clip start; None = no punch-in
    amount: float = 0.12
    ramp: float = 0.25
    hold: float = 1.0

    def expr(self) -> str:
        """Scale factor as a function of t: quick punch-in at the peak, slower release."""
        if self.peak is None or self.amount <= 0:
            return "1"
        p, a, r, h = self.peak, self.amount, self.ramp, self.hold
        return (
            f"1+{a}*(between(t,{p - r:.3f},{p:.3f})*(t-{p - r:.3f})/{r}"
            f"+between(t,{p:.3f},{p + h:.3f})*(1-(t-{p:.3f})/{h}))"
        )


@dataclass(frozen=True)
class LayoutOptions:
    zoom: float = 1.0  # jogo_cheio / facecam game crop: >1 crops tighter
    foco_x: float = 0.5  # 0 = left edge, 1 = right edge of the source


def _even(v: float) -> int:
    v = round(v)
    return max(2, v - v % 2)


def rect_to_pixels(rect, src_w: int, src_h: int) -> tuple[int, int, int, int]:
    """Normalized (0-1) facecam rect -> source pixels (clamped, even sizes)."""
    x, y, w, h = (float(v) for v in rect)
    x, y = min(max(x, 0.0), 0.98), min(max(y, 0.0), 0.98)
    w, h = min(max(w, 0.02), 1 - x), min(max(h, 0.02), 1 - y)
    return round(x * src_w), round(y * src_h), _even(w * src_w), _even(h * src_h)


def _zoomed(label_in: str, label_out: str, w: int, h: int, zoom: Zoom) -> str:
    z = zoom.expr()
    if z == "1":
        return f"[{label_in}]null[{label_out}]"
    return (
        f"[{label_in}]scale=w='trunc({w}*({z})/2)*2':h='trunc({h}*({z})/2)*2':eval=frame,"
        f"crop={w}:{h}:'(iw-{w})/2':'(ih-{h})/2'[{label_out}]"
    )


def _crop_aspect(src_w: int, src_h: int, out_w: int, out_h: int, opts: LayoutOptions) -> str:
    """Crop of the source with the output aspect, `zoom` tighter, shifted by `foco_x`."""
    z = max(1.0, opts.zoom)
    ch = src_h / z
    cw = ch * out_w / out_h
    if cw > src_w:
        cw = src_w / z
        ch = cw * out_h / out_w
    cw, ch = _even(cw), _even(ch)
    x = round((src_w - cw) * min(max(opts.foco_x, 0.0), 1.0))
    y = round((src_h - ch) / 2)
    return f"crop={cw}:{ch}:{x}:{y},scale={out_w}:{out_h}"


def layout_blur(src_w: int, src_h: int, zoom: Zoom, w: int = OUT_W, h: int = OUT_H) -> list[str]:
    """Game at full width over a blurred, darkened copy of itself. The background is blurred at
    1/4 resolution and scaled up: same look, a fraction of the cost of blurring 1080x1920."""
    fg_h = _even(w * src_h / src_w)
    if fg_h > h:  # taller than 9:16: fit height instead
        fg_w, fg_h = _even(h * src_w / src_h), h
    else:
        fg_w = w
    bw, bh = _even(w / 4), _even(h / 4)
    return [
        "[0:v]setsar=1,split=2[s1][s2]",
        f"[s1]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
        f"boxblur=8:2,eq=brightness=-0.10:saturation=1.15,scale={w}:{h}:flags=bilinear[bg]",
        f"[s2]scale={fg_w}:{fg_h}[fg]",
        _zoomed("fg", "fgz", fg_w, fg_h, zoom),
        "[bg][fgz]overlay=(W-w)/2:(H-h)/2[v0]",
    ]


def layout_facecam(src_w: int, src_h: int, rect, zoom: Zoom, opts: LayoutOptions,
                   w: int = OUT_W, h: int = OUT_H) -> list[str]:  # fmt: skip
    """Game on top (2/3 of the height), facecam crop below."""
    x, y, cw, ch = rect_to_pixels(rect, src_w, src_h)
    game_h = _even(h * GAME_H / OUT_H)
    cam_h = h - game_h
    return [
        "[0:v]setsar=1,split=2[s1][s2]",
        f"[s1]{_crop_aspect(src_w, src_h, w, game_h, opts)}[g]",
        _zoomed("g", "gz", w, game_h, zoom),
        f"[s2]crop={cw}:{ch}:{x}:{y},scale={w}:{cam_h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{cam_h}[cam]",
        "[gz][cam]vstack=inputs=2[v0]",
    ]


def layout_jogo_cheio(src_w: int, src_h: int, zoom: Zoom, opts: LayoutOptions,
                      w: int = OUT_W, h: int = OUT_H) -> list[str]:  # fmt: skip
    """Center 9:16 crop of the game, `zoom` tighter and shifted horizontally by `foco_x`."""
    return [
        f"[0:v]setsar=1,{_crop_aspect(src_w, src_h, w, h, opts)}[base]",
        _zoomed("base", "v0", w, h, zoom),
    ]


def vertical_graph(
    layout: str,
    src_w: int,
    src_h: int,
    duration: float,
    zoom: Zoom,
    audio_filter: str,
    *,
    rect=None,
    opts: LayoutOptions | None = None,
    out_size: tuple[int, int] = (OUT_W, OUT_H),
    captions: bool = True,
) -> str:
    """Full filter_complex with outputs [v] and [a], built directly at `out_size` (the preview
    uses the same proportions, so it matches the final; the ASS is authored at 1080x1920 and
    libass scales it to the frame)."""
    opts = opts or LayoutOptions()
    w, h = out_size
    if layout == "facecam_embaixo" and rect:
        parts = layout_facecam(src_w, src_h, rect, zoom, opts, w, h)
    elif layout == "jogo_cheio":
        parts = layout_jogo_cheio(src_w, src_h, zoom, opts, w, h)
    else:
        parts = layout_blur(src_w, src_h, zoom, w, h)
    fade_out = max(0.0, duration - 0.25)
    tail = f"[v0]fade=t=in:d=0.15,fade=t=out:st={fade_out:.3f}:d=0.25"
    if captions:
        tail += f",subtitles={SUBS_FILE}:fontsdir={FONTS_SUBDIR}"
    parts.append(tail + ",format=yuv420p[v]")
    parts.append(f"[0:a]{audio_filter}[a]")
    return ";".join(parts)


def horizontal_graph(src_w: int, src_h: int, duration: float, audio_filter: str,
                     captions: bool = True) -> str:  # fmt: skip
    """16:9 1920x1080 segment for the long compilation (letterbox if needed)."""
    fade_out = max(0.0, duration - 0.4)
    v = (
        "[0:v]setsar=1,scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fade=t=in:d=0.3,fade=t=out:st={fade_out:.3f}:d=0.4"
    )
    if captions:
        v += f",subtitles={SUBS_FILE}:fontsdir={FONTS_SUBDIR}"
    return f"{v},format=yuv420p[v];[0:a]{audio_filter}[a]"


def encoder_args(encoder: str, preview: bool = False) -> list[str]:
    if encoder == "h264_nvenc":
        if preview:
            v = ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "30", "-b:v", "0"]
        else:
            v = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq",
                 "21", "-b:v", "0", "-maxrate", "14M", "-bufsize", "28M",
                 "-profile:v", "high"]  # fmt: skip
    elif preview:
        v = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30"]
    else:
        v = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high"]
    audio = ["-c:a", "aac", "-b:a", "128k" if preview else "192k", "-ar", "48000"]
    return [*v, "-pix_fmt", "yuv420p", *audio, "-movflags", "+faststart"]


def output_rate_args(fps: float, max_fps: int) -> list[str]:
    """Constant frame rate output (Twitch VODs are often VFR): source fps, capped."""
    rate = min(float(fps or 30), float(max_fps))
    rate = 30.0 if rate < 10 else rate
    return ["-fps_mode", "cfr", "-r", f"{rate:.3f}".rstrip("0").rstrip(".")]
