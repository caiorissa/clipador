"""Automatic facecam detection (step 5C).

1. Sample ~16 frames spread over the video (ffmpeg, 640 px wide), skipping the first/last seconds.
2. Find faces in each frame (OpenCV YuNet if the ONNX model is in data/models, else the Haar
   cascade bundled with OpenCV). CPU only.
3. Cluster detections by position: the facecam is where a face shows up again and again at
   (almost) the same place. Faces that move around (game characters) end up in small, scattered
   clusters and are dropped.
4. Expand the face box to the camera box: look outward from the face for the camera's borders,
   using the temporal STD map (the camera's background is quiet, the game around it moves) and
   persistent edges in the temporal MEAN image (overlay frames, hard borders). Sides without a
   border fall back to a margin proportional to the face, clamped to the frame and preferring to
   touch the nearest corner.

Everything except the face detectors themselves is plain numpy/scipy, so it is testable with
synthetic frames and a fake detector.
"""

from __future__ import annotations

import logging
import struct
import subprocess
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import ndimage

from clipador.errors import ClipadorError, ToolError

log = logging.getLogger(__name__)

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE
)
SAMPLE_WIDTH = 640
DEFAULT_SAMPLES = 16
MIN_CONFIDENCE = 0.45
MIN_HITS = 3  # frames that must confirm the face at the same place

PxBox = tuple[float, float, float, float]  # x, y, w, h in pixels of the sampled frame

# Border search thresholds (gray levels 0-255 / robust z-scores).
_Z_MIN = 5.0
_STD_STEP_MIN = 6.0  # quiet camera vs moving game
_EDGE_MIN = 12.0  # persistent edge in the temporal mean
_Z_EDGE_ONLY = 8.0  # stricter when there is no motion contrast at all
_STD_SMOOTH = 5  # box filter (px) applied to the temporal STD map


@dataclass(frozen=True)
class FacecamBox:
    """Facecam rectangle NORMALIZED to the frame (0..1), plus how sure the detector is."""

    x: float
    y: float
    w: float
    h: float
    confidence: float  # 0..1
    frames: int  # sampled frames that confirmed the face at this place
    total_frames: int = 0
    method: str = ""  # "bordas" | "bordas+margem" | "margem"

    def as_list(self) -> list[float]:
        return [round(self.x, 4), round(self.y, 4), round(self.w, 4), round(self.h, 4)]


@dataclass
class DetectionResult:
    box: FacecamBox | None  # None = no facecam (or confidence too low)
    confidence: float  # of the best attempt, also when `box` is None
    frames: int
    total_frames: int
    face: PxBox | None = None  # median face box (pixels of the sampled frames)
    camera_px: PxBox | None = None  # expanded box (pixels), also when below the threshold
    detector: str = ""
    message: str = ""


# --- face detectors --------------------------------------------------------------------------


class FaceDetector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[PxBox]:
        """Faces in a BGR uint8 frame, as pixel boxes (x, y, w, h)."""
        ...


def _import_cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - dependency is in pyproject
        raise ClipadorError(
            "OpenCV não está instalado (rode `uv sync`): sem ele não dá para detectar a facecam.",
            detail=str(e),
        ) from e
    return cv2


class HaarFaceDetector:
    """Frontal-face Haar cascade bundled with OpenCV (no download needed)."""

    name = "haar"

    def __init__(self) -> None:
        cv2 = _import_cv2()
        if not hasattr(cv2, "CascadeClassifier"):  # OpenCV 5 moved it out of the main module
            raise ClipadorError(
                "Este OpenCV não tem o detector Haar (OpenCV 5?): instale "
                "opencv-python-headless<5 ou deixe o modelo YuNet ser baixado."
            )
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cv2 = cv2
        self._clf = cv2.CascadeClassifier(str(path))
        if self._clf.empty():
            raise ClipadorError("Não consegui carregar o detector de rostos do OpenCV.")

    def detect(self, frame: np.ndarray) -> list[PxBox]:
        cv2 = self._cv2
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        side = max(20, frame.shape[0] // 18)
        faces = self._clf.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(side, side)
        )
        return [(float(x), float(y), float(w), float(h)) for x, y, w, h in faces]


class YuNetFaceDetector:
    """OpenCV's YuNet (cv2.FaceDetectorYN): much better than Haar on profiles/headsets."""

    name = "yunet"

    def __init__(self, model: Path, score_threshold: float = 0.75) -> None:
        cv2 = _import_cv2()
        self._det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score_threshold, 0.3, 50)

    def detect(self, frame: np.ndarray) -> list[PxBox]:
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        if faces is None:
            return []
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in faces]


def _download(url: str, dest: Path, timeout: float = 30.0) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    if tmp.stat().st_size < 50_000:  # an HTML error page, not the model
        tmp.unlink()
        raise ClipadorError("download do modelo YuNet veio incompleto")
    tmp.replace(dest)


def make_face_detector(
    models_dir: Path, download: bool = True, log_cb: Callable[[str], None] | None = None
) -> FaceDetector:
    """YuNet if its ONNX model is (or can be downloaded to) `models_dir`, else Haar."""
    say = log_cb or log.info
    cv2 = _import_cv2()
    model = models_dir / YUNET_FILE
    if not model.is_file() and download:
        try:
            say("Baixando o modelo de rostos YuNet (~230 KB)…")
            _download(YUNET_URL, model)
        except Exception as e:
            say(f"Sem o modelo YuNet ({e}); usando o detector Haar do OpenCV.")
    if model.is_file() and hasattr(cv2, "FaceDetectorYN"):
        try:
            return YuNetFaceDetector(model)
        except Exception as e:  # corrupt model, old OpenCV...
            say(f"YuNet não carregou ({e}); usando o detector Haar do OpenCV.")
    return HaarFaceDetector()


# --- frame sampling --------------------------------------------------------------------------


def sample_times(
    duration: float,
    n: int = DEFAULT_SAMPLES,
    start: float | None = None,
    end: float | None = None,
) -> list[float]:
    """`n` instants spread over [start, end], skipping the first/last ~8% (max 30 s)."""
    a = max(0.0, start or 0.0)
    b = min(duration, end) if end else duration
    if b <= a:
        a, b = 0.0, duration
    span = max(0.0, b - a)
    margin = min(30.0, span * 0.08)
    lo, hi = a + margin, b - margin
    n = max(1, n)
    if hi <= lo or n == 1:
        return [round(a + span / 2, 3)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 3) for i in range(n)]


def sample_size(src_w: int, src_h: int, width: int = SAMPLE_WIDTH) -> tuple[int, int]:
    w = max(2, min(width, src_w) // 2 * 2)
    h = max(2, round(w * src_h / src_w / 2) * 2)
    return w, h


def sample_frames(
    video: Path,
    times: Sequence[float],
    src_w: int,
    src_h: int,
    width: int = SAMPLE_WIDTH,
    ffmpeg: str = "ffmpeg",
) -> list[np.ndarray]:
    """BGR uint8 frames (H, W, 3) at the given instants; unreadable instants are skipped."""
    w, h = sample_size(src_w, src_h, width)
    size = w * h * 3
    frames: list[np.ndarray] = []
    for t in times:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{max(0.0, t):.3f}", "-i", str(video), "-frames:v", "1",
            "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]  # fmt: skip
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=60)
        except FileNotFoundError as e:
            raise ToolError("ffmpeg não encontrado (rode `clipador doctor`).", detail=str(e)) from e
        except subprocess.TimeoutExpired:
            continue
        if res.returncode != 0 or len(res.stdout) < size:
            continue
        frames.append(np.frombuffer(res.stdout[:size], np.uint8).reshape(h, w, 3).copy())
    return frames


# --- clustering ------------------------------------------------------------------------------


def iou(a: PxBox, b: PxBox) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class FaceCluster:
    boxes: list[PxBox] = field(default_factory=list)
    frames: set[int] = field(default_factory=set)

    def add(self, frame_idx: int, box: PxBox) -> None:
        self.boxes.append(box)
        self.frames.add(frame_idx)

    @property
    def hits(self) -> int:
        return len(self.frames)

    def median(self) -> PxBox:
        m = np.median(np.asarray(self.boxes, dtype=float), axis=0)
        return (float(m[0]), float(m[1]), float(m[2]), float(m[3]))

    def spread(self) -> float:
        """Std of the face centers, in face widths (0 = never moved)."""
        arr = np.asarray(self.boxes, dtype=float)
        if len(arr) < 2:
            return 0.0
        cx, cy = arr[:, 0] + arr[:, 2] / 2, arr[:, 1] + arr[:, 3] / 2
        fw = float(np.median(arr[:, 2])) or 1.0
        return float(np.hypot(cx.std(), cy.std()) / fw)

    def stability(self) -> float:
        return float(np.clip(1.0 - self.spread() / 0.5, 0.0, 1.0))


def cluster_faces(detections: Sequence[Sequence[PxBox]], iou_min: float = 0.3) -> list[FaceCluster]:
    """Group per-frame detections by position (IoU against the cluster's median box). A frame
    counts at most once per cluster. Sorted by hits, most first."""
    clusters: list[FaceCluster] = []
    for i, boxes in enumerate(detections):
        for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
            best, best_iou = None, 0.0
            for c in clusters:
                if i in c.frames:
                    continue
                v = iou(b, c.median())
                if v > best_iou:
                    best, best_iou = c, v
            if best is not None and best_iou >= iou_min:
                best.add(i, b)
            else:
                clusters.append(FaceCluster([b], {i}))
    clusters.sort(key=lambda c: -c.hits)
    return clusters


def best_cluster(clusters: Sequence[FaceCluster]) -> FaceCluster | None:
    if not clusters:
        return None
    return max(clusters, key=lambda c: c.hits * (0.5 + 0.5 * c.stability()))


# --- face box -> camera box ------------------------------------------------------------------


def _gray(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    if f.ndim == 2:
        return f
    return 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]  # BGR


def _zscore(p: np.ndarray) -> np.ndarray:
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med))) * 1.4826 + 1.0
    return (p - med) / mad


def _best_line(edge: np.ndarray, step: np.ndarray) -> int | None:
    """Index of the strongest border in a 1-D search profile, or None.

    `edge` = mean |gradient| of the temporal-mean image along each candidate line;
    `step` = (outside STD - inside STD) along it: positive when the game moves outside and the
    camera is quiet inside. Motion contrast wins; persistent edges alone need a stronger z."""
    if edge.size == 0:
        return None
    ze, zs = _zscore(edge), _zscore(step)
    ok = (step >= _STD_STEP_MIN) & (zs >= _Z_MIN)
    if ok.any():
        score = np.where(ok, zs + 0.5 * np.maximum(ze, 0), -np.inf)
        return int(np.argmax(score))
    ok = (edge >= _EDGE_MIN) & (ze >= _Z_EDGE_ONLY)
    if ok.any():
        return int(np.argmax(np.where(ok, ze, -np.inf)))
    return None


@dataclass
class _Maps:
    ex: np.ndarray  # (H, W-1): |mean[:, c+1] - mean[:, c]|
    ey: np.ndarray  # (H-1, W)
    std: np.ndarray  # (H, W) smoothed temporal std


def _maps(frames: Sequence[np.ndarray]) -> _Maps:
    gray = np.stack([_gray(f) for f in frames])
    mean = gray.mean(axis=0)
    if len(frames) >= 3:
        std = ndimage.uniform_filter(gray.std(axis=0), size=_STD_SMOOTH)
    else:
        std = np.zeros_like(mean)
    return _Maps(ex=np.abs(np.diff(mean, axis=1)), ey=np.abs(np.diff(mean, axis=0)), std=std)


def _std_probes(idx: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Where to read the STD on each side of boundary `idx` (between pixel idx and idx+1).

    The STD map is smoothed, so a sharp quiet/moving border becomes a ramp `_STD_SMOOTH` px wide;
    comparing adjacent pixels would only see a fraction of the contrast. Probe past the ramp."""
    off = _STD_SMOOTH // 2 + 1
    return np.clip(idx - off, 0, n - 1), np.clip(idx + 1 + off, 0, n - 1)


def _clip_span(a: float, b: float, n: int) -> tuple[int, int]:
    lo, hi = max(0, int(a)), min(n, int(np.ceil(b)))
    return (lo, hi) if hi > lo else (0, n)


def _search(m: _Maps, side: str, face: PxBox, span: tuple[int, int], reach: float) -> int | None:
    """Border position (pixel coordinate of the camera edge) on one side, or None.

    Boundary index c separates pixel c from c+1; the camera edge coordinate is c+1."""
    fx, fy, fw, fh = face
    H, W = m.std.shape
    r0, r1 = span
    if side in ("left", "right"):
        gap = 0.1 * fw
        if side == "left":
            hi = int(fx - gap) - 1
            lo = max(0, int(fx - reach))
        else:
            lo = int(fx + fw + gap)
            hi = min(W - 2, int(fx + fw + reach))
        if hi <= lo:
            return None
        cs = np.arange(lo, hi + 1)
        edge = m.ex[r0:r1, :][:, cs].mean(axis=0)
        before, after = _std_probes(cs, W)
        left_std = m.std[r0:r1, :][:, before].mean(axis=0)
        right_std = m.std[r0:r1, :][:, after].mean(axis=0)
        step = left_std - right_std if side == "left" else right_std - left_std
    else:
        gap = 0.1 * fh
        if side == "top":
            hi = int(fy - gap) - 1
            lo = max(0, int(fy - reach))
        else:
            lo = int(fy + fh + gap)
            hi = min(H - 2, int(fy + fh + reach))
        if hi <= lo:
            return None
        rs = np.arange(lo, hi + 1)
        edge = m.ey[:, r0:r1][rs, :].mean(axis=1)
        before, after = _std_probes(rs, H)
        top_std = m.std[:, r0:r1][before, :].mean(axis=1)
        bot_std = m.std[:, r0:r1][after, :].mean(axis=1)
        step = top_std - bot_std if side == "top" else bot_std - top_std
    k = _best_line(edge, step)
    return None if k is None else int(cs[k] if side in ("left", "right") else rs[k]) + 1


def _fallback_box(face: PxBox, W: int, H: int) -> list[float]:
    """Margin proportional to the face (head + shoulders), clamped, touching the nearest corner
    when it is close."""
    fx, fy, fw, fh = face
    box = [fx - 0.9 * fw, fy - 0.6 * fh, fx + 1.9 * fw, fy + 2.0 * fh]  # l, t, r, b
    cx, cy = fx + fw / 2, fy + fh / 2
    if cx < W / 2:
        box[0] = 0.0 if box[0] < 1.5 * fw else box[0]
    else:
        box[2] = float(W) if W - box[2] < 1.5 * fw else box[2]
    if cy < H / 2:
        box[1] = 0.0 if box[1] < 1.5 * fh else box[1]
    else:
        box[3] = float(H) if H - box[3] < 1.5 * fh else box[3]
    return [max(0.0, box[0]), max(0.0, box[1]), min(float(W), box[2]), min(float(H), box[3])]


def expand_to_camera(frames: Sequence[np.ndarray], face: PxBox) -> tuple[PxBox, int]:
    """Camera box (pixels) around a face + how many of the 4 sides were found as real borders
    (a side resting on the frame edge counts half)."""
    H, W = frames[0].shape[:2]
    fx, fy, fw, fh = face
    m = _maps(frames)
    fb = _fallback_box(face, W, H)
    credit = 0.0
    sides: dict[str, float] = {}

    rows = _clip_span(fy - 0.25 * fh, fy + 1.25 * fh, H)
    for side, reach, frame_edge, near in (
        ("left", 3.0 * fw, 0.0, fx),
        ("right", 3.0 * fw, float(W), W - (fx + fw)),
    ):
        pos = _search(m, side, face, rows, reach)
        if pos is not None:
            sides[side], credit = float(pos), credit + 1
        elif near <= 1.5 * fw:
            sides[side], credit = frame_edge, credit + 0.5
        else:
            sides[side] = fb[0] if side == "left" else fb[2]

    cols = _clip_span(sides["left"] + 2, sides["right"] - 2, W)
    for side, reach, frame_edge, near in (
        ("top", 2.0 * fh, 0.0, fy),
        ("bottom", 4.0 * fh, float(H), H - (fy + fh)),
    ):
        pos = _search(m, side, face, cols, reach)
        if pos is not None:
            sides[side], credit = float(pos), credit + 1
        elif near <= (1.5 if side == "top" else 3.0) * fh:
            sides[side], credit = frame_edge, credit + 0.5
        else:
            sides[side] = fb[1] if side == "top" else fb[3]

    left, top, right, bottom = sides["left"], sides["top"], sides["right"], sides["bottom"]
    w, h = right - left, bottom - top
    plausible = (
        w >= 1.2 * fw and h >= 1.2 * fh and 0.5 <= w / max(h, 1e-6) <= 3.0 and w * h <= 0.45 * W * H
    )
    if not plausible and not (w >= 0.95 * W and h >= 0.95 * H):  # full-frame "just chatting"
        left, top, right, bottom = fb
        credit = 0.0
    return (left, top, right - left, bottom - top), round(credit)


def _normalize(box: PxBox, W: int, H: int) -> tuple[float, float, float, float]:
    x = round(min(max(box[0] / W, 0.0), 0.97), 4)
    y = round(min(max(box[1] / H, 0.0), 0.97), 4)
    w = round(min(max(box[2] / W, 0.03), 1.0 - x), 4)
    h = round(min(max(box[3] / H, 0.03), 1.0 - y), 4)
    return x, y, w, h


def analyze_frames(
    frames: Sequence[np.ndarray],
    detector: FaceDetector,
    min_confidence: float = MIN_CONFIDENCE,
) -> DetectionResult:
    total = len(frames)
    name = getattr(detector, "name", "")
    if total == 0:
        return DetectionResult(None, 0.0, 0, 0, detector=name, message="nenhum quadro lido")
    detections = [detector.detect(f) for f in frames]
    best = best_cluster(cluster_faces(detections))
    if best is None:
        return DetectionResult(None, 0.0, 0, total, detector=name, message="nenhum rosto")
    face = best.median()
    hits, stability = best.hits, best.stability()
    camera, found = expand_to_camera(frames, face)
    det_part = min(1.0, hits / (0.6 * total))
    conf = round(det_part * (0.55 + 0.45 * stability) * (0.8 + 0.05 * found), 3)
    H, W = frames[0].shape[:2]
    if hits < min(MIN_HITS, total) or conf < min_confidence:
        msg = f"rosto em {hits}/{total} quadros, confiança {conf:.0%}: sem facecam fixa"
        return DetectionResult(
            None, conf, hits, total, face=face, camera_px=camera, detector=name, message=msg
        )
    method = "bordas" if found >= 3 else ("bordas+margem" if found > 0 else "margem")
    box = FacecamBox(*_normalize(camera, W, H), confidence=conf, frames=hits,
                     total_frames=total, method=method)  # fmt: skip
    return DetectionResult(
        box, conf, hits, total, face=face, camera_px=camera, detector=name, message=method
    )


def detect_facecam(
    video: Path,
    detector: FaceDetector,
    *,
    duration: float,
    src_w: int,
    src_h: int,
    start: float | None = None,
    end: float | None = None,
    samples: int = DEFAULT_SAMPLES,
    ffmpeg: str = "ffmpeg",
    min_confidence: float = MIN_CONFIDENCE,
) -> FacecamBox | None:
    """FacecamBox (normalized 0..1) or None when there is no (confident) facecam."""
    frames = sample_frames(video, sample_times(duration, samples, start, end), src_w, src_h,
                           ffmpeg=ffmpeg)  # fmt: skip
    return analyze_frames(frames, detector, min_confidence).box


# --- preview PNG (no OpenCV needed) ------------------------------------------------------------


def draw_detection(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    """RGB copy of a BGR frame with the camera box (green; red when rejected) and the face."""
    img = np.ascontiguousarray(frame[..., ::-1]).copy()
    H, W = img.shape[:2]

    def rect(b: PxBox, color: tuple[int, int, int], t: int) -> None:
        x0, y0 = int(max(0, b[0])), int(max(0, b[1]))
        x1, y1 = int(min(W - 1, b[0] + b[2])), int(min(H - 1, b[1] + b[3]))
        if x1 <= x0 or y1 <= y0:
            return
        img[y0 : y0 + t, x0 : x1 + 1] = color
        img[max(y0, y1 - t + 1) : y1 + 1, x0 : x1 + 1] = color
        img[y0 : y1 + 1, x0 : x0 + t] = color
        img[y0 : y1 + 1, max(x0, x1 - t + 1) : x1 + 1] = color

    if result.face:
        rect(result.face, (255, 210, 0), 1)
    if result.box is not None:
        b = result.box
        rect((b.x * W, b.y * H, b.w * W, b.h * H), (0, 230, 60), 3)
    elif result.camera_px:
        rect(result.camera_px, (230, 40, 40), 2)
    return img


def write_png(path: Path, rgb: np.ndarray) -> Path:
    """Minimal 8-bit RGB PNG writer (zlib), so the preview needs no imaging library."""
    img = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    return path
