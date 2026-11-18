"""YouTube Data API v3 (official), with plain httpx: OAuth (installed app, loopback + PKCE),
resumable upload of a final render, and the statistics of what was published (step 6).

Setup (once): Google Cloud console -> create a project -> enable "YouTube Data API v3" ->
OAuth consent screen (External, add yourself as test user) -> Credentials -> OAuth client ID ->
"Desktop app" -> download the JSON and point CLIPADOR_YOUTUBE_CLIENT_SECRETS to it. Then
"Conectar YouTube" on /publicar. Tokens are stored in data/youtube_token.json (never in git).

Nothing is ever uploaded without an explicit click on an approved clip.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx

from clipador.config import Settings
from clipador.errors import ClipadorError

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/youtube/v3"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
SCOPES = (
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
)
CHUNK = 8 * 1024 * 1024
GAMING_CATEGORY = "20"
PRIVACY = ("private", "unlisted", "public")
PRIVACY_LABELS = {"private": "Privado", "unlisted": "Não listado", "public": "Público"}


class YouTubeError(ClipadorError):
    pass


SETUP_HINT = (
    "Crie um cliente OAuth 'App para computador' no Google Cloud (com a YouTube Data API v3 "
    "ativada), baixe o JSON e coloque-o em data/google_clients/ (ou defina "
    "CLIPADOR_YOUTUBE_CLIENT_SECRETS no .env). Passo a passo no README (Publicar no YouTube)."
)


# --- credentials ---------------------------------------------------------------------------------


@dataclass
class Client:
    client_id: str
    client_secret: str
    name: str = ""  # file name: one Google Cloud project = one daily upload quota


def _read_client(path: Path) -> Client:
    try:
        data = json.loads(path.read_text("utf-8"))
        block = data.get("installed") or data.get("web") or data
        return Client(block["client_id"], block["client_secret"], path.stem)
    except (ValueError, KeyError, OSError) as e:
        raise YouTubeError(f"O arquivo de credenciais '{path.name}' não é válido. "
                           + SETUP_HINT) from e  # fmt: skip


def clients_dir(settings: Settings) -> Path:
    """Drop one OAuth JSON per Google Cloud project here (each has its own ~6 uploads/day)."""
    return settings.data_dir / "google_clients"


def load_clients(settings: Settings) -> list[Client]:
    """Every configured Google Cloud project: CLIPADOR_YOUTUBE_CLIENT_SECRETS (a file, or a
    folder of JSONs) + data/google_clients/*.json. Same client id twice counts once."""
    paths: list[Path] = []
    cfg = settings.youtube_client_secrets
    if cfg:
        cfg = Path(cfg)
        paths += sorted(cfg.glob("*.json")) if cfg.is_dir() else ([cfg] if cfg.is_file() else [])
    if clients_dir(settings).is_dir():
        paths += sorted(clients_dir(settings).glob("*.json"))
    out: dict[str, Client] = {}
    for p in paths:
        c = _read_client(p)
        out.setdefault(c.client_id, c)
    return list(out.values())


def load_client(settings: Settings, client_id: str | None = None) -> Client:
    """The project with `client_id` (the one an account was connected with), or the first."""
    clients = load_clients(settings)
    if not clients:
        raise YouTubeError("YouTube não configurado. " + SETUP_HINT)
    if client_id:
        for c in clients:
            if c.client_id == client_id:
                return c
        raise YouTubeError(
            "O projeto do Google com que esta conta foi conectada não está mais configurado "
            "(o JSON sumiu de data/google_clients ou do .env). Recoloque o arquivo ou conecte "
            "a conta de novo."
        )
    return clients[0]


class TokenStore(Protocol):
    """Where an account's OAuth token lives (the social_account row; tests use memory)."""

    def load(self) -> dict[str, Any] | None: ...
    def save(self, token: dict[str, Any]) -> None: ...


class MemoryStore:
    def __init__(self, token: dict[str, Any] | None = None):
        self.token = token

    def load(self) -> dict[str, Any] | None:
        return self.token

    def save(self, token: dict[str, Any]) -> None:
        self.token = token


def token_path(settings: Settings) -> Path:
    """Single-account token file of the first version (imported into an account, then gone)."""
    return settings.data_dir / "youtube_token.json"


def load_token(settings: Settings) -> dict[str, Any] | None:
    p = token_path(settings)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except ValueError:
        return None


def save_token(settings: Settings, token: dict[str, Any]) -> None:
    p = token_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(token), "utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    tmp.replace(p)


def disconnect(settings: Settings) -> None:
    token_path(settings).unlink(missing_ok=True)


def connected(settings: Settings) -> bool:
    t = load_token(settings)
    return bool(t and t.get("refresh_token"))


# --- OAuth (loopback + PKCE) -------------------------------------------------------------------


def new_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge.rstrip(b"=").decode()


def auth_url(client: Client, redirect_uri: str, state: str, challenge: str) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_id": client.client_id, "redirect_uri": redirect_uri, "response_type": "code",
        "scope": " ".join(SCOPES), "access_type": "offline", "prompt": "consent",
        "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
    })  # fmt: skip


def _token_request(data: dict[str, str], http: httpx.Client) -> dict[str, Any]:
    r = http.post(TOKEN_URL, data=data, timeout=30)
    if r.status_code != 200:
        raise YouTubeError("O Google recusou a autorização do YouTube.", detail=r.text[:1000])
    tok = r.json()
    tok["expires_at"] = time.time() + float(tok.get("expires_in", 3600)) - 60
    return tok


def exchange_code(settings: Settings, code: str, redirect_uri: str, verifier: str,
                  http: httpx.Client | None = None,
                  client_id: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Code -> token (with refresh token, and the project's client_id) + {"channel": {id,
    title}}. The caller stores it."""
    client = load_client(settings, client_id)
    with http or httpx.Client() as h:
        form = {"code": code, "client_id": client.client_id,
                "client_secret": client.client_secret, "redirect_uri": redirect_uri,
                "grant_type": "authorization_code", "code_verifier": verifier}  # fmt: skip
        tok = _token_request(form, h)
        if not tok.get("refresh_token"):
            raise YouTubeError("O Google não devolveu um refresh token; tente conectar de novo.")
        tok["client_id"] = client.client_id
        store = MemoryStore(tok)
        tok["channel"] = YouTube(settings, h, store).channel()
    return store.token


# --- API ---------------------------------------------------------------------------------------


class YouTube:
    def __init__(self, settings: Settings, http: httpx.Client | None = None,
                 store: TokenStore | None = None):  # fmt: skip
        self.settings = settings
        self.http = http or httpx.Client()
        self.store = store or MemoryStore(load_token(settings))

    def _access_token(self) -> str:
        tok = self.store.load()
        if not tok or not tok.get("refresh_token"):
            raise YouTubeError("Esta conta do YouTube perdeu a autorização: conecte de novo.")
        if tok.get("access_token") and tok.get("expires_at", 0) > time.time():
            return tok["access_token"]
        client = load_client(self.settings, tok.get("client_id"))
        fresh = _token_request({"client_id": client.client_id,
                                "client_secret": client.client_secret,
                                "refresh_token": tok["refresh_token"],
                                "grant_type": "refresh_token"}, self.http)  # fmt: skip
        tok.update({k: v for k, v in fresh.items() if k != "refresh_token" or v})
        self.store.save(tok)
        return tok["access_token"]

    def _headers(self, **extra) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token()}", **extra}

    @staticmethod
    def _check(r: httpx.Response, what: str) -> None:
        if r.status_code < 400:
            return
        try:
            reason = r.json()["error"]["errors"][0].get("reason", "")
        except (ValueError, KeyError, IndexError, TypeError):
            reason = ""
        if reason in ("quotaExceeded", "uploadLimitExceeded"):
            raise YouTubeError(
                "Limite diário da API do YouTube atingido (cerca de 6 envios por dia na cota "
                "gratuita). Tente amanhã.", detail=r.text[:1000],
            )  # fmt: skip
        retry = r.status_code >= 500 or r.status_code == 429
        raise YouTubeError(f"YouTube recusou {what} (HTTP {r.status_code} {reason}).".strip(),
                           detail=r.text[:2000], retryable=retry)  # fmt: skip

    def channel(self) -> dict[str, str]:
        r = self.http.get(f"{API}/channels", params={"part": "snippet", "mine": "true"},
                          headers=self._headers(), timeout=30)  # fmt: skip
        self._check(r, "a consulta do canal")
        items = r.json().get("items") or []
        if not items:
            raise YouTubeError("Esta conta Google não tem canal no YouTube.")
        return {"id": items[0]["id"], "title": items[0]["snippet"]["title"]}

    def upload(self, video: Path, meta: dict[str, Any],
               progress_cb: Callable[[float], None] = lambda f: None) -> str:  # fmt: skip
        """Resumable upload; returns the video id. `meta` = videos.insert body."""
        size = video.stat().st_size
        r = self.http.post(
            UPLOAD_URL, params={"uploadType": "resumable", "part": "snippet,status"},
            headers=self._headers(**{"Content-Type": "application/json; charset=UTF-8",
                                     "X-Upload-Content-Type": "video/mp4",
                                     "X-Upload-Content-Length": str(size)}),
            content=json.dumps(meta).encode(), timeout=60,
        )  # fmt: skip
        self._check(r, "o início do envio")
        session_url = r.headers.get("Location")
        if not session_url:
            raise YouTubeError("O YouTube não abriu a sessão de envio.", retryable=True)
        sent = 0
        with video.open("rb") as f:
            while sent < size:
                f.seek(sent)
                chunk = f.read(CHUNK)
                end = sent + len(chunk) - 1
                r = self.http.put(session_url, content=chunk, timeout=300, headers=self._headers(
                    **{"Content-Length": str(len(chunk)),
                       "Content-Range": f"bytes {sent}-{end}/{size}"}))  # fmt: skip
                if r.status_code == 308:  # resume incomplete: continue after what Google has
                    rng = r.headers.get("Range")
                    sent = int(rng.split("-")[1]) + 1 if rng else 0
                    progress_cb(sent / size)
                    continue
                self._check(r, "o envio do vídeo")
                progress_cb(1.0)
                return r.json()["id"]
        raise YouTubeError("O envio terminou sem resposta do YouTube.", retryable=True)

    def stats(self, video_ids: list[str]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for i in range(0, len(video_ids), 50):
            r = self.http.get(f"{API}/videos", headers=self._headers(), timeout=30, params={
                "part": "statistics,status", "id": ",".join(video_ids[i : i + 50])})  # fmt: skip
            self._check(r, "a consulta das estatísticas")
            for it in r.json().get("items", []):
                st = it.get("statistics", {})
                out[it["id"]] = {k: int(st.get(f"{k}Count", 0) or 0)
                                 for k in ("view", "like", "comment")}  # fmt: skip
                out[it["id"]]["privacy"] = it.get("status", {}).get("privacyStatus")
        return out


def video_meta(title: str, description: str, tags: list[str], privacy: str,
               publish_at: str | None = None) -> dict[str, Any]:  # fmt: skip
    """videos.insert body. Shorts are detected by YouTube from the vertical format (+#shorts).
    A scheduled video must be private until `publish_at` (RFC 3339)."""
    tags_clean = [t.lstrip("#") for t in tags if t.strip("#")]
    hashtags = " ".join(f"#{t}" for t in tags_clean[:6])
    desc = description.strip()
    if hashtags and hashtags not in desc:
        desc = f"{desc}\n\n{hashtags}".strip()
    status: dict[str, Any] = {"privacyStatus": "private" if publish_at else privacy,
                              "selfDeclaredMadeForKids": False}  # fmt: skip
    if publish_at:
        status["publishAt"] = publish_at
    return {
        "snippet": {"title": title.strip()[:100] or "Clip", "description": desc[:5000],
                    "tags": tags_clean[:15], "categoryId": GAMING_CATEGORY},
        "status": status,
    }  # fmt: skip
