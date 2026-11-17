"""TikTok Content Posting API (official), with plain httpx: OAuth (Login Kit, PKCE), direct
post by FILE_UPLOAD from this computer (no public URL needed), publish status and the numbers.

Setup (once, free): developers.tiktok.com -> create an app -> add "Login Kit" and "Content
Posting API" (Direct Post) -> scopes user.info.basic, video.publish, video.list -> add the
redirect URI shown on /publicar -> put the client key/secret in .env
(CLIPADOR_TIKTOK_CLIENT_KEY / CLIPADOR_TIKTOK_CLIENT_SECRET).

Until TikTok audits the app, its posts can only be private (SELF_ONLY): Clipador falls back to
that automatically and says so. After the audit, public posting works.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.services.youtube import MemoryStore, TokenStore

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
API = "https://open.tiktokapis.com/v2"
TOKEN_URL = f"{API}/oauth/token/"
SCOPES = ("user.info.basic", "video.publish", "video.list")
MIN_CHUNK = 5 * 1024 * 1024
MAX_SINGLE = 64 * 1024 * 1024
CHUNK = 10 * 1024 * 1024
PRIVACY = ("SELF_ONLY", "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR", "PUBLIC_TO_EVERYONE")
PRIVACY_LABELS = {
    "SELF_ONLY": "Só eu (privado)",
    "MUTUAL_FOLLOW_FRIENDS": "Amigos",
    "FOLLOWER_OF_CREATOR": "Seguidores",
    "PUBLIC_TO_EVERYONE": "Público",
}
FROM_GENERIC = {
    "private": "SELF_ONLY",
    "unlisted": "FOLLOWER_OF_CREATOR",
    "public": "PUBLIC_TO_EVERYONE",
}
SETUP_HINT = (
    "Crie um app em developers.tiktok.com (Login Kit + Content Posting API), adicione a URL de "
    "retorno mostrada em Publicar e defina CLIPADOR_TIKTOK_CLIENT_KEY e "
    "CLIPADOR_TIKTOK_CLIENT_SECRET no .env. Passo a passo no README."
)


class TikTokError(ClipadorError):
    pass


@dataclass
class Client:
    key: str
    secret: str


def load_client(settings: Settings) -> Client:
    if not settings.tiktok_client_key or not settings.tiktok_client_secret:
        raise TikTokError("TikTok não configurado. " + SETUP_HINT)
    return Client(settings.tiktok_client_key, settings.tiktok_client_secret)


def new_pkce() -> tuple[str, str]:
    """TikTok's PKCE uses the HEX SHA-256 of the verifier (not base64url)."""
    verifier = secrets.token_urlsafe(48)[:64]
    return verifier, hashlib.sha256(verifier.encode()).hexdigest()


def auth_url(client: Client, redirect_uri: str, state: str, challenge: str) -> str:
    return (
        AUTH_URL
        + "?"
        + urlencode(
            {
                "client_key": client.key,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": ",".join(SCOPES),
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
    )


def _token(http: httpx.Client, form: dict[str, str]) -> dict[str, Any]:
    r = http.post(
        TOKEN_URL,
        data=form,
        timeout=30,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    data = r.json() if r.content else {}
    if r.status_code != 200 or "access_token" not in data:
        raise TikTokError("O TikTok recusou a autorização.", detail=r.text[:1000])
    data["expires_at"] = time.time() + float(data.get("expires_in", 86400)) - 60
    return data


def exchange_code(
    settings: Settings,
    code: str,
    redirect_uri: str,
    verifier: str,
    http: httpx.Client | None = None,
) -> dict[str, Any]:
    """Code -> token + {"user": {id, name}}. The caller stores it."""
    client = load_client(settings)
    with http or httpx.Client() as h:
        tok = _token(
            h,
            {
                "client_key": client.key,
                "client_secret": client.secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
        )
        store = MemoryStore(tok)
        tok["user"] = TikTok(settings, h, store).user()
    return store.token


class TikTok:
    def __init__(
        self, settings: Settings, http: httpx.Client | None = None, store: TokenStore | None = None
    ):
        self.settings = settings
        self.http = http or httpx.Client()
        self.store = store or MemoryStore()

    def _access_token(self) -> str:
        tok = self.store.load()
        if not tok or not tok.get("refresh_token"):
            raise TikTokError("Esta conta do TikTok perdeu a autorização: conecte de novo.")
        if tok.get("access_token") and tok.get("expires_at", 0) > time.time():
            return tok["access_token"]
        client = load_client(self.settings)
        fresh = _token(
            self.http,
            {
                "client_key": client.key,
                "client_secret": client.secret,
                "grant_type": "refresh_token",
                "refresh_token": tok["refresh_token"],
            },
        )
        tok.update(fresh)
        self.store.save(tok)
        return tok["access_token"]

    def _call(self, method: str, path: str, what: str, **kw) -> dict[str, Any]:
        r = self.http.request(
            method,
            f"{API}{path}",
            timeout=60,
            headers={
                "Authorization": f"Bearer {self._access_token()}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            **kw,
        )
        try:
            body = r.json()
        except ValueError:
            body = {}
        err = (body.get("error") or {}) if isinstance(body, dict) else {}
        code = err.get("code", "ok" if r.status_code < 400 else "http_error")
        if code != "ok":
            if code in ("spam_risk_too_many_posts", "rate_limit_exceeded"):
                raise TikTokError(
                    "Limite de posts do TikTok atingido por agora; tente mais tarde.",
                    detail=r.text[:1000],
                    retryable=False,
                )
            if code == "unaudited_client_can_only_post_to_private_accounts":
                raise TikTokError(
                    "O app do TikTok ainda não foi aprovado pelo TikTok: ele só pode postar em "
                    "contas privadas. Deixe a conta do TikTok privada ou peça a aprovação "
                    "(audit) do app.",
                    detail=r.text[:1000],
                )
            retry = r.status_code >= 500 or r.status_code == 429
            raise TikTokError(
                f"TikTok recusou {what} ({code}: {err.get('message', '')}).",
                detail=r.text[:2000],
                retryable=retry,
            )
        return body.get("data") or {}

    def user(self) -> dict[str, str]:
        d = self._call(
            "GET", "/user/info/", "a consulta do usuário", params={"fields": "open_id,display_name"}
        )
        u = d.get("user") or {}
        return {"id": u.get("open_id", ""), "title": u.get("display_name") or "TikTok"}

    def creator_info(self) -> dict[str, Any]:
        return self._call(
            "POST", "/post/publish/creator_info/query/", "a consulta do criador", json={}
        )

    def upload(
        self,
        video: Path,
        caption: str,
        privacy: str,
        progress_cb: Callable[[float], None] = lambda f: None,
    ) -> tuple[str, str]:
        """Direct post. Returns (publish_id, privacy actually used)."""
        info = self.creator_info()
        options = info.get("privacy_level_options") or ["SELF_ONLY"]
        used = (
            privacy
            if privacy in options
            else ("SELF_ONLY" if "SELF_ONLY" in options else options[0])
        )
        size = video.stat().st_size
        if size <= MAX_SINGLE:
            chunk, count = size, 1
        else:
            chunk, count = CHUNK, size // CHUNK  # the last chunk takes the remainder
        d = self._call(
            "POST",
            "/post/publish/video/init/",
            "o início do envio",
            json={
                "post_info": {
                    "title": caption[:2200],
                    "privacy_level": used,
                    "disable_comment": False,
                    "disable_duet": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": size,
                    "chunk_size": chunk,
                    "total_chunk_count": count,
                },
            },
        )
        publish_id, url = d.get("publish_id"), d.get("upload_url")
        if not publish_id or not url:
            raise TikTokError("O TikTok não abriu o envio.", retryable=True)
        with video.open("rb") as f:
            for i in range(count):
                start = i * chunk
                end = size - 1 if i == count - 1 else start + chunk - 1
                f.seek(start)
                data = f.read(end - start + 1)
                r = self.http.put(
                    url,
                    content=data,
                    timeout=300,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(len(data)),
                        "Content-Range": f"bytes {start}-{end}/{size}",
                    },
                )
                if r.status_code >= 400:
                    raise TikTokError(
                        f"Falha ao enviar o vídeo ao TikTok (HTTP {r.status_code}).",
                        detail=r.text[:1000],
                        retryable=True,
                    )
                progress_cb((i + 1) / count)
        return publish_id, used

    def status(self, publish_id: str) -> dict[str, Any]:
        return self._call(
            "POST",
            "/post/publish/status/fetch/",
            "a consulta do envio",
            json={"publish_id": publish_id},
        )

    def wait_published(
        self, publish_id: str, timeout: float = 600, sleep: Callable[[float], None] = time.sleep
    ) -> str | None:
        """Poll until TikTok finishes processing. Returns the post id (when public)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            st = self.status(publish_id)
            s = st.get("status")
            if s == "PUBLISH_COMPLETE":
                ids = st.get("publicaly_available_post_id") or st.get("publicly_available_post_id")
                return str(ids[0]) if ids else None
            if s == "FAILED":
                raise TikTokError(f"O TikTok recusou o vídeo ({st.get('fail_reason', '?')}).")
            sleep(5)
        raise TikTokError(
            "O TikTok ainda está processando o vídeo; confira no app.", retryable=False
        )

    def stats(self, video_ids: list[str]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for i in range(0, len(video_ids), 20):
            d = self._call(
                "POST",
                "/video/query/",
                "a consulta dos números",
                params={"fields": "id,view_count,like_count,comment_count"},
                json={"filters": {"video_ids": video_ids[i : i + 20]}},
            )
            for v in d.get("videos", []):
                out[str(v["id"])] = {
                    "view": int(v.get("view_count") or 0),
                    "like": int(v.get("like_count") or 0),
                    "comment": int(v.get("comment_count") or 0),
                }
        return out


def caption(description: str, hashtags: list[str]) -> str:
    tags = " ".join(t if t.startswith("#") else f"#{t}" for t in hashtags if t.strip("#"))
    text = description.strip()
    return f"{text} {tags}".strip() if tags and tags not in text else text
