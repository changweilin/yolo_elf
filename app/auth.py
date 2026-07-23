from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

from app.config import Settings

COOKIE_NAME = "yolo_elf_session"


class AuthGuard:
    """Shared-token access control with signed, expiring session cookies.

    A single ``AUTH_TOKEN`` is the credential. Browsers exchange it once on the
    login page for an HttpOnly signed cookie that then authenticates pages, REST
    and WebSocket alike (the browser sends the cookie on the same-origin WS
    handshake, so the token never lands in a URL or access log). Non-browser
    clients can instead send ``Authorization: Bearer <token>``.

    Empty token = disabled: every check returns unauthenticated, and callers are
    expected to gate on :attr:`enabled` so the app stays open (current behaviour).
    """

    def __init__(self, settings: Settings) -> None:
        self.token = settings.auth_token
        self.enabled = bool(self.token)
        self.ttl = int(settings.auth_session_ttl)
        # Derive the signing key from the token itself so there is no second
        # secret to manage; rotating AUTH_TOKEN thus invalidates old cookies.
        self._key = hashlib.sha256(
            b"yolo-elf-auth-v1:" + self.token.encode("utf-8")
        ).digest()

    # --- credential + cookie primitives -------------------------------------

    def verify_credential(self, candidate: str | None) -> bool:
        if not self.enabled or not candidate:
            return False
        return hmac.compare_digest(candidate, self.token)

    def issue_cookie(self, now: float | None = None) -> str:
        expiry = int((time.time() if now is None else now) + self.ttl)
        return f"{expiry}.{self._sign(expiry)}"

    def valid_cookie(self, value: str | None, now: float | None = None) -> bool:
        if not self.enabled or not value or "." not in value:
            return False
        raw_expiry, _, signature = value.partition(".")
        try:
            expiry = int(raw_expiry)
        except ValueError:
            return False
        if not hmac.compare_digest(signature, self._sign(expiry)):
            return False
        return expiry > (time.time() if now is None else now)

    def _sign(self, expiry: int) -> str:
        return hmac.new(
            self._key, str(expiry).encode("ascii"), hashlib.sha256
        ).hexdigest()

    # --- request / websocket checks -----------------------------------------

    @staticmethod
    def _bearer(header: str | None) -> str | None:
        if not header:
            return None
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
        return None

    def request_ok(self, request: Any) -> bool:
        if self.valid_cookie(request.cookies.get(COOKIE_NAME)):
            return True
        return self.verify_credential(self._bearer(request.headers.get("authorization")))

    def websocket_ok(self, websocket: Any) -> bool:
        if self.valid_cookie(websocket.cookies.get(COOKIE_NAME)):
            return True
        if self.verify_credential(self._bearer(websocket.headers.get("authorization"))):
            return True
        # Last-resort fallback for non-browser clients that can set neither a
        # cookie nor a header. Browsers use the cookie, so their token never hits
        # a URL; a query token here would be logged, so prefer the cookie/header.
        return self.verify_credential(websocket.query_params.get("token"))
