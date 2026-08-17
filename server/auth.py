"""Optional HTTP Basic authentication for public GSStudio deployments."""
from __future__ import annotations

import base64
import binascii
import hmac
import json
from collections.abc import Iterable

from starlette.types import ASGIApp, Receive, Scope, Send


class BasicAuthMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        username: str = "",
        password: str = "",
        service_token: str = "",
        exempt_paths: Iterable[str] = ("/api/health",),
    ) -> None:
        self.app = app
        self.username = username
        self.password = password
        self.service_token = service_token
        self.enabled = bool((username and password) or service_token)
        self.exempt_paths = frozenset(exempt_paths)

    def _authorized(self, scope: Scope) -> bool:
        for name, value in scope.get("headers", []):
            if name.lower() != b"authorization":
                continue
            try:
                scheme, encoded = value.decode("latin-1").split(" ", 1)
                if scheme.lower() == "bearer":
                    return bool(self.service_token) and hmac.compare_digest(encoded, self.service_token)
                if scheme.lower() == "basic":
                    decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
                    username, password = decoded.split(":", 1)
                    return (
                        bool(self.username and self.password)
                        and hmac.compare_digest(username, self.username)
                        and hmac.compare_digest(password, self.password)
                    )
                return False
            except (ValueError, UnicodeDecodeError, binascii.Error):
                return False
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            not self.enabled
            or scope["type"] not in {"http", "websocket"}
            or scope.get("path") in self.exempt_paths
            or self._authorized(scope)
        ):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401, "reason": "Authentication required"})
            return

        body = json.dumps({"detail": "Authentication required"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b'Basic realm="GSStudio"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
