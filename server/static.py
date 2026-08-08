"""Static file serving with safe client-side routing fallback."""
from __future__ import annotations

from pathlib import PurePosixPath

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class SPAStaticFiles(StaticFiles):
    @staticmethod
    def _should_fallback(path: str) -> bool:
        normalized = path.lstrip("/")
        return not normalized.startswith("api/") and not PurePosixPath(normalized).suffix

    async def get_response(self, path: str, scope: Scope):
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or not self._should_fallback(path):
                raise
            return await super().get_response("index.html", scope)

        if response.status_code == 404 and self._should_fallback(path):
            return await super().get_response("index.html", scope)
        return response
