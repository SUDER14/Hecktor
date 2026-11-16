"""The Studio's HTTP surface: a page, its static files, and three read-only JSON / binary endpoints.

Licence rules enforced here rather than left to the page:
  * requests are answered only when addressed to 127.0.0.1 or localhost (a page on another site cannot reach the
    server by DNS rebinding), and no cross-origin header is ever sent;
  * a Content-Security-Policy of 'self' stops the page from contacting any other host;
  * static files are served from an explicit list of suffixes, so nothing else in the package is reachable;
  * there is no endpoint that writes, exports or downloads: the display volume carries no filename header and
    is marked no-store.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.studio import cases
from app.studio.settings import HOST, Settings, load_settings

STATIC = Path(__file__).resolve().parent / "static"
SERVED_SUFFIXES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                   ".js": "text/javascript; charset=utf-8", ".woff2": "font/woff2"}
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


def served_files() -> list[Path]:
    """Every file the server can send. The licence test scans exactly this list for outside URLs."""
    return sorted(p for p in STATIC.rglob("*") if p.is_file() and p.suffix in SERVED_SUFFIXES)


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or load_settings()
    app = FastAPI(title="ACF Studio", docs_url=None, redoc_url=None, openapi_url=None)   # the docs pages load a CDN

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "").rsplit(":", 1)[0]
        if host not in ALLOWED_HOSTS:
            return JSONResponse({"detail": "this server answers only on 127.0.0.1"}, status_code=403)
        if request.method not in ("GET", "HEAD"):
            return JSONResponse({"detail": "read-only"}, status_code=405)
        resp = await call_next(request)
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    def _static(rel: str) -> Response:
        p = (STATIC / rel).resolve()
        if STATIC.resolve() not in p.parents or not p.is_file() or p.suffix not in SERVED_SUFFIXES:
            raise HTTPException(404, "not found")
        return Response(p.read_bytes(), media_type=SERVED_SUFFIXES[p.suffix])

    @app.get("/")
    def page() -> Response:
        return _static("index.html")

    @app.get("/static/{rel:path}")
    def static(rel: str) -> Response:
        return _static(rel)

    @app.get("/api/index")
    def index() -> dict:
        return cases.build_index(s)

    @app.get("/api/cases/{pid}")
    def meta(pid: str) -> dict:
        try:
            return cases.case_meta(s, pid)
        except cases.MissingSource as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/cases/{pid}/volume")
    def volume(pid: str) -> Response:
        try:
            header, arrays = cases.display_volume(s, pid)
        except cases.MissingSource as e:
            raise HTTPException(404, str(e)) from e
        return Response(cases.pack_volume(header, arrays), media_type="application/octet-stream")

    return app


def run(settings: Settings | None = None) -> None:
    import uvicorn

    s = settings or load_settings()
    print(f"http://{HOST}:{s.port}", flush=True)
    uvicorn.run(create_app(s), host=HOST, port=s.port, log_level="warning", access_log=False)
