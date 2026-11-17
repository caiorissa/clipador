"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from clipador.config import Settings, get_settings
from clipador.db import make_engine, run_migrations
from clipador.web.deps import STATIC_DIR
from clipador.web.routes import analysis, jobs, live, review, sources, streamers, system


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    run_migrations(settings)

    app = FastAPI(title="Clipador", docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.engine = make_engine(settings)
    from clipador.web.deps import templates

    templates.env.globals["default_cut_silences"] = settings.render_cut_silences

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    for module in (
        streamers,
        sources,
        analysis,
        review,
        live,
        jobs,
        system,
    ):  # sources before analysis
        app.include_router(module.router)

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/fontes", status_code=303)

    @app.get("/arquivos/{rel_path:path}", include_in_schema=False)
    def data_file(rel_path: str):
        """Serve files from the data dir (thumbnails now, clips later). No path traversal."""
        base = settings.data_dir.resolve()
        target = (base / rel_path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target)

    return app
