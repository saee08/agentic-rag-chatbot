"""Programmatic Alembic entry point (``python -m app.db.migrate`` == ``alembic upgrade head``)."""

from __future__ import annotations

from alembic import command
from alembic.config import Config

from app.core.config import BACKEND_DIR, get_settings


def alembic_config(database_url: str | None = None) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.attributes["database_url"] = database_url or get_settings().database_url
    cfg.attributes["configure_logger"] = False
    return cfg


def upgrade(database_url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


if __name__ == "__main__":
    upgrade()
    print("Database migrated to head.")
