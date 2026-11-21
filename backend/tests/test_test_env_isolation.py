"""Verify pytest global fixture isolates config, SQLite data dir, KB paths, and logs."""

from __future__ import annotations

import os
from pathlib import Path

from core import config, logger
from services.kb import indexer
from services.storage import job_tracker, knowledge, paths, resume_history, review


def test_default_paths_are_isolated_from_repo_backend():
    backend_root = Path(paths.backend_root()).resolve()
    repo_root = backend_root.parent

    config_path = Path(config.CONFIG_FILE).resolve()
    data_path = Path(paths.data_dir()).resolve()
    log_path = Path(logger.LOG_DIR).resolve()

    assert config_path != (backend_root / "config.json").resolve()
    assert not config_path.is_relative_to(backend_root)
    assert not data_path.is_relative_to(backend_root)
    assert not log_path.is_relative_to(repo_root)

    for db_path in (
        job_tracker.DB_PATH,
        knowledge.DB_PATH,
        resume_history.DB_PATH,
        review.DB_PATH,
    ):
        assert Path(db_path).resolve().is_relative_to(data_path)

    cfg = config.get_config()
    assert indexer.resolve_path(cfg.kb_dir).resolve().is_relative_to(data_path)
    assert indexer.resolve_path(cfg.kb_db_path).resolve().is_relative_to(data_path)
    assert indexer.resolve_path(cfg.kb_cache_dir).resolve().is_relative_to(data_path)


def test_sqlite_path_does_not_migrate_legacy_root_db_when_data_dir_overridden(tmp_path: Path, monkeypatch):
    fake_backend = tmp_path / "backend"
    fake_backend.mkdir()
    legacy_db = fake_backend / "legacy.db"
    legacy_db.write_bytes(b"real-user-data")

    isolated_data = tmp_path / "isolated_data"
    monkeypatch.setattr(paths, "_BACKEND_ROOT", str(fake_backend))
    monkeypatch.setenv("IA_DATA_DIR", str(isolated_data))

    resolved = Path(paths.sqlite_path("legacy.db"))
    assert resolved == isolated_data.resolve() / "legacy.db"
    assert legacy_db.exists()
    assert not resolved.exists()
    assert os.path.isdir(isolated_data)


def test_realistic_interview_script_imports_without_optional_aiohttp(monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setitem(sys.modules, "aiohttp", None)
    script_path = Path(__file__).with_name("test_realistic_interview.py")
    spec = importlib.util.spec_from_file_location("test_realistic_interview_probe", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "run_test")
