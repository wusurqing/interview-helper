"""P1-13 + P2-3 回归测试：消除 async def 阻塞事件循环、KB force/增量索引、截屏解码、泳道隔离与 lifespan 规范停机。"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def test_review_asr_correction_and_sql_routes_run_off_event_loop(monkeypatch: pytest.MonkeyPatch):
    review_router = importlib.import_module("api.review.router")
    tids: dict[str, int] = {}

    def fake_check(question: str, candidate_answer: str):
        tids["asr"] = threading.get_ident()
        return {"ok": True, "original": candidate_answer, "corrected": candidate_answer, "changed": False}

    def fake_list(page: int, page_size: int):
        tids["list"] = threading.get_ident()
        return {"items": [], "total": 0, "page": page, "page_size": page_size}

    monkeypatch.setattr(review_router.review_analysis, "run_asr_correction_check", fake_check)
    monkeypatch.setattr(review_router.review, "list_sessions", fake_list)

    async def exercise():
        loop_tid = threading.get_ident()
        await review_router.test_asr_correction(
            review_router.AsrCorrectionTestRequest(question="Q", answer="A")
        )
        await review_router.list_sessions(1, 20)
        return loop_tid

    loop_tid = asyncio.run(exercise())
    assert tids["asr"] != loop_tid
    assert tids["list"] != loop_tid


def test_review_manual_length_limit_and_update_error_sanitization(monkeypatch: pytest.MonkeyPatch):
    review_router = importlib.import_module("api.review.router")
    app = FastAPI()
    app.include_router(review_router.router, prefix="/api")

    monkeypatch.setattr(
        review_router.review,
        "get_session_detail",
        lambda _sid: (_ for _ in ()).throw(RuntimeError("internal-sql-secret-detail")),
    )

    with TestClient(app) as client:
        too_long = client.post(
            "/api/review/sessions/manual",
            json={"transcript": "Q: " + "x" * 200_005},
        )
        assert too_long.status_code == 422

        err = client.patch("/api/review/sessions/1", json={"title": "new"})
        assert err.status_code == 500
        assert "internal-sql-secret-detail" not in err.json()["detail"]


def test_assist_screen_capture_and_cancel_run_off_event_loop(monkeypatch: pytest.MonkeyPatch):
    assist_routes = importlib.import_module("api.assist.routes")
    tids: dict[str, int] = {}

    def fake_capture():
        tids["capture"] = threading.get_ident()
        return "data:image/png;base64,fake"

    def fake_cancel(**_kwargs):
        tids["cancel"] = threading.get_ident()

    monkeypatch.setattr("services.capture.capture_primary_left_half_data_url", fake_capture)
    monkeypatch.setattr(assist_routes, "cancel_answer_work", fake_cancel)
    monkeypatch.setattr(assist_routes.mmpg, "clear_turns", lambda: None)
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(written_exam_mode=False))

    async def exercise():
        loop_tid = threading.get_ident()
        await assist_routes.api_capture_server_screen()
        await assist_routes.api_ask_cancel()
        return loop_tid

    loop_tid = asyncio.run(exercise())
    assert tids["capture"] != loop_tid
    assert tids["cancel"] != loop_tid


def test_common_devices_enumerates_once_off_loop_and_sanitizes_errors(monkeypatch: pytest.MonkeyPatch):
    common_router = importlib.import_module("api.common.router")
    calls = {"list_devices": 0, "tid": None}
    fake_devs = [{"id": 1, "name": "Loopback", "is_loopback": True}]

    def fake_list_devices():
        calls["list_devices"] += 1
        calls["tid"] = threading.get_ident()
        return fake_devs

    monkeypatch.setattr(common_router.AudioCapture, "list_devices", staticmethod(fake_list_devices))

    async def exercise():
        loop_tid = threading.get_ident()
        res = await common_router.api_devices()
        return loop_tid, res

    loop_tid, res = asyncio.run(exercise())
    assert calls["list_devices"] == 1
    assert calls["tid"] != loop_tid
    assert res["devices"] == fake_devs
    assert res["platform"]["has_loopback"] is True

    # 错误脱敏：不透出原始异常文本
    def broken_devices():
        raise RuntimeError("secret-driver-internal-pointer-0xdeadbeef")

    monkeypatch.setattr(common_router.AudioCapture, "list_devices", staticmethod(broken_devices))
    err_res = asyncio.run(common_router.api_devices())
    assert err_res["devices"] == []
    assert "0xdeadbeef" not in err_res["platform"]["error"]


def test_common_network_info_closes_udp_socket_on_error(monkeypatch: pytest.MonkeyPatch):
    common_router = importlib.import_module("api.common.router")
    closed = {"flag": False}

    class _FailingSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed["flag"] = True

        def connect(self, _addr):
            raise OSError("network unreachable")

    monkeypatch.setattr(common_router.socket, "socket", lambda *_args, **_kwargs: _FailingSocket())
    assert common_router._detect_lan_ip() == "127.0.0.1"
    assert closed["flag"] is True


def test_jobs_router_routes_are_sync_and_enforce_text_limits():
    import inspect

    jobs_router = importlib.import_module("api.jobs.router")
    route_fns = [
        jobs_router.api_stages,
        jobs_router.api_list_applications,
        jobs_router.api_create_application,
        jobs_router.api_batch_stage,
        jobs_router.api_reorder_stage,
        jobs_router.api_patch_application,
        jobs_router.api_application_reviews,
        jobs_router.api_delete_application,
        jobs_router.api_list_offers,
        jobs_router.api_create_offer,
        jobs_router.api_patch_offer,
        jobs_router.api_delete_offer,
        jobs_router.api_compare,
    ]
    for fn in route_fns:
        assert not inspect.iscoroutinefunction(fn), f"{fn.__name__} should be sync def"

    app = FastAPI()
    app.include_router(jobs_router.router, prefix="/api")
    with TestClient(app) as client:
        res = client.post("/api/job-tracker/applications", json={"company": "A" * 201})
        assert res.status_code == 422
        res2 = client.post("/api/job-tracker/applications", json={"notes": "N" * 50_001})
        assert res2.status_code == 422


def test_analytics_router_validates_pagination_and_offloads_to_threadpool(monkeypatch: pytest.MonkeyPatch):
    analytics_router = importlib.import_module("api.analytics.router")
    seen_tid = {}

    def fake_history(page, page_size, session_types):
        seen_tid["tid"] = threading.get_ident()
        return {"records": [], "total": 0, "page": page, "page_size": page_size}

    monkeypatch.setattr(analytics_router, "get_history", fake_history)

    async def exercise():
        loop_tid = threading.get_ident()
        await analytics_router.api_knowledge_history(page=1, page_size=20)
        return loop_tid

    loop_tid = asyncio.run(exercise())
    assert seen_tid["tid"] != loop_tid

    app = FastAPI()
    app.include_router(analytics_router.router, prefix="/api")
    with TestClient(app) as client:
        assert client.get("/api/knowledge/history?page=0").status_code == 422
        assert client.get("/api/knowledge/history?page_size=0").status_code == 422
        assert client.get("/api/knowledge/history?page_size=201").status_code == 422
        assert client.get("/api/knowledge/history?page=1&page_size=20").status_code == 200


def test_screen_capture_handles_non_ascii_stdout_without_unicode_error(monkeypatch: pytest.MonkeyPatch):
    sc = importlib.import_module("services.capture.screen_capture")

    # 1. 非 ASCII 杂音行 + 有效 JSON 行：不抛 UnicodeDecodeError，成功提取
    img_b64 = base64.b64encode(b"x" * 120).decode("ascii")
    payload = json.dumps({"mime": "image/jpeg", "data": img_b64}).encode("utf-8")
    noisy_stdout = b"\xff\xfe \xe4\xb8\xad\xe6\x96\x87\xe8\xad\xa6\xe5\x91\x8a\n" + payload

    monkeypatch.setattr(
        sc.subprocess,
        "run",
        lambda **_kwargs: subprocess.CompletedProcess(args=[], returncode=0, stdout=noisy_stdout, stderr=b""),
    )
    url = sc.capture_primary_region_data_url("left_half")
    assert url == f"data:image/jpeg;base64,{img_b64}"

    # 2. 纯非 ASCII 乱码 stdout：转为受控 ScreenCaptureError，绝不泄漏 UnicodeDecodeError
    garbage = b"\xff\xfe\x80\x90" * 60
    monkeypatch.setattr(
        sc.subprocess,
        "run",
        lambda **_kwargs: subprocess.CompletedProcess(args=[], returncode=0, stdout=garbage, stderr=b""),
    )
    with pytest.raises(sc.ScreenCaptureError):
        sc.capture_primary_region_data_url("left_half")


def test_kb_reindex_force_and_incremental_reindex_file_and_limit_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from core.config import get_config
    from services.kb import indexer
    from api import kb as kb_api

    cfg = get_config()
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    monkeypatch.setattr(cfg, "kb_dir", str(kb_dir), raising=False)
    monkeypatch.setattr(cfg, "kb_db_path", str(tmp_path / "kb.sqlite"), raising=False)
    monkeypatch.setattr(cfg, "kb_cache_dir", str(tmp_path / "kb_cache"), raising=False)
    monkeypatch.setattr(cfg, "kb_enabled", True, raising=False)

    (kb_dir / "a.md").write_text("# A\n\nAlpha content", encoding="utf-8")
    (kb_dir / "b.md").write_text("# B\n\nBeta content", encoding="utf-8")
    indexer.reset()

    app = FastAPI()
    app.include_router(kb_api.router, prefix="/api")

    with TestClient(app) as client:
        # 首次 reindex 索引 2 个文件
        r1 = client.post("/api/kb/reindex", json={"force": False})
        assert r1.status_code == 200
        assert r1.json()["reindexed_files"] == 2

        # 再次 force=False 跳过未改动文件
        r2 = client.post("/api/kb/reindex", json={"force": False})
        assert r2.status_code == 200
        assert r2.json()["reindexed_files"] == 0

        # force=True 强制重建全部文件
        r3 = client.post("/api/kb/reindex", json={"force": True})
        assert r3.status_code == 200
        assert r3.json()["reindexed_files"] == 2

        # limit < 1 被 Query(ge=1) 拒绝返回 422
        assert client.get("/api/kb/docs?limit=0").status_code == 422
        assert client.get("/api/kb/docs?limit=-5").status_code == 422
        assert client.get("/api/kb/hits/recent?limit=0").status_code == 422

    # 增量 reindex_file 不扫描目录内其他未索引文件
    (kb_dir / "untracked.md").write_text("# Untracked\n\nShould not be scanned", encoding="utf-8")
    (kb_dir / "target.md").write_text("# Target\n\nOnly this one", encoding="utf-8")
    info = indexer.reindex_file("target.md")
    assert info["reindexed_files"] == 1
    indexed_paths = {d["path"] for d in indexer.list_docs()}
    assert "target.md" in indexed_paths
    assert "untracked.md" not in indexed_paths
    indexer.reset()


def test_resource_lanes_isolate_kb_queries_from_long_running_llm_tasks():
    rl = importlib.import_module("core.resource_lanes")
    rl.shutdown_resource_lanes(wait=True)

    llm_started = threading.Event()
    release_llm = threading.Event()

    def slow_llm_job():
        llm_started.set()
        release_llm.wait(timeout=5.0)

    try:
        assert rl.submit_low_priority_background(slow_llm_job) is True
        assert llm_started.wait(timeout=2.0)

        # 后台 LLM 泳道被长任务占满期间，KB 查询泳道与索引泳道仍可立即完成
        t0 = time.monotonic()
        res_query = asyncio.run(rl.run_kb_query(lambda: "kb-ok"))
        res_index = asyncio.run(rl.run_kb_index(lambda: "index-ok"))
        elapsed = time.monotonic() - t0
        assert res_query == "kb-ok"
        assert res_index == "index-ok"
        assert elapsed < 1.0
    finally:
        release_llm.set()
        rl.shutdown_resource_lanes(wait=True)


def test_lifespan_shutdown_runs_stop_off_loop_awaits_ws_and_shuts_down_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import main as main_mod
    from services.storage import review
    rl = importlib.import_module("core.resource_lanes")

    monkeypatch.setattr(review, "DB_PATH", str(tmp_path / "review.db"))
    review.init_db()
    monkeypatch.setattr("services.storage.resume_history.restore_active_resume", lambda: False)

    stop_tids: list[int] = []
    shutdown_lanes_called = {"count": 0}
    real_shutdown_lanes = rl.shutdown_resource_lanes

    def spy_stop_loop():
        stop_tids.append(threading.get_ident())

    def spy_shutdown_lanes(*, wait: bool = False):
        shutdown_lanes_called["count"] += 1
        return real_shutdown_lanes(wait=wait)

    monkeypatch.setattr(main_mod.assist, "stop_interview_loop", spy_stop_loop)
    monkeypatch.setattr(main_mod.assist, "shutdown_background_workers", lambda: None)
    monkeypatch.setattr(rl, "shutdown_resource_lanes", spy_shutdown_lanes)

    with TestClient(main_mod.app) as client:
        assert client.get("/api/options").status_code == 200

    assert len(stop_tids) == 1
    assert shutdown_lanes_called["count"] == 1
    # 关闭后再提交仍可懒重建，不影响同进程后续请求/测试
    assert asyncio.run(rl.run_kb_query(lambda: 42)) == 42
    rl.shutdown_resource_lanes(wait=True)


def test_resume_optimize_enforces_jd_max_length_and_knowledge_tags_strip_markdown_fences(
    monkeypatch: pytest.MonkeyPatch,
):
    from core.config import KnowledgeAnalysisSettings, ModelConfig
    from services import knowledge_analysis
    from services.storage import knowledge
    resume_router = importlib.import_module("api.resume.router")

    app = FastAPI()
    app.include_router(resume_router.router, prefix="/api")
    with TestClient(app) as client:
        too_long = client.post("/api/resume/optimize", json={"jd": "x" * 50_005})
        assert too_long.status_code == 422

    monkeypatch.setattr(
        knowledge_analysis.streaming,
        "chat_completion_create",
        lambda *_args, **_kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='```json\n["Redis", "缓存穿透", "布隆过滤器"]\n```'
                    )
                )
            ]
        ),
    )
    record = knowledge.AnalysisRecord(
        id=1,
        question="Redis 如何防穿透？",
        answer="布隆过滤器",
        candidate_answer="我讲了布隆过滤器和空值缓存",
        tags=None,
    )
    tags = knowledge_analysis._extract_tags(
        record,
        ModelConfig(name="M", model="m", api_key="k"),
        KnowledgeAnalysisSettings(),
        "sys",
        {},
        "job-1",
    )
    assert tags == ["Redis", "缓存穿透", "布隆过滤器"]
