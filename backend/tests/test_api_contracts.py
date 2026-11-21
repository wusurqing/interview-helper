from __future__ import annotations

import asyncio
from pathlib import Path
import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

main_mod = importlib.import_module("main")
assist_routes = importlib.import_module("api.assist.routes")


def test_is_path_within_dir_rejects_same_prefix_sibling(tmp_path: Path):
    safe_dir = tmp_path / "dist"
    sibling_dir = tmp_path / "dist-evil"
    safe_dir.mkdir()
    sibling_dir.mkdir()

    assert main_mod._is_path_within_dir(safe_dir, sibling_dir / "index.html") is False


def test_is_path_within_dir_rejects_symlink_escape(tmp_path: Path):
    import os

    safe_dir = tmp_path / "dist"
    outside_dir = tmp_path / "outside"
    safe_dir.mkdir()
    outside_dir.mkdir()
    secret = outside_dir / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link_dir = safe_dir / "escape"
    try:
        os.symlink(outside_dir, link_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform == "win32":
            import _winapi

            _winapi.CreateJunction(str(outside_dir), str(link_dir))
        else:
            pytest.skip("cannot create directory link on this platform")

    assert main_mod._is_path_within_dir(safe_dir, link_dir / "secret.txt") is False


def test_auth_info_and_network_info_block_cross_origin_and_dns_rebinding(monkeypatch: pytest.MonkeyPatch):
    common_router = importlib.import_module("api.common.router")
    monkeypatch.setattr(main_mod, "is_auth_disabled", lambda: False)
    monkeypatch.setattr(main_mod, "get_token", lambda: "lan-secret-token")
    monkeypatch.setattr(common_router, "is_auth_disabled", lambda: False)
    monkeypatch.setattr(common_router, "get_token", lambda: "lan-secret-token")

    def make_req(*, client_host="127.0.0.1", origin=None, hostname="127.0.0.1", port=18080):
        return SimpleNamespace(
            client=SimpleNamespace(host=client_host),
            headers=({"origin": origin} if origin else {}),
            url=SimpleNamespace(scheme="http", hostname=hostname, port=port),
        )

    # Cross-origin browser request from loopback must be denied token.
    cross_origin = make_req(origin="https://evil.example")
    auth_resp = asyncio.run(main_mod.api_auth_info(cross_origin))
    assert getattr(auth_resp, "status_code", 200) == 403
    net_resp = asyncio.run(common_router.api_network_info(cross_origin))
    assert "lan-secret-token" not in net_resp["url"]

    # DNS-rebinding Host header (non-loopback hostname) must be denied token.
    rebound = make_req(hostname="attacker.example")
    auth_rebound = asyncio.run(main_mod.api_auth_info(rebound))
    assert getattr(auth_rebound, "status_code", 200) == 403
    net_rebound = asyncio.run(common_router.api_network_info(rebound))
    assert "lan-secret-token" not in net_rebound["url"]

    # Legitimate same-origin loopback receives token.
    local_ok = make_req(origin="http://127.0.0.1:18080")
    assert asyncio.run(main_mod.api_auth_info(local_ok)) == {"required": True, "token": "lan-secret-token"}
    net_ok = asyncio.run(common_router.api_network_info(local_ok))
    assert net_ok["url"].endswith("/?t=lan-secret-token")


def _request_with_origin(origin: str | None, host: str = "127.0.0.1", port: int = 18080):
    return SimpleNamespace(
        headers=({"origin": origin} if origin else {}),
        url=SimpleNamespace(scheme="http", hostname=host, port=port),
    )


def test_loopback_bypass_allows_non_browser_requests():
    assert main_mod._origin_allows_loopback_bypass(_request_with_origin(None)) is True


def test_loopback_bypass_rejects_cross_origin_browser_requests():
    req = _request_with_origin("http://localhost:5173")

    assert main_mod._origin_allows_loopback_bypass(req) is False


def test_loopback_bypass_allows_same_port_loopback_origin():
    req = _request_with_origin("http://localhost:18080")

    assert main_mod._origin_allows_loopback_bypass(req) is True


def test_api_start_rejects_non_integer_device_id():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(assist_routes.api_start({"device_id": "abc"}))

    assert exc_info.value.status_code == 400
    assert "device_id" in str(exc_info.value.detail)


def test_api_ask_returns_error_when_no_answer_model_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(assist_routes, "submit_answer_task", lambda task, **_kwargs: False)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            assist_routes.api_ask(assist_routes.ManualQuestion(text="解释一下 CAP 定理"))
        )

    assert exc_info.value.status_code == 503
    assert "模型" in str(exc_info.value.detail)


@pytest.mark.parametrize("capture_mode", [False, True])
def test_screen_capture_reports_its_workspace(monkeypatch: pytest.MonkeyPatch, capture_mode: bool):
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(written_exam_mode=capture_mode))
    monkeypatch.setattr("services.capture.capture_primary_left_half_data_url", lambda: "data:image/png;base64,fake")

    result = asyncio.run(assist_routes.api_capture_server_screen())

    assert result["image"] == "data:image/png;base64,fake"
    assert result["written_exam_mode"] is capture_mode


@pytest.mark.parametrize("capture_mode", [False, True])
def test_screen_batch_rejects_another_workspace_before_any_answer_side_effect(
    monkeypatch: pytest.MonkeyPatch, capture_mode: bool,
):
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(written_exam_mode=not capture_mode))

    def unexpected(*_args, **_kwargs):
        pytest.fail("a batch from a different workspace must not touch answer work")

    monkeypatch.setattr(assist_routes, "pick_model_index", unexpected)
    monkeypatch.setattr(assist_routes, "cancel_answer_work", unexpected)
    monkeypatch.setattr(assist_routes, "submit_answer_task", unexpected)
    body = assist_routes.MultiServerScreenQuestion(images=["image"], expected_written_exam_mode=capture_mode)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(assist_routes.api_ask_from_server_screens(body))
    assert exc.value.status_code == 409
    assert "重新截图" in exc.value.detail


def test_api_ask_from_server_screens_submits_multi_image_task(monkeypatch: pytest.MonkeyPatch):
    submitted = []
    cancel_calls = []

    monkeypatch.setattr("services.llm.has_vision_model", lambda: True)
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(
        language="zh",
        screen_capture_region="left_half",
        written_exam_mode=False,
    ))
    monkeypatch.setattr(assist_routes, "pick_model_index", lambda task, busy, **_kwargs: 0)
    monkeypatch.setattr(assist_routes, "cancel_answer_work", lambda reset_session_data=False: cancel_calls.append(reset_session_data))
    monkeypatch.setattr(assist_routes, "submit_answer_task", lambda task, **_kwargs: submitted.append(task) or True)

    res = asyncio.run(
        assist_routes.api_ask_from_server_screens(
            assist_routes.MultiServerScreenQuestion(images=["data:image/png;base64,a", "data:image/png;base64,b"])
        )
    )

    assert res == {"ok": True}
    assert len(submitted) == 1
    text, images, manual, source, meta = submitted[0]
    assert "2 张连续截图" in text
    assert images == ["data:image/png;base64,a", "data:image/png;base64,b"]
    assert manual is True
    assert source == "server_screen_multi"
    assert meta["image_count"] == 2
    assert "override_think_mode" not in meta
    assert cancel_calls == [False]


def test_api_ask_from_server_screens_skips_cancel_in_written_exam(monkeypatch: pytest.MonkeyPatch):
    submitted = []
    cancel_calls = []

    monkeypatch.setattr("services.llm.has_vision_model", lambda: True)
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(
        language="zh",
        screen_capture_region="left_half",
        written_exam_mode=True,
    ))
    monkeypatch.setattr(assist_routes, "pick_model_index", lambda task, busy, **_kwargs: 0)
    monkeypatch.setattr(assist_routes, "cancel_answer_work", lambda reset_session_data=False: cancel_calls.append(reset_session_data))
    monkeypatch.setattr(assist_routes, "submit_answer_task", lambda task, **_kwargs: submitted.append(task) or True)

    res = asyncio.run(
        assist_routes.api_ask_from_server_screens(
            assist_routes.MultiServerScreenQuestion(
                images=["data:image/png;base64,a"],
                override_think_mode=True,
            )
        )
    )

    assert res == {"ok": True}
    assert len(submitted) == 1
    meta = submitted[0][4]
    assert meta["override_think_mode"] is True
    assert cancel_calls == []


def test_api_ask_from_server_screen_cancels_existing_generation(monkeypatch: pytest.MonkeyPatch):
    submitted = []
    cancel_calls = []

    monkeypatch.setattr("services.llm.has_vision_model", lambda: True)
    monkeypatch.setattr("services.capture.capture_primary_left_half_data_url", lambda: "data:image/jpeg;base64,a")
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(
        language="zh",
        screen_capture_region="left_half",
        written_exam_mode=False,
    ))
    monkeypatch.setattr(assist_routes, "pick_model_index", lambda task, busy, **_kwargs: 0)
    monkeypatch.setattr(assist_routes, "cancel_answer_work", lambda reset_session_data=False: cancel_calls.append(reset_session_data))
    monkeypatch.setattr(assist_routes, "submit_answer_task", lambda task, **_kwargs: submitted.append(task) or True)

    res = asyncio.run(assist_routes.api_ask_from_server_screen())

    assert res == {"ok": True}
    assert len(submitted) == 1
    assert submitted[0][1] == "data:image/jpeg;base64,a"
    assert submitted[0][3] == "server_screen_left"
    assert "override_think_mode" not in submitted[0][4]
    assert cancel_calls == [False]


def test_api_ask_from_server_screen_skips_cancel_in_written_exam(monkeypatch: pytest.MonkeyPatch):
    submitted = []
    cancel_calls = []

    monkeypatch.setattr("services.llm.has_vision_model", lambda: True)
    monkeypatch.setattr("services.capture.capture_primary_left_half_data_url", lambda: "data:image/jpeg;base64,a")
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(
        language="zh",
        screen_capture_region="left_half",
        written_exam_mode=True,
    ))
    monkeypatch.setattr(assist_routes, "pick_model_index", lambda task, busy, **_kwargs: 0)
    monkeypatch.setattr(assist_routes, "cancel_answer_work", lambda reset_session_data=False: cancel_calls.append(reset_session_data))
    monkeypatch.setattr(assist_routes, "submit_answer_task", lambda task, **_kwargs: submitted.append(task) or True)

    res = asyncio.run(
        assist_routes.api_ask_from_server_screen(
            assist_routes.ServerScreenQuestion(override_think_mode=False)
        )
    )

    assert res == {"ok": True}
    assert len(submitted) == 1
    assert submitted[0][4]["override_think_mode"] is False
    assert cancel_calls == []


def test_api_preflight_replay_validates_params_and_returns_result(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(
        stt_provider="doubao", doubao_stt_api_key="test-key",
    ))
    monkeypatch.setattr(
        assist_routes,
        "get_session",
        lambda: SimpleNamespace(is_recording=False, is_paused=False),
    )

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    replay_calls = {}

    def fake_replay(device_id, *, repeats=3, gap_sec=0.25, audio_path=None, expected_phrase=None):
        replay_calls.update(
            {
                "device_id": device_id,
                "repeats": repeats,
                "gap_sec": gap_sec,
                "audio_path": audio_path,
                "expected_phrase": expected_phrase,
            }
        )
        return {"device_id": device_id, "success_count": repeats, "failure_count": 0}

    monkeypatch.setattr(assist_routes, "run_in_threadpool", fake_run_in_threadpool)
    monkeypatch.setattr("api.assist.sound_test.replay_preflight_capture_stt", fake_replay)

    from api.assist import sound_test

    result = asyncio.run(
        assist_routes.api_preflight_replay({
            "device_id": 7,
            "repeats": 4,
            "gap_sec": 0.1,
            "fixture_id": "default",
            "expected_phrase": "你好",
        })
    )

    assert result == {"device_id": 7, "success_count": 4, "failure_count": 0}
    assert replay_calls == {
        "device_id": 7,
        "repeats": 4,
        "gap_sec": 0.1,
        "audio_path": sound_test.PREFLIGHT_AUDIO_PATH.resolve(),
        "expected_phrase": "你好",
    }


@pytest.mark.parametrize(
    "audio_path",
    [
        "",
        "tmp/sample.wav",
        "C:\\Windows\\win.ini",
        "\\\\attacker\\share\\probe.wav",
        "//attacker/share/probe.wav",
        "../secret.wav",
    ],
)
def test_api_preflight_replay_rejects_arbitrary_audio_path(monkeypatch: pytest.MonkeyPatch, audio_path: str):
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(
        stt_provider="doubao", doubao_stt_api_key="test-key",
    ))
    monkeypatch.setattr(
        assist_routes,
        "get_session",
        lambda: SimpleNamespace(is_recording=False, is_paused=False),
    )

    def should_not_run(*_args, **_kwargs):
        pytest.fail("replay should not be invoked when arbitrary audio_path is supplied")

    monkeypatch.setattr("api.assist.sound_test.replay_preflight_capture_stt", should_not_run)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(assist_routes.api_preflight_replay({"device_id": 7, "audio_path": audio_path}))

    assert exc_info.value.status_code == 400
    assert audio_path not in str(exc_info.value.detail) or not audio_path


@pytest.mark.parametrize("fixture_id", ["unknown", "../preflight_phrase", "C:evil.wav", 123])
def test_api_preflight_replay_rejects_invalid_fixture_id(monkeypatch: pytest.MonkeyPatch, fixture_id):
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(
        stt_provider="doubao", doubao_stt_api_key="test-key",
    ))
    monkeypatch.setattr(
        assist_routes,
        "get_session",
        lambda: SimpleNamespace(is_recording=False, is_paused=False),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(assist_routes.api_preflight_replay({"device_id": 7, "fixture_id": fixture_id}))

    assert exc_info.value.status_code == 400


def test_api_preflight_replay_hides_filesystem_paths_in_response_and_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(assist_routes, "get_config", lambda: SimpleNamespace(
        stt_provider="doubao", doubao_stt_api_key="test-key",
    ))
    monkeypatch.setattr(
        assist_routes,
        "get_session",
        lambda: SimpleNamespace(is_recording=False, is_paused=False),
    )

    async def fake_run_in_threadpool(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(assist_routes, "run_in_threadpool", fake_run_in_threadpool)
    monkeypatch.setattr(
        "api.assist.sound_test.replay_preflight_capture_stt",
        lambda device_id, **kwargs: {
            "device_id": device_id,
            "audio_path": "E:\\secret\\server\\assets\\preflight_phrase.wav",
            "success_count": 1,
            "failure_count": 0,
        },
    )

    result = asyncio.run(assist_routes.api_preflight_replay({"device_id": 7}))
    assert result["audio_path"] == "preflight_phrase.wav"

    def fail_with_path(*_args, **_kwargs):
        raise RuntimeError("测试音频不存在: E:\\secret\\server\\assets\\preflight_phrase.wav")

    monkeypatch.setattr("api.assist.sound_test.replay_preflight_capture_stt", fail_with_path)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(assist_routes.api_preflight_replay({"device_id": 7}))
    assert exc_info.value.status_code == 503
    assert "secret" not in str(exc_info.value.detail)


def test_api_preflight_replay_rejects_when_recording(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        assist_routes,
        "get_session",
        lambda: SimpleNamespace(is_recording=True, is_paused=False),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(assist_routes.api_preflight_replay({"device_id": 7}))

    assert exc_info.value.status_code == 409


def test_api_exam_preflight_rejects_active_or_paused_session(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        assist_routes,
        "get_session",
        lambda: SimpleNamespace(is_recording=True, is_paused=True),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(assist_routes.api_exam_preflight_run())

    assert exc_info.value.status_code == 409
    assert "结束当前面试或笔试" in str(exc_info.value.detail)


def test_api_start_rejects_while_exam_preflight_is_running(
    monkeypatch: pytest.MonkeyPatch,
):
    exam_test = importlib.import_module("api.assist.exam_test")
    monkeypatch.setattr(exam_test, "is_exam_preflight_running", lambda: True)
    started: list[object] = []
    monkeypatch.setattr(
        assist_routes,
        "start_nonblocking",
        lambda device_id: started.append(device_id),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(assist_routes.api_start({"device_id": 1}))

    assert exc_info.value.status_code == 409
    assert "链路检测正在运行" in str(exc_info.value.detail)
    assert started == []


def test_prompt_catalog_and_settings_api_keep_runtime_data_out(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.common.router import router
    from core import config

    config_path = tmp_path / "config.json"
    cfg = config.AppConfig(resume_text="private-resume-sentinel", position="private-role-sentinel", max_tokens=640)
    monkeypatch.setattr(config, "CONFIG_FILE", config_path)
    monkeypatch.setattr(config, "_config", cfg)
    app = FastAPI()
    app.include_router(router, prefix="/api")

    with TestClient(app) as client:
        catalog = client.get("/api/prompts")
        assert catalog.status_code == 200
        assert "private-" not in catalog.text
        scenes = {row["id"]: row for row in catalog.json()["scenes"]}
        assert len(scenes) == 12
        assert "assist_realtime_short" not in scenes
        assert scenes["assist_manual"]["max_output_tokens"] == 640
        assert "summary_markdown" in scenes["review_summary"]["output_contract"]
        assert scenes["assist_manual"]["default_prompt"]

        body = {"system_prompts": {"assist_manual": "Use {literal} and bold text."}}
        saved = client.post("/api/config", json=body)
        assert saved.status_code == 200
        assert saved.json()["system_prompts"] == body["system_prompts"]
        assert "private-resume-sentinel" not in config_path.read_text(encoding="utf-8")

        bad = client.post("/api/config", json={"system_prompts": {"unknown": "rules"}})
        assert 400 <= bad.status_code < 500
        assert client.get("/api/config").json()["system_prompts"] == body["system_prompts"]
        reset = client.post("/api/config", json={"system_prompts": {"assist_manual": None}})
        assert reset.json()["system_prompts"] == {}
