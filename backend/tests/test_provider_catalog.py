import importlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def catalog():
    return importlib.import_module("api.common.provider_catalog")


class Reply:
    def __init__(self, body, status=200):
        self.status_code = status
        self.body = body

    def json(self):
        return self.body


def get_provider(**changes):
    fields = {
        "id": "test-ark",
        "api_type": "openai",
        "api_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": "fake-test-ark-key",
    }
    return SimpleNamespace(**(fields | changes))


def make_client(catalog, monkeypatch):
    monkeypatch.setattr(catalog, "get_config", lambda: SimpleNamespace(model_providers=[get_provider()], models=[]))
    app = FastAPI()
    app.include_router(catalog.router)
    return TestClient(app)


def test_catalog_checks_all_saved_models_with_one_get(catalog, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Reply({"data": [{"id": "kimi"}]})

    def forbid_generation(*args, **kwargs):
        pytest.fail("目录重检应只发送 GET")

    def model(model_id):
        return SimpleNamespace(name=model_id, model=model_id, enabled=True, **vars(get_provider()))

    monkeypatch.setattr(catalog.requests, "get", get)
    monkeypatch.setattr(catalog.requests, "post", forbid_generation)
    monkeypatch.setattr(catalog, "get_config", lambda: SimpleNamespace(models=[model("kimi"), model("ep-private")]))
    results = catalog.check_catalog_models()
    assert results["catalog"] == {0: "visible", 1: "not_listed"}
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://ark.cn-beijing.volces.com/api/v3/models"
    assert kwargs["headers"]["Authorization"] == "Bearer fake-test-ark-key"
    assert kwargs["allow_redirects"] is False


def test_catalog_follows_pages_and_reports_explicit_audio_only_model(catalog, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs.get("params")))
        if not kwargs.get("params"):
            return Reply({
                "data": [{"id": "audio", "domain": "Audio", "modalities": {"output_modalities": ["audio"]}}],
                "last_id": "audio",
                "has_more": True,
            })
        return Reply({
            "data": [{"id": "vision", "capabilities": {"image_input": {"supported": True}}}],
            "has_more": False,
        })

    monkeypatch.setattr(catalog.requests, "get", get)
    client = make_client(catalog, monkeypatch)
    response = client.post("/model-providers/catalog", json={
        "api_type": "anthropic",
        "api_base_url": "https://api.anthropic.com",
        "api_key": "fake-test-anthropic-key",
    })
    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()["models"]}
    assert models["audio"]["incompatible_reason"]
    assert models["vision"]["supports_vision"] is True
    assert calls == [("https://api.anthropic.com/v1/models", None),
                     ("https://api.anthropic.com/v1/models", {"after_id": "audio"})]


def test_missing_model_results_are_not_exposed_from_invalid_catalog(catalog, monkeypatch):
    monkeypatch.setattr(catalog.requests, "get", lambda *args, **kwargs: Reply({"unrelated": []}))
    model = SimpleNamespace(model="unknown", enabled=True, **vars(get_provider()))
    monkeypatch.setattr(catalog, "get_config", lambda: SimpleNamespace(models=[model]))
    assert catalog.check_catalog_models()["catalog"] == {0: "unverified"}


def test_verification_requires_real_text_and_is_bound_to_credentials(catalog, monkeypatch):
    calls = []
    replies = iter([
        Reply({"choices": [{"message": {"content": "OK"}}]}),
        Reply({"choices": [{"message": {"content": {"secret": "not text"}}}]}),
    ])

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return next(replies)

    monkeypatch.setattr(catalog.requests, "post", post)
    client = make_client(catalog, monkeypatch)
    connection = {
        "provider_id": "test-ark",
        "api_type": "openai",
        "api_base_url": "https://ark.cn-beijing.volces.com/api/v3/",
        "model": "ep-private",
    }
    response = client.post("/model-providers/verify", json=connection)
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert catalog.was_verified(get_provider(), "ep-private")
    assert not catalog.was_verified(get_provider(api_key="new-test-key"), "ep-private")
    assert not catalog.was_verified(get_provider(), "another-model")
    assert calls[0][0] == "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    assert calls[0][1]["json"] == {
        "model": "ep-private",
        "messages": [{"role": "user", "content": "Reply OK."}],
        "max_tokens": 16,
        "stream": False,
    }
    assert calls[0][1]["allow_redirects"] is False

    response = client.post("/model-providers/verify", json=connection)
    assert response.json()["ok"] is False
    assert not catalog.was_verified(get_provider(), "ep-private")
    assert len(calls) == 2


def test_saved_credential_is_not_forwarded_to_edited_address(catalog, monkeypatch):
    def forbid_request(*args, **kwargs):
        pytest.fail("不应把已存凭据发送到修改后的地址")

    monkeypatch.setattr(catalog.requests, "get", forbid_request)
    client = make_client(catalog, monkeypatch)
    response = client.post("/model-providers/catalog", json={
        "provider_id": "test-ark",
        "api_type": "openai",
        "api_base_url": "https://modified.example/v1",
    })
    assert response.status_code == 409


def test_model_not_found_is_not_retried_at_an_alternate_path(catalog, monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return Reply({"error": {"code": "model_not_found", "message": "private upstream message"}}, 404)

    monkeypatch.setattr(catalog.requests, "post", post)
    client = make_client(catalog, monkeypatch)
    result = client.post("/model-providers/verify", json={
        "api_type": "openai",
        "api_base_url": "https://api.example.test",
        "api_key": "fake-key",
        "model": "not-found",
    }).json()
    assert result["ok"] is False
    assert "private upstream message" not in result["detail"]
    assert calls == ["https://api.example.test/v1/chat/completions"]


def test_advanced_probe_rejects_stale_index_before_generation(monkeypatch):
    from api.common import model_health
    from core.config import ModelListChanged

    model = SimpleNamespace(model="actual", enabled=True, **vars(get_provider(id="different-model-id")))
    monkeypatch.setattr(model_health, "get_config", lambda: SimpleNamespace(models=[model]))

    def forbid_generation(*args, **kwargs):
        pytest.fail("不应测试当前索引上的其他模型")

    monkeypatch.setattr(model_health.requests, "post", forbid_generation)
    with pytest.raises(ModelListChanged):
        model_health.probe_single_model(0, expected_model_id="previous-model-id")


def test_legacy_provider_key_resolves_by_its_migrated_identity(catalog, monkeypatch):
    from core.config import AppConfig, ModelConfig, get_model_providers

    cfg = AppConfig(models=[ModelConfig(
        id="legacy-model-id",
        api_base_url="https://api.example.test/v1",
        api_key="legacy-fixture-key",
        model="example",
    )])
    assert cfg.model_providers is None
    monkeypatch.setattr(catalog, "get_config", lambda: cfg)
    received = []

    def get(url, **kwargs):
        received.append((url, kwargs["headers"]["Authorization"]))
        return Reply({"data": []})

    monkeypatch.setattr(catalog.requests, "get", get)
    app = FastAPI()
    app.include_router(catalog.router)
    response = TestClient(app).post("/model-providers/catalog", json={
        "provider_id": get_model_providers(cfg)[0].id,
        "api_type": "openai",
        "api_base_url": "https://api.example.test/v1",
    })
    assert response.status_code == 200
    assert received == [("https://api.example.test/v1/models", "Bearer legacy-fixture-key")]


@pytest.mark.parametrize(("base_url", "api_type", "model_id", "params"), [
    ("https://api.deepseek.com/v1", "openai", "deepseek-v4-flash", {"thinking": {"type": "disabled"}}),
    ("https://api.deepseek.com/v1", "responses", "deepseek-v4-flash", {"reasoning": {"effort": "none"}}),
    ("https://ark.cn-beijing.volces.com/api/v3", "openai", "kimi-k3-1-pro-261024", {"thinking": {"type": "disabled"}}),
    ("https://ark.cn-beijing.volces.com/api/v3", "responses", "kimi-k3-1-pro-261024", {"reasoning": {"effort": "minimal"}}),
])
def test_official_reasoning_models_get_a_bounded_text_probe(catalog, monkeypatch, base_url, api_type, model_id, params):
    sent = []

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        if api_type == "responses":
            return Reply({"status": "completed", "output": [{"content": [{"type": "output_text", "text": "OK"}]}]})
        return Reply({"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(catalog.requests, "post", post)
    response = catalog.provider_verify_model(catalog.VerificationRequest(
        api_type=api_type, api_base_url=base_url, api_key="test-reasoning-key", model=model_id,
    ))
    assert response["ok"] is True
    assert len(sent) == 1
    assert all(sent[0].get(key) == value for key, value in params.items())
    assert "temperature" not in sent[0]


def test_verification_records_billable_usage_without_exposing_credentials(catalog, monkeypatch, tmp_path):
    import json

    from services.llm import trace

    key = "sensitive-trace-fixture-key"
    monkeypatch.setattr(trace, "LOG_DIR", tmp_path)
    monkeypatch.setattr(catalog.requests, "post", lambda *args, **kwargs: Reply({
        "choices": [{"message": {"content": f"OK {key}"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }))
    result = catalog.provider_verify_model(catalog.VerificationRequest(
        api_base_url="https://api.example.test/v1", api_key=key, model="trace-model",
    ))
    assert result["ok"] is True
    recorded = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    assert key not in recorded
    request, completed = [json.loads(line) for line in recorded.splitlines()]
    assert request["context"]["scene"] == "model_verification"
    assert completed["tokens"]["total"] == 7


def test_superseded_queued_verification_does_not_generate(catalog, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    waiting = Event()
    release = Event()
    slot_lock = Lock()
    calls = []

    class DelayedFirstSlot:
        entered = 0

        def __enter__(self):
            with slot_lock:
                self.entered += 1
                first = self.entered == 1
            if first:
                waiting.set()
                assert release.wait(timeout=5)

        def __exit__(self, *exc_info):
            return None

    def post(url, **kwargs):
        calls.append(url)
        return Reply({"choices": [{"message": {"content": "OK"}}]})

    monkeypatch.setattr(catalog, "_VERIFICATION_SLOTS", DelayedFirstSlot())
    monkeypatch.setattr(catalog.requests, "post", post)
    body = catalog.VerificationRequest(
        api_base_url="https://api.example.test/v1", api_key="queued-fixture-key", model="queued-model",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        older = executor.submit(catalog.provider_verify_model, body)
        try:
            assert waiting.wait(timeout=5)
            newer = executor.submit(catalog.provider_verify_model, body)
            assert newer.result(timeout=5)["ok"] is True
        finally:
            release.set()
        assert older.result(timeout=5)["ok"] is False
    assert calls == ["https://api.example.test/v1/chat/completions"]
    assert catalog.was_verified(body, "queued-model")


@pytest.mark.parametrize(("route", "bad_input"), [
    ("/model-providers/catalog", {}),
    ("/model-providers/catalog", {"api_base_url": "https://example.test", "api_type": "wrong"}),
    ("/model-providers/verify", {"api_base_url": "https://example.test"}),
    ("/model-providers/verify", {"api_base_url": "https://example.test", "model": 123}),
    ("/models/list", {}),
    ("/config", {"models": {"api_key": "sentinel-reflection-fixture-key"}}),
])
def test_malformed_provider_input_never_echoes_submitted_key(catalog, route, bad_input):
    from api.common.router import router

    app = FastAPI()
    app.include_router(router)
    key = "sentinel-reflection-fixture-key"
    response = TestClient(app).post(route, json={"api_key": key, **bad_input})
    assert response.status_code == 422
    assert key not in response.text


def test_catalog_pagination_enforces_total_timeout_budget(catalog, monkeypatch):
    clock = [100.0]
    seen_timeouts = []
    page_num = [0]

    def fake_monotonic():
        return clock[0]

    def fake_get(url, **kwargs):
        seen_timeouts.append(kwargs["timeout"])
        page_num[0] += 1
        clock[0] += 12.0  # simulate slow upstream pages consuming budget
        return Reply({
            "data": [{"id": f"m-{page_num[0]}"}],
            "last_id": f"m-{page_num[0]}",
            "has_more": True,
        })

    monkeypatch.setattr(catalog, "monotonic", fake_monotonic)
    monkeypatch.setattr(catalog, "_CATALOG_TOTAL_TIMEOUT_SEC", 30.0)
    monkeypatch.setattr(catalog, "_CATALOG_PAGE_TIMEOUT_SEC", 15.0)
    monkeypatch.setattr(catalog.requests, "get", fake_get)

    conn = catalog.Connection("openai", "https://api.example.test/v1", "sk-test-key")
    with pytest.raises(catalog.ProviderRequestError, match="超时"):
        catalog.fetch_provider_catalog(conn)

    # 1st page: min(15, 30)=15 -> clock=112; 2nd: min(15, 18)=15 -> clock=124; 3rd: min(15, 6)=6 -> clock=136 (>130) -> aborts
    assert seen_timeouts == [15.0, 15.0, 6.0]
    assert page_num[0] == 3
