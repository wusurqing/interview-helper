from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.common.config_payload import build_config_payload  # noqa: E402
from core import config  # noqa: E402
from services.llm.protocols import build_chat_payload  # noqa: E402


def test_scene_overrides_survive_restart_and_merge_independently(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"position":"Backend developer"}', encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", config_path)
    monkeypatch.setattr(config, "_config", None)
    assert build_config_payload(config.get_config()).get("system_prompts") == {}

    patches = [
        {"assist_manual": "Answer in three bullets."},
        {"review_summary": "Write a short narrative."},
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda patch: config.update_config({"system_prompts": patch}), patches))
    assert config._load_config().system_prompts == {key: text for patch in patches for key, text in patch.items()}

    config.update_config({"system_prompts": {"assist_manual": None, "resume_match": ""}})
    assert config._load_config().system_prompts == {
        "review_summary": "Write a short narrative.",
        "resume_match": "",
    }


@pytest.mark.parametrize("patch", [{"unknown": "rules"}, {"assist_manual": 42}, ["rules"]])
def test_invalid_scene_patch_is_atomic(monkeypatch, tmp_path, patch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_FILE", config_path)
    monkeypatch.setattr(config, "_config", config.AppConfig())
    with pytest.raises(ValueError):
        config.update_config({"temperature": 1.0, "system_prompts": patch})
    assert config.get_config().temperature == 0.5
    assert not config_path.exists()


@pytest.mark.parametrize("protocol", ["openai", "anthropic", "responses"])
def test_system_instruction_keeps_task_input_in_user_message(protocol):
    messages = [
        {"role": "system", "content": "Use three bullets."},
        {"role": "user", "content": "Explain CAP."},
    ]
    payload = build_chat_payload(
        protocol,
        "model",
        messages,
        stream=True,
        think_params={"messages": [{"role": "user", "content": "Wrong question."}]},
    )
    if protocol == "openai":
        assert payload["messages"] == messages
    else:
        system_key, input_key = ("system", "messages") if protocol == "anthropic" else ("instructions", "input")
        assert payload[system_key] == "Use three bullets."
        assert payload[input_key] == messages[1:]


def test_openai_sdk_extra_body_cannot_replace_system_or_user_message(monkeypatch):
    import httpx
    from openai import OpenAI

    from services.llm import streaming

    captured = []

    def send(request):
        import json
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    model = config.ModelConfig(think_disabled_params={"messages": [{"role": "user", "content": "Wrong"}]})
    monkeypatch.setattr(config, "_config", config.AppConfig(models=[model]))
    question = {"role": "user", "content": "Actual question"}
    with OpenAI(
        api_key="offline-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(send)),
    ) as client:
        monkeypatch.setattr(streaming, "get_client_for_model", lambda _model: client)
        list(streaming.chat_stream_single_model(model, [question], system_prompt="Use bullets."))
    assert captured[0]["messages"] == [{"role": "system", "content": "Use bullets."}, question]
