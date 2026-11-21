"""
测试 review 分析功能（不需要真实 API key）
"""
import pytest
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import json

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services import review_analysis
from core.config import ModelConfig


def _mock_review_config():
    mock_config = MagicMock()
    mock_config.get_review_model.return_value = ModelConfig(
        name="lite-ark",
        api_base_url="https://api.example.com/v1",
        api_key="test-key-123",
        model="doubao-seed-2-0-lite-260428",
    )
    return mock_config


@pytest.mark.parametrize(("scene", "source", "summary"), [
    ("review_turn", "assist", False),
    ("review_summary", "assist", True),
    ("exam_review_turn", "written_exam", False),
    ("exam_review_summary", "written_exam", True),
])
def test_review_scene_uses_custom_system_prompt_and_preserves_output_contract(scene, source, summary):
    model_config = _mock_review_config()
    custom = "我的复盘规则：使用 {自定义维度}。"
    model_config.system_prompts = {scene: custom}
    if summary:
        result = {"summary_markdown": "## 概述", "strong_points": [], "weak_points": []}
        contract_field = "summary_markdown"
    else:
        result = {
            "strengths": [],
            "risks": [],
            "scorecard": {"自定义维度": 8.5},
            "evidence": {"improvement_advice": "", "follow_up_questions": [], "tags": []},
        }
        contract_field = "scorecard"

    response = MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(result)))])
    with patch("services.review_analysis.get_config", return_value=model_config), \
         patch("services.llm.streaming.chat_completion_create", return_value=response) as completion:
        if summary:
            actual = review_analysis.generate_summary(
                [{"question_text": "输入问题 XYZ", "reference_answer_text": "参考答案"}],
                review_source=source,
            )
        else:
            actual = review_analysis.analyze_turn(
                "输入问题 XYZ", "口述回答", "参考答案", apply_asr_correction=False,
                review_source=source,
            )

    system, user = completion.call_args.args[1]
    assert system["role"] == "system"
    assert system["content"].startswith(custom)
    assert contract_field in system["content"]
    assert "输入问题 XYZ" not in system["content"]
    assert user["role"] == "user"
    assert "输入问题 XYZ" in user["content"]
    assert contract_field not in user["content"]
    assert actual[contract_field] == result[contract_field]


def test_analyze_turn_no_active_model():
    """模型配置错误需要由后台任务处理。"""
    with patch('services.review_analysis.get_review_model_cfg') as mock_model:
        mock_model.side_effect = ValueError("No API key")

        with pytest.raises(ValueError, match="No API key"):
            review_analysis.analyze_turn(
                question="什么是 Python GIL？",
                candidate_answer="GIL 是全局解释器锁",
                reference_answer="参考答案",
                apply_asr_correction=False,
            )


def test_analyze_turn_returns_asr_correction_evidence():
    response = MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
        "strengths": [], "risks": [], "scorecard": {}, "evidence": {},
    })))])
    with patch('services.review_analysis.correct_asr_errors') as mock_correct, \
         patch('services.review_analysis.get_config', return_value=_mock_review_config()), \
         patch('services.llm.streaming.chat_completion_create', return_value=response):
        mock_correct.return_value = "Redis 有字符串和哈希"

        result = review_analysis.analyze_turn(
            question="Redis 有哪些数据结构？",
            candidate_answer="red 地址有字符串和哈希",
            reference_answer="参考答案",
        )

        assert result["corrected_answer"] == "Redis 有字符串和哈希"
        assert result["evidence"]["asr_correction"]["original"] == "red 地址有字符串和哈希"


def test_analyze_turn_preserves_structured_evidence():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='''```json
{
  "strengths": ["术语准确"],
  "risks": ["缺少扩容方案"],
  "scorecard": {"准确性": 8, "深度": 5, "表达": 7},
  "evidence": {
    "improvement_advice": "补充扩容后的缓存击穿处理",
    "follow_up_questions": ["热点 key 失效时怎么保护数据库？"],
    "tags": ["Redis", "缓存击穿"]
  }
}
```'''))]
    )

    with patch('services.review_analysis.correct_asr_errors', return_value="Redis 回答"), \
         patch('services.review_analysis.get_config', return_value=_mock_review_config()), \
         patch('services.review_analysis.get_review_model_cfg', return_value=_mock_review_config().get_review_model()),          patch('services.llm.streaming.chat_completion_create', return_value=mock_client.chat.completions.create.return_value) as mock_llm_create:
        result = review_analysis.analyze_turn(
            question="Redis 缓存击穿怎么处理？",
            candidate_answer="Redis 回答",
            reference_answer="参考答案",
        )

    assert result["scorecard"]["深度"] == 5
    assert result["evidence"]["improvement_advice"] == "补充扩容后的缓存击穿处理"
    assert result["evidence"]["follow_up_questions"] == ["热点 key 失效时怎么保护数据库？"]
    assert result["evidence"]["tags"] == ["Redis", "缓存击穿"]


def test_analyze_written_exam_turn_reviews_generated_answer():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='''```json
{
  "strengths": ["复杂度清晰"],
  "risks": ["缺少空数组用例"],
  "scorecard": {"正确性": 8, "完整性": 6, "可提交性": 7},
  "evidence": {
    "improvement_advice": "先补边界用例",
    "follow_up_questions": ["空数组时返回什么？"],
    "tags": ["哈希表"]
  }
}
```'''))]
    )

    with patch('services.review_analysis.correct_asr_errors') as mock_correct, \
         patch('services.review_analysis.get_config', return_value=_mock_review_config()), \
         patch('services.review_analysis.get_review_model_cfg', return_value=_mock_review_config().get_review_model()),          patch('services.llm.streaming.chat_completion_create', return_value=mock_client.chat.completions.create.return_value) as mock_llm_create:
        result = review_analysis.analyze_turn(
            question="截图题：两数之和",
            candidate_answer="",
            reference_answer="用哈希表一次遍历。",
            review_source="written_exam",
        )

    mock_correct.assert_not_called()
    kwargs = mock_llm_create.call_args.kwargs
    messages = mock_llm_create.call_args.args[1]
    assert "笔试辅导教练" in messages[0]["content"]
    user_prompt = messages[-1]["content"]
    assert "截图题目/题面" in user_prompt
    assert "本次生成答案" in user_prompt
    assert "用哈希表一次遍历。" in user_prompt
    assert "候选人回答" not in user_prompt
    assert result["scorecard"]["可提交性"] == 7
    assert result["evidence"]["review_mode"] == "written_exam"
    assert result["evidence"]["improvement_advice"] == "先补边界用例"


def test_generate_summary_no_active_model():
    """模型配置错误需要由后台任务处理。"""
    with patch('services.review_analysis.get_review_model_cfg') as mock_model:
        mock_model.side_effect = ValueError("No API key")

        with pytest.raises(ValueError, match="No API key"):
            review_analysis.generate_summary(
                turns=[{"question_text": "Q1", "candidate_answer_text": "A1"}],
            )


def test_generate_summary_empty_turns():
    """测试空 turns 时返回提示"""
    result = review_analysis.generate_summary(turns=[])

    assert "未录制" in result["summary_markdown"]
    assert result["strong_points"] == []
    assert result["weak_points"] == []


def test_summary_markdown_can_contain_code_fences():
    summary = "## 代码改进例子\n\n```python\nprint(1)\n```"
    content = json.dumps({"summary_markdown": summary, "strong_points": [], "weak_points": []})
    response = MagicMock(choices=[MagicMock(message=MagicMock(content=content))])
    with patch('services.review_analysis.get_config', return_value=_mock_review_config()), \
         patch('services.llm.streaming.chat_completion_create', return_value=response):
        result = review_analysis.generate_summary([{"question_text": "题目"}])

    assert result["summary_markdown"] == summary


def test_generate_summary_for_written_exam_uses_generated_answers():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='''```json
{
  "summary_markdown": "## 笔试答题质量\\n\\n整体可提交。",
  "strong_points": ["代码可提交"],
  "weak_points": ["边界用例不足"]
}
```'''))]
    )

    with patch('services.review_analysis.get_config', return_value=_mock_review_config()), \
         patch('services.review_analysis.get_review_model_cfg', return_value=_mock_review_config().get_review_model()),          patch('services.llm.streaming.chat_completion_create', return_value=mock_client.chat.completions.create.return_value) as mock_llm_create:
        result = review_analysis.generate_summary(
            turns=[
                {
                    "question_text": "截图题：两数之和",
                    "candidate_answer_text": "",
                    "reference_answer_text": "用哈希表一次遍历。",
                    "strengths": ["复杂度清晰"],
                    "risks": ["缺少空数组用例"],
                }
            ],
            review_source="written_exam",
        )

    kwargs = mock_llm_create.call_args.kwargs
    messages = mock_llm_create.call_args.args[1]
    assert "笔试辅导教练" in messages[0]["content"]
    user_prompt = messages[-1]["content"]
    assert "本次笔试练习共 1 道截图题" in user_prompt
    assert "生成答案：用哈希表一次遍历。" in user_prompt
    assert "回答：" not in user_prompt
    assert result["strong_points"] == ["代码可提交"]


def test_run_asr_correction_check_success():
    mock_llm_return = MagicMock(
        choices=[MagicMock(message=MagicMock(content="我做过 Redis 缓存，MySQL 查询会先看布隆过滤器。"))]
    )

    with patch('services.review_analysis.get_config', return_value=_mock_review_config()), \
         patch('services.review_analysis.get_review_model_cfg', return_value=_mock_review_config().get_review_model()), \
         patch('services.llm.streaming.chat_completion_create', return_value=mock_llm_return) as mock_llm_create:
        result = review_analysis.run_asr_correction_check(
            question="请介绍缓存优化",
            candidate_answer="我做过 red 地址缓存，麦 SQL 查询会先看布隆过绿器。",
        )

    assert result["ok"] is True
    assert result["model_name"] == "lite-ark"
    assert result["model"] == "doubao-seed-2-0-lite-260428"
    assert result["changed"] is True
    assert "Redis" in result["corrected"]
    kwargs = mock_llm_create.call_args.kwargs
    assert kwargs["think_params"] == {
        "thinking": {"type": "disabled"},
        "think_mode": False,
        "enable_thinking": False,
    }


def test_run_asr_correction_check_returns_blocked_error():
    _blocked = PermissionError("Your request was blocked")

    with patch('services.review_analysis.get_config', return_value=_mock_review_config()), \
         patch('services.review_analysis.get_review_model_cfg', return_value=_mock_review_config().get_review_model()), \
         patch('services.llm.streaming.chat_completion_create', side_effect=_blocked):
        result = review_analysis.run_asr_correction_check(
            question="请介绍缓存优化",
            candidate_answer="我做过 red 地址缓存，麦 SQL 查询会先看布隆过绿器。",
        )

    assert result["ok"] is False
    assert result["changed"] is False
    assert result["corrected"] == result["original"]
    assert "blocked" in result["detail"]


def test_get_active_llm_client():
    """测试获取激活模型客户端"""
    with patch('services.review_analysis.get_config') as mock_get_config:
        mock_config = MagicMock()
        mock_model = ModelConfig(
            name="Test Model",
            api_base_url="https://api.example.com/v1",
            api_key="test-key-123",
            model="test-model",
        )
        mock_config.get_review_model.return_value = mock_model
        mock_get_config.return_value = mock_config

        client, model_name = review_analysis.get_active_llm_client()

        assert client is not None
        assert model_name == "test-model"
        assert "api.example.com" in str(client.base_url)


def test_run_asr_correction_check_handles_empty_choices():
    empty_response = MagicMock(choices=[])
    with patch("services.review_analysis.get_config", return_value=_mock_review_config()), \
         patch("services.review_analysis.get_review_model_cfg", return_value=_mock_review_config().get_review_model()), \
         patch("services.llm.streaming.chat_completion_create", return_value=empty_response):
        result = review_analysis.run_asr_correction_check(
            question="请介绍缓存优化",
            candidate_answer="我做过 red 地址缓存，麦 SQL 查询会先看布隆过绿器。",
        )
    assert result["ok"] is False
    assert result["detail"] == "模型返回为空"


def test_request_review_json_tolerates_fences_with_prose_and_inner_code_blocks():
    summary = "## 建议\n\n```python\nprint('ok')\n```"
    fenced_with_inner_fence = f"这里是复盘 JSON：\n```json\n{json.dumps({'summary_markdown': summary, 'strong_points': ['清晰'], 'weak_points': []})}\n```\n以上。"
    response = MagicMock(choices=[MagicMock(message=MagicMock(content=fenced_with_inner_fence))])
    with patch("services.review_analysis.get_config", return_value=_mock_review_config()), \
         patch("services.llm.streaming.chat_completion_create", return_value=response):
        result = review_analysis.generate_summary([{"question_text": "题目"}])
    assert result["summary_markdown"] == summary
    assert result["strong_points"] == ["清晰"]

