"""STT text processing: vocabulary banks, normalization, ASR question classification."""

import re
from typing import Optional, Literal

TERM_CORRECTIONS = {
    r"(?i)radex|reddis|redist": "Redis",
    r"(?i)(?<![a-zA-Z])sequel(?![a-zA-Z])|sequles?|sequals?": "SQL",
    r"(?i)py\s*p+line|pypline|pipe\s*line": "pipeline",
    r"(?i)time\s*out": "timeout",
    r"(?i)(?<![a-zA-Z])C\s*SET(?![a-zA-Z])": "ZSET",
    r"(?i)my\s*sequel": "MySQL",
    r"(?i)post\s*gre": "PostgreSQL",
    r"(?i)dacker|docker\s*r": "Docker",
    r"(?i)kubernetes|k\s*8\s*s": "Kubernetes",
    r"(?i)(?<![a-zA-Z])g\s*i\s*l(?![a-zA-Z])": "GIL",
    r"(?i)fast\s*a\s*p\s*i": "FastAPI",
    r"(?i)web\s*socket": "WebSocket",
    r"(?i)spring\s*boot": "SpringBoot",
    r"(?i)my\s*batis": "MyBatis",
    r"(?i)mongo\s*d\s*b": "MongoDB",
    r"(?i)rabbit\s*m\s*q": "RabbitMQ",
    r"(?i)elastic\s*search": "Elasticsearch",
    r"(?i)ngin\s*x|engine\s*x": "Nginx",
    r"(?i)pie\s*test|pie\s*tests": "Pytest",
    r"(?i)pie\s*torch": "PyTorch",
    r"(?i)tensor\s*flow": "TensorFlow",
    r"(?i)git\s*hub": "GitHub",
    r"(?i)go\s*lang": "Golang",
    r"(?i)zookeeper": "Zookeeper",
    r"(?i)click\s*house": "ClickHouse",
    r"(?i)kafka": "Kafka",
    r"(?i)(?<![a-zA-Z])z\s*set(?![a-zA-Z])": "ZSET",
    r"(?i)sort(?:ed)?\s*set": "Sorted Set",
    r"(?i)spring\s*cloud": "Spring Cloud",
    r"(?i)concurrent\s*hash\s*map": "ConcurrentHashMap",
    r"(?i)hash\s*map": "HashMap",
    r"(?i)thread\s*pool": "ThreadPool",
    r"(?i)dead\s*lock": "deadlock",
    r"(?i)k\s*eight\s*s|kates|k8s": "Kubernetes",
    r"(?i)(?<![a-zA-Z])mq(?![a-zA-Z])": "MQ",
    r"(?i)my\s*sql": "MySQL",
    r"(?i)(?<![a-zA-Z])circle(?![a-zA-Z])": "SQL",
    r"(?i)Rose(?=和\s*Skills|和\s*skills)": "rules",
    r"(?i)Rose\s*和\s*SQL(?=的区别)": "rules和Skills",
    r"(?i)如\s*Redis\s*和\s*SQL\s*的区别.{0,4}什么": "rules和Skills的区别是什么",
    r"(?i)如\s*Redis\s*[、,，和]\s*ZSET\s*的区别是什么": "rules和Skills的区别是什么",
    r"(?i)SQL\s*所以失效": "SQL 索引失效",
    r"(?i)(?<![a-zA-Z])setcode(?![a-zA-Z])": "SQL code",
    r"(?i)SekoAI": "SQL AI",
    r"下皮|下屏|下期|沙皮": "虾皮",
    r"自动词史|自动价史|自动价是|自动价值": "自动驾驶",
    r"简率|减力": "简历",
    r"纳机回收|拿去回收": "垃圾回收",
    r"类诊陷漏|类诊泄漏": "内存泄漏",
    r"偶尔M": "OOM",
    r"财与": "参与",
    r"平衫|平侧|平测|频测|评课|平分": "评测",
    r"材料圆模型": "裁判员模型",
    r"机座模型": "基座模型",
    r"侧开": "测开",
    r"新脑": "新老",
    r"选心": "选型",
    r"(?<![a-zA-Z])AZNT(?![a-zA-Z])": "Agent",
    r"(?<![a-zA-Z])R\s*N\s*A\s*G(?![a-zA-Z])": "RAG",
    r"大家的车载": "大疆的车载",
    r"大家是它的": "大疆是它的",
    r"大家控股": "大疆控股",
    r"街单": "接单",
    r"路件": "路径",
    r"购件": "构建",
    r"体校": "提效",
    r"用力(?!气|量|心|来|劲)": "用例",
    r"说1[。.]下": "说一下",
    r"介绍1[。.]下": "介绍一下",
    r"解释1[。.]下": "解释一下",
    r"展开讲1[。.]下": "展开讲一下",
    r"举个例子[!！]1": "举个例子",
    r"结合效果母": "结合项目",
    r"那准备的验重": "那怎么验证",
}

INTERVIEW_TERM_CORRECTIONS = {
    r"(?i)(?<![a-zA-Z])child(?![a-zA-Z])": "pipeline",
    r"(?i)色吧": "server",
}

# ---------------------------------------------------------------------------
# Traditional-to-Simplified Chinese converter
# ---------------------------------------------------------------------------

_t2s_converter = None


def _get_t2s_converter():
    global _t2s_converter
    if _t2s_converter is None:
        try:
            import opencc
            _t2s_converter = opencc.OpenCC("t2s")
        except ImportError:
            _t2s_converter = False
    return _t2s_converter


# ---------------------------------------------------------------------------
# CJK slow-speech punctuation normalization
# ---------------------------------------------------------------------------

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"

_PAT_CJK_COMMA_FW = re.compile(rf"([{_CJK}])，([{_CJK}])")
_PAT_CJK_COMMA_ASCII = re.compile(rf"([{_CJK}]),([{_CJK}])")
_PAT_CJK_PERIOD_FW = re.compile(rf"([{_CJK}])。([{_CJK}])")
_PAT_CJK_PERIOD_ASCII = re.compile(rf"([{_CJK}])\.([{_CJK}])")
_PAT_CJK_PUNCT_SHORT_FOLLOWUP = re.compile(rf"([{_CJK}])[，,。\.]([0-9A-Za-z{_CJK}]{{1,3}})([{_CJK}])")
_PAT_ENG_PERIOD_ENG = re.compile(r"([A-Za-z])。([A-Za-z])")
_PAT_ENG_COMMA_ENG = re.compile(r"([A-Za-z])[，,]([A-Za-z])")
_PAT_DIGIT_PERIOD_DIGIT = re.compile(r"(\d)。(\d)")
_PAT_CJK_PERIOD_ENG = re.compile(rf"([{_CJK}])。([A-Za-z])")
_PAT_ENG_PERIOD_CJK = re.compile(rf"([A-Za-z])。([{_CJK}])")


def _normalize_slow_speech_intraword_punct(text: str) -> str:
    """Remove erroneous intra-word punctuation from slow-speech ASR output."""
    if not text:
        return text

    def count_adjacent_cjk_punct(s: str) -> int:
        n = len(_PAT_CJK_COMMA_FW.findall(s)) + len(_PAT_CJK_COMMA_ASCII.findall(s))
        n += len(_PAT_CJK_PERIOD_FW.findall(s)) + len(_PAT_CJK_PERIOD_ASCII.findall(s))
        n += len(_PAT_CJK_PUNCT_SHORT_FOLLOWUP.findall(s))
        return n

    def cjk_len(s: str) -> int:
        return len(re.findall(rf"[{_CJK}]", s))

    adj = count_adjacent_cjk_punct(text)
    n_cjk = cjk_len(text)
    short_followup = bool(_PAT_CJK_PUNCT_SHORT_FOLLOWUP.search(text))
    should_merge_all = adj >= 2
    should_merge_short = adj == 1 and n_cjk <= 3
    if not (should_merge_all or should_merge_short or short_followup):
        return text

    t = text
    prev = None
    while prev != t:
        prev = t
        t = _PAT_CJK_COMMA_FW.sub(r"\1\2", t)
        t = _PAT_CJK_COMMA_ASCII.sub(r"\1\2", t)
        t = _PAT_CJK_PERIOD_FW.sub(r"\1\2", t)
        t = _PAT_CJK_PERIOD_ASCII.sub(r"\1\2", t)
        t = _PAT_CJK_PUNCT_SHORT_FOLLOWUP.sub(r"\1\2\3", t)
        t = _PAT_ENG_PERIOD_ENG.sub(r"\1\2", t)
        t = _PAT_ENG_COMMA_ENG.sub(r"\1\2", t)
        t = _PAT_DIGIT_PERIOD_DIGIT.sub(r"\1\2", t)
        t = _PAT_CJK_PERIOD_ENG.sub(r"\1\2", t)
        t = _PAT_ENG_PERIOD_CJK.sub(r"\1\2", t)
    return t


_normalize_slow_speech_commas = _normalize_slow_speech_intraword_punct


_REPEAT_CHAR = re.compile(r"([\u4e00-\u9fffA-Za-z,，.。!?？！])\1{4,}")
_REPEAT_PHRASE = re.compile(r"(.{2,8}?)\1{3,}")


def _strip_feedback_loop(text: str) -> str:
    t = _REPEAT_CHAR.sub("", text)
    prev = None
    while prev != t:
        prev = t
        t = _REPEAT_PHRASE.sub(r"\1", t)
    return t.strip()


def _postprocess(text: str, context: str = "") -> str:
    """Apply term corrections, t2s conversion, slow-speech normalization, and feedback-loop filter."""
    text = _strip_feedback_loop(text)
    converter = _get_t2s_converter()
    if converter and converter is not False:
        text = converter.convert(text)
    for pattern, replacement in TERM_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text)
    if context == "interview":
        for pattern, replacement in INTERVIEW_TERM_CORRECTIONS.items():
            text = re.sub(pattern, replacement, text)
    text = _normalize_slow_speech_intraword_punct(text)
    return text.strip()


def postprocess_interview_transcription(text: str) -> str:
    """Apply interview-context-only term corrections on top of already-postprocessed text."""
    if not text:
        return text
    for pattern, replacement in INTERVIEW_TERM_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text)
    return text


def transcription_significant_len(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(rf"[{_CJK}A-Za-z0-9]", text))


def transcription_for_publish(text: str, min_significant_chars: int = 2) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    if is_interview_boilerplate_text(t):
        return None
    need = max(1, int(min_significant_chars))
    if transcription_significant_len(t) < need:
        return None
    return t


def join_transcription_fragments(parts: list[str]) -> str:
    qs = [p.strip() for p in parts if p and str(p).strip()]
    if not qs:
        return ""
    acc = qs[0]
    for q in qs[1:]:
        overlap_len = _suffix_prefix_overlap_len(acc, q)
        if overlap_len > 0:
            q = q[overlap_len:].lstrip()
            if not q:
                continue
        need_space = bool(re.search(r"[a-zA-Z0-9]$", acc) and re.match(r"^[a-zA-Z0-9]", q))
        acc = f"{acc} {q}" if need_space else f"{acc}{q}"
    return acc.strip()


def _suffix_prefix_overlap_len(prev: str, cur: str) -> int:
    left = (prev or "").strip()
    right = (cur or "").strip()
    if not left or not right:
        return 0
    max_overlap = min(len(left), len(right), 32)
    for size in range(max_overlap, 0, -1):
        if left[-size:] == right[:size]:
            significant = transcription_significant_len(right[:size])
            if significant >= 4 or size >= 6:
                return size
    return 0


# ---------------------------------------------------------------------------
# ASR question classification
# ---------------------------------------------------------------------------

_FILLER_PREFIX = re.compile(
    r"^(?:"
    r"嗯+|恩+|啊+|呃+|额+|诶+|欸+|唉+|"
    r"那个|这个|就是|然后|所以|"
    r"我想想|让我想想|等一下|等会|等会儿|稍等"
    r")(?:[\s,，。.!！?？;；:：、]+|$)",
    re.IGNORECASE,
)
_BACKCHANNEL = re.compile(
    r"^(?:"
    r"对+|对对+|没错|是的|好|好的|行|可以|嗯+|恩+|啊+|哦+|噢+|"
    r"明白了|知道了|收到|你说得对|你说的对|确实|继续|继续吧|然后呢|还有吗"
    r")(?:[\s,，。.!！?？;；:：、]+|$)",
    re.IGNORECASE,
)
_QUESTION_CUE = re.compile(
    r"(什么|怎么|如何|为什么|为何|区别|原理|作用|流程|实现|设计|优化|排查|处理|"
    r"介绍(?:一下|下)?|说(?:一下|下)?|讲(?:一下|下)?|聊(?:一下|下)?|解释(?:一下|下)?|"
    r"分析(?:一下|下)?|怎么做|怎么办|有哪些|是否|能不能|可不可以|对比(?:一下|下)?|"
    r"展开讲讲|详细说说|举个例子|继续讲|接着说)",
    re.IGNORECASE,
)
_DIRECTIVE_QUESTION_CUE = re.compile(
    r"(?:请|你|麻烦|帮我|能否|能不能|可不可以).{0,24}"
    r"(?:介绍|说|讲|聊|解释|分析|设计|实现|排查|优化|比较|总结|展开)(?:一下|下)?",
    re.IGNORECASE,
)
_INCOMPLETE_TAIL = re.compile(
    r"(?:因为|所以|然后|但是|并且|或者|以及|如果|比如|例如|就是|那个|这个|首先|其次|再然后)$",
    re.IGNORECASE,
)
_LOW_VALUE_FRAGMENT = re.compile(
    r"^(?:"
    r"和|因为这个|因为这个呢|然后呢|还有呢|就是这个|什么|什么呢|那个|那个呢|"
    r"那个还是那个东西|继续说|继续说说|说完了是吧|完了是吧|"
    r"还有什么|还是什么|对吧|是吧|"
    r"对(?:[，,。.]?在.*里面)?|嗯+|啊+|哦+|噢+|好+|OK|ok|Uh"
    r")$",
    re.IGNORECASE,
)
_QUESTION_CONSTRAINT_TAIL = re.compile(
    r"(?:不要|别|先不|不用|不需要|无需).{0,12}(?:结合|联系|套|带入|展开).{0,12}(?:项目|简历|经历)|"
    r"(?:不要|别|先不|不用|不需要|无需).{0,12}(?:项目|简历|经历)|"
    r"(?:只讲|就讲|单独讲).{0,12}(?:概念|原理|区别|核心|技术)",
    re.IGNORECASE,
)
_QUESTION_SPLIT = re.compile(r"[？?]+")
_QUESTION_CONNECTOR_SPLIT = re.compile(
    r"(?:[，,；;]\s*|\s+)(?=(?:再说|再讲|再聊|然后|另外|还有|还有一个|顺便|"
    r"第二个问题|另一个问题|下一个问题))",
    re.IGNORECASE,
)
_SENTENCE_START_AFTER_SPLIT = re.compile(
    r"^(?:那|那么|然后|所以|但是|不过|另外|还有|以及|就是|呃|额|哎|噢|"
    r"请问|能不能|可不可以|为什么|怎么|如何|什么)",
)
_ASR_GARBAGE_PUNCT = re.compile(r"(?<=[\u4e00-\u9fffA-Za-z0-9])[？?](?=[\u4e00-\u9fffA-Za-z0-9])")
_ASR_TRAILING_TECH_TAG = re.compile(
    r"(?:[。.!！?？,，;；:：、\s]|^)(JVM|SQL|Redis|Java|Python|MySQL|Kafka|HTTP)$",
    re.IGNORECASE,
)
_INTERVIEW_BOILERPLATE = re.compile(
    r"(?:"
    r"欢迎参与(?:ai)?面试|欢迎参加.*面试|我是本次的面试官|这是一个相互了解的机会|"
    r"为保障公平|全程面试记录将被存档|取消资格|永久拉黑|"
    r"如果音量过大或过小请调整|请调整你的设备音量|"
    r"时间差不多了|今天的面试就先|我们今天的面试就先|"
    r"突发情况|超时将自动结束|返回超时将自动结束|"
    r"谢谢参加本次面试|好再见|请问你.*还有什么其他的问题"
    r")",
    re.IGNORECASE,
)

AsrQuestionCandidate = Literal["ignore", "candidate", "promote"]


def _strip_filler_prefix(text: str) -> str:
    t = (text or "").strip()
    prev = None
    while t and t != prev:
        prev = t
        t = _FILLER_PREFIX.sub("", t).strip()
    return t


def normalize_transcription_for_analysis(text: str) -> str:
    t = _strip_filler_prefix((text or "").strip())
    t = _ASR_GARBAGE_PUNCT.sub("", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" \t\r\n,，。.!！?？;；:：、")
    t = _ASR_TRAILING_TECH_TAG.sub("", t).strip(" \t\r\n,，。.!！?？;；:：、")
    return t


def is_interview_boilerplate_text(text: str) -> bool:
    normalized = normalize_transcription_for_analysis(text)
    if not normalized:
        return True
    if transcription_significant_len(normalized) < 6:
        return False
    return bool(_INTERVIEW_BOILERPLATE.search(normalized))


def split_question_like_text(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    raw_parts = [p for p in _QUESTION_SPLIT.split(raw) if p and p.strip()]
    if not raw_parts:
        raw_parts = [raw]
    merged: list[str] = []
    for part in raw_parts:
        stripped = part.strip()
        if not stripped:
            continue
        if merged and not _SENTENCE_START_AFTER_SPLIT.match(stripped):
            merged[-1] = merged[-1] + "？" + stripped
        else:
            merged.append(stripped)
    out: list[str] = []
    for part in merged:
        subparts = _QUESTION_CONNECTOR_SPLIT.split(part)
        for sub in subparts:
            cleaned = normalize_transcription_for_analysis(sub)
            if cleaned:
                out.append(cleaned)
    deduped: list[str] = []
    for item in out:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped


def is_backchannel_text(text: str) -> bool:
    normalized = normalize_transcription_for_analysis(text)
    if not normalized:
        return True
    normalized = re.sub(r"^(?:嗯+|恩+|啊+|呃+|额+|诶+|欸+|哦+|噢+)", "", normalized).strip()
    if not normalized:
        return True
    return bool(_BACKCHANNEL.fullmatch(normalized))


def classify_asr_question_candidate(
    text: str,
    min_significant_chars: int = 4,
) -> tuple[AsrQuestionCandidate, str]:
    normalized = normalize_transcription_for_analysis(text)
    if not normalized:
        return "ignore", ""
    if is_interview_boilerplate_text(normalized):
        return "ignore", normalized
    backchannel_cleaned = re.sub(r"^(?:嗯+|恩+|啊+|呃+|额+|诶+|欸+|哦+|噢+)", "", normalized).strip()
    if is_backchannel_text(normalized):
        return "ignore", backchannel_cleaned or normalized
    sig = transcription_significant_len(normalized)
    need = max(2, int(min_significant_chars))
    has_question_mark = "?" in text or "？" in text
    has_question_cue = bool(_QUESTION_CUE.search(normalized))
    has_directive_cue = bool(_DIRECTIVE_QUESTION_CUE.search(normalized))
    has_tail_question = bool(re.search(r"(?:吗|么|呢)$", normalized))
    incomplete_tail = bool(_INCOMPLETE_TAIL.search(normalized))
    low_value_fragment = bool(_LOW_VALUE_FRAGMENT.fullmatch(normalized))
    constraint_tail = bool(_QUESTION_CONSTRAINT_TAIL.search(normalized))
    if constraint_tail and sig >= max(3, need):
        return "candidate", normalized
    if low_value_fragment:
        return "ignore", normalized
    if has_question_mark and sig >= 2:
        return "promote", normalized
    if has_question_cue or has_directive_cue:
        return "promote", normalized
    if has_tail_question and sig >= max(3, need - 1):
        return "promote", normalized
    if incomplete_tail:
        if sig >= need:
            return "candidate", normalized
        return "ignore", normalized
    if not has_question_mark and not has_question_cue and not has_directive_cue and not has_tail_question:
        if sig < 8:
            return "ignore", normalized
    if sig >= need:
        return "candidate", normalized
    return "ignore", normalized


def is_asr_question_constraint_tail(text: str) -> bool:
    normalized = normalize_transcription_for_analysis(text)
    if not normalized:
        return False
    return bool(_QUESTION_CONSTRAINT_TAIL.search(normalized))


def is_viable_asr_question_group(
    texts: list[str],
    min_significant_chars: int = 4,
) -> bool:
    cleaned_parts: list[str] = []
    saw_promote = False
    for text in texts:
        parts = split_question_like_text(text) or [normalize_transcription_for_analysis(text)]
        for part in parts:
            if not part:
                continue
            kind, cleaned = classify_asr_question_candidate(part, min_significant_chars)
            if cleaned:
                cleaned_parts.append(cleaned)
            if kind == "promote":
                saw_promote = True
    if saw_promote:
        return True
    if len(cleaned_parts) >= 2:
        return True
    if len(cleaned_parts) == 1:
        return False
    merged = "".join(cleaned_parts)
    return transcription_significant_len(merged) >= max(3, int(min_significant_chars))


def build_asr_question_group_text(texts: list[str], max_items: int = 4) -> str:
    items: list[str] = []
    for text in texts:
        parts = split_question_like_text(text) or [normalize_transcription_for_analysis(text)]
        for part in parts:
            if not part:
                continue
            kind, cleaned = classify_asr_question_candidate(part, 2)
            if kind == "ignore":
                continue
            part = cleaned or part
            if not items or items[-1] != part:
                items.append(part)
    items = [it for it in items if len(it) >= 3 or it == items[0]]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) > max_items:
        items = [items[0], *items[-(max_items - 1):]]
    followups = "\n".join(f"{idx}. {item}" for idx, item in enumerate(items[1:], start=1))
    return (
        "以下内容来自同一轮实时面试中的连续追问，请合并理解，优先回答最后一个追问，并兼顾前文上下文。\n\n"
        f"主问题：{items[0]}\n\n"
        f"连续追问：\n{followups}"
    )


# ---------------------------------------------------------------------------
# Follow-up detection (cross-turn)
# ---------------------------------------------------------------------------

_FOLLOWUP_CUES = re.compile(
    r"(?:那|那么|那个|这个|刚才|刚刚|你刚说|上面|前面|接着|继续|"
    r"展开|详细|具体|深入|补充|举个例子|比如说|为什么不|"
    r"真的吗|确定吗|怎么说|啥意思|什么意思)",
    re.IGNORECASE,
)
_ENGLISH_FOLLOWUP_CUES = re.compile(
    r"(?:"
    r"\byou\s+(?:just\s+)?(?:said|mentioned)\b|"
    r"\b(?:previous|last)\s+(?:one|round|question|answer)\b|"
    r"\bfollow(?:ing)?\s*up\b|"
    r"\b(?:can|could)\s+you\s+elaborate\b|"
    r"\btell\s+me\s+more\b|"
    r"\bgive\s+(?:me\s+)?an\s+example\b|"
    r"\bwhy\s+not\b|"
    r"\bhow\s+(?:did|would|do)\s+you\s+(?:validate|verify|test|handle|solve|improve)\s+(?:it|that|this)\b"
    r")",
    re.IGNORECASE,
)

_PRONOUN_FOLLOWUP = re.compile(
    r"^(?:那|那么|然后|所以|但是|不过|另外|还有|以及).*(?:呢|吗|么|吧|\?|？)$"
)


def classify_followup(
    text: str,
    prev_question: str,
    prev_answer: str,
) -> bool:
    """Heuristically decide whether *text* is a follow-up to the previous Q&A."""
    if not text or not prev_question:
        return False
    normalized = normalize_transcription_for_analysis(text)
    if not normalized:
        return False

    if _FOLLOWUP_CUES.search(normalized):
        return True

    if _ENGLISH_FOLLOWUP_CUES.search(normalized):
        return True

    if len(normalized) < 15 and _PRONOUN_FOLLOWUP.match(normalized):
        return True

    if len(normalized) < 12 and not any(c in normalized for c in "？?"):
        prev_norm = normalize_transcription_for_analysis(prev_question)
        if prev_norm:
            prev_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}", prev_norm))
            cur_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,}", normalized))
            if prev_keywords and cur_keywords and cur_keywords & prev_keywords:
                return True

    return False
