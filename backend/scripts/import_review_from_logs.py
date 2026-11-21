"""
从日志文件中提取历史面试记录，导入到 review 数据库
用法：python -m scripts.import_review_from_logs
"""
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.storage import review
from core.logger import get_logger
from scripts.recover_review_from_logs import _extract_repr

logger = get_logger(__name__)


def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """
    解析日志文件，提取面试会话
    只提取有 conversation_loopback 的会话（表示有双音轨对话）
    返回会话列表，每个会话包含 turns
    """
    sessions = []
    current_session = None
    current_turns = []
    current_qa = {}
    has_conversation = False  # 标记当前 session 是否有真实对话

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 检测会话开始：INTERVIEW_START
            if 'INTERVIEW_START' in line:
                # 如果有正在记录的 session（例如前一场异常中断未打印 INTERVIEW_STOP），先完整落盘最后一个 QA
                if current_session:
                    if current_qa.get('question'):
                        current_turns.append(current_qa)
                    if current_turns and has_conversation:
                        current_session['turns'] = current_turns
                        sessions.append(current_session)

                # 提取时间戳
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if ts_match:
                    dt = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                    started_at = dt.timestamp()
                else:
                    started_at = time.time()

                current_session = {
                    'started_at': started_at,
                    'ended_at': None,
                    'source': 'log_import',
                }
                current_turns = []
                current_qa = {}
                has_conversation = False

            # 检测会话结束：INTERVIEW_STOP
            elif current_session and 'INTERVIEW_STOP' in line:
                # 保存当前 QA（如果有）
                if current_qa.get('question'):
                    current_turns.append(current_qa)
                    current_qa = {}

                ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if ts_match:
                    dt = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                    current_session['ended_at'] = dt.timestamp()
                if current_turns and has_conversation:
                    current_session['turns'] = current_turns
                    sessions.append(current_session)
                current_session = None
                current_turns = []
                has_conversation = False

            # 检测真实对话：source=conversation_loopback
            elif current_session and 'source=conversation_loopback' in line:
                has_conversation = True

            # 提取 ASR_QUESTION：面试官提的问题
            elif current_session and 'ASR_QUESTION' in line and 'text=' in line:
                # 保存上一个 QA
                if current_qa.get('question'):
                    current_turns.append(current_qa)

                # 提取问题文本（使用 _extract_repr 兼容单引号、双引号与转义字符）
                question_text = _extract_repr(line, 'text')
                if question_text:
                    # 去掉连续追问的前缀
                    if '以下内容来自同一轮实时面试' in question_text:
                        # 提取最关键的部分
                        parts = question_text.split('连续追问：')
                        if len(parts) > 1:
                            # 取连续追问的最后一条
                            追问s = [item.strip() for item in parts[1].strip().splitlines() if item.strip()]
                            if 追问s:
                                question_text = 追问s[-1].split('. ', 1)[-1] if '. ' in 追问s[-1] else 追问s[-1]
                        else:
                            # 取主问题
                            main_q = question_text.split('主问题：')
                            if len(main_q) > 1:
                                question_text = main_q[1].splitlines()[0].strip()

                    current_qa = {
                        'question': question_text[:500],
                        'answer': '',
                        'candidate_answer': '',
                    }

            # 提取 ANSWER_DONE：LLM 给出的参考答案（answer_len）
            elif current_session and current_qa and 'ANSWER_DONE' in line and 'answer_len=' in line:
                # 这里只能记录答案存在，实际文本在流式输出中，这里只标记有答案
                current_qa['answer'] = '(系统参考答案)'

    # 保存最后一个 session（只保存有真实对话的）
    if current_session and has_conversation:
        if current_qa.get('question'):
            current_turns.append(current_qa)
        if current_turns:
            current_session['turns'] = current_turns
            sessions.append(current_session)

    return sessions


def _collect_existing_started_ats() -> list[float]:
    started_ats: list[float] = []
    fetched = 0
    page = 1
    page_size = review.MAX_REVIEW_PAGE_SIZE
    while True:
        batch = review.list_sessions(page=page, page_size=page_size)
        items = batch.get('items') or []
        fetched += len(items)
        for item in items:
            started = item.get('started_at')
            if started is not None:
                started_ats.append(float(started))
        if fetched >= int(batch.get('total') or 0) or not items:
            break
        page += 1
    return started_ats


def import_sessions_to_db(sessions: List[Dict[str, Any]]) -> int:
    """
    导入会话到数据库（按 started_at 去重，状态置为 recorded）
    返回成功导入的会话数
    """
    imported_count = 0
    seen_started_ats = _collect_existing_started_ats()

    for sess in sessions:
        turns = sess.get('turns', [])
        if not turns:
            continue
        started_at = float(sess['started_at'])
        if any(abs(existing - started_at) < 1.0 for existing in seen_started_ats):
            logger.info("Skipped duplicate session started_at=%s", started_at)
            continue

        try:
            # 创建 session
            session_id = review.create_session(
                started_at=started_at,
                interviewer_enabled=True,
                candidate_enabled=True,
                source=str(sess.get('source') or 'log_import'),
            )

            # 添加 turns
            for idx, turn in enumerate(turns, start=1):
                review.add_turn(
                    session_id=session_id,
                    qa_id=f"log_import_{session_id}_{idx}",
                    seq=idx,
                    question_text=turn.get('question', ''),
                    candidate_answer_text=turn.get('candidate_answer', ''),
                    reference_answer_text=turn.get('answer', ''),
                    duration_ms=0,
                    is_partial=not bool((turn.get('candidate_answer') or '').strip()),
                    analysis_status='pending',
                )

            # 结束 session：未运行分析前保持 recorded 状态
            ended_at = sess.get('ended_at') or (started_at + len(turns) * 300)
            review.end_session(session_id=session_id, status='recorded', ended_at=ended_at)

            seen_started_ats.append(started_at)
            imported_count += 1
            logger.info(f"Imported session {session_id} with {len(turns)} turns")

        except Exception as e:
            logger.error(f"Failed to import session: {e}")

    return imported_count


def main():
    """主函数"""
    backend_dir = Path(__file__).parent.parent
    log_dir = backend_dir.parent / 'log'

    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        return

    print(f"正在扫描日志目录: {log_dir}")

    all_sessions = []
    # 只处理 interview.log 开头的日志文件
    log_files = sorted(log_dir.glob('interview.log*'))

    for log_file in log_files:
        print(f"解析: {log_file.name}")
        try:
            sessions = parse_log_file(str(log_file))
            all_sessions.extend(sessions)
            print(f"  发现 {len(sessions)} 个会话")
        except Exception as e:
            print(f"  解析失败: {e}")

    print(f"\n总共发现 {len(all_sessions)} 个会话")

    if not all_sessions:
        print("没有可导入的会话")
        return

    # 过滤有效会话：至少有 2 个 turn（过滤掉只是测试的会话）
    valid_sessions = [s for s in all_sessions if len(s.get('turns', [])) >= 2]
    print(f"有效会话（至少 2 个问答）: {len(valid_sessions)}")

    if not valid_sessions:
        print("没有有效会话可导入")
        return

    # 导入到数据库
    print("\n开始导入到数据库...")
    imported = import_sessions_to_db(valid_sessions)
    print(f"[OK] 成功导入 {imported} 个会话")

    # 显示统计
    try:
        total = review.list_sessions(page=1, page_size=1)['total']
        print(f"\n数据库中现有 {total} 场面试记录")
    except Exception:
        print("\n导入完成")


if __name__ == '__main__':
    main()
