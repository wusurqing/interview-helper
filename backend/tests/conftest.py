"""pytest 全局夹具。

必须在任何测试 import main/core/services 之前执行：
1. 将默认 IA_CONFIG_PATH 指向进程级隔离临时文件，绝不读取或改写开发者本机 backend/config.json；
2. 将 IA_DATA_DIR 与日志目录指向隔离临时目录，绝不读写真实 backend/data/ SQLite 或项目 log/；
3. 每个用例前后复位内存配置缓存与临时配置文件，避免跨用例状态污染。
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile

import pytest

_TEST_ENV_ROOT = tempfile.mkdtemp(prefix="ia-pytest-env-")
_TEST_LOG_DIR = os.path.join(_TEST_ENV_ROOT, "logs")
_TEST_DATA_DIR = os.path.join(_TEST_ENV_ROOT, "data")
_TEST_CONFIG_PATH = os.path.join(_TEST_ENV_ROOT, "config.json")

os.makedirs(_TEST_LOG_DIR, exist_ok=True)
os.makedirs(_TEST_DATA_DIR, exist_ok=True)

# 强制覆盖进程默认路径，确保单测绝不触碰开发者真实配置与数据库。
os.environ["IA_CONFIG_PATH"] = _TEST_CONFIG_PATH
os.environ["IA_DATA_DIR"] = _TEST_DATA_DIR


def _cleanup_test_env() -> None:
    try:
        logging.shutdown()
    except Exception:
        pass
    shutil.rmtree(_TEST_ENV_ROOT, ignore_errors=True)


atexit.register(_cleanup_test_env)

import core.logger as _logger  # noqa: E402

_logger.LOG_DIR = _TEST_LOG_DIR

import core.config as _config  # noqa: E402

_config.CONFIG_FILE = _TEST_CONFIG_PATH


def _reset_default_config_state() -> None:
    _config.CONFIG_FILE = _TEST_CONFIG_PATH
    _config._config = None
    _config._config_corrupt_state = None
    try:
        if os.path.exists(_TEST_CONFIG_PATH):
            os.remove(_TEST_CONFIG_PATH)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _isolate_default_config():
    _reset_default_config_state()
    yield
    _reset_default_config_state()
