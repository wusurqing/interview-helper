"""pytest 全局夹具。

必须在任何测试 import main/core 之前执行：把日志目录指到系统临时目录，
避免测试进程（import main 会触发 setup_logging）把转录/答案/错误日志
写进项目根目录 log/ 污染生产日志。
"""

import os
import tempfile

# Storage modules initialize databases on import. Isolate BEFORE test collection,
# including legacy fixtures which delete rows through an imported DB_PATH.
_TEST_DATA = tempfile.TemporaryDirectory(prefix="ia-test-data-", ignore_cleanup_errors=True)
os.environ["IA_DATA_DIR"] = _TEST_DATA.name
os.environ["IA_CONFIG_PATH"] = os.path.join(_TEST_DATA.name, "config.json")

_TEST_LOG_DIR = os.path.join(tempfile.gettempdir(), "ia-test-logs")

import core.logger as _logger  # noqa: E402

_logger.LOG_DIR = _TEST_LOG_DIR
