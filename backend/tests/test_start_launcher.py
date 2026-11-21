"""start.py 桌面模式启动监控：成功隐藏窗口 / 失败保留窗口并展示日志。"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
START = ROOT / "start.py"


def load_start():
    spec = importlib.util.spec_from_file_location("start_launcher", START)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dependency_check_installs_requirements_when_upload_support_is_missing(monkeypatch):
    start = load_start()
    installed = False

    def import_package(name):
        if name == "multipart.multipart" and not installed:
            raise ModuleNotFoundError(name)

    def install_requirements(command):
        nonlocal installed
        assert command[:4] == [sys.executable, "-m", "pip", "install"]
        assert command[4:6] == ["-r", str(ROOT / "backend" / "requirements.txt")]
        installed = True
        return start.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(start, "__import__", import_package, raising=False)
    monkeypatch.setattr(start.subprocess, "run", install_requirements)
    start.ensure_python_deps()
    assert installed


def _patch_common(monkeypatch, tmp_path, *, ready, poll_results):
    start = load_start()
    log_path = tmp_path / "startup-console.log"
    monkeypatch.setattr(start, "STARTUP_LOG", str(log_path))
    monkeypatch.setattr(start, "_find_npx", lambda: "npx")
    monkeypatch.setattr(start, "ensure_electron", lambda: True)
    monkeypatch.setattr(start, "kill_port", lambda _port: None)
    monkeypatch.setattr(start, "get_local_ip", lambda: "127.0.0.1")
    monkeypatch.delenv(start.HIDE_CONSOLE_ENV, raising=False)
    monkeypatch.setattr(start.platform, "system", lambda: "Windows")
    monkeypatch.setattr(start, "_app_ready", lambda port, timeout=1.5: ready())
    popen_calls = []

    class FakeProc:
        def __init__(self):
            self._polls = iter(poll_results)
            self.waited = False

        def poll(self):
            try:
                return next(self._polls)
            except StopIteration:
                return None

        def wait(self, timeout=None):
            self.waited = True
            return 0

    procs = []

    def _popen(*a, **k):
        proc = FakeProc()
        popen_calls.append(k)
        procs.append(proc)
        return proc

    monkeypatch.setattr(start.subprocess, "Popen", _popen)
    monkeypatch.setattr(start.time, "sleep", lambda *_a: None)
    hide_calls = []
    monkeypatch.setattr(start, "_hide_own_console", lambda: hide_calls.append(True))
    return start, popen_calls, hide_calls, log_path, procs


def test_desktop_run_hides_console_when_ready(monkeypatch, tmp_path):
    start, popen_calls, hide_calls, log_path, procs = _patch_common(
        monkeypatch, tmp_path, ready=lambda: True, poll_results=[None],
    )
    monkeypatch.setattr(start.os, "environ", {**os.environ})

    with pytest.raises(SystemExit) as ei:
        start.run_desktop_mode(18080)
    assert ei.value.code == 0
    assert hide_calls == [True]
    # 隐藏后驻留：等子进程（应用）退出才结束
    assert procs and procs[0].waited is True
    # 子进程带隐藏标志启动且输出进日志文件
    kwargs = popen_calls[0]
    assert kwargs["env"][start.HIDE_CONSOLE_ENV] == "1"
    assert kwargs["creationflags"] == getattr(start.subprocess, "CREATE_NO_WINDOW", 0x08000000)
    assert kwargs["stdout"] != start.subprocess.DEVNULL


def test_desktop_run_keeps_console_and_shows_log_on_failure(monkeypatch, tmp_path, capsys):
    start, _popen_calls, hide_calls, log_path, _procs = _patch_common(
        monkeypatch, tmp_path, ready=lambda: False, poll_results=[1],
    )
    log_path.write_text("[ERROR] npm install 失败\n", encoding="utf-8")
    monkeypatch.setattr(start.os, "environ", {**os.environ})
    monkeypatch.setattr("builtins.input", lambda *_a: "")

    with pytest.raises(SystemExit) as ei:
        start.run_desktop_mode(18080)
    assert ei.value.code == 1
    assert hide_calls == []  # 失败不隐藏，用户要看日志
    out = capsys.readouterr().out
    assert "启动失败" in out
    assert "npm install 失败" in out


def test_app_ready_accepts_any_http_response(monkeypatch):
    import urllib.error

    start = load_start()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(start.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert start._app_ready(18080) is True

    def _raise_http_error(*a, **k):
        raise urllib.error.HTTPError("u", 401, "unauthorized", None, None)

    monkeypatch.setattr(start.urllib.request, "urlopen", _raise_http_error)
    assert start._app_ready(18080) is True

    monkeypatch.setattr(start.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    assert start._app_ready(18080) is False


def test_parse_windows_listening_pids_matches_exact_local_port_only():
    start = load_start()
    sample = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:8080           0.0.0.0:0              LISTENING       9001
  TCP    0.0.0.0:80             0.0.0.0:0              LISTENING       1001
  TCP    [::]:80                [::]:0                 LISTENING       1001
  TCP    127.0.0.1:8000         0.0.0.0:0              LISTENING       9002
  TCP    192.168.1.10:51234     10.0.0.1:80            LISTENING       9003
  TCP    127.0.0.1:80           10.0.0.2:443           ESTABLISHED     9004
  TCP    [fe80::1%12]:18080     [::]:0                 LISTENING       2002
  TCP    0.0.0.0:18080          0.0.0.0:0              LISTENING       2003
  TCP    0.0.0.0:1808           0.0.0.0:0              LISTENING       2004
  UDP    0.0.0.0:80             *:*                                    9005
  TCP    0.0.0.0:80             0.0.0.0:0              LISTENING       0
"""
    assert start._parse_windows_listening_pids(sample, 80) == [1001]
    assert start._parse_windows_listening_pids(sample, 1808) == [2004]
    assert start._parse_windows_listening_pids(sample, 18080) == [2002, 2003]
    assert start._parse_windows_listening_pids(sample, 9999) == []
    # Malformed / portless IPv6 or empty-host addresses must never match a port
    assert start._extract_port("::1") is None
    assert start._extract_port("[::1") is None
    assert start._extract_port(":80") is None
    assert start._extract_port("[]:80") is None
    assert start._extract_port("[::]:80") == 80
    assert start._extract_port("127.0.0.1:8080") == 8080


def test_kill_port_windows_kills_exact_port_and_skips_prefix_and_self(monkeypatch):
    start = load_start()
    self_pid = os.getpid()
    netstat_stdout = (
        f"  TCP    0.0.0.0:8080     0.0.0.0:0    LISTENING    7777\n"
        f"  TCP    0.0.0.0:80       0.0.0.0:0    LISTENING    {self_pid}\n"
        f"  TCP    127.0.0.1:80     0.0.0.0:0    LISTENING    4321\n"
        f"  TCP    [::]:80          [::]:0       LISTENING    4321\n"
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "netstat":
            return subprocess.CompletedProcess(cmd, 0, stdout=netstat_stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(start.platform, "system", lambda: "Windows")
    monkeypatch.setattr(start.subprocess, "run", fake_run)
    monkeypatch.setattr(start.time, "sleep", lambda _sec: None)

    start.kill_port(80)

    assert calls == [
        ["netstat", "-ano"],
        ["taskkill", "/F", "/PID", "4321"],
    ]


def test_kill_port_posix_queries_listening_tcp_only_and_deduplicates(monkeypatch):
    start = load_start()
    self_pid = os.getpid()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "lsof":
            return subprocess.CompletedProcess(cmd, 0, stdout=f"5555\n{self_pid}\n5555\ninvalid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(start.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(start.subprocess, "run", fake_run)
    monkeypatch.setattr(start.time, "sleep", lambda _sec: None)

    start.kill_port(18080)

    assert calls == [
        ["lsof", "-tiTCP:18080", "-sTCP:LISTEN"],
        ["kill", "-9", "5555"],
    ]


def test_get_local_ip_closes_udp_socket_on_connect_error(monkeypatch):
    start = load_start()
    closed = []

    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            closed.append(True)
            return False

        def connect(self, _addr):
            raise OSError("network unreachable")

        def getsockname(self):
            return ("192.168.1.99", 12345)

    monkeypatch.setattr(start.socket, "socket", MagicMock(return_value=DummySocket()))

    assert start.get_local_ip() == "127.0.0.1"
    assert closed == [True]


def test_get_local_ip_returns_ip_and_closes_socket_on_success(monkeypatch):
    start = load_start()
    closed = []

    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            closed.append(True)
            return False

        def connect(self, _addr):
            return None

        def getsockname(self):
            return ("192.168.1.42", 54321)

    monkeypatch.setattr(start.socket, "socket", MagicMock(return_value=DummySocket()))

    assert start.get_local_ip() == "192.168.1.42"
    assert closed == [True]


def test_kill_port_does_not_sleep_or_report_ok_when_kill_fails(monkeypatch, capsys):
    start = load_start()
    slept = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "netstat":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="  TCP    0.0.0.0:18080    0.0.0.0:0    LISTENING    4321\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="not found")

    monkeypatch.setattr(start.platform, "system", lambda: "Windows")
    monkeypatch.setattr(start.subprocess, "run", fake_run)
    monkeypatch.setattr(start.time, "sleep", lambda sec: slept.append(sec))

    start.kill_port(18080)

    assert slept == []
    out = capsys.readouterr().out
    assert "[OK]" not in out
    assert "[WARN]" in out
