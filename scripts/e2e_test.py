#!/usr/bin/env python3
"""End-to-end smoke test: backend API + frontend build."""

import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")


def log(msg: str):
    print(f"  [E2E] {msg}")


def run_cmd(cmd: list, cwd: str = ROOT, timeout: int = 120) -> tuple[int, str]:
    exe = cmd[0]
    if exe == "npm":
        exe = shutil.which("npm") or shutil.which("npm.cmd") or "npm"
    cmd = [exe] + cmd[1:]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def wait_for_server(url: str, process, max_wait: int = 30) -> bool:
    for _ in range(max_wait):
        if not process.is_alive():
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    r.read()
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    return False


def serve_backend(listener: socket.socket, sandbox: str):
    import uvicorn

    os.environ.update(
        PORT=str(listener.getsockname()[1]),
        IA_CONFIG_PATH=os.path.join(sandbox, "config.json"),
        IA_DATA_DIR=os.path.join(sandbox, "data"),
        IA_LLM_TRACE="0",
    )
    os.chdir(BACKEND_DIR)
    sys.path.insert(0, BACKEND_DIR)
    from core import logger

    logger.LOG_DIR = os.environ.get("IA_SMOKE_LOG_DIR") or os.path.join(sandbox, "logs")
    uvicorn.Server(uvicorn.Config("main:app")).run(sockets=[listener])


@contextmanager
def smoke_backend():
    # Keep the socket bound from port allocation through server shutdown.
    with tempfile.TemporaryDirectory(prefix="ia-smoke-") as sandbox, socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        process = multiprocessing.get_context("spawn").Process(target=serve_backend, args=(listener, sandbox))
        process.start()
        try:
            yield f"http://127.0.0.1:{port}", process
        finally:
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def main() -> int:
    print("\n=== 端到端自测 ===\n")

    # 1. Frontend build
    dist_index = os.path.join(FRONTEND_DIR, "dist", "index.html")
    if os.path.isfile(dist_index):
        log("1. 前端已构建，跳过 build")
    else:
        log("1. 构建前端...")
        code, out = run_cmd(["npm", "run", "build"], cwd=FRONTEND_DIR)
        if code != 0:
            log(f"FAIL: npm run build 失败\n{out}")
            return 1
        log("OK 前端构建成功")

    # 2. Check frontend dist
    if not os.path.isfile(dist_index):
        log("FAIL: dist/index.html 不存在")
        return 1
    log("OK dist 文件存在")

    log("2. 启动后端服务...")
    with smoke_backend() as (base, process):
        if not wait_for_server(f"{base}/api/config", process, max_wait=25):
            log("FAIL: 后端未在 25 秒内就绪")
            return 1
        log("OK 后端已就绪")

        return check_endpoints(base)


def check_endpoints(base: str) -> int:
    # 4. Test API endpoints
    log("3. 测试 API...")

    tests = [
        ("GET", "/api/config", None),
        ("GET", "/api/options", None),
        ("GET", "/api/devices", None),
        ("GET", "/api/network-info", None),
        ("GET", "/api/stt/status", None),
        ("GET", "/api/session", None),
        ("POST", "/api/clear", "{}"),
        ("POST", "/api/ask/cancel", "{}"),
        # 能力分析 / 求职看板（api 分包后路径不变）
        ("GET", "/api/knowledge/summary", None),
        ("GET", "/api/job-tracker/stages", None),
        ("GET", "/api/job-tracker/applications", None),
        ("GET", "/api/resume/history", None),
    ]

    for method, path, body in tests:
        req = urllib.request.Request(f"{base}{path}", method=method)
        if body:
            req.add_header("Content-Type", "application/json")
            req.data = body.encode("utf-8")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status != 200:
                    log(f"FAIL {path} -> {r.status}")
                    return 1
                json.load(r)
        except urllib.error.HTTPError as e:
            log(f"FAIL {method} {path} -> {e.code}")
            return 1
        except (OSError, ValueError) as e:
            log(f"FAIL {method} {path} -> {e}")
            return 1
        log(f"  OK {method} {path}")

    # 5. Test SPA page
    log("4. 测试前端页面...")
    try:
        with urllib.request.urlopen(f"{base}/", timeout=5) as r:
            if r.status != 200:
                log(f"FAIL / -> {r.status}")
                return 1
            html = r.read().decode("utf-8", errors="replace")
            if "<!DOCTYPE html>" not in html and "<html" not in html.lower():
                log("FAIL / 返回内容不是 HTML")
                return 1
    except Exception as e:
        log(f"FAIL / -> {e}")
        return 1
    log("OK 前端页面可访问")

    print("\n=== 全部通过 ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
