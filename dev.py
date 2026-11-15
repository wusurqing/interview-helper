#!/usr/bin/env python3
"""Run the local development servers and the Electron app from one terminal."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import webbrowser

from start import BACKEND_DIR, DESKTOP_DIR, FRONTEND_DIR, ROOT, _find_npm, ensure_python_deps


HOST = "127.0.0.1"
HTTP = build_opener(ProxyHandler({}))


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def check_port(port: int, option: str) -> None:
    with socket.socket() as sock:
        if os.name == "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((HOST, port))
        except OSError as error:
            raise RuntimeError(f"端口 {port} 无法使用，请设置 {option} 指定其他端口。") from error


def prepare_python() -> str:
    python = sys.executable
    if sys.prefix == sys.base_prefix and not os.environ.get("CONDA_PREFIX"):
        venv_dir = Path(ROOT) / ".venv"
        relative_python = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        local_python = venv_dir / relative_python
        if not local_python.is_file():
            print("[dev] 创建项目虚拟环境 .venv", flush=True)
            subprocess.run([python, "-m", "venv", str(venv_dir)], check=True)
        python = str(local_python)

    if python == sys.executable:
        ensure_python_deps()
    else:
        subprocess.run(
            [python, "-c", "from start import ensure_python_deps; ensure_python_deps()"],
            cwd=ROOT,
            check=True,
        )
    return python


def install_node_deps(directory: str) -> None:
    npm = _find_npm()
    if npm is None:
        raise RuntimeError("找不到 npm，请安装 Node.js 后重试。")
    print(f"[dev] 安装 {Path(directory).name} 依赖", flush=True)
    subprocess.run([npm, "ci", "--include=dev"], cwd=directory, check=True)


def electron_binary(node: str) -> str:
    def installed_binary():
        result = subprocess.run(
            [node, "-p", "require('electron')"], cwd=DESKTOP_DIR, capture_output=True, text=True,
        )
        binary = result.stdout.strip()
        if result.returncode == 0 and Path(binary).is_file():
            return binary
        return None

    binary = installed_binary()
    if binary is None:
        install_node_deps(DESKTOP_DIR)
        binary = installed_binary()
    if binary is None:
        raise RuntimeError("找不到 Electron 可执行文件，请检查 desktop 的 npm 安装输出。")
    return binary


def stop_process(process: subprocess.Popen) -> None:
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            if process.poll() is None:
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def stop_backend(process: subprocess.Popen) -> None:
    if os.name == "nt" and process.poll() is None:
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=4)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    stop_process(process)


def start_process(
    stack: ExitStack, command: list[str], cwd: str, env: dict, *, cleanup=stop_process,
) -> subprocess.Popen:
    if os.name == "nt":
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        options = {"start_new_session": True}
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, **options)
    stack.callback(cleanup, process)
    return process


def python_sources() -> dict[str, int]:
    modified = {}
    for directory, subdirs, filenames in os.walk(BACKEND_DIR):
        subdirs[:] = [name for name in subdirs if name not in {"__pycache__", "data", "tmp", ".venv"}]
        for filename in filenames:
            if filename.endswith(".py"):
                path = os.path.join(directory, filename)
                try:
                    modified[path] = os.stat(path).st_mtime_ns
                except FileNotFoundError:
                    pass  # An editor may replace the file while we walk the directory.
    return modified


def check_services(services: dict[str, subprocess.Popen]) -> None:
    for name, process in services.items():
        if process.poll() is not None:
            raise RuntimeError(f"{name} 已退出（退出码 {process.returncode}），请查看上方日志。")


def wait_until_ready(url: str, services: dict[str, subprocess.Popen]) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        check_services(services)
        try:
            with HTTP.open(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (URLError, OSError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"等待 {url} 就绪超时，请查看上方日志。")


def run(args: argparse.Namespace) -> int:
    check_port(args.backend_port, "--backend-port")
    check_port(args.frontend_port, "--frontend-port")
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("找不到 node，请安装 Node.js 后重试。")

    python = prepare_python()
    vite = Path(FRONTEND_DIR) / "node_modules/vite/bin/vite.js"
    if not vite.is_file():
        install_node_deps(FRONTEND_DIR)
    electron = None if args.web else electron_binary(node)
    frontend_url = f"http://{HOST}:{args.frontend_port}"
    backend_url = f"http://{HOST}:{args.backend_port}"
    env = {
        **os.environ,
        "PORT": str(args.backend_port),
        "IA_DEV_BACKEND_PORT": str(args.backend_port),
        "IA_DEV_FRONTEND_PORT": str(args.frontend_port),
        "PYTHONUNBUFFERED": "1",
    }

    backend_command = [
        python, "-m", "uvicorn", "main:app", "--host", HOST, "--port", str(args.backend_port),
        "--timeout-graceful-shutdown", "2",
    ]
    if os.environ.get("IA_ACCESS_LOG", "1").strip().lower() in {"0", "false", "no"}:
        backend_command.append("--no-access-log")

    with ExitStack() as backend_stack, ExitStack() as stack:
        sources = python_sources()
        backend = start_process(backend_stack, backend_command, BACKEND_DIR, env, cleanup=stop_backend)
        frontend = start_process(stack, [
            node, str(vite), "--host", HOST, "--port", str(args.frontend_port),
            "--strictPort", "--clearScreen", "false",
        ], FRONTEND_DIR, env)
        services = {"后端": backend, "Vite": frontend}
        wait_until_ready(f"{frontend_url}/@vite/client", services)
        wait_until_ready(f"{backend_url}/api/options", services)

        print(f"[dev] {frontend_url} · Ctrl+C 停止调试", flush=True)
        if electron is None:
            webbrowser.open(frontend_url)
            desktop = None
        else:
            desktop = start_process(stack, [electron, "."], DESKTOP_DIR, env)

        reported_backend_exit = False
        while desktop is None or desktop.poll() is None:
            check_services({"Vite": frontend})
            updated_sources = python_sources()
            if updated_sources != sources:
                # Uvicorn's Windows reloader sends Ctrl+C to its shared console.
                # Restart the API process directly so the terminal and UI stay up.
                print("[dev] Python 源码有改动，正在重启后端。", flush=True)
                backend_stack.close()
                backend = start_process(backend_stack, backend_command, BACKEND_DIR, env, cleanup=stop_backend)
                sources = updated_sources
                reported_backend_exit = False
            if backend.poll() is not None and not reported_backend_exit:
                print(f"[dev] 后端已退出（退出码 {backend.returncode}），保存 Python 源码后会重试。", flush=True)
                reported_backend_exit = True
            time.sleep(0.5)
        return desktop.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="启动本地开发环境（Vite + Uvicorn + Electron）")
    parser.add_argument("--web", action="store_true", help="在默认浏览器打开应用")
    parser.add_argument("--backend-port", type=port_number, default=18080)
    parser.add_argument("--frontend-port", type=port_number, default=5173)
    args = parser.parse_args()
    if args.backend_port == args.frontend_port:
        parser.error("前端和后端需要使用不同端口。")

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[dev] 已停止。")
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"[dev] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
