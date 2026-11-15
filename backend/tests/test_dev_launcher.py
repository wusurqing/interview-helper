import socket
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("service", ["backend", "frontend"])
def test_dev_launcher_refuses_to_take_over_a_listening_port(service):
    root = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with socket.socket() as spare:
            spare.bind(("127.0.0.1", 0))
            available = spare.getsockname()[1]
        ports = {"backend": available, "frontend": available}
        ports[service] = listener.getsockname()[1]
        result = subprocess.run(
            [
                sys.executable, "dev.py", "--web",
                "--backend-port", str(ports["backend"]),
                "--frontend-port", str(ports["frontend"]),
            ],
            cwd=root,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert f"--{service}-port".encode("ascii") in result.stderr
        assert str(ports[service]).encode("ascii") in result.stderr
        with socket.create_connection(listener.getsockname(), timeout=1):
            pass
