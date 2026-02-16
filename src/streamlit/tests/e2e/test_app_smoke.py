"""
Smoke end-to-end test for Streamlit app startup.

Starts the app in headless mode and checks that the home page responds.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest


def _find_repo_root() -> Path:
    """Find repository root by locating src/streamlit/app.py."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "streamlit" / "app.py").exists():
            return parent
    raise RuntimeError("Could not locate repository root from test file path.")


def _free_port() -> int:
    """Reserve and return a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.e2e
@pytest.mark.smoke
def test_streamlit_homepage_is_reachable():
    """App bootstraps and serves the homepage."""
    repo_root = _find_repo_root()
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/streamlit/app.py",
        "--server.headless=true",
        f"--server.port={port}",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        deadline = time.time() + 45
        last_error = None

        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("Streamlit process exited before becoming healthy.")

            try:
                with urlopen(url, timeout=2) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
                assert resp.status == 200
                assert "streamlit" in body.lower() or "rakuten" in body.lower()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(1)

        pytest.fail(f"Streamlit app did not become ready in time. Last error: {last_error}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
