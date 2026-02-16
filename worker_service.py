"""Overmind Worker Service — auto-updating watchdog for remote worker deployment.

Runs the WorkerClient and automatically pulls latest code from git.
When code changes are detected, restarts the worker process.

Usage:
    python worker_service.py --server ws://192.168.1.100:8000/ws/worker --machine-id mac-builder

    # With custom workspace and pull interval:
    python worker_service.py \
        --server ws://HOST:8000/ws/worker \
        --machine-id remote-1 \
        --workspace ./my_workspace \
        --pull-interval 30
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker_service")

# Directory where this script lives (repo root)
REPO_DIR = Path(__file__).resolve().parent


def git_pull() -> bool:
    """Run git pull and return True if files changed."""
    try:
        # Get current HEAD before pull
        before = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=10,
        )
        head_before = before.stdout.strip()

        # Pull
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=30,
        )

        if result.returncode != 0:
            log.warning("git pull failed: %s", result.stderr.strip())
            return False

        # Get HEAD after pull
        after = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=10,
        )
        head_after = after.stdout.strip()

        if head_before != head_after:
            log.info("Code updated: %s -> %s", head_before[:8], head_after[:8])
            return True

        return False

    except subprocess.TimeoutExpired:
        log.warning("git pull timed out")
        return False
    except FileNotFoundError:
        log.warning("git not found — auto-update disabled")
        return False
    except Exception as e:
        log.warning("git pull error: %s", e)
        return False


def start_worker(server: str, machine_id: str, workspace: str) -> subprocess.Popen:
    """Start the worker client as a subprocess."""
    cmd = [
        sys.executable, "-m", "overmind.workers.client",
        "--server", server,
        "--machine-id", machine_id,
        "--workspace", workspace,
    ]
    env = os.environ.copy()
    # Ensure src/ is in PYTHONPATH
    src_dir = str(REPO_DIR / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else src_dir

    log.info("Starting worker: %s", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overmind Worker Service — auto-updating watchdog",
    )
    parser.add_argument(
        "--server", required=True,
        help="WebSocket URL of the Overmind server (e.g. ws://192.168.1.100:8000/ws/worker)",
    )
    parser.add_argument(
        "--machine-id", required=True,
        help="Unique identifier for this worker machine",
    )
    parser.add_argument(
        "--workspace", default=".",
        help="Workspace directory for file operations (default: current directory)",
    )
    parser.add_argument(
        "--pull-interval", type=int, default=60,
        help="Seconds between git pull checks (default: 60)",
    )
    args = parser.parse_args()

    workspace = str(Path(args.workspace).resolve())
    Path(workspace).mkdir(parents=True, exist_ok=True)

    running = True
    worker_proc: subprocess.Popen | None = None

    def handle_signal(sig, frame):
        nonlocal running
        running = False
        log.info("Shutdown signal received")
        if worker_proc and worker_proc.poll() is None:
            worker_proc.terminate()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Initial pull
    git_pull()

    # Start worker
    worker_proc = start_worker(args.server, args.machine_id, workspace)
    last_pull = time.time()

    log.info("Worker service started (pull interval: %ds)", args.pull_interval)

    while running:
        # Check if worker process died
        if worker_proc.poll() is not None:
            exit_code = worker_proc.returncode
            log.warning("Worker exited with code %d, restarting in 3s...", exit_code)
            time.sleep(3)
            if running:
                worker_proc = start_worker(args.server, args.machine_id, workspace)
                last_pull = time.time()

        # Periodic git pull
        if time.time() - last_pull >= args.pull_interval:
            changed = git_pull()
            last_pull = time.time()

            if changed and running:
                log.info("Code updated — restarting worker...")
                worker_proc.terminate()
                try:
                    worker_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker_proc.kill()
                    worker_proc.wait()
                worker_proc = start_worker(args.server, args.machine_id, workspace)

        time.sleep(1)

    # Cleanup
    if worker_proc and worker_proc.poll() is None:
        log.info("Stopping worker...")
        worker_proc.terminate()
        try:
            worker_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker_proc.kill()
            worker_proc.wait()

    log.info("Worker service stopped.")


if __name__ == "__main__":
    main()
