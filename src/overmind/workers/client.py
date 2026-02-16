"""Overmind Worker Client — run on remote machines to register as a worker node.

This script connects to the Overmind server via WebSocket, registers as a worker,
and processes tool call requests by executing them locally using ToolExecutor.

Multiple instances can run on the same machine with different machine IDs
and workspace directories for testing.

Usage:
    python -m overmind.workers.client \\
        --server ws://overmind-host:8000/ws/worker \\
        --machine-id desktop-001

    # Multiple workers on the same machine (testing):
    python -m overmind.workers.client \\
        --server ws://localhost:8000/ws/worker \\
        --machine-id test-worker-1 \\
        --workspace ./workspace2

    python -m overmind.workers.client \\
        --server ws://localhost:8000/ws/worker \\
        --machine-id test-worker-2 \\
        --workspace ./workspace3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import socket
import sys
from pathlib import Path

# Allow running from project root without installing
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from overmind.agents.tools import ToolExecutor
from overmind.utils.llm_client import ToolCall, ToolResultWithImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("overmind.worker")


class WorkerClient:
    """Connects to Overmind server and executes tool calls locally."""

    def __init__(
        self,
        server_url: str,
        machine_id: str,
        workspace_dir: Path,
    ):
        self.server_url = server_url
        self.machine_id = machine_id
        self.workspace_dir = workspace_dir.resolve()
        self.executor = ToolExecutor(self.workspace_dir)
        self._running = True

    def _detect_capabilities(self) -> list[str]:
        """Detect what this worker can do based on available libraries."""
        caps = ["bash", "file_ops", "search"]
        try:
            import pyautogui  # noqa: F401
            caps.extend(["screenshot", "gui"])
        except ImportError:
            pass
        try:
            import pygetwindow  # noqa: F401
            caps.append("window_management")
        except ImportError:
            pass
        return caps

    async def run(self) -> None:
        """Connect to server with auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_serve()
            except Exception as e:
                if not self._running:
                    break
                log.warning("Connection lost: %s. Reconnecting in 3s...", e)
                await asyncio.sleep(3)

    async def _connect_and_serve(self) -> None:
        """Single connection lifecycle: connect, register, serve, handle disconnect."""
        try:
            import websockets
        except ImportError:
            log.error("websockets package not installed. Run: pip install websockets")
            self._running = False
            return

        log.info("Connecting to %s ...", self.server_url)

        async with websockets.connect(
            self.server_url,
            ping_interval=20,
            ping_timeout=30,
            max_size=50 * 1024 * 1024,  # 50MB for screenshots
        ) as ws:
            # Register
            capabilities = self._detect_capabilities()
            hostname = socket.gethostname()

            await ws.send(json.dumps({
                "type": "register",
                "machine_id": self.machine_id,
                "hostname": hostname,
                "capabilities": capabilities,
            }))

            reg = json.loads(await ws.recv())
            if not reg.get("ok"):
                log.error("Registration failed: %s", reg)
                self._running = False
                return

            log.info(
                "Registered as '%s' (hostname=%s, workspace=%s, capabilities=%s)",
                self.machine_id, hostname, self.workspace_dir, capabilities,
            )

            # Main loop: process incoming messages
            async for raw in ws:
                if not self._running:
                    break

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("Received invalid JSON: %s", raw[:200])
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "tool_call":
                    # Execute tool call in background to avoid blocking
                    asyncio.create_task(
                        self._handle_tool_call(ws, msg)
                    )
                elif msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                else:
                    log.debug("Unknown message type: %s", msg_type)

    async def _handle_tool_call(self, ws: Any, msg: dict) -> None:
        """Execute a tool call and send the result back."""
        request_id = msg.get("request_id", "?")
        tool_name = msg.get("tool_name", "")
        arguments = msg.get("arguments", {})

        log.info("Tool call: %s(%s) [%s]", tool_name, str(arguments)[:100], request_id)

        try:
            tc = ToolCall(id=request_id, name=tool_name, arguments=arguments)
            result = await self.executor.execute(tc)

            if isinstance(result, ToolResultWithImage):
                response = {
                    "type": "tool_result",
                    "request_id": request_id,
                    "text": result.text,
                    "image_base64": result.image_base64,
                }
            else:
                response = {
                    "type": "tool_result",
                    "request_id": request_id,
                    "text": result,
                    "image_base64": None,
                }

            await ws.send(json.dumps(response))
            result_preview = result.text if isinstance(result, ToolResultWithImage) else result
            log.info("Tool result: %s → %s [%s]", tool_name, result_preview[:100], request_id)

        except Exception as e:
            log.exception("Tool call failed: %s", tool_name)
            await ws.send(json.dumps({
                "type": "tool_result",
                "request_id": request_id,
                "text": f"Error executing {tool_name}: {type(e).__name__}: {e}",
                "image_base64": None,
            }))

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._running = False
        log.info("Worker stopping...")


# Need Any for ws type hint in _handle_tool_call
from typing import Any  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overmind Worker Client — execute tasks on this machine",
    )
    parser.add_argument(
        "--server",
        required=True,
        help="WebSocket URL of the Overmind server (e.g. ws://localhost:8000/ws/worker)",
    )
    parser.add_argument(
        "--machine-id",
        required=True,
        help="Unique identifier for this worker machine",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace directory for file operations (default: current directory)",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    client = WorkerClient(
        server_url=args.server,
        machine_id=args.machine_id,
        workspace_dir=workspace,
    )

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def handle_signal(sig: int, frame: Any) -> None:
        client.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(client.run())
    except KeyboardInterrupt:
        client.stop()
    finally:
        loop.close()

    log.info("Worker shut down.")


if __name__ == "__main__":
    main()
