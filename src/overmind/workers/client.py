"""Overmind Worker Client — run on remote machines to register as a worker node.

This script connects to the Overmind server via WebSocket, registers as a worker,
and processes requests: legacy tool calls + OpenCode HTTP proxy + SSE forwarding.

OpenCode integration:
    If `opencode` is in PATH, the worker will:
    1. Start `opencode serve` on localhost
    2. Proxy HTTP requests from the server to opencode serve
    3. Forward SSE events from opencode serve back through WebSocket

Usage:
    python -m overmind.workers.client \\
        --server ws://overmind-host:8000/ws/worker \\
        --machine-id desktop-001

    # With custom opencode port:
    python -m overmind.workers.client \\
        --server ws://localhost:8000/ws/worker \\
        --machine-id desktop-001 \\
        --opencode-port 4096
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

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
    """Connects to Overmind server and executes tool calls locally.

    If OpenCode is installed, also manages an `opencode serve` subprocess
    and proxies HTTP/SSE between the server and OpenCode.
    """

    def __init__(
        self,
        server_url: str,
        machine_id: str,
        workspace_dir: Path,
        opencode_port: int = 4096,
    ):
        self.server_url = server_url
        self.machine_id = machine_id
        self.workspace_dir = workspace_dir.resolve()
        self.executor = ToolExecutor(self.workspace_dir)
        self._running = True
        self._current_ws: Any = None  # Track active WebSocket for forced shutdown

        # OpenCode
        self._opencode_port = opencode_port
        self._opencode_process: asyncio.subprocess.Process | None = None
        self._opencode_ready = False
        self._opencode_base_url = f"http://127.0.0.1:{opencode_port}"
        self._http_session: Any = None  # aiohttp.ClientSession, lazy init
        self._sse_task: asyncio.Task | None = None

    # ── Capability Detection ──────────────────────────────────────────────

    def _detect_capabilities(self) -> list[str]:
        """Detect what this worker can do based on available tools."""
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
        if self._has_opencode():
            caps.append("opencode")
        return caps

    @staticmethod
    def _has_opencode() -> bool:
        """Check if the opencode binary is available in PATH."""
        return shutil.which("opencode") is not None

    # ── OpenCode Serve Management ─────────────────────────────────────────

    async def _start_opencode_serve(self) -> bool:
        """Start opencode serve subprocess. Returns True if started successfully."""
        if not self._has_opencode():
            return False

        log.info("Starting opencode serve on port %d ...", self._opencode_port)

        env = os.environ.copy()
        # Auto-approve all permissions for autonomous operation
        env["OPENCODE_PERMISSION"] = '{"*":"allow"}'

        try:
            opencode_bin = shutil.which("opencode") or "opencode"
            # On Windows, npm installs .CMD wrappers that can't be exec'd directly
            if sys.platform == "win32" and opencode_bin.lower().endswith((".cmd", ".bat")):
                cmd = [opencode_bin, "serve", "--port", str(self._opencode_port), "--hostname", "0.0.0.0"]
                self._opencode_process = await asyncio.create_subprocess_shell(
                    " ".join(cmd),
                    env=env,
                    cwd=str(self.workspace_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                self._opencode_process = await asyncio.create_subprocess_exec(
                    opencode_bin, "serve",
                    "--port", str(self._opencode_port),
                    "--hostname", "0.0.0.0",
                    env=env,
                    cwd=str(self.workspace_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
        except FileNotFoundError:
            log.warning("opencode binary not found at: %s", shutil.which("opencode"))
            return False
        except Exception as e:
            log.error("Failed to start opencode serve: %s", e)
            return False

        # Wait for it to accept connections
        if await self._wait_opencode_ready(timeout=30):
            log.info("opencode serve is ready at %s", self._opencode_base_url)
            self._opencode_ready = True
            return True

        log.error("opencode serve failed to become ready within timeout")
        await self._stop_opencode_serve()
        return False

    async def _wait_opencode_ready(self, timeout: float = 30) -> bool:
        """Poll until opencode serve is accepting connections."""
        import aiohttp

        start = time.time()
        while time.time() - start < timeout:
            # Check if process died
            if self._opencode_process and self._opencode_process.returncode is not None:
                stderr = ""
                if self._opencode_process.stderr:
                    stderr = (await self._opencode_process.stderr.read()).decode(errors="replace")
                log.error("opencode serve exited with code %d: %s",
                          self._opencode_process.returncode, stderr[:500])
                return False

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self._opencode_base_url}/path",
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as resp:
                        if resp.status == 200:
                            return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return False

    async def _stop_opencode_serve(self) -> None:
        """Stop the opencode serve subprocess."""
        self._opencode_ready = False
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
            self._sse_task = None

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        if self._opencode_process:
            try:
                self._opencode_process.terminate()
                await asyncio.wait_for(self._opencode_process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._opencode_process.kill()
                await self._opencode_process.wait()
            except Exception:
                pass
            self._opencode_process = None
            log.info("opencode serve stopped")

    async def _get_http_session(self) -> Any:
        """Get or create a shared aiohttp ClientSession."""
        import aiohttp

        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._http_session

    # ── OpenCode HTTP Proxy ───────────────────────────────────────────────

    async def _handle_opencode_http(self, ws: Any, msg: dict) -> None:
        """Proxy an HTTP request to local opencode serve and return the response."""
        request_id = msg.get("request_id", "?")
        method = msg.get("method", "GET").upper()
        path = msg.get("path", "/")
        body = msg.get("body")

        url = f"{self._opencode_base_url}{path}"
        log.info("OpenCode HTTP: %s %s [%s]", method, path, request_id)

        if not self._opencode_ready:
            await ws.send(json.dumps({
                "type": "opencode_http_response",
                "request_id": request_id,
                "status": 503,
                "body": {"error": "opencode serve is not running"},
            }))
            return

        try:
            session = await self._get_http_session()
            kwargs: dict[str, Any] = {
                "headers": {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            }
            if body is not None:
                kwargs["json"] = body

            async with session.request(method, url, **kwargs) as resp:
                content_type = resp.content_type or ""
                if "json" in content_type:
                    resp_body = await resp.json()
                else:
                    text = await resp.text()
                    resp_body = text if text else None

                await ws.send(json.dumps({
                    "type": "opencode_http_response",
                    "request_id": request_id,
                    "status": resp.status,
                    "body": resp_body,
                }))

            log.debug("OpenCode HTTP response: %s %s → %d [%s]",
                      method, path, resp.status, request_id)

        except Exception as e:
            log.error("OpenCode HTTP proxy error: %s", e)
            await ws.send(json.dumps({
                "type": "opencode_http_response",
                "request_id": request_id,
                "status": 500,
                "body": {"error": f"{type(e).__name__}: {e}"},
            }))

    # ── OpenCode SSE Event Forwarding ─────────────────────────────────────

    async def _subscribe_opencode_events(self, ws: Any) -> None:
        """Subscribe to opencode SSE events and forward them via WebSocket.

        Runs indefinitely with auto-reconnect. Call as a background task.
        """
        import aiohttp

        while self._running and self._opencode_ready:
            try:
                session = await self._get_http_session()
                async with session.get(
                    f"{self._opencode_base_url}/event",
                    headers={"Accept": "text/event-stream"},
                    timeout=aiohttp.ClientTimeout(
                        total=None,  # SSE is long-lived
                        sock_read=60,  # Expect heartbeat every 30s
                    ),
                ) as resp:
                    if resp.status != 200:
                        log.warning("SSE endpoint returned %d", resp.status)
                        await asyncio.sleep(2)
                        continue

                    log.info("Connected to opencode SSE event stream")
                    buffer = ""
                    async for chunk in resp.content.iter_any():
                        if not self._running:
                            break
                        buffer += chunk.decode(errors="replace")
                        # Parse SSE: events separated by blank lines
                        while "\n\n" in buffer:
                            event_text, buffer = buffer.split("\n\n", 1)
                            await self._parse_and_forward_sse(ws, event_text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                log.warning("SSE connection lost: %s. Reconnecting in 2s...", e)
                await asyncio.sleep(2)

    async def _parse_and_forward_sse(self, ws: Any, event_text: str) -> None:
        """Parse a single SSE event block and forward it as opencode_event."""
        data_lines = []
        for line in event_text.split("\n"):
            if line.startswith("data: "):
                data_lines.append(line[6:])
            elif line.startswith("data:"):
                data_lines.append(line[5:])
            # Ignore comment lines (: keepalive) and other fields

        if not data_lines:
            return

        data_str = "\n".join(data_lines)
        try:
            event_data = json.loads(data_str)
        except json.JSONDecodeError:
            return

        # Skip heartbeats to reduce noise
        event_type = event_data.get("type", "")
        if event_type == "server.heartbeat":
            return

        props = event_data.get("properties", {})
        session_id = props.get("sessionID", "<none>")
        log.info("SSE event: %s (sessionID=%s)", event_type, session_id)

        try:
            await ws.send(json.dumps({
                "type": "opencode_event",
                "machine_id": self.machine_id,
                "event": event_data,
            }))
        except Exception as e:
            log.warning("Failed to forward SSE event: %s", e)

    # ── Legacy Tool Call Handling ─────────────────────────────────────────

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

    async def _handle_file_read(self, ws: Any, msg: dict) -> None:
        """Read a file as base64 and send it back."""
        request_id = msg.get("request_id", "?")
        file_path = msg.get("file_path", "")

        try:
            import base64
            import mimetypes

            path = Path(file_path)
            if not path.exists():
                await ws.send(json.dumps({
                    "type": "file_read_result",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"File not found: {file_path}",
                }))
                return

            size = path.stat().st_size
            if size > 10 * 1024 * 1024:
                await ws.send(json.dumps({
                    "type": "file_read_result",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"File too large: {size} bytes (limit 10MB)",
                }))
                return

            data = base64.b64encode(path.read_bytes()).decode()
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

            await ws.send(json.dumps({
                "type": "file_read_result",
                "request_id": request_id,
                "ok": True,
                "data": data,
                "size": size,
                "mime": mime,
            }))
            log.info("File read: %s (%d bytes, %s) [%s]", file_path, size, mime, request_id)

        except Exception as e:
            log.exception("File read failed: %s", file_path)
            await ws.send(json.dumps({
                "type": "file_read_result",
                "request_id": request_id,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }))

    # ── Main Connection Loop ──────────────────────────────────────────────

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
            self._current_ws = ws
            # Detect capabilities
            capabilities = self._detect_capabilities()
            hostname = socket.gethostname()

            # Register with server
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

            # Start opencode serve if available
            if "opencode" in capabilities:
                started = await self._start_opencode_serve()
                if started:
                    # Start SSE forwarding in background
                    self._sse_task = asyncio.create_task(
                        self._subscribe_opencode_events(ws)
                    )

            try:
                # Main message loop
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
                        asyncio.create_task(self._handle_tool_call(ws, msg))
                    elif msg_type == "opencode_http":
                        asyncio.create_task(self._handle_opencode_http(ws, msg))
                    elif msg_type == "file_read":
                        asyncio.create_task(self._handle_file_read(ws, msg))
                    elif msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                    else:
                        log.debug("Unknown message type: %s", msg_type)
            finally:
                self._current_ws = None
                await self._stop_opencode_serve()

    def stop(self) -> None:
        """Signal the worker to stop and force-close the WebSocket."""
        if not self._running:
            return  # Already stopping
        self._running = False
        log.info("Worker stopping...")
        # Force-close WebSocket to unblock the async message loop
        ws = self._current_ws
        if ws:
            try:
                # Schedule an async close on the event loop
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(asyncio.ensure_future, ws.close())
            except Exception:
                pass
            # Also try transport close as fallback
            try:
                if hasattr(ws, 'transport') and ws.transport:
                    asyncio.get_event_loop().call_soon_threadsafe(ws.transport.close)
            except Exception:
                pass
        # Cancel SSE task if running
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()


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
    parser.add_argument(
        "--opencode-port",
        type=int,
        default=4096,
        help="Port for opencode serve (default: 4096)",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    client = WorkerClient(
        server_url=args.server,
        machine_id=args.machine_id,
        workspace_dir=workspace,
        opencode_port=args.opencode_port,
    )

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
        # Clean up opencode serve before exiting
        try:
            loop.run_until_complete(client._stop_opencode_serve())
        except Exception:
            pass
        loop.close()

    log.info("Worker shut down.")


if __name__ == "__main__":
    main()
