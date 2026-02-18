"""Tool definitions and executor for sub-agents."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from overmind.utils.llm_client import ToolCall, ToolResultWithImage

log = logging.getLogger(__name__)

# Tool output truncation limits (matches OpenCode defaults)
MAX_OUTPUT_LINES = 2000
MAX_OUTPUT_BYTES = 50_000
# Tools whose output should NOT be truncated
_NO_TRUNCATE_TOOLS = {"send_to_boss"}

# OpenRouter / OpenAI-compatible tool schema
CODER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the file content as text. For large files, use offset and limit to read specific line ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or workspace-relative file path"},
                    "offset": {"type": "integer", "description": "Line number to start reading from (1-based, default: 1)"},
                    "limit": {"type": "integer", "description": "Maximum number of lines to return (default: all)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific string in a file. The old_string must appear exactly once in the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "Exact string to find and replace"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a regex pattern in file contents. Returns matching lines with file paths and line numbers. Use this to quickly find relevant code instead of manually browsing with list_directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in"},
                    "glob": {"type": "string", "description": "File glob filter, e.g. '*.py', '*.rs'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash/shell command. Use for git, build tools, tests, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_boss",
            "description": (
                "Send a message (and optionally an image) to the Boss in the chat. "
                "Use this to share screenshots, progress updates, or any content with the Boss. "
                "If you took a screenshot with the screenshot tool and the Boss asked to see it, "
                "use this tool with the image_path pointing to the saved screenshot file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Text message to send to the Boss",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Optional path to an image file to attach (PNG/JPG)",
                    },
                    "take_screenshot": {
                        "type": "boolean",
                        "description": "If true, takes a fresh screenshot and attaches it to the message",
                    },
                },
                "required": ["message"],
            },
        },
    },
]


GUI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "screenshot",
            "description": "Take a screenshot of the screen. Returns the image so you can see what's on screen.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_click",
            "description": "Click at screen coordinates (x, y).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button (default: left)"},
                    "clicks": {"type": "integer", "description": "Number of clicks (default: 1)"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_move",
            "description": "Move the mouse to screen coordinates (x, y) without clicking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyboard_type",
            "description": "Type text using the keyboard. For special keys or shortcuts, use key_press instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "key_press",
            "description": "Press a key or key combination. Examples: 'enter', 'tab', 'ctrl+c', 'ctrl+shift+s', 'alt+f4'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {"type": "string", "description": "Key or combination, e.g. 'ctrl+c', 'enter', 'alt+tab'"},
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_window",
            "description": "Find a window by title. Returns position, size, and whether it's active.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Window title or substring to search for"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activate_window",
            "description": "Bring a window to the foreground by title. Use before screenshot to ensure the right window is visible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Window title or substring to search for"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_windows",
            "description": "List all visible windows on the screen. Useful to see what applications are open.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# Combined tool set for OPERATOR agents
OPERATOR_TOOLS: list[dict[str, Any]] = CODER_TOOLS + GUI_TOOLS

# Read-only tool set for EXPLORER agents (no write_file, no edit_file)
_EXPLORER_BASH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a read-only shell command. Use for git log, git diff, git blame, "
            "find, type/cat, tree, etc. Do NOT run commands that modify files or state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute (read-only)"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
            },
            "required": ["command"],
        },
    },
}

# Pick read-only tools from CODER_TOOLS by name
_EXPLORER_TOOL_NAMES = {"read_file", "list_directory", "search_files", "send_to_boss"}
EXPLORER_TOOLS: list[dict[str, Any]] = [
    t for t in CODER_TOOLS if t["function"]["name"] in _EXPLORER_TOOL_NAMES
] + [_EXPLORER_BASH]

# Tools for L1 agents to spawn child (L2) agents
CHILD_AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "spawn_child_agent",
            "description": (
                "Spawn a child sub-agent to handle a sub-task in the background. "
                "Returns immediately with the child agent's ID. "
                "Use get_child_result to check if the child has finished and retrieve its result. "
                "Child agents are useful for: research, reading docs, searching code, "
                "running tests in parallel, reviewing code, etc. "
                "Don't spawn children for trivial tasks you can do yourself quickly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_prompt": {
                        "type": "string",
                        "description": "Detailed task description for the child agent",
                    },
                    "agent_type": {
                        "type": "string",
                        "enum": ["CODER", "TESTER", "REVIEWER", "EXPLORER"],
                        "description": "Type of child agent (OPERATOR not allowed for children). Use EXPLORER for read-only codebase search and analysis.",
                    },
                },
                "required": ["task_prompt", "agent_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_child_result",
            "description": (
                "Check if a child agent has finished and get its result. "
                "Returns the result summary if completed, or a 'still running' message. "
                "Call this after spawning a child agent to retrieve its findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The child agent ID returned by spawn_child_agent",
                    },
                },
                "required": ["agent_id"],
            },
        },
    },
]


def get_agent_tools(agent_type_str: str, nesting_level: int = 0) -> list[dict[str, Any]]:
    """Get the tool set for an agent based on its type and nesting level.

    L1 agents (nesting_level=0) get base tools + CHILD_AGENT_TOOLS.
    L2 agents (nesting_level>=1) get base tools only (no further nesting).
    """
    if agent_type_str == "OPERATOR":
        base = OPERATOR_TOOLS
    elif agent_type_str == "EXPLORER":
        base = EXPLORER_TOOLS
    else:
        base = CODER_TOOLS

    if nesting_level == 0:
        return base + CHILD_AGENT_TOOLS
    return base


class ToolExecutor:
    """Executes tool calls within a workspace directory."""

    def __init__(
        self,
        workspace_dir: Path,
        send_to_boss: Any | None = None,
    ):
        self.workspace_dir = workspace_dir.resolve()
        self.files_changed: list[str] = []
        self._send_to_boss = send_to_boss  # async callback: (text, image_data?) -> None

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_dir / p
        return p.resolve()

    def _truncate_output(self, output: str, tool_name: str) -> str:
        """Truncate large tool output, saving full version to file."""
        if tool_name in _NO_TRUNCATE_TOOLS:
            return output
        lines = output.splitlines()
        total_bytes = len(output.encode("utf-8", errors="replace"))
        if len(lines) <= MAX_OUTPUT_LINES and total_bytes <= MAX_OUTPUT_BYTES:
            return output

        # Save full output to file
        output_dir = self.workspace_dir / ".overmind" / "tool-output"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tool_{int(time.time())}_{tool_name}.txt"
        output_path = output_dir / filename
        output_path.write_text(output, encoding="utf-8", errors="replace")

        # Truncate: keep first MAX_OUTPUT_LINES lines within byte budget
        kept = []
        byte_count = 0
        for line in lines[:MAX_OUTPUT_LINES]:
            line_bytes = len(line.encode("utf-8", errors="replace")) + 1
            if byte_count + line_bytes > MAX_OUTPUT_BYTES:
                break
            kept.append(line)
            byte_count += line_bytes

        removed = len(lines) - len(kept)
        truncated = "\n".join(kept)
        truncated += (
            f"\n\n...{removed} lines truncated...\n"
            f"Full output saved to: {output_path}\n"
            "Use search_files to search the full content, "
            "or read_file with offset/limit to view specific sections."
        )

        # Cleanup old files (>7 days)
        try:
            cutoff = time.time() - 7 * 86400
            for old in output_dir.iterdir():
                if old.stat().st_mtime < cutoff:
                    old.unlink()
        except Exception:
            pass

        return truncated

    async def execute(self, tool_call: ToolCall) -> str | ToolResultWithImage:
        # Check for malformed JSON args from LLM
        if "_parse_error" in tool_call.arguments:
            return (
                f"Error: {tool_call.arguments['_parse_error']}\n"
                "Please retry the tool call with valid JSON arguments."
            )

        # Try exact match first, then case-insensitive repair
        name = tool_call.name
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            # Case repair: Read_file → read_file, Bash → bash
            lower = name.lower()
            handler = getattr(self, f"_tool_{lower}", None)
            if handler is not None:
                log.warning("Repaired tool name: %s → %s", name, lower)
                name = lower
            else:
                available = [
                    a[6:] for a in dir(self)
                    if a.startswith("_tool_") and callable(getattr(self, a))
                ]
                return (
                    f"Tool '{name}' does not exist. "
                    f"Available tools: {', '.join(sorted(available))}. "
                    "Please retry with the correct tool name."
                )
        try:
            result = await handler(**tool_call.arguments)
            # Apply truncation to string results
            if isinstance(result, str):
                result = self._truncate_output(result, name)
            return result
        except TypeError as e:
            # Handle unexpected keyword arguments gracefully
            return (
                f"Error calling {name}: {e}\n"
                "Please check the argument names and try again."
            )
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    async def _tool_read_file(self, path: str, offset: int = 1, limit: int = 0) -> str:
        p = self._resolve_path(path)
        if not p.exists():
            return f"File not found: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset/limit if specified
        start = max(0, offset - 1)  # Convert 1-based to 0-based
        if limit > 0:
            end = min(start + limit, total_lines)
        else:
            end = total_lines
        lines = lines[start:end]

        # Format with line numbers for context
        result_lines = []
        for i, line in enumerate(lines, start + 1):
            result_lines.append(f"{i}: {line}")
        result = "\n".join(result_lines)

        # Truncation safety net
        if len(result) > 100_000:
            result = result[:100_000] + "\n... (truncated)"

        # Add file info header
        if start > 0 or (limit > 0 and end < total_lines):
            result = f"[{path} lines {start + 1}-{end} of {total_lines}]\n{result}"

        return result

    async def _tool_write_file(self, path: str, content: str) -> str:
        p = self._resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.files_changed.append(str(p.relative_to(self.workspace_dir)))
        return f"Written {len(content)} bytes to {path}"

    async def _tool_edit_file(self, path: str, old_string: str, new_string: str) -> str:
        p = self._resolve_path(path)
        if not p.exists():
            return f"File not found: {path}"
        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return f"old_string not found in {path}"
        if count > 1:
            return f"old_string found {count} times in {path}, must be unique. Provide more context."
        content = content.replace(old_string, new_string, 1)
        p.write_text(content, encoding="utf-8")
        self.files_changed.append(str(p.relative_to(self.workspace_dir)))
        return f"Edited {path}: replaced 1 occurrence"

    async def _tool_list_directory(self, path: str = ".") -> str:
        p = self._resolve_path(path)
        if not p.exists():
            return f"Directory not found: {path}"
        if not p.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines = []
        for entry in entries[:200]:
            prefix = "d " if entry.is_dir() else "f "
            size = ""
            if entry.is_file():
                size = f" ({entry.stat().st_size} bytes)"
            lines.append(f"{prefix}{entry.name}{size}")
        return "\n".join(lines) if lines else "(empty directory)"

    async def _tool_search_files(
        self, pattern: str, path: str = ".", glob: str = ""
    ) -> str:
        """Cross-platform file content search using Python (no grep dependency)."""
        import fnmatch
        import re

        p = self._resolve_path(path)
        if not p.exists():
            return f"Path not found: {path}"

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # Skip binary / large / hidden directories
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                     "target", "dist", "build", ".next", ".nuxt", "vendor"}
        matches: list[str] = []
        max_matches = 200
        files_searched = 0

        def should_search(fp: Path) -> bool:
            if glob:
                return fnmatch.fnmatch(fp.name, glob)
            # Default: skip binary extensions
            binary_exts = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff",
                           ".woff2", ".ttf", ".eot", ".exe", ".dll", ".so",
                           ".dylib", ".zip", ".tar", ".gz", ".lock", ".pdf"}
            return fp.suffix.lower() not in binary_exts

        if p.is_file():
            files_to_search = [p]
        else:
            files_to_search = []
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                for f in files:
                    fp = Path(root) / f
                    if should_search(fp):
                        files_to_search.append(fp)

        for fp in files_to_search:
            if len(matches) >= max_matches:
                break
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel = fp.relative_to(p) if p.is_dir() else fp.name
                        matches.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                        if len(matches) >= max_matches:
                            break
                files_searched += 1
            except (OSError, PermissionError):
                continue

        if not matches:
            return f"No matches found for '{pattern}' in {files_searched} files"
        result = "\n".join(matches)
        if len(matches) >= max_matches:
            result += f"\n... (truncated at {max_matches} matches)"
        return result

    async def _tool_bash(self, command: str, timeout: int = 120) -> str:
        return await self._run_command(command, timeout=timeout)

    async def _tool_send_to_boss(
        self, message: str, image_path: str = "", take_screenshot: bool = False,
    ) -> str:
        """Send a message (and optional image) to the Boss via chat."""
        if not self._send_to_boss:
            return "send_to_boss is not available (no callback configured)."

        image_data = ""
        if take_screenshot:
            # Take a fresh screenshot and attach it
            try:
                result = await self._tool_screenshot()
                image_data = result.image_base64
            except Exception as e:
                return f"Failed to take screenshot: {e}"
        elif image_path:
            p = self._resolve_path(image_path)
            if p.exists():
                import base64
                image_data = base64.b64encode(p.read_bytes()).decode()
            else:
                return f"Image file not found: {image_path}"

        try:
            await self._send_to_boss(message, image_data)
            result_msg = f"Message sent to Boss: {message[:100]}"
            if image_data:
                result_msg += " (with image attached)"
            return result_msg
        except Exception as e:
            return f"Failed to send to Boss: {e}"

    # --- GUI tools ---

    async def _tool_screenshot(self) -> ToolResultWithImage:
        import base64
        import io
        import pyautogui
        img = pyautogui.screenshot()
        # Resize to control token cost
        max_w = 1280
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return ToolResultWithImage(
            text=f"Screenshot captured ({img.width}x{img.height})",
            image_base64=b64,
        )

    async def _tool_mouse_click(
        self, x: int, y: int, button: str = "left", clicks: int = 1
    ) -> str:
        import pyautogui
        pyautogui.click(x, y, clicks=clicks, button=button)
        return f"Clicked ({x}, {y}) button={button} clicks={clicks}"

    async def _tool_mouse_move(self, x: int, y: int) -> str:
        import pyautogui
        pyautogui.moveTo(x, y)
        return f"Mouse moved to ({x}, {y})"

    async def _tool_keyboard_type(self, text: str) -> str:
        import pyautogui
        pyautogui.write(text, interval=0.02)
        return f"Typed {len(text)} characters"

    async def _tool_key_press(self, keys: str) -> str:
        import pyautogui
        parts = [k.strip() for k in keys.split("+")]
        if len(parts) == 1:
            pyautogui.press(parts[0])
        else:
            pyautogui.hotkey(*parts)
        return f"Pressed {keys}"

    async def _tool_find_window(self, title: str) -> str:
        import pygetwindow as gw
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"No window found matching '{title}'"
        lines = []
        for w in windows[:10]:
            lines.append(
                f"Title: {w.title}\n"
                f"  Position: ({w.left}, {w.top})\n"
                f"  Size: {w.width}x{w.height}\n"
                f"  Active: {w.isActive}\n"
                f"  Visible: {w.visible}"
            )
        return "\n".join(lines)

    async def _tool_activate_window(self, title: str) -> str:
        import pygetwindow as gw
        import time
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"No window found matching '{title}'"
        w = windows[0]
        try:
            if w.isMinimized:
                w.restore()
            w.activate()
            time.sleep(0.5)  # Wait for window to come to foreground
            return f"Activated window: {w.title} ({w.width}x{w.height})"
        except Exception as e:
            return f"Found window '{w.title}' but failed to activate: {e}"

    async def _tool_list_windows(self) -> str:
        import pygetwindow as gw
        windows = gw.getAllWindows()
        lines = []
        for w in windows:
            if w.title and w.visible:
                active = " [ACTIVE]" if w.isActive else ""
                lines.append(f"{w.title}{active} ({w.width}x{w.height})")
        return "\n".join(lines) if lines else "No visible windows found"

    # --- Shell execution ---

    async def _run_command(self, command: str, timeout: int = 120) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_dir),
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            output = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            result = ""
            if output:
                result += output
            if err:
                result += f"\nSTDERR:\n{err}"
            if proc.returncode != 0:
                result += f"\n(exit code: {proc.returncode})"
            if len(result) > 50_000:
                result = result[:50_000] + "\n... (truncated)"
            return result.strip() or "(no output)"
        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s"
