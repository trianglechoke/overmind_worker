"""OpenRouter LLM client with tool-use agentic loop support.

Features (inspired by OpenClaw architecture):
- Error classification & exponential backoff retry (429, 5xx, timeout)
- Context window protection & auto-compaction
- Agent cancel support via is_cancelled callback
- Message steering (inject messages mid-loop) via get_steered_messages callback
- Parallel tool execution for read-only tool calls
- StuckDetector: 5-pattern stuck loop detection (inspired by OpenHands)
- BudgetControl: per-agent cost ceiling
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import httpx

from overmind.utils.stuck_detector import StuckDetector

log = logging.getLogger(__name__)

# Tools that are safe to execute in parallel (read-only, no side effects)
READONLY_TOOLS = frozenset({
    "read_file", "list_directory", "search_files",
    "screenshot", "list_windows", "find_window",
    "get_child_result",
})

# Kimi K2.5 context window size (tokens)
MODEL_CONTEXT_WINDOW = 262_000
# Compact when messages exceed this fraction of context window
COMPACT_THRESHOLD = 0.75
# Reserve tokens for output
RESERVED_OUTPUT_TOKENS = 16_384


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultWithImage:
    """Tool result that includes an image (e.g. screenshot)."""
    text: str
    image_base64: str


@dataclass
class LLMResponse:
    content: str
    reasoning: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict[str, Any]
    raw: dict[str, Any]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class CostTracker:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0

    def record(self, usage: dict[str, Any]) -> None:
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.total_cost_usd += usage.get("cost", 0.0)
        self.call_count += 1

    @property
    def summary(self) -> str:
        return (
            f"Calls: {self.call_count} | "
            f"Tokens: {self.total_prompt_tokens}+{self.total_completion_tokens} | "
            f"Cost: ${self.total_cost_usd:.4f}"
        )


class LLMApiError(Exception):
    """Classified LLM API error."""

    def __init__(self, message: str, status_code: int = 0, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class AgentCancelled(Exception):
    """Raised when an agent is cancelled mid-execution."""
    pass


class AgentStuck(Exception):
    """Raised when the stuck detector finds the agent is looping."""
    def __init__(self, pattern: str, detail: str):
        self.pattern = pattern
        self.detail = detail
        super().__init__(f"Agent stuck ({pattern}): {detail}")


class AgentBudgetExceeded(Exception):
    """Raised when the agent exceeds its cost budget."""
    def __init__(self, spent: float, limit: float):
        self.spent = spent
        self.limit = limit
        super().__init__(f"Agent budget exceeded: ${spent:.4f} > ${limit:.4f}")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3 chars per token (mix of English and Chinese)."""
    return max(1, len(text) // 3)


def _estimate_messages_tokens(
    messages: list[dict[str, Any]], system: str | None = None,
) -> int:
    """Estimate total tokens in a message list."""
    total = 0
    if system:
        total += _estimate_tokens(system)
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            # Content blocks (e.g. image + text)
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += _estimate_tokens(block.get("text", ""))
                    elif block.get("type") == "image_url":
                        total += 1000  # rough estimate for image tokens
        # Tool calls in assistant messages
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            total += _estimate_tokens(func.get("name", ""))
            total += _estimate_tokens(func.get("arguments", ""))
    return total


class LLMClient:
    """Async OpenRouter API client with tool-use support."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "moonshotai/kimi-k2.5",
        timeout: float = 300.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cost_tracker = CostTracker()
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        max_retries: int = 3,
        on_chunk: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Make a chat completion call with optional streaming.

        Args:
            on_chunk: Async callback for each streaming chunk (type, content).
                      Types: 'content', 'reasoning', 'tool_call'
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": on_chunk is not None,
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}, *messages]
        if tools:
            payload["tools"] = tools

        if on_chunk:
            return await self._chat_stream(payload, on_chunk, max_retries)

        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    raise RuntimeError(f"LLM API error: {data['error']}")

                choice = data["choices"][0]
                message = choice["message"]
                usage = data.get("usage", {})
                self.cost_tracker.record(usage)

                tool_calls = []
                for tc in message.get("tool_calls", []):
                    func = tc["function"]
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=func["name"],
                        arguments=args,
                    ))

                return LLMResponse(
                    content=message.get("content", "") or "",
                    reasoning=message.get("reasoning", "") or "",
                    tool_calls=tool_calls,
                    finish_reason=choice.get("finish_reason", ""),
                    usage=usage,
                    raw=data,
                )

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                last_error = e

                if status == 429:
                    retry_after = float(
                        e.response.headers.get("retry-after", 2 ** attempt)
                    )
                    wait = min(retry_after, 60.0)
                    log.warning(
                        "Rate limited (429), retrying in %.1fs (attempt %d/%d)",
                        wait, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue

                elif status >= 500:
                    wait = min(2 ** attempt, 30.0)
                    log.warning(
                        "Server error (%d), retrying in %.1fs (attempt %d/%d)",
                        status, wait, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue

                elif status in (400, 401, 403):
                    raise LLMApiError(
                        f"API error {status}: {e.response.text[:500]}",
                        status_code=status,
                        retryable=False,
                    ) from e

                else:
                    raise LLMApiError(
                        f"HTTP {status}: {e.response.text[:500]}",
                        status_code=status,
                        retryable=False,
                    ) from e

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = min(2 ** attempt, 30.0)
                    log.warning(
                        "Connection error (%s), retrying in %.1fs (attempt %d/%d)",
                        type(e).__name__, wait, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise LLMApiError(
                    f"Connection failed after {max_retries + 1} attempts: {e}",
                    retryable=False,
                ) from e

        raise LLMApiError(
            f"All {max_retries + 1} attempts failed: {last_error}",
            retryable=False,
        )

    async def _chat_stream(
        self,
        payload: dict[str, Any],
        on_chunk: Callable[[str, str], Awaitable[None]],
        max_retries: int,
    ) -> LLMResponse:
        """Streaming chat completion with chunk callbacks."""
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                    resp.raise_for_status()

                    content_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    tool_calls_raw: dict[str, dict[str, Any]] = {}
                    finish_reason = ""
                    usage: dict[str, Any] = {}
                    stream_done = False

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            stream_done = True
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if "error" in data:
                            raise RuntimeError(f"LLM API error: {data['error']}")

                        choice = data.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason") or finish_reason
                        usage = data.get("usage", usage)

                        if "content" in delta and delta["content"]:
                            chunk = delta["content"]
                            content_parts.append(chunk)
                            await on_chunk("content", chunk)

                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            chunk = delta["reasoning_content"]
                            reasoning_parts.append(chunk)
                            await on_chunk("reasoning", chunk)

                        for tc in delta.get("tool_calls", []):
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_raw:
                                tool_calls_raw[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.get("id"):
                                tool_calls_raw[idx]["id"] = tc["id"]
                            func = tc.get("function", {})
                            if func.get("name"):
                                tool_calls_raw[idx]["name"] = func["name"]
                            if func.get("arguments"):
                                tool_calls_raw[idx]["arguments"] += func["arguments"]
                                await on_chunk("tool_call", f"{tool_calls_raw[idx]['name']}: {func['arguments'][-50:]}")

                    # Detect stream interruption: no [DONE], no finish_reason, and no content
                    if not stream_done and not finish_reason and not content_parts and not tool_calls_raw:
                        log.warning("Stream interrupted (no [DONE], no content, attempt %d/%d)", attempt + 1, max_retries + 1)
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)
                            continue  # retry
                        raise LLMApiError("Stream interrupted: no [DONE] received and empty response", retryable=False)

                    if usage:
                        self.cost_tracker.record(usage)

                    tool_calls = []
                    for idx in sorted(tool_calls_raw.keys()):
                        tc = tool_calls_raw[idx]
                        args = tc["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        tool_calls.append(ToolCall(
                            id=tc["id"] or f"tc_{idx}",
                            name=tc["name"],
                            arguments=args,
                        ))

                    return LLMResponse(
                        content="".join(content_parts),
                        reasoning="".join(reasoning_parts),
                        tool_calls=tool_calls,
                        finish_reason=finish_reason,
                        usage=usage,
                        raw={},
                    )

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                last_error = e
                if status in (429, 500, 502, 503) and attempt < max_retries:
                    wait = min(2 ** attempt, 30.0)
                    log.warning("Stream error %d, retrying in %.1fs", status, wait)
                    await asyncio.sleep(wait)
                    continue
                raise LLMApiError(f"Stream error {status}", status_code=status, retryable=False)

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise LLMApiError(f"Stream connection failed: {e}", retryable=False)

        raise LLMApiError(f"All streaming attempts failed: {last_error}", retryable=False)

    async def _compact_messages(
        self,
        messages: list[dict[str, Any]],
        system: str,
    ) -> list[dict[str, Any]]:
        """Compact old messages by summarizing them to free context space.

        Keeps the first user message and the last few messages intact,
        replaces everything in between with a summary.
        """
        if len(messages) <= 6:
            return messages  # Too few to compact

        # Keep first message (original task) and last 4 messages
        keep_head = 1
        keep_tail = 4
        to_summarize = messages[keep_head:-keep_tail]

        if not to_summarize:
            return messages

        # Build summary request
        summary_text_parts = []
        for msg in to_summarize:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if content:
                summary_text_parts.append(f"[{role}] {content[:500]}")
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                summary_text_parts.append(
                    f"[tool_call] {func.get('name', '?')}({func.get('arguments', '')[:100]})"
                )

        summary_input = "\n".join(summary_text_parts[-50:])  # limit input size

        try:
            summary_response = await self.chat(
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarize the following agent conversation history concisely. "
                        "Focus on: what was done, what files were changed, what errors occurred, "
                        "and what the current state is. Keep it under 500 words.\n\n"
                        f"{summary_input}"
                    ),
                }],
                max_tokens=2048,
                max_retries=1,
            )
            summary = summary_response.content
        except Exception as e:
            log.warning("Failed to generate compaction summary: %s", e)
            # Fallback: just truncate
            summary = f"[Previous {len(to_summarize)} messages compacted - summary unavailable]"

        # Build compacted message list
        compacted = []
        compacted.extend(messages[:keep_head])  # original task
        compacted.append({
            "role": "user",
            "content": (
                f"[Context compacted: {len(to_summarize)} messages summarized]\n\n"
                f"{summary}\n\n"
                "Continue from where you left off."
            ),
        })
        compacted.extend(messages[-keep_tail:])  # recent messages

        old_tokens = _estimate_messages_tokens(messages, system)
        new_tokens = _estimate_messages_tokens(compacted, system)
        log.info(
            "Context compacted: %d messages → %d messages, ~%d tokens → ~%d tokens",
            len(messages), len(compacted), old_tokens, new_tokens,
        )

        return compacted

    async def run_agent_loop(
        self,
        system: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[ToolCall], Awaitable["str | ToolResultWithImage"]],
        max_iterations: int = 80,
        on_llm_response: Callable[[str, str, int], Awaitable[None]] | None = None,
        on_llm_start: Callable[[], Awaitable[None]] | None = None,
        on_llm_chunk: Callable[[str, str], Awaitable[None]] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        is_completed: Callable[[], bool] | None = None,
        get_steered_messages: Callable[[], list[str]] | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
        max_cost_usd: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Run a complete agentic tool-use loop.

        Args:
            system: System prompt.
            user_prompt: Initial user message.
            tools: Tool definitions.
            tool_executor: Async function to execute tool calls.
            max_iterations: Maximum number of tool call iterations.
            on_llm_response: Callback after each LLM response (content, reasoning, tool_count).
            on_llm_start: Callback before each LLM call (shows thinking status).
            on_llm_chunk: Callback for each streaming chunk (type, content).
            is_cancelled: Callback to check if the agent has been cancelled.
            get_steered_messages: Callback to get injected messages from Boss mid-loop.
            initial_messages: If provided, use these messages instead of creating from user_prompt.
                Used for agent continuation (continue_run) to preserve prior context.
            max_cost_usd: Maximum cost budget for this agent loop. Raises AgentBudgetExceeded
                when exceeded. Default $5.

        Returns the full message history.
        """
        if initial_messages is not None:
            messages = initial_messages
        else:
            messages = [
                {"role": "user", "content": user_prompt},
            ]
        iteration = 0
        empty_retries = 0
        cost_at_start = self.cost_tracker.total_cost_usd
        stuck_detector = StuckDetector()
        compact_threshold_tokens = int(
            (MODEL_CONTEXT_WINDOW - RESERVED_OUTPUT_TOKENS) * COMPACT_THRESHOLD
        )

        while iteration < max_iterations:
            # --- Cancel check ---
            if is_cancelled and is_cancelled():
                log.info("Agent cancelled before LLM call (iteration %d)", iteration)
                raise AgentCancelled("Agent was cancelled by user")

            # --- Inject steered messages ---
            if get_steered_messages:
                steered = get_steered_messages()
                for msg_text in steered:
                    messages.append({
                        "role": "user",
                        "content": f"[Boss update] {msg_text}",
                    })
                    log.info("Injected steered message: %s", msg_text[:100])

            # --- Context window protection ---
            estimated_tokens = _estimate_messages_tokens(messages, system)
            if estimated_tokens > compact_threshold_tokens:
                log.warning(
                    "Context approaching limit (~%d tokens > %d threshold), compacting...",
                    estimated_tokens, compact_threshold_tokens,
                )
                messages = await self._compact_messages(messages, system)

            # --- LLM call ---
            if on_llm_start:
                await on_llm_start()
            response = await self.chat(
                messages=messages,
                system=system,
                tools=tools,
                max_tokens=16384,
                on_chunk=on_llm_chunk,
            )

            # Build assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response.content:
                assistant_msg["content"] = response.content
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
            if not assistant_msg.get("content") and not assistant_msg.get("tool_calls"):
                assistant_msg["content"] = ""
            messages.append(assistant_msg)

            # Notify caller about LLM response
            if on_llm_response:
                await on_llm_response(
                    response.content,
                    response.reasoning,
                    len(response.tool_calls),
                )

            if not response.has_tool_calls:
                # If the LLM returned completely empty (no content, no tool_calls)
                # and we're early in the loop, it might be a transient issue — retry once.
                is_empty = not response.content and not response.reasoning
                if is_empty and iteration <= 3 and empty_retries < 1:
                    log.warning(
                        "LLM returned empty response at iteration %d (no content, no tool_calls). "
                        "Retrying once.",
                        iteration,
                    )
                    # Remove the empty assistant message we just appended
                    messages.pop()
                    empty_retries += 1
                    continue  # retry the LLM call
                break

            # --- Execute tool calls ---
            tool_calls = response.tool_calls
            all_readonly = all(tc.name in READONLY_TOOLS for tc in tool_calls)

            if all_readonly and len(tool_calls) > 1:
                # Parallel execution for read-only tools
                results = await self._execute_tools_parallel(
                    tool_calls, tool_executor, is_cancelled,
                )
            else:
                # Sequential execution
                results = await self._execute_tools_sequential(
                    tool_calls, tool_executor, is_cancelled,
                )

            # Append results to messages
            for tc, result in results:
                iteration += 1
                if isinstance(result, ToolResultWithImage):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.text,
                    })
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here is the screenshot:"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{result.image_base64}"
                            }},
                        ],
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

            log.info(
                "Agent loop iteration %d/%d, last tool: %s",
                iteration, max_iterations,
                tool_calls[-1].name if tool_calls else "none",
            )

            # --- Early exit if task_complete was called ---
            if is_completed and is_completed():
                log.info("Agent task_complete detected, exiting loop at iteration %d", iteration)
                break

            # --- Stuck detection (5 patterns) ---
            stuck_result = stuck_detector.check(messages)
            if stuck_result.is_stuck:
                log.warning(
                    "StuckDetector: %s — %s", stuck_result.pattern, stuck_result.detail
                )
                raise AgentStuck(stuck_result.pattern, stuck_result.detail)

            # --- Budget control ---
            spent = self.cost_tracker.total_cost_usd - cost_at_start
            if spent > max_cost_usd:
                log.warning(
                    "Agent budget exceeded: $%.4f > $%.4f limit", spent, max_cost_usd
                )
                raise AgentBudgetExceeded(spent, max_cost_usd)

        return messages

    async def _execute_tools_sequential(
        self,
        tool_calls: list[ToolCall],
        tool_executor: Callable[[ToolCall], Awaitable["str | ToolResultWithImage"]],
        is_cancelled: Callable[[], bool] | None = None,
    ) -> list[tuple[ToolCall, "str | ToolResultWithImage"]]:
        """Execute tool calls one by one."""
        results: list[tuple[ToolCall, str | ToolResultWithImage]] = []
        for tc in tool_calls:
            if is_cancelled and is_cancelled():
                results.append((tc, "Agent cancelled by user."))
                break
            try:
                result = await tool_executor(tc)
            except Exception as e:
                result = f"Error executing {tc.name}: {e}"
                log.error("Tool execution error: %s(%s) -> %s", tc.name, tc.arguments, e)
            results.append((tc, result))
        return results

    async def _execute_tools_parallel(
        self,
        tool_calls: list[ToolCall],
        tool_executor: Callable[[ToolCall], Awaitable["str | ToolResultWithImage"]],
        is_cancelled: Callable[[], bool] | None = None,
    ) -> list[tuple[ToolCall, "str | ToolResultWithImage"]]:
        """Execute read-only tool calls in parallel using asyncio.gather."""
        if is_cancelled and is_cancelled():
            return [(tc, "Agent cancelled by user.") for tc in tool_calls]

        log.info("Executing %d read-only tool calls in parallel", len(tool_calls))

        async def _safe_execute(tc: ToolCall) -> tuple[ToolCall, str | ToolResultWithImage]:
            try:
                result = await tool_executor(tc)
            except Exception as e:
                result = f"Error executing {tc.name}: {e}"
                log.error("Tool execution error: %s(%s) -> %s", tc.name, tc.arguments, e)
            return (tc, result)

        gathered = await asyncio.gather(
            *[_safe_execute(tc) for tc in tool_calls],
            return_exceptions=False,
        )
        return list(gathered)

    async def close(self) -> None:
        await self._client.aclose()
