"""Stuck detection for agent loops.

Detects 5 patterns of an agent being stuck in a loop, inspired by OpenHands' StuckDetector
but adapted for Overmind's message-based (dict) conversation format.

Patterns detected:
1. Repeating action: LLM calls the same tool with same args N times
2. Repeating observation: Same result from non-exploration tools N times
3. Repeating error: Same error output from non-exploration tools N times
4. Empty response: LLM returns empty content with no tool calls N times
5. Action-observation pair loop: Same (tool_call, result) pair alternates in a cycle
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass

log = logging.getLogger(__name__)

# How many repetitions trigger each pattern
DEFAULT_REPEAT_THRESHOLD = 3
# How many recent messages to scan (keeps cost low on long conversations)
DEFAULT_SCAN_WINDOW = 30

# Tools that represent read-only exploration — exempt from observation-based
# stuck detection (Patterns 2 & 3).  An agent reading many files or polling
# child status is researching, not stuck.
EXPLORATION_TOOLS = frozenset({
    "read_file", "list_directory", "search_files",
    "screenshot", "list_windows", "find_window",
    "get_child_result",
})

# Tool results that are effectively empty — many legitimate commands produce
# no output (Start-Process, sleep, mkdir, environment setup, etc.).  These
# should not count as "repeated observations" because the repetition is
# meaningless; real stuck loops repeat a *substantive* output or error.
EMPTY_RESULTS = frozenset({
    "(no output)", "", "(empty directory)",
})


@dataclass
class StuckResult:
    """Result of a stuck check."""
    is_stuck: bool
    pattern: str = ""       # e.g. "repeating_action", "empty_response"
    repeat_count: int = 0
    detail: str = ""


@dataclass
class _ActionInfo:
    """Internal: action hash paired with tool names in the batch."""
    hash: str
    tool_names: frozenset[str]


@dataclass
class _ObsInfo:
    """Internal: observation hash paired with the tool that produced it."""
    hash: str
    tool_name: str


class StuckDetector:
    """Detects when an agent loop is stuck in repetitive patterns.

    Works on Overmind's message format: list of dicts with
    role="assistant" (with optional tool_calls) and role="tool" (with content).
    """

    def __init__(
        self,
        repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD,
        scan_window: int = DEFAULT_SCAN_WINDOW,
    ):
        self.repeat_threshold = repeat_threshold
        self.scan_window = scan_window
        # Pre-compute hashes for empty results so we can skip them cheaply
        self._empty_hashes = frozenset(
            self._observation_hash(r) for r in EMPTY_RESULTS
        )

    def check(self, messages: list[dict]) -> StuckResult:
        """Check if the conversation shows signs of being stuck.

        Args:
            messages: The full message history from the agent loop.

        Returns:
            StuckResult with is_stuck=True and pattern details if stuck.
        """
        # Only look at recent messages
        window = messages[-self.scan_window:] if len(messages) > self.scan_window else messages

        # Extract structured data from messages
        actions = self._extract_actions(window)
        obs_infos = self._extract_observations_with_context(window)
        obs_hashes = [o.hash for o in obs_infos]
        assistant_msgs = self._extract_assistant_messages(window)

        # Pattern 1: Repeating action (same tool call repeated)
        result = self._check_repeating_actions(actions)
        if result.is_stuck:
            return result

        # Pattern 2: Repeating observation (same tool result repeated)
        # Only checks non-exploration tools to avoid false positives
        # when agents read many files or poll child status.
        result = self._check_repeating_observations(obs_infos)
        if result.is_stuck:
            return result

        # Pattern 3: Repeating error in tool results
        # Only checks non-exploration tools.
        result = self._check_repeating_errors(obs_infos)
        if result.is_stuck:
            return result

        # Pattern 4: Empty response loop
        result = self._check_empty_responses(assistant_msgs)
        if result.is_stuck:
            return result

        # Pattern 5: Action-observation pair cycle (A1,O1,A2,O2,A1,O1,A2,O2)
        result = self._check_pair_cycle(actions, obs_hashes)
        if result.is_stuck:
            return result

        return StuckResult(is_stuck=False)

    # --- Extractors ---

    @staticmethod
    def _action_hash(tool_calls: list[dict]) -> str:
        """Hash tool calls to a short fingerprint for comparison."""
        normalized = []
        for tc in sorted(tool_calls, key=lambda t: t.get("function", {}).get("name", "")):
            func = tc.get("function", {})
            normalized.append(f"{func.get('name', '')}:{func.get('arguments', '')}")
        return hashlib.md5("|".join(normalized).encode()).hexdigest()[:12]

    @staticmethod
    def _observation_hash(content: str) -> str:
        """Hash tool result content."""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_actions(self, messages: list[dict]) -> list[_ActionInfo]:
        """Extract action hashes with tool names from assistant messages."""
        actions = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tcs = msg["tool_calls"]
                names = frozenset(
                    tc.get("function", {}).get("name", "")
                    for tc in tcs
                )
                actions.append(_ActionInfo(
                    hash=self._action_hash(tcs),
                    tool_names=names,
                ))
        return actions

    def _extract_observations_with_context(self, messages: list[dict]) -> list[_ObsInfo]:
        """Extract observation hashes paired with the tool that produced them.

        Maps each tool result message back to the tool name via tool_call_id,
        so downstream checks can exclude exploration-only tools.
        """
        # Build tool_call_id → tool_name mapping from assistant messages
        tc_id_to_name: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("function", {}).get("name", "")
                    if tc_id:
                        tc_id_to_name[tc_id] = tc_name

        # Extract observations with tool context
        result: list[_ObsInfo] = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                tc_id = msg.get("tool_call_id", "")
                tool_name = tc_id_to_name.get(tc_id, "unknown")
                result.append(_ObsInfo(
                    hash=self._observation_hash(content),
                    tool_name=tool_name,
                ))
        return result

    @staticmethod
    def _extract_assistant_messages(messages: list[dict]) -> list[dict]:
        """Extract assistant messages."""
        return [m for m in messages if m.get("role") == "assistant"]

    # --- Pattern checks ---

    def _check_repeating_actions(self, actions: list[_ActionInfo]) -> StuckResult:
        """Pattern 1: Same tool call repeated N times in a row.

        Skips detection when all tools in the repeated batch are exploration
        tools (e.g. get_child_result polling), since repetition is expected.
        """
        if len(actions) < self.repeat_threshold:
            return StuckResult(is_stuck=False)

        tail = actions[-self.repeat_threshold:]
        if len(set(a.hash for a in tail)) == 1:
            # If every tool in the repeated batch is an exploration tool, skip
            all_exploration = all(
                a.tool_names.issubset(EXPLORATION_TOOLS) for a in tail
            )
            if all_exploration:
                return StuckResult(is_stuck=False)

            return StuckResult(
                is_stuck=True,
                pattern="repeating_action",
                repeat_count=self.repeat_threshold,
                detail=f"Same tool call repeated {self.repeat_threshold} times consecutively.",
            )
        return StuckResult(is_stuck=False)

    def _check_repeating_observations(self, obs_infos: list[_ObsInfo]) -> StuckResult:
        """Pattern 2: Same tool result repeated N times (excludes exploration tools).

        Exploration tools (read_file, list_directory, get_child_result, etc.)
        are filtered out because sequential file reads or status polling are
        normal research behaviour, not stuck loops.

        Empty results like "(no output)" are also excluded — many legitimate
        commands (Start-Process, sleep, mkdir) produce no output, and their
        repetition is not a sign of being stuck.
        """
        actionable = [o for o in obs_infos
                       if o.tool_name not in EXPLORATION_TOOLS
                       and o.hash not in self._empty_hashes]
        if len(actionable) < self.repeat_threshold:
            return StuckResult(is_stuck=False)

        tail = actionable[-self.repeat_threshold:]
        if len(set(o.hash for o in tail)) == 1:
            return StuckResult(
                is_stuck=True,
                pattern="repeating_observation",
                repeat_count=self.repeat_threshold,
                detail=f"Same tool result repeated {self.repeat_threshold} times consecutively (from action tools).",
            )
        return StuckResult(is_stuck=False)

    def _check_repeating_errors(self, obs_infos: list[_ObsInfo]) -> StuckResult:
        """Pattern 3: Same error hash repeated N times (not necessarily consecutive).

        Scans the last 10 non-exploration observations for any hash appearing
        >= threshold times.  This is a superset of the old Sisyphus detection.
        Empty results are excluded (same rationale as Pattern 2).
        """
        actionable = [o for o in obs_infos
                       if o.tool_name not in EXPLORATION_TOOLS
                       and o.hash not in self._empty_hashes]
        recent = actionable[-10:] if len(actionable) > 10 else actionable
        if len(recent) < self.repeat_threshold:
            return StuckResult(is_stuck=False)

        counts = Counter(o.hash for o in recent)
        for h, count in counts.most_common(1):
            if count >= self.repeat_threshold:
                return StuckResult(
                    is_stuck=True,
                    pattern="repeating_error",
                    repeat_count=count,
                    detail=f"Same tool output hash appeared {count} times in last {len(recent)} action results.",
                )
        return StuckResult(is_stuck=False)

    def _check_empty_responses(self, assistant_msgs: list[dict]) -> StuckResult:
        """Pattern 4: LLM returns empty content with no tool calls N times in a row."""
        if len(assistant_msgs) < self.repeat_threshold:
            return StuckResult(is_stuck=False)

        tail = assistant_msgs[-self.repeat_threshold:]
        all_empty = all(
            (not msg.get("content", "").strip()) and (not msg.get("tool_calls"))
            for msg in tail
        )
        if all_empty:
            return StuckResult(
                is_stuck=True,
                pattern="empty_response",
                repeat_count=self.repeat_threshold,
                detail=f"LLM returned {self.repeat_threshold} consecutive empty responses.",
            )
        return StuckResult(is_stuck=False)

    def _check_pair_cycle(self, actions: list[_ActionInfo], observations: list[str]) -> StuckResult:
        """Pattern 5: Alternating (action, observation) pairs cycle.

        Detects patterns like: (A1,O1), (A2,O2), (A1,O1), (A2,O2), (A1,O1), (A2,O2)
        where two distinct action-observation pairs alternate 3+ times.
        """
        # Need at least 6 pairs to detect a 2-step cycle repeated 3 times
        min_pairs = min(len(actions), len(observations))
        if min_pairs < 6:
            return StuckResult(is_stuck=False)

        # Build pairs from the tail (use action hash, not full _ActionInfo)
        action_hashes = [a.hash for a in actions[-6:]]
        pairs = list(zip(action_hashes, observations[-6:]))

        # Check if pairs[0]==pairs[2]==pairs[4] and pairs[1]==pairs[3]==pairs[5]
        even_same = pairs[0] == pairs[2] == pairs[4]
        odd_same = pairs[1] == pairs[3] == pairs[5]

        if even_same and odd_same and pairs[0] != pairs[1]:
            return StuckResult(
                is_stuck=True,
                pattern="pair_cycle",
                repeat_count=3,
                detail="Alternating action-observation pair cycle detected (2-step pattern repeated 3 times).",
            )
        return StuckResult(is_stuck=False)
