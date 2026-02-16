"""Stuck detection for agent loops.

Detects 5 patterns of an agent being stuck in a loop, inspired by OpenHands' StuckDetector
but adapted for Overmind's message-based (dict) conversation format.

Patterns detected:
1. Repeating action: LLM calls the same tool with same args N times
2. Repeating observation: Tool returns the same result N times but LLM doesn't adjust
3. Repeating error: Same error output N times (replaces old Sisyphus MD5 detection)
4. Empty response: LLM returns empty content with no tool calls N times
5. Action-observation pair loop: Same (tool_call, result) pair alternates in a cycle
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# How many repetitions trigger each pattern
DEFAULT_REPEAT_THRESHOLD = 3
# How many recent messages to scan (keeps cost low on long conversations)
DEFAULT_SCAN_WINDOW = 30


@dataclass
class StuckResult:
    """Result of a stuck check."""
    is_stuck: bool
    pattern: str = ""       # e.g. "repeating_action", "empty_response"
    repeat_count: int = 0
    detail: str = ""


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
        observations = self._extract_observations(window)
        assistant_msgs = self._extract_assistant_messages(window)

        # Pattern 1: Repeating action (same tool call repeated)
        result = self._check_repeating_actions(actions)
        if result.is_stuck:
            return result

        # Pattern 2: Repeating observation (same tool result repeated)
        result = self._check_repeating_observations(observations)
        if result.is_stuck:
            return result

        # Pattern 3: Repeating error in tool results
        result = self._check_repeating_errors(observations)
        if result.is_stuck:
            return result

        # Pattern 4: Empty response loop
        result = self._check_empty_responses(assistant_msgs)
        if result.is_stuck:
            return result

        # Pattern 5: Action-observation pair cycle (A1,O1,A2,O2,A1,O1,A2,O2)
        result = self._check_pair_cycle(actions, observations)
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

    def _extract_actions(self, messages: list[dict]) -> list[str]:
        """Extract action hashes from assistant messages that have tool_calls."""
        hashes = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                hashes.append(self._action_hash(msg["tool_calls"]))
        return hashes

    def _extract_observations(self, messages: list[dict]) -> list[str]:
        """Extract observation hashes from tool result messages."""
        hashes = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                hashes.append(self._observation_hash(content))
        return hashes

    @staticmethod
    def _extract_assistant_messages(messages: list[dict]) -> list[dict]:
        """Extract assistant messages."""
        return [m for m in messages if m.get("role") == "assistant"]

    # --- Pattern checks ---

    def _check_repeating_actions(self, actions: list[str]) -> StuckResult:
        """Pattern 1: Same tool call repeated N times in a row."""
        if len(actions) < self.repeat_threshold:
            return StuckResult(is_stuck=False)

        tail = actions[-self.repeat_threshold:]
        if len(set(tail)) == 1:
            return StuckResult(
                is_stuck=True,
                pattern="repeating_action",
                repeat_count=self.repeat_threshold,
                detail=f"Same tool call repeated {self.repeat_threshold} times consecutively.",
            )
        return StuckResult(is_stuck=False)

    def _check_repeating_observations(self, observations: list[str]) -> StuckResult:
        """Pattern 2: Same tool result repeated N times."""
        if len(observations) < self.repeat_threshold:
            return StuckResult(is_stuck=False)

        tail = observations[-self.repeat_threshold:]
        if len(set(tail)) == 1:
            return StuckResult(
                is_stuck=True,
                pattern="repeating_observation",
                repeat_count=self.repeat_threshold,
                detail=f"Same tool result repeated {self.repeat_threshold} times consecutively.",
            )
        return StuckResult(is_stuck=False)

    def _check_repeating_errors(self, observations: list[str]) -> StuckResult:
        """Pattern 3: Same error hash repeated N times (not necessarily consecutive).

        Scans the last 10 observations for any hash appearing >= threshold times.
        This is a superset of the old Sisyphus detection.
        """
        recent = observations[-10:] if len(observations) > 10 else observations
        if len(recent) < self.repeat_threshold:
            return StuckResult(is_stuck=False)

        # Count occurrences
        from collections import Counter
        counts = Counter(recent)
        for h, count in counts.most_common(1):
            if count >= self.repeat_threshold:
                return StuckResult(
                    is_stuck=True,
                    pattern="repeating_error",
                    repeat_count=count,
                    detail=f"Same tool output hash appeared {count} times in last {len(recent)} results.",
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

    def _check_pair_cycle(self, actions: list[str], observations: list[str]) -> StuckResult:
        """Pattern 5: Alternating (action, observation) pairs cycle.

        Detects patterns like: (A1,O1), (A2,O2), (A1,O1), (A2,O2), (A1,O1), (A2,O2)
        where two distinct action-observation pairs alternate 3+ times.
        """
        # Need at least 6 pairs to detect a 2-step cycle repeated 3 times
        min_pairs = min(len(actions), len(observations))
        if min_pairs < 6:
            return StuckResult(is_stuck=False)

        # Build pairs from the tail
        pairs = list(zip(actions[-6:], observations[-6:]))

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
