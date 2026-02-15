"""Conversation state for the support assistant.

Provides:
- SpeakerRole enum (CUSTOMER, BOT, HUMAN_AGENT)
- Turn dataclass (speaker, text, timestamp, meta)
- Session dataclass with methods to append turns and manage escalation

Designed for Python 3.11+ and easy extension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SpeakerRole(Enum):
    """Role of a message speaker in a session."""

    CUSTOMER = "customer"
    BOT = "bot"
    HUMAN_AGENT = "human_agent"

    def __str__(self) -> str:  # helpful for logging/printing
        return self.value


@dataclass
class Turn:
    """One turn in the conversation.

    Attributes:
        speaker: Which role produced this turn.
        text: The spoken / written content.
        timestamp: Time the turn was created (UTC, tz-aware).
        meta: Arbitrary metadata (intent, confidence, source, etc.).
    """

    speaker: SpeakerRole
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    meta: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.timestamp.isoformat()} {self.speaker.name}: {self.text}"


@dataclass
class Session:
    """Conversation session holding turns and state.

    Minimal, extendable model for managing handoffs and summaries.
    """

    session_id: str
    turns: List[Turn] = field(default_factory=list)
    active_role: SpeakerRole = SpeakerRole.BOT
    issue_summary: Optional[str] = None
    escalated: bool = False
    resolved: bool = False

    def add_turn(self, speaker: SpeakerRole, text: str, meta: Optional[Dict[str, Any]] = None) -> Turn:
        """Append a new Turn and return it.

        The method will append the turn and update `active_role` in a
        sensible, minimal way (customer -> bot, bot/agent -> customer).
        Escalation overrides automatic advancement (if escalated, the
        `active_role` remains `HUMAN_AGENT`).
        """
        if not isinstance(speaker, SpeakerRole):
            raise TypeError("speaker must be an instance of SpeakerRole")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        turn = Turn(speaker=speaker, text=text, meta=meta or {})
        self.turns.append(turn)

        # sensible auto-advance of active_role (doesn't change `escalated`)
        if self.escalated:
            # when escalated, human agent remains the active responder
            self.active_role = SpeakerRole.HUMAN_AGENT
        else:
            if speaker == SpeakerRole.CUSTOMER:
                self.active_role = SpeakerRole.BOT
            else:
                # BOT or HUMAN_AGENT just spoke -> expect customer next
                self.active_role = SpeakerRole.CUSTOMER

        return turn

    def get_recent_context(self, n: int = 10) -> str:
        """Return the last `n` turns as a single string (oldest->newest).

        Each turn is rendered as: "<iso-timestamp> <SPEAKER>: <text>" on its own line.
        """
        recent = self.turns[-max(0, n) :]
        return "\n".join(t.__str__() for t in recent)

    def update_issue_summary(self, summary: str) -> None:
        """Store a short, user-provided issue summary."""
        if summary is None:
            raise TypeError("summary must be a string")
        self.issue_summary = summary.strip()

    def mark_escalated(self) -> None:
        """Mark session as escalated and make a human agent the active role."""
        self.escalated = True
        self.active_role = SpeakerRole.HUMAN_AGENT

    def hand_back_to_bot(self) -> None:
        """Return handling to the bot (clear escalation)."""
        self.escalated = False
        self.active_role = SpeakerRole.BOT


__all__ = ["SpeakerRole", "Turn", "Session"]
