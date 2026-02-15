"""Simple console demo for the support assistant.

- Single `Session` from `dialogue.state`.
- Commands: `/human` (escalate), `/bot` (hand back), `/quit` (exit).
- No real LLM calls — bot replies are simple echoes.

Run with: `python -m src.interfaces.console_app` or `python src/interfaces/console_app.py`
"""
from __future__ import annotations

import uuid
from typing import Optional

from src.dialogue.state import Session, SpeakerRole
from src.dialogue.llm_client import generate_bot_reply


PROMPT_CUSTOMER = "Customer> "
PROMPT_HUMAN = "HUMAN_AGENT> "




def _print_instructions() -> None:
    print("--- Support Assistant Console Demo ---")
    print("Type customer text and press Enter.")
    print("Commands:")
    print("  /human  -> escalate to human agent")
    print("  /bot    -> hand back to bot")
    print("  /quit   -> exit")
    print("--------------------------------------")


def main(session_id: Optional[str] = None) -> None:
    session_id = session_id or str(uuid.uuid4())
    session = Session(session_id=session_id)

    print(f"Starting session {session.session_id} (active_role={session.active_role.name})")
    _print_instructions()

    try:
        while True:
            try:
                text = input(PROMPT_CUSTOMER).strip()
            except EOFError:
                print("\nEOF received — exiting.")
                break

            if not text:
                continue

            # global commands
            if text == "/quit":
                print("Quitting...")
                break
            if text == "/human":
                session.mark_escalated()
                print("System> Escalating to HUMAN_AGENT...")
                continue
            if text == "/bot":
                session.hand_back_to_bot()
                print("System> Handing conversation back to BOT...")
                continue

            # regular customer input -> record it
            session.add_turn(SpeakerRole.CUSTOMER, text)

            # responder behavior depends on active_role
            if session.active_role == SpeakerRole.BOT:
                # build history from session turns
                history: list[dict] = []
                for t in session.turns:
                    if t.speaker == SpeakerRole.CUSTOMER:
                        history.append({"role": "user", "content": t.text})
                    else:
                        # BOT and HUMAN_AGENT are treated as assistant messages
                        history.append({"role": "assistant", "content": t.text})

                system_prompt = (
                    "You are a support assistant in a customer service call center. "
                    "Answer briefly and politely."
                )

                reply_text = generate_bot_reply(
                    model="phi3",
                    system_prompt=system_prompt,
                    history=history,
                    user_message=text,
                )

                print(f"BOT> {reply_text}")
                session.add_turn(SpeakerRole.BOT, reply_text)

            elif session.active_role == SpeakerRole.HUMAN_AGENT:
                # ask for manual human reply
                try:
                    human_reply = input(PROMPT_HUMAN).strip()
                except EOFError:
                    print("\nEOF received from human — returning to loop.")
                    continue

                # allow human to issue quick control commands while replying
                if human_reply == "/bot":
                    session.hand_back_to_bot()
                    print("[SYSTEM] Handed back to BOT. Active role=BOT")
                    continue
                if human_reply == "/quit":
                    print("Quitting...")
                    break

                session.add_turn(SpeakerRole.HUMAN_AGENT, human_reply)
                print(f"HUMAN_AGENT> {human_reply}")

            else:
                # unexpected active role (e.g., CUSTOMER) — just report it
                print(f"[SYSTEM] No responder configured for active role: {session.active_role}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt — exiting.")

    # final session dump
    print("\n--- Session Ended ---")
    print(f"session_id: {session.session_id}")
    print(f"turns: {len(session.turns)} | escalated: {session.escalated} | resolved: {session.resolved}")
    print("Recent context:\n")
    print(session.get_recent_context(50))


if __name__ == "__main__":
    main()
