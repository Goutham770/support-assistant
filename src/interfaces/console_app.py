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
from src.rag.simple_faq_rag import get_rag_context


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
    print("--- Support Assistant Console Demo ---")
    print("Type customer text and press Enter.")
    print("Commands:")
    print("  /human  -> escalate to human agent")
    print("  /bot    -> hand back to bot")
    print("  /quit   -> exit")
    print("--------------------------------------")

    system_prompt = (
        "You are a polite, concise customer support assistant in a call center. "
        "Answer briefly and clearly."
    )

    try:
        while True:
            try:
                raw = input(PROMPT_CUSTOMER).strip()
            except EOFError:
                print("\nEOF received — exiting.")
                break

            if not raw:
                continue

            # 1) Handle commands BEFORE calling the LLM
            if raw == "/human":
                session.active_role = SpeakerRole.HUMAN_AGENT
                print("System> Connecting you to a human agent (ringing...)")
                print("System> Human agent joined the conversation.")
                continue

            if raw == "/bot":
                session.active_role = SpeakerRole.BOT
                print("System> Handing the conversation back to the bot.")
                continue

            if raw == "/quit":
                print("System> Goodbye.")
                break

            # 2) Normal customer message
            customer_text = raw
            session.add_turn(SpeakerRole.CUSTOMER, customer_text)

            if session.active_role == SpeakerRole.BOT:
                # Build history for the LLM
                history: list[dict] = []
                for turn in session.turns:
                    if turn.speaker == SpeakerRole.CUSTOMER:
                        history.append({"role": "user", "content": turn.text})
                    else:
                        history.append({"role": "assistant", "content": turn.text})

                # Retrieve short RAG context and augment the user message
                rag_context = get_rag_context(customer_text)
                augmented_user_message = f"""
                Docs context:

                {rag_context}

                Customer question:
                {customer_text}
                """.strip()

                reply_text = generate_bot_reply(
                    model="phi3",
                    system_prompt=system_prompt,
                    history=history,
                    user_message=augmented_user_message,
                )
                print(f"BOT> {reply_text}")
                session.add_turn(SpeakerRole.BOT, reply_text)

            elif session.active_role == SpeakerRole.HUMAN_AGENT:
                human_text = input(PROMPT_HUMAN).strip()
                session.add_turn(SpeakerRole.HUMAN_AGENT, human_text)
                print(f"HUMAN_AGENT> {human_text}")

            else:
                print(f"System> No responder configured for active role: {session.active_role}")

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
