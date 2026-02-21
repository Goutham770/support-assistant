"""High-level I/O loop bridging audio modules to the dialogue Session.

This demo module simulates audio input by asking the developer to type a
fake ASR transcription. That text is fed into the same dialogue logic used
by the console app: a `Session` is maintained, history built for the LLM,
and replies are sent to the `speak()` TTS stub.

This is intentionally lightweight and synchronous for testing and demo
purposes; real audio capture / streaming and ASR integration will be
plugged in later.
"""

from __future__ import annotations

import uuid
from typing import Optional

from src.dialogue.state import Session, SpeakerRole
from src.dialogue.ollama_client import generate_with_ollama
from src.rag.simple_faq_rag import get_rag_context
from src.audio.asr import transcribe_chunk
from src.audio.tts import speak


# Debug flag: set to True to log RAG context retrieval during development
DEBUG_RAG = True

# Example ASR inputs to test RAG and cover all FAQ sections:
# - I want to change my mobile plan, what should I say?
# - A customer wants to cancel their broadband, what should I say?
# - The customer has a billing dispute, how do I handle it?
# - The customer paid their bill late, what should I say?
# - The customer wants to upgrade their mobile plan, what should I say?
# - The customer lost their phone, what are the steps?
# - The customer's SIM card is not working, what should I do?

PROMPT_SIMULATED_ASR = "ASR text (or /quit): "
PROMPT_HUMAN = "HUMAN_AGENT> "


def demo_turn_based_audio_session() -> None:
    """Run a simple turn-based audio session demo.

    Behavior:
      - Prompt the developer to type a fake ASR transcription line.
      - If the line is `/quit`, exit the demo.
      - Otherwise treat the typed line as the customer's spoken text,
        add it to a `Session`, build history, call `generate_bot_reply`,
        and send the resulting reply to `speak()`.

    The function reuses the same system prompt / history construction
    logic used in the console app so behavior is consistent.
    """

    session = Session(session_id=str(uuid.uuid4()))

    print(f"Starting simulated audio session {session.session_id} (active_role={session.active_role.name})")
    print("Type simulated ASR transcriptions (or /quit to exit).")

    system_prompt = """
    You are a telecom customer support coach.
    You talk to a human agent (not the customer) and tell them what to say.
    Always follow the Docs context from support_faq.md over any other knowledge.
    When answering, give 3–6 concise bullet points the agent can read out.
    If the Docs context does not cover the issue, say that clearly and suggest the agent escalate or check with a supervisor.
    """.strip()

    try:
        while True:
            try:
                raw = input(PROMPT_SIMULATED_ASR).strip()
            except EOFError:
                print("\nEOF received — exiting demo.")
                break

            if not raw:
                continue

            # Allow quitting immediately
            if raw == "/quit":
                print("System> Goodbye.")
                break

            # The user is typing a simulated ASR transcription. In a real
            # implementation we would call `transcribe_chunk` with audio
            # bytes; here the developer directly supplies the transcription.
            customer_text = raw

            # Show the ASR output for clarity, then record the customer turn
            print(f"Customer (ASR)> {customer_text}")
            session.add_turn(SpeakerRole.CUSTOMER, customer_text)

            if session.active_role == SpeakerRole.BOT:
                # Build history from session turns
                history: list[dict] = []
                for turn in session.turns:
                    if turn.speaker == SpeakerRole.CUSTOMER:
                        history.append({"role": "user", "content": turn.text})
                    else:
                        history.append({"role": "assistant", "content": turn.text})

                # Retrieve short RAG context from the FAQ and include it
                rag_context = get_rag_context(customer_text)
                
                if DEBUG_RAG:
                    print("=== RAG DEBUG ===")
                    print("QUESTION:", customer_text)
                    print("CONTEXT PREVIEW:")
                    for line in rag_context.splitlines()[:8]:
                        print(line)
                    print("=================")
                
                augmented_message = f"""
                Docs context:

                {rag_context}

                Instructions:
                - Answer ONLY using the Docs context above.
                - If the Docs context is missing information, say that explicitly
                  and suggest the agent escalate or check with a supervisor.

                Customer question:
                {customer_text}
                """.strip()

                reply_text = generate_with_ollama(
                    system_prompt=system_prompt,
                    history=history,
                    user_message=augmented_message,
                )

                # Print the bot reply, then send it to the TTS stub and log it
                print(f"BOT> {reply_text}")
                speak(reply_text)
                session.add_turn(SpeakerRole.BOT, reply_text)

            elif session.active_role == SpeakerRole.HUMAN_AGENT:
                # Prompt for manual human reply in the simulated environment
                try:
                    human_text = input(PROMPT_HUMAN).strip()
                except EOFError:
                    print("\nEOF received from human — returning to loop.")
                    continue

                if human_text == "/quit":
                    print("System> Goodbye.")
                    break

                session.add_turn(SpeakerRole.HUMAN_AGENT, human_text)
                # Optionally speak the human reply as well so the audio
                # pipeline can be exercised in tests.
                speak(human_text)

            else:
                print(f"System> No responder configured for active role: {session.active_role}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt — exiting demo.")


def run_full_voice_test(duration: float = 4.0, sample_rate: int = 16000) -> None:
    """Record a short clip, run ASR -> LLM -> TTS, then exit.

    This function captures audio from the default microphone for
    `duration` seconds, encodes it as WAV bytes in-memory, calls
    `transcribe_chunk` to obtain ASR text, then runs the dialogue logic
    (generate_bot_reply) and speaks the resulting reply via `speak()`.
    """

    try:
        import io
        import wave
        import sounddevice as sd
        import numpy as np
    except Exception as exc:
        print("Error: full voice test requires 'sounddevice' and 'numpy'.")
        print("Install with: pip install sounddevice numpy")
        print(f"Detail: {exc}")
        return

    channels = 1

    try:
        print("Recording...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
        sd.wait()
        print("Done.")
        print("STEP 1: recorded audio")

        # Normalize and convert to int16 PCM
        audio = np.asarray(recording)
        if audio.ndim > 1:
            audio = audio.reshape(-1, channels)
        audio = np.clip(audio, -1.0, 1.0)
        int_data = (audio * 32767.0).astype(np.int16)

        # Write WAV to in-memory bytes buffer
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int_data.tobytes())

        wav_bytes = buf.getvalue()

        # Save WAV to disk for inspection
        try:
            import os

            out_path = os.path.abspath("full_voice_test.wav")
            with open(out_path, "wb") as outf:
                outf.write(wav_bytes)
            print(f"Saved WAV to {out_path}")
        except Exception:
            # non-fatal: continue even if saving fails
            pass

        # Call ASR (use Whisper on the saved WAV file)
        print("STEP 2: sending audio to ASR...")
        try:
            from src.audio.asr_whisper import transcribe_file

            asr_text = transcribe_file(out_path)
        except Exception as exc:
            print("Error: Whisper ASR failed:", exc)
            return

        print(f"STEP 3: ASR text: {asr_text}")

        # Build session and history, then call LLM
        session = Session(session_id=str(uuid.uuid4()))
        system_prompt = """
        You are a telecom customer support coach.
        You talk to a human agent (not the customer) and tell them what to say.
        Always follow the Docs context from support_faq.md over any other knowledge.
        When answering, give 3–6 concise bullet points the agent can read out.
        If the Docs context does not cover the issue, say that clearly and suggest the agent escalate or check with a supervisor.
        """.strip()

        session.add_turn(SpeakerRole.CUSTOMER, asr_text)

        history: list[dict] = []
        for turn in session.turns:
            if turn.speaker == SpeakerRole.CUSTOMER:
                history.append({"role": "user", "content": turn.text})
            else:
                history.append({"role": "assistant", "content": turn.text})

        print("STEP 4: sending text to LLM...")
        # Add RAG context to the LLM input
        rag_context = get_rag_context(asr_text)
        
        if DEBUG_RAG:
            print("=== RAG DEBUG ===")
            print("QUESTION:", asr_text)
            print("CONTEXT PREVIEW:")
            for line in rag_context.splitlines()[:8]:
                print(line)
            print("=================")
        
        augmented_message = f"""
        Docs context:

        {rag_context}

        Instructions:
        - Answer ONLY using the Docs context above.
        - If the Docs context is missing information, say that explicitly
          and suggest the agent escalate or check with a supervisor.

        Customer question:
        {asr_text}
        """.strip()

        reply_text = generate_with_ollama(
            system_prompt=system_prompt,
            history=history,
            user_message=augmented_message,
        )

        # Trim to first sentence before speaking
        import re

        reply_text = reply_text.strip()
        parts = re.split(r'(?<=[.!?])\s+', reply_text)
        if parts:
            reply_text = parts[0]

        print(f"STEP 5: BOT reply (trimmed): {reply_text}")
        print("STEP 6: sending reply to TTS...")
        print(f"BOT> {reply_text}")
        speak(reply_text)
        session.add_turn(SpeakerRole.BOT, reply_text)

    except Exception as exc:
        print("Error during full voice test:")
        print(exc)


def load_test_scenarios(path: str = "scenarios.txt") -> list[str]:
    """Read a scenarios file and return non-empty lines.

    Each non-empty line in `path` is treated as one test scenario.
    """
    scenarios: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    scenarios.append(s)
    except FileNotFoundError:
        print(f"Scenarios file not found: {path}")
    except Exception as exc:
        print(f"Error reading scenarios file {path}: {exc}")

    return scenarios


def run_scenario_tests(path: str = "scenarios.txt") -> None:
    """Run through scenarios from a file and print LLM replies (no TTS).

    For each scenario we create an isolated Session, send the scenario
    text to the LLM using the same system rules as the voice pipeline,
    trim the reply to the first sentence, and print it. TTS is not called.
    """
    scenarios = load_test_scenarios(path)
    if not scenarios:
        print("No scenarios to run.")
        return

    system_prompt = """
    You are a telecom customer support coach.
    You talk to a human agent (not the customer) and tell them what to say.
    Always follow the Docs context from support_faq.md over any other knowledge.
    When answering, give 3–6 concise bullet points the agent can read out.
    If the Docs context does not cover the issue, say that clearly and suggest the agent escalate or check with a supervisor.
    """.strip()

    import re

    for idx, text in enumerate(scenarios, start=1):
        print("---")
        print(f"Scenario {idx}:")
        print(f"Customer (ASR)> {text}")

        # Isolated session per scenario
        session = Session(session_id=str(uuid.uuid4()))
        session.add_turn(SpeakerRole.CUSTOMER, text)

        history: list[dict] = []
        for turn in session.turns:
            if turn.speaker == SpeakerRole.CUSTOMER:
                history.append({"role": "user", "content": turn.text})
            else:
                history.append({"role": "assistant", "content": turn.text})

        # Add RAG context to the LLM input for scenario tests
        rag_context = get_rag_context(text)
        
        if DEBUG_RAG:
            print("=== RAG DEBUG ===")
            print("QUESTION:", text)
            print("CONTEXT PREVIEW:")
            for line in rag_context.splitlines()[:8]:
                print(line)
            print("=================")
        
        augmented_message = f"""
        Docs context:

        {rag_context}

        Instructions:
        - Answer ONLY using the Docs context above.
        - If the Docs context is missing information, say that explicitly
          and suggest the agent escalate or check with a supervisor.

        Customer question:
        {text}
        """.strip()

        reply_text = generate_with_ollama(
            system_prompt=system_prompt,
            history=history,
            user_message=augmented_message,
        )

        # Post-process reply: strip surrounding quotes and keep first sentence
        reply_text = reply_text.strip()
        reply_text = reply_text.strip('"').strip("'")
        parts = re.split(r'(?<=[.!?])\s+', reply_text)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            reply_text = parts[0]

        print(f"BOT> {reply_text}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulated audio I/O demo")
    parser.add_argument("--mic-test", action="store_true", help="Record a short mic test clip and exit")
    parser.add_argument(
        "--full-voice-test",
        action="store_true",
        help="Record from mic, run ASR -> LLM -> TTS once and exit",
    )
    parser.add_argument(
        "--scenario-tests",
        dest="scenario_tests",
        action="store_true",
        help="Run LLM replies for scenarios from scenarios.txt (no TTS)",
    )
    args = parser.parse_args()

    if args.mic_test:
        # Import here so missing mic-test dependencies don't affect normal demo
        from src.audio.mic_test import record_test_clip

        record_test_clip()
    elif args.full_voice_test:
        run_full_voice_test()
    elif getattr(args, "scenario_tests", False):
        run_scenario_tests()
    else:
        demo_turn_based_audio_session()
