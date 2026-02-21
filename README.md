# Support Assistant: Local Telecom Coach with RAG + Ollama

A lightweight, offline-first customer support coaching assistant that uses **Retrieval-Augmented Generation (RAG)** over a telecom FAQ knowledge base to guide support agents. Built entirely locally with Python, Ollama, and sentence-transformers—no cloud dependencies.

## Overview

This project combines:

- **RAG Pipeline** → Embeds and retrieves relevant FAQ chunks based on customer questions
- **Local LLM** → Ollama (llama3.2) generates grounded, contextual coaching responses
- **Text & Voice Interfaces** → Turn-based console demo + full voice recording/ASR pipeline (with voice support optional)
- **Evaluation Tools** → Manual smoke tests and full RAG roundtrip evaluation for answer grounding

A support agent or supervisor can ask a question (e.g., "Customer wants to cancel their broadband"), and the assistant returns bullet points from the FAQ to guide the conversation.

## Quickstart

### Prerequisites

- **Python 3.11+**
- **Ollama** installed and running locally (https://ollama.ai)
- **GPU** (recommended, though CPU is supported)

### 1. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.ai
# Then pull the model
ollama pull llama3.2
```

### 2. Clone and install dependencies

```bash
git clone <repo-url>
cd support-assistant
pip install -r requirements.txt
```

### 3. Run the assistant

In one terminal, start Ollama:

```bash
ollama run llama3.2
```

In another terminal, start the support assistant:

```bash
python -m src.audio.io_loop
```

You'll be prompted to type simulated ASR text (or real voice input with dependencies). Try one of the example questions below:

- `I want to change my mobile plan, what should I say?`
- `A customer wants to cancel their broadband, what should I say?`
- `The customer has a billing dispute, how do I handle it?`
- `The customer paid their bill late, what should I say?`
- `The customer lost their phone, what are the steps?`

Or type `/quit` to exit.

## Project Structure

```
support-assistant/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .env.example                           # Environment placeholder
├── .gitignore                             # Git ignore rules
│
├── data/
│   └── support_faq.md                     # Telecom FAQ knowledge base (markdown)
│
├── src/
│   ├── __init__.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── simple_faq_rag.py             # RAG: chunk, embed, retrieve
│   ├── dialogue/
│   │   ├── __init__.py
│   │   ├── state.py                      # Session and conversation state
│   │   ├── ollama_client.py              # HTTP wrapper for Ollama /api/chat
│   │   └── llm_client.py                 # (Legacy) OpenAI-style client
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── io_loop.py                    # Main text/voice I/O loop (ENTRY POINT)
│   │   ├── asr.py                        # ASR stubs
│   │   ├── asr_whisper.py                # Whisper integration (optional)
│   │   ├── tts.py                        # TTS stubs
│   │   ├── vad.py                        # Voice Activity Detection
│   │   └── mic_test.py                   # Microphone testing utility
│   └── interfaces/
│       ├── __init__.py
│       └── console_app.py                # Text-only console demo (alternate entry point)
│
├── tests/
│   └── manual_rag_smoke.py               # Smoke test: print RAG context for sample queries
│
└── scripts/
    └── eval_rag_roundtrip.py             # Evaluation: run full RAG+LLM answers for fixed scenarios
```

## RAG & Prompt Design

### Knowledge Base

- **File**: `data/support_faq.md`
- **Structure**: Sections like "Change mobile plan", "Cancel broadband", "Billing dispute", etc.
- **Format**: Markdown with bullet-point guidance for agents.

### RAG Pipeline (`src/rag/simple_faq_rag.py`)

1. Parses support_faq.md into sections (split by `#` headings)
2. Embeds each section using `sentence-transformers/all-MiniLM-L6-v2`
3. On query, retrieves top-k (default k=3) sections by cosine similarity
4. Returns concatenated text of relevant chunks

### Augmented Prompt

Each LLM call includes:

```
Docs context:
[Retrieved FAQ sections]

Instructions:
- Answer ONLY using the Docs context above.
- If the Docs context is missing information, say explicitly and suggest escalation.

Customer question:
[User question]
```

This design ensures answers are grounded in the FAQ and minimizes hallucination.

### System Prompt

```
You are a telecom customer support coach.
You talk to a human agent (not the customer) and tell them what to say.
Always follow the Docs context from support_faq.md over any other knowledge.
When answering, give 3–6 concise bullet points the agent can read out.
If the Docs context does not cover the issue, say that clearly and suggest escalation.
```

## Testing & Evaluation

### Smoke Test: RAG Context Retrieval

View raw FAQ chunks retrieved for sample queries:

```bash
python -m tests.manual_rag_smoke
```

Outputs question → retrieved context for each query in `QUERIES` list.

### Full Roundtrip Evaluation

Run 6 canonical test cases through the complete RAG+LLM pipeline and compare answers to expected behavior:

```bash
python -m scripts.eval_rag_roundtrip
```

Requires Ollama running. Outputs:

```
================================================================================
TEST ID: plan_change
QUESTION: I want to change my mobile plan, what should I tell the customer?
EXPECTED: Verify identity, explain options, mention contract impacts, ...

CONTEXT PREVIEW:
[8 lines of FAQ context]

MODEL ANSWER:
[Full LLM response]
================================================================================
```

### Debug Mode

Set `DEBUG_RAG = True` in `src/audio/io_loop.py` (default) to see RAG context logged for each turn:

```
=== RAG DEBUG ===
QUESTION: [customer question]
CONTEXT PREVIEW:
[first 8 lines of FAQ sections used]
=================
```

## Alternate Entry Points

### Text-Only Console Demo

```bash
python -m src.interfaces.console_app
```

Simple turn-based text interface. No voice/ASR.

### Voice-Based Full Test (requires audio dependencies)

```bash
python -m src.audio.io_loop --full-voice-test
```

Records 4 seconds of audio, transcribes with Whisper, generates reply, and speaks via TTS.

### Scenario Tests (LLM replies without TTS)

```bash
python -m src.audio.io_loop --scenario-tests
```

Runs scenarios from `scenarios.txt` and prints LLM replies for batch evaluation.

## Configuration

### Environment Variables

Currently, none are required for local operation. See `.env.example` for placeholders.

### Model Selection

Edit `src/dialogue/ollama_client.py`:

```python
MODEL_NAME = "llama3.2"  # Change to another Ollama model
```

### Ollama Endpoint

Default: `http://localhost:11434/api/chat`  
Edit `src/dialogue/ollama_client.py` if using a remote Ollama server.

## Dependencies

See `requirements.txt`. Core requirements:

- `requests` – HTTP client for Ollama API
- `sentence-transformers` – Embedding model for RAG
- `ollama` – (Legacy) Direct Ollama Python SDK

Optional (for voice/audio features):

- `sounddevice` – Microphone recording
- `numpy` – Audio processing
- `openai` – Whisper ASR (can be local or remote)

## Development Notes

### Adding to the FAQ

Edit `data/support_faq.md` and add new sections:

```markdown
# New Topic

- Bullet point guidance 1
- Bullet point guidance 2
- etc.
```

RAG will automatically pick up new sections on next run.

### Tuning RAG

Adjust `k` in `get_rag_context(question, k=3)` calls:

- `k=1` → Single most relevant section
- `k=3` (default) → Top 3 sections
- `k=5+` → More context, slower retrieval

### System Prompt

Modify `SYSTEM_PROMPT` in `src/audio/io_loop.py` to adjust coaching style (e.g., more verbose, different tone).

## Troubleshooting

| Issue                                    | Solution                                                                               |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| `Connection refused` when calling Ollama | Ensure Ollama is running: `ollama run llama3.2` in a separate terminal                 |
| Model not found                          | Run: `ollama pull llama3.2` (or your selected model)                                   |
| Slow embeddings on first run             | Sentence-transformers model downloads on first use; subsequent calls are fast (cached) |
| Audio/microphone not working             | Install optional dependencies: `pip install sounddevice numpy`                         |
| Embeddings model errors                  | Try: `pip install --upgrade sentence-transformers`                                     |

## License

[Add your license here]

## Contact & Contributions

[Add contact info or contribution guidelines]
