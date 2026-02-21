"""Lightweight RAG evaluation script for manual answer grounding assessment.

Run with: python -m scripts.eval_rag_roundtrip

For each test case, this script:
1. Retrieves context from the FAQ using simple_faq_rag.
2. Builds the augmented_user_message with instructions (same as io_loop.py).
3. Calls generate_with_ollama to get an answer.
4. Prints formatted output for manual review.
"""

from src.rag.simple_faq_rag import get_rag_context
from src.dialogue.ollama_client import generate_with_ollama


# System prompt (same as in io_loop.py)
SYSTEM_PROMPT = """
You are a telecom customer support coach.
You talk to a human agent (not the customer) and tell them what to say.
Always follow the Docs context from support_faq.md over any other knowledge.
When answering, give 3–6 concise bullet points the agent can read out.
If the Docs context does not cover the issue, say that clearly and suggest the agent escalate or check with a supervisor.
""".strip()


TEST_CASES = [
    {
        "id": "plan_change",
        "question": "I want to change my mobile plan, what should I tell the customer?",
        "expected": "Verify identity, explain options, mention contract impacts, confirm change, send confirmation.",
    },
    {
        "id": "cancel_broadband",
        "question": "A customer wants to cancel their broadband, what should I say?",
        "expected": "Check reason, confirm contract/fees, discuss troubleshooting or downgrade, explain cancellation steps.",
    },
    {
        "id": "billing_dispute",
        "question": "The customer is disputing their bill because of unexpected charges.",
        "expected": "Listen to concern, identify the charge, check for error, process credit or escalate for dispute investigation.",
    },
    {
        "id": "late_payment",
        "question": "The customer has paid their bill late, what should I say?",
        "expected": "Confirm balance, explain due date and fees, offer payment options and late fee waiver if applicable.",
    },
    {
        "id": "lost_phone",
        "question": "The customer has lost their phone, what are the steps?",
        "expected": "Ask for phone details, recommend device tracking service, block SIM, discuss insurance and replacement.",
    },
    {
        "id": "sim_not_working",
        "question": "The customer's SIM card is not working, what should I do?",
        "expected": "Check insertion, restart device, verify error messages, offer replacement or escalate to technical support.",
    },
]


def run_test_case(test_case: dict) -> None:
    """Run a single RAG evaluation test case and print formatted output."""
    test_id = test_case["id"]
    question = test_case["question"]
    expected = test_case["expected"]

    # Retrieve RAG context
    context = get_rag_context(question, k=3)

    # Build augmented message (same as io_loop.py)
    augmented_user_message = f"""
Docs context:

{context}

Instructions:
- Answer ONLY using the Docs context above.
- If the Docs context is missing information, say that explicitly
  and suggest the agent escalate or check with a supervisor.

Customer question:
{question}
""".strip()

    # Call LLM with empty history (isolated test)
    answer = generate_with_ollama(
        system_prompt=SYSTEM_PROMPT,
        history=[],
        user_message=augmented_user_message,
    )

    # Print formatted output
    print("=" * 80)
    print(f"TEST ID: {test_id}")
    print(f"QUESTION: {question}")
    print(f"EXPECTED: {expected}")
    print()
    print("CONTEXT PREVIEW:")
    for line in context.splitlines()[:8]:
        print(line)
    print()
    print("MODEL ANSWER:")
    print(answer)
    print("=" * 80)
    print()


if __name__ == "__main__":
    print("RAG Evaluation Roundtrip")
    print("Running all test cases...\n")
    for tc in TEST_CASES:
        run_test_case(tc)
    print("Done.")
