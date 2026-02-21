from src.rag.simple_faq_rag import get_rag_context

QUERIES = [
    "I want to change my mobile plan, what should I say?",
    "A customer wants to cancel their broadband, what should I say?",
    "The customer has a billing dispute, how do I handle it?",
    "The customer has paid their bill late, what should I say?",
    "The customer wants to upgrade their mobile plan, how do I handle it?",
    "A customer wants to cancel their broadband service, what should I say?",
    "The customer is disputing their bill because of unexpected charges.",
    "The customer has lost their phone, what are the steps?",
    "The customer's SIM card is not working, what should I do?",
]


if __name__ == "__main__":
    for q in QUERIES:
        ctx = get_rag_context(q, k=3)
        print("=" * 80)
        print("QUESTION:", q)
        print("CONTEXT:\n", ctx)
