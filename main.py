from debate_graph import verify_claim


def main():
    print("=" * 60)
    print("  Multi-Agent News Fact Verification")
    print("  Agent 1 : GPT-4o-mini  (Web Search)")
    print("  Agent 2 : Grok 4 Fast  (Wikipedia)")
    print("  Agent 3 : Gemini 2.5   (Reasoning)")
    print("  Judge   : Claude Sonnet")
    print("=" * 60)
    print("  Type any news claim and press Enter.")
    print("  Type 'quit' to exit.\n")

    while True:
        claim = input("Claim: ").strip()

        if not claim:
            continue

        if claim.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break

        print("\nRunning debate...\n")

        result = verify_claim(claim)

        print("\n" + "=" * 60)
        print(f"  VERDICT   : {result['verdict']}")
        print(f"  CONFIDENCE: {result['confidence']:.0%}")
        print(f"  ROUNDS    : {result['debate_rounds']}")
        print(f"\n  REASONING :\n  {result['reasoning']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()