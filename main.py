from debate_graph import verify_claim              # import the verify_claim function from debate_graph.py


def main():
    print("=" * 60)                                # print a line of 60 = signs for visual separation
    print("  Multi-Agent News Fact Verification")  # print the app title
    print("  Agent 1 : GPT-4o-mini  (Web Search)") # show Agent 1 model
    print("  Agent 2 : Grok 4 Fast  (Wikipedia)")  # show Agent 2 model
    print("  Agent 3 : Gemini 2.5   (Reasoning)")  # show Agent 3 model
    print("  Judge   : Claude Sonnet")             # show Judge model
    print("=" * 60)                                # closing separator
    print("  Type any news claim and press Enter.") # instructions
    print("  Type 'quit' to exit.\n")              # tell user how to exit

    while True:                                    # keep looping forever until user types quit
        claim = input("Claim: ").strip()           # ask user to type a claim — .strip() removes accidental spaces at start and end

        if not claim:                              # if user just pressed Enter without typing anything
            continue                               # go back to top of loop and ask again

        if claim.lower() in ["quit", "exit", "q"]: # if user typed quit or exit or q in any case
            print("Goodbye.")                      # say goodbye
            break                                  # exit the while loop — program ends

        print("\nRunning debate...\n")             # tell user the system is working — this takes 20-40 seconds

        result = verify_claim(claim)               # run the full multi-agent debate — calls debate_graph.py

        print("\n" + "=" * 60)                     # top separator for result display
        print(f"  VERDICT   : {result['verdict']}")            # SUPPORTED / REFUTED / INSUFFICIENT_EVIDENCE
        print(f"  CONFIDENCE: {result['confidence']:.0%}")     # convert 0.87 to 87% for readability
        print(f"  ROUNDS    : {result['debate_rounds']}")      # how many debate rounds happened
        print(f"\n  REASONING :\n  {result['reasoning']}")     # judge's full explanation
        print("=" * 60 + "\n")                                 # bottom separator


if __name__ == "__main__":                         # only run main() if we execute THIS file directly
    main()                                         # if someone imports this file elsewhere main() won't auto-run