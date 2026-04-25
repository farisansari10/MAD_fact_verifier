import json                                        # built-in Python library — used to read claims.json and save results
import time                                        # built-in Python library — used to measure how long each claim takes
from debate_graph import verify_claim              # import our main debate function from debate_graph.py


def load_claims(filepath: str = "claims.json") -> list:
    # loads claims from claims.json file instead of hardcoding them
    # filepath = path to the JSON file — defaults to claims.json in same folder
    # returns  = a list of dicts each with "claim" and "label" keys

    with open(filepath, "r") as f:                 # open claims.json file for reading
        claims = json.load(f)                      # parse the JSON file into a Python list

    print(f"  Loaded {len(claims)} claims from {filepath}\n")  # tell user how many claims were loaded
    return claims                                  # return the list so run_evaluation() can use it


def run_evaluation():
    results    = []                                # empty list — we will append each claim's result here
    correct    = 0                                 # counter for how many claims the system got right
    total_time = 0                                 # counter for total time taken across all claims

    print("=" * 60)                                # visual separator
    print("  Multi-Agent Debate — Evaluation Run") # title
    print("=" * 60)                                # visual separator

    TEST_CLAIMS = load_claims("claims.json")       # load claims from file — not hardcoded

    for i, test in enumerate(TEST_CLAIMS):         # loop through each claim — i = index (0,1,2...), test = the dict
        print(f"\nClaim {i+1}/{len(TEST_CLAIMS)}: {test['claim']}")  # print which claim we are on
        print(f"Expected : {test['label']}")        # print the correct answer we expect

        start   = time.time()                      # record start time before running the debate
        output  = verify_claim(test["claim"])      # run the full multi-agent debate on this claim
        elapsed = round(time.time() - start, 1)    # calculate how many seconds it took — rounded to 1 decimal
        total_time += elapsed                      # add to running total time

        predicted  = output["verdict"]             # get what the system predicted
        is_correct = (predicted == test["label"])  # True if prediction matches expected label — False if not

        if is_correct:                             # if the system got it right
            correct += 1                           # increment correct counter by 1

        print(f"Predicted: {predicted}  (confidence: {output['confidence']:.2f})")  # show prediction and confidence
        print(f"Rounds   : {output['debate_rounds']}  |  Time: {elapsed}s")         # show rounds used and time taken
        print(f"Result   : {'✓ CORRECT' if is_correct else '✗ WRONG'}")             # show if right or wrong

        results.append({                           # save this claim's full result to the list
            "claim":         test["claim"],        # original claim text
            "expected":      test["label"],        # correct label
            "predicted":     predicted,            # what the system said
            "confidence":    output["confidence"], # judge's confidence score
            "correct":       is_correct,           # True or False
            "debate_rounds": output["debate_rounds"],  # how many debate rounds happened
            "reasoning":     output["reasoning"],  # judge's reasoning text
            "time_seconds":  elapsed,              # how long this claim took in seconds
        })

    # ── Calculate final metrics ───────────────────────────────────────────────

    accuracy   = correct / len(TEST_CLAIMS) * 100                          # accuracy = correct / total * 100
    avg_conf   = sum(r["confidence"] for r in results) / len(results)      # average confidence across all claims
    avg_rounds = sum(r["debate_rounds"] for r in results) / len(results)   # average debate rounds used per claim

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy          : {accuracy:.1f}%  ({correct}/{len(TEST_CLAIMS)})")  # e.g. 80.0% (8/10)
    print(f"  Avg Confidence    : {avg_conf:.2f}")                                   # e.g. 0.83
    print(f"  Avg Debate Rounds : {avg_rounds:.1f}")                                 # e.g. 1.2
    print(f"  Total Time        : {total_time:.0f}s")                                # e.g. 142s
    print("=" * 60)

    # ── Save results to JSON file ─────────────────────────────────────────────

    output_data = {                                # build dict with all evaluation data
        "accuracy":          accuracy,             # overall accuracy percentage
        "avg_confidence":    avg_conf,             # average confidence score
        "avg_debate_rounds": avg_rounds,           # average debate rounds used
        "total_time":        total_time,           # total time in seconds
        "results":           results,              # full per-claim results list
    }

    with open("evaluation_results.json", "w") as f:   # open evaluation_results.json for writing
        json.dump(output_data, f, indent=2, ensure_ascii=False)            # write dict as formatted JSON into the file

    print("\n  Results saved to evaluation_results.json")  # tell user where results are saved
    print("  Use these numbers in your IEEE paper.\n")     # reminder for the paper


if __name__ == "__main__":                         # only run if we execute this file directly
    run_evaluation()                               # call the evaluation function