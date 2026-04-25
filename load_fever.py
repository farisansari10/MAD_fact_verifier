import json                                        # built-in Python library — used to save claims to claims.json
from datasets import load_dataset                  # HuggingFace library — used to download the dataset


def load_fever_claims(num_claims: int = 15, output_file: str = "claims.json"):
    # num_claims  = how many claims to pull — 20 is enough for evaluation
    # output_file = where to save them — defaults to claims.json

    print("Downloading climate_fever dataset from HuggingFace...")
    print("This may take a minute on first run.\n")

    dataset = load_dataset(                        # download dataset from HuggingFace
        "climate_fever",                           # dataset name
        split="test",                              # use test split
    )

    # label mapping — climate_fever uses integers not strings
    # 0 = SUPPORTED, 1 = REFUTED, 2 = INSUFFICIENT_EVIDENCE
    LABEL_MAP = {                                  # dictionary mapping integer labels to our string labels
        0: "SUPPORTED",                            # 0 means the claim is supported by evidence
        1: "REFUTED",                              # 1 means the claim is refuted by evidence
        2: "INSUFFICIENT_EVIDENCE",                # 2 means not enough evidence to decide
    }

    claims  = []                                   # empty list to collect our selected claims
    counts  = {                                    # track how many of each label we collected
        "SUPPORTED":             0,
        "REFUTED":               0,
        "INSUFFICIENT_EVIDENCE": 0
    }
    target  = num_claims // 3                      # how many of each label we want — equal split e.g. 20//3 = 6

    for row in dataset:                            # loop through every row in the dataset
        if len(claims) >= num_claims:              # if we already have enough claims
            break                                  # stop looping

        label_int = row["claim_label"]             # get the integer label for this row e.g. 0, 1, or 2

        if label_int not in LABEL_MAP:             # if label is not 0, 1, or 2
            continue                               # skip this row — unknown label

        our_label = LABEL_MAP[label_int]           # convert integer to our string label e.g. 0 → "SUPPORTED"

        if counts[our_label] >= target:            # if we already have enough of this label type
            continue                               # skip — we want balanced labels

        claim_text = row["claim"]                  # get the actual claim text

        if len(claim_text) < 10:                   # skip very short claims — usually garbage
            continue

        claims.append({                            # add this claim to our list
            "claim": claim_text,                   # the claim text
            "label": our_label,                    # our converted string label
        })

        counts[our_label] += 1                     # increment counter for this label type

    # if we still need more claims after equal split fill remaining slots
    if len(claims) < num_claims:                   # if we still need more
        for row in dataset:                        # loop through dataset again
            if len(claims) >= num_claims:          # if we now have enough
                break

            label_int = row["claim_label"]         # get integer label

            if label_int not in LABEL_MAP:         # skip unknown labels
                continue

            our_label = LABEL_MAP[label_int]       # convert to string label

            claim_text = row["claim"]              # get claim text

            if len(claim_text) < 10:               # skip short claims
                continue

            if {"claim": claim_text, "label": our_label} not in claims:  # avoid duplicates
                claims.append({
                    "claim": claim_text,
                    "label": our_label,
                })

    with open(output_file, "w") as f:              # open claims.json for writing
        json.dump(claims, f, indent=2, ensure_ascii=False)            # write claims list as formatted JSON

    print(f"Saved {len(claims)} claims to {output_file}")
    print(f"  SUPPORTED            : {counts['SUPPORTED']}")
    print(f"  REFUTED              : {counts['REFUTED']}")
    print(f"  INSUFFICIENT_EVIDENCE: {counts['INSUFFICIENT_EVIDENCE']}")
    print(f"\nNow run: python3 main.py")


if __name__ == "__main__":                         # only run if we execute this file directly
    load_fever_claims(num_claims=15)              # load 8 claims