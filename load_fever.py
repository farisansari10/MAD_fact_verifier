import json
from datasets import load_dataset


def load_fever_claims(num_claims: int = 15, output_file: str = "claims.json"):

    print("Downloading climate_fever dataset from HuggingFace...")
    print("This may take a minute on first run.\n")

    dataset = load_dataset(
        "climate_fever",
        split="test",
    )

    LABEL_MAP = {
        0: "SUPPORTED",
        1: "REFUTED",
        2: "INSUFFICIENT_EVIDENCE",
    }

    claims  = []
    counts  = {
        "SUPPORTED":             0,
        "REFUTED":               0,
        "INSUFFICIENT_EVIDENCE": 0
    }
    target  = num_claims // 3

    for row in dataset:
        if len(claims) >= num_claims:
            break

        label_int = row["claim_label"]

        if label_int not in LABEL_MAP:
            continue

        our_label = LABEL_MAP[label_int]

        if counts[our_label] >= target:
            continue

        claim_text = row["claim"]

        if len(claim_text) < 10:
            continue

        claims.append({
            "claim": claim_text,
            "label": our_label,
        })

        counts[our_label] += 1

    if len(claims) < num_claims:
        for row in dataset:
            if len(claims) >= num_claims:
                break

            label_int = row["claim_label"]

            if label_int not in LABEL_MAP:
                continue

            our_label = LABEL_MAP[label_int]

            claim_text = row["claim"]

            if len(claim_text) < 10:
                continue

            if {"claim": claim_text, "label": our_label} not in claims:
                claims.append({
                    "claim": claim_text,
                    "label": our_label,
                })

    with open(output_file, "w") as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(claims)} claims to {output_file}")
    print(f"  SUPPORTED            : {counts['SUPPORTED']}")
    print(f"  REFUTED              : {counts['REFUTED']}")
    print(f"  INSUFFICIENT_EVIDENCE: {counts['INSUFFICIENT_EVIDENCE']}")
    print(f"\nNow run: python3 main.py")


if __name__ == "__main__":
    load_fever_claims(num_claims=15)