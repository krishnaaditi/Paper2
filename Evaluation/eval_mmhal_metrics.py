# eval_mmhal_metrics.py
import json
import re
import argparse
import collections


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    args = ap.parse_args()

    total = 0
    hallucinated = 0

    by_cat = collections.defaultdict(lambda: {"n": 0, "hall": 0})

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            pred = tokenize(row.get("answer", ""))
            ref = tokenize(row.get("reference", ""))
            cat = row.get("category", "other")

            total += 1
            by_cat[cat]["n"] += 1

            # lightweight proxy:
            # count hallucination if many content words appear in prediction but not in reference
            extra_tokens = [t for t in pred if t not in ref and len(t) > 2]

            hall_flag = 1 if len(extra_tokens) >= 3 else 0
            hallucinated += hall_flag
            by_cat[cat]["hall"] += hall_flag

    overall_hall = 100.0 * hallucinated / max(total, 1)
    overall_score = 10.0 * (1.0 - hallucinated / max(total, 1))

    result = {
        "Overall_score_proxy": round(overall_score, 4),
        "Hallucination_rate": round(overall_hall, 4),
        "by_category": {}
    }

    for cat, stats in sorted(by_cat.items()):
        rate = 100.0 * stats["hall"] / max(stats["n"], 1)
        score = 10.0 * (1.0 - stats["hall"] / max(stats["n"], 1))
        result["by_category"][cat] = {
            "Hallucination_rate": round(rate, 4),
            "Score_proxy": round(score, 4),
            "count": stats["n"]
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
