import json
import argparse
from collections import defaultdict


def normalize_gt(x):
    x = str(x).strip()
    if x.upper() in {"A", "B", "C", "D"}:
        return x.upper()

    mapping = {
        "0": "A",
        "1": "B",
        "2": "C",
        "3": "D",
    }
    if x in mapping:
        return mapping[x]

    return x.upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.pred_json, "r", encoding="utf-8"))

    total = 0
    correct = 0

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in rows:
        gt = normalize_gt(row["gt"])
        pred = str(row["pred"]).strip().upper()
        cat = str(row.get("category", "unknown"))

        ok = int(pred == gt)
        total += 1
        correct += ok

        by_cat[cat]["correct"] += ok
        by_cat[cat]["total"] += 1

    out = {
        "Accuracy": round(100.0 * correct / max(total, 1), 4),
        "count": total,
        "by_category": {}
    }

    for cat in sorted(by_cat.keys()):
        c = by_cat[cat]["correct"]
        n = by_cat[cat]["total"]
        out["by_category"][cat] = round(100.0 * c / max(n, 1), 4)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
