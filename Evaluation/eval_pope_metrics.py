import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.pred_json, "r", encoding="utf-8"))

    tp = fp = tn = fn = 0

    for r in rows:
        gt = str(r["label"]).strip().lower()
        pred = str(r["answer"]).strip().lower()

        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1

    total = tp + fp + tn + fn

    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    yes_ratio = (tp + fp) / max(total, 1)

    result = {
        "Accuracy": round(acc * 100, 4),
        "Precision": round(prec * 100, 4),
        "Recall": round(rec * 100, 4),
        "F1": round(f1 * 100, 4),
        "Yes_ratio": round(yes_ratio * 100, 4),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
