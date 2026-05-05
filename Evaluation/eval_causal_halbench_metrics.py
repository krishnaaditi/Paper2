# eval_causal_halbench_metrics.py
import json
import argparse
from collections import defaultdict


def safe_acc(correct, total):
    return 100.0 * correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True, help="model response JSON")
    ap.add_argument("--qa_json", required=True, help="official qa.json")
    args = ap.parse_args()

    preds = json.load(open(args.pred_json, "r", encoding="utf-8"))
    qa_rows = json.load(open(args.qa_json, "r", encoding="utf-8"))

    pred_by_id = {x["id"]: x for x in preds}

    # We map benchmark items into four groups:
    # Qc      : target + origin
    # Qa      : absent + origin
    # Qc_prime: target + edited/inpainted
    # Qa_prime: absent + edited/inpainted
    #
    # This matches the paper-style reporting:
    # Qc, Qa, Q'c, Q'a, Acc, ΔQc, ΔQa, CHR

    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in qa_rows:
        rid = row["id"]
        if rid not in pred_by_id:
            continue

        pred = str(pred_by_id[rid]["answer"]).lower()
        gt = str(row["answer"]).lower() if "answer" in row else str(row.get("label", "")).lower()

        qtype = row["type"].lower()   # target / absent
        tag = row["tag"].lower()      # origin / edited / inpainted / ...

        # define group
        edited_flag = 0 if tag == "origin" else 1

        if qtype == "target" and edited_flag == 0:
            group = "Qc"
        elif qtype == "absent" and edited_flag == 0:
            group = "Qa"
        elif qtype == "target" and edited_flag == 1:
            group = "Qc_prime"
        elif qtype == "absent" and edited_flag == 1:
            group = "Qa_prime"
        else:
            continue

        stats[group]["total"] += 1
        if pred == gt:
            stats[group]["correct"] += 1

    Qc = safe_acc(stats["Qc"]["correct"], stats["Qc"]["total"])
    Qa = safe_acc(stats["Qa"]["correct"], stats["Qa"]["total"])
    Qcp = safe_acc(stats["Qc_prime"]["correct"], stats["Qc_prime"]["total"])
    Qap = safe_acc(stats["Qa_prime"]["correct"], stats["Qa_prime"]["total"])

    all_correct = sum(v["correct"] for v in stats.values())
    all_total = sum(v["total"] for v in stats.values())
    acc = safe_acc(all_correct, all_total)

    delta_qc = abs(Qc - Qcp)
    delta_qa = abs(Qa - Qap)

    # CHR proxy / paper-aligned interpretation:
    # hallucination under counterfactual absence
    # use 100 - Qa' accuracy
    chr_score = 100.0 - Qap

    result = {
        "Qc": round(Qc, 4),
        "Qa": round(Qa, 4),
        "Q'_c": round(Qcp, 4),
        "Q'_a": round(Qap, 4),
        "Accuracy": round(acc, 4),
        "Delta_Qc": round(delta_qc, 4),
        "Delta_Qa": round(delta_qa, 4),
        "CHR": round(chr_score, 4)
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
