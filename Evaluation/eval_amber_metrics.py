# eval_amber_metrics.py
import json
import re
import argparse


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.pred_json, "r", encoding="utf-8"))

    # discriminative
    tp = fp = tn = fn = 0

    # generative
    gen_total = 0
    gen_cover = 0.0
    gen_hal = 0.0
    gen_cog = 0.0

    for r in rows:
        task = r.get("task", "disc")

        if task == "disc":
            gt = str(r.get("label", "")).lower()
            pred = str(r.get("answer", "")).lower()

            if gt == "yes" and pred == "yes":
                tp += 1
            elif gt == "no" and pred == "yes":
                fp += 1
            elif gt == "no" and pred == "no":
                tn += 1
            elif gt == "yes" and pred == "no":
                fn += 1

        else:
            gt_objs = set(x.lower() for x in r.get("objects_present", []))
            pred_toks = tokenize(r.get("answer", ""))

            gen_total += 1

            if len(gt_objs) > 0:
                matched = 0
                for obj in gt_objs:
                    obj_toks = obj.split()
                    if any(tok in pred_toks for tok in obj_toks):
                        matched += 1

                cover = matched / max(len(gt_objs), 1)
                gen_cover += cover

                # simple hallucination proxy
                hal = 1.0 - cover
                gen_hal += hal

            # placeholder proxy for Cog
            gen_cog += 0.0

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    result = {
        "disc_Accuracy": round(acc * 100, 4),
        "disc_Precision": round(prec * 100, 4),
        "disc_Recall": round(rec * 100, 4),
        "disc_F1": round(f1 * 100, 4),
    }

    if gen_total > 0:
        result.update({
            "gen_CHAIR_proxy": round(100 * gen_hal / gen_total, 4),
            "gen_Cover": round(100 * gen_cover / gen_total, 4),
            "gen_Hal": round(100 * gen_hal / gen_total, 4),
            "gen_Cog": round(100 * gen_cog / max(gen_total, 1), 4),
        })

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
