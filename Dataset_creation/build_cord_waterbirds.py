#!/usr/bin/env python3
import os
import csv
import json
import random
import argparse
from typing import Dict, List

from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image(img, path, resize=384):
    ensure_dir(os.path.dirname(path))
    if resize and resize > 0:
        img = img.resize((resize, resize), Image.Resampling.BICUBIC)
    img.save(path)


def read_csv_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waterbirds_image_root", required=True)
    ap.add_argument("--waterbirds_csv", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--train_per_group", type=int, default=100)
    ap.add_argument("--test_per_group", type=int, default=20)
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--wb_image_col", type=str, default="img_filename")
    ap.add_argument("--wb_label_col", type=str, default="y")
    ap.add_argument("--wb_place_col", type=str, default="place")
    ap.add_argument("--wb_cue_place_value", type=str, default="1")
    ap.add_argument("--class_names", type=str, default="landbird,waterbird")
    args = ap.parse_args()

    ensure_dir(args.output_root)
    random.seed(args.seed)

    rows = read_csv_rows(args.waterbirds_csv)
    labels = sorted({str(r[args.wb_label_col]) for r in rows})
    names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    label_to_name = {lab: names[i] if i < len(names) else f"class_{lab}" for i, lab in enumerate(labels)}

    summary = []

    for i, lab in enumerate(labels):
        class_name = label_to_name[lab]
        groups = {"retain": [], "forget": [], "forget_neg": [], "both_pos": []}

        for row in rows:
            C = int(str(row[args.wb_label_col]) == lab)
            S = int(str(row[args.wb_place_col]) == str(args.wb_cue_place_value))

            if C == 1 and S == 0:
                groups["retain"].append(row)      # C ∩ S^c
            elif C == 0 and S == 1:
                groups["forget"].append(row)      # C^c ∩ S
            elif C == 0 and S == 0:
                groups["forget_neg"].append(row)  # C^c ∩ S^c
            else:
                groups["both_pos"].append(row)    # C ∩ S

        for g in groups:
            random.Random(args.seed + i + len(g)).shuffle(groups[g])

        for split_name, n in [("train", args.train_per_group), ("test", args.test_per_group)]:
            for group_name in ["retain", "forget", "forget_neg", "both_pos"]:
                selected = groups[group_name][:args.train_per_group] if split_name == "train" \
                    else groups[group_name][args.train_per_group:args.train_per_group + args.test_per_group]

                out_dir = os.path.join(args.output_root, "images", split_name, group_name, class_name)
                ensure_dir(out_dir)

                for row in selected:
                    img_path = os.path.join(args.waterbirds_image_root, row[args.wb_image_col])
                    if not os.path.exists(img_path):
                        continue
                    img = Image.open(img_path).convert("RGB")
                    base = os.path.splitext(os.path.basename(row[args.wb_image_col]))[0]
                    out_name = f"{base}_{class_name}_place{row[args.wb_place_col]}.png"
                    save_image(img, os.path.join(out_dir, out_name), args.resize)

        summary.append({
            "class_name": class_name,
            "retain_train": min(len(groups["retain"]), args.train_per_group),
            "forget_train": min(len(groups["forget"]), args.train_per_group),
            "forget_neg_train": min(len(groups["forget_neg"]), args.train_per_group),
            "both_pos_train": min(len(groups["both_pos"]), args.train_per_group),
        })

        print(f"[CORD][{class_name}] retain={len(groups['retain'])} forget={len(groups['forget'])} forget_neg={len(groups['forget_neg'])} both_pos={len(groups['both_pos'])}")

    with open(os.path.join(args.output_root, "method3_waterbirds_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
