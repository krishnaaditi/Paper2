#!/usr/bin/env python3
import os
import json
import random
import argparse
from collections import Counter

from PIL import Image
from pycocotools.coco import COCO


COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image(img, path, resize=384):
    ensure_dir(os.path.dirname(path))
    if resize and resize > 0:
        img = img.resize((resize, resize), Image.Resampling.BICUBIC)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_image_root", required=True)
    ap.add_argument("--coco_ann_json", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--train_per_group", type=int, default=100)
    ap.add_argument("--test_per_group", type=int, default=20)
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_names", type=str, default="")
    args = ap.parse_args()

    ensure_dir(args.output_root)
    random.seed(args.seed)

    coco = COCO(args.coco_ann_json)
    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {c["id"]: c["name"] for c in cats}
    name_to_id = {c["name"]: c["id"] for c in cats}

    img_ids = coco.getImgIds()
    image_to_cats = {}
    for iid in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], iscrowd=False))
        image_to_cats[iid] = set(a["category_id"] for a in anns)

    classes = COCO80_CLASSES
    if args.class_names:
        wanted = {x.strip() for x in args.class_names.split(",") if x.strip()}
        classes = [c for c in COCO80_CLASSES if c in wanted]

    summary = []

    for ci, target_name in enumerate(classes):
        if target_name not in name_to_id:
            continue
        target_id = name_to_id[target_name]

        # simple cue approximation: most frequent co-occurring category
        counter = Counter()
        for iid, cats_present in image_to_cats.items():
            if target_id in cats_present:
                for c in cats_present:
                    if c != target_id:
                        counter[c] += 1

        if not counter:
            print(f"[skip] no cue for {target_name}")
            continue

        cue_id, _ = counter.most_common(1)[0]
        cue_name = id_to_name[cue_id]

        groups = {"retain": [], "forget": [], "forget_neg": [], "both_pos": []}
        for iid, cats_present in image_to_cats.items():
            C = int(target_id in cats_present)
            S = int(cue_id in cats_present)

            if C == 1 and S == 0:
                groups["retain"].append(iid)      # C ∩ S^c
            elif C == 0 and S == 1:
                groups["forget"].append(iid)      # C^c ∩ S
            elif C == 0 and S == 0:
                groups["forget_neg"].append(iid)  # C^c ∩ S^c
            else:
                groups["both_pos"].append(iid)    # C ∩ S

        for g in groups:
            random.Random(args.seed + ci + len(g)).shuffle(groups[g])

        for split_name, n in [("train", args.train_per_group), ("test", args.test_per_group)]:
            for group_name in ["retain", "forget", "forget_neg", "both_pos"]:
                ids = groups[group_name]
                if split_name == "train":
                    selected = ids[:args.train_per_group]
                else:
                    selected = ids[args.train_per_group:args.train_per_group + args.test_per_group]

                out_dir = os.path.join(args.output_root, "images", split_name, group_name, target_name)
                ensure_dir(out_dir)

                for iid in selected:
                    info = coco.loadImgs([iid])[0]
                    img_path = os.path.join(args.coco_image_root, info["file_name"])
                    if not os.path.exists(img_path):
                        continue
                    img = Image.open(img_path).convert("RGB")
                    base = os.path.splitext(os.path.basename(info["file_name"]))[0]
                    out_name = f"{base}_{target_name}_{cue_name}.png"
                    save_image(img, os.path.join(out_dir, out_name), args.resize)

        summary.append({
            "class_name": target_name,
            "cue_name": cue_name,
            "retain_train": min(len(groups["retain"]), args.train_per_group),
            "forget_train": min(len(groups["forget"]), args.train_per_group),
            "forget_neg_train": min(len(groups["forget_neg"]), args.train_per_group),
            "both_pos_train": min(len(groups["both_pos"]), args.train_per_group),
        })

        print(f"[CORD][{target_name}] cue={cue_name} retain={len(groups['retain'])} forget={len(groups['forget'])} forget_neg={len(groups['forget_neg'])} both_pos={len(groups['both_pos'])}")

    with open(os.path.join(args.output_root, "method3_coco_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
