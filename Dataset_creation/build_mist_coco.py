#!/usr/bin/env python3
import os
import random
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
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


def save_np_image(arr: np.ndarray, path: str, resize: Optional[int] = None):
    ensure_dir(os.path.dirname(path))
    img = Image.fromarray(arr.astype(np.uint8))
    if resize is not None and resize > 0:
        img = img.resize((resize, resize), Image.Resampling.BICUBIC)
    img.save(path)


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    return x1, y1, x2, y2


def bbox_xyxy_from_coco(bbox):
    x, y, w, h = bbox
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


def union_bbox_from_anns(anns: List[Dict], img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    xs1, ys1, xs2, ys2 = [], [], [], []
    for ann in anns:
        x1, y1, x2, y2 = bbox_xyxy_from_coco(ann["bbox"])
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, img_w, img_h)
        if x2 > x1 and y2 > y1:
            xs1.append(x1)
            ys1.append(y1)
            xs2.append(x2)
            ys2.append(y2)
    if not xs1:
        return None
    return min(xs1), min(ys1), max(xs2), max(ys2)


def build_retain_white(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    out = np.full_like(img_np, 255)
    out[y1:y2, x1:x2] = img_np[y1:y2, x1:x2]
    return out


def build_forget_whitebox(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    out = img_np.copy()
    out[y1:y2, x1:x2] = 255
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_image_root", required=True)
    ap.add_argument("--coco_ann_json", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--samples_per_class", type=int, default=100)
    ap.add_argument("--resize", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_names", type=str, default="")
    ap.add_argument("--min_bbox_w", type=int, default=12)
    ap.add_argument("--min_bbox_h", type=int, default=12)
    args = ap.parse_args()

    random.seed(args.seed)
    ensure_dir(args.output_root)

    coco = COCO(args.coco_ann_json)
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {c["name"]: c["id"] for c in cats}
    all_img_ids = coco.getImgIds()

    classes = COCO80_CLASSES
    if args.class_names:
        wanted = {x.strip() for x in args.class_names.split(",") if x.strip()}
        classes = [c for c in COCO80_CLASSES if c in wanted]

    for ci, class_name in enumerate(classes):
        if class_name not in name_to_id:
            print(f"[skip] {class_name} not found")
            continue

        cat_id = name_to_id[class_name]
        ann_ids = coco.getAnnIds(catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        anns = [a for a in anns if a["bbox"][2] >= args.min_bbox_w and a["bbox"][3] >= args.min_bbox_h]

        img_to_anns = {}
        for ann in anns:
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

        pos_ids = list(img_to_anns.keys())
        target_img_ids = set(coco.getImgIds(catIds=[cat_id]))
        neg_ids = [iid for iid in all_img_ids if iid not in target_img_ids]

        random.Random(args.seed + ci).shuffle(pos_ids)
        random.Random(args.seed + 1000 + ci).shuffle(neg_ids)
        pos_ids = pos_ids[:args.samples_per_class]
        neg_ids = neg_ids[:args.samples_per_class]

        retain_dir = os.path.join(args.output_root, "retain", class_name)
        forget_dir = os.path.join(args.output_root, "forget", class_name)
        forget_neg_dir = os.path.join(args.output_root, "forget_neg", class_name)
        ensure_dir(retain_dir)
        ensure_dir(forget_dir)
        ensure_dir(forget_neg_dir)

        saved_pos, saved_neg = 0, 0

        for iid in pos_ids:
            img_info = coco.loadImgs([iid])[0]
            img_path = os.path.join(args.coco_image_root, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            img_np = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_np.shape[:2]
            bbox = union_bbox_from_anns(img_to_anns[iid], w, h)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox

            retain = build_retain_white(img_np, x1, y1, x2, y2)
            forget = build_forget_whitebox(img_np, x1, y1, x2, y2)

            base = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
            name = f"{base}_{class_name}.png"
            save_np_image(retain, os.path.join(retain_dir, name), args.resize)
            save_np_image(forget, os.path.join(forget_dir, name), args.resize)
            saved_pos += 1

        for iid in neg_ids:
            img_info = coco.loadImgs([iid])[0]
            img_path = os.path.join(args.coco_image_root, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            img_np = np.array(Image.open(img_path).convert("RGB"))
            base = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
            name = f"{base}_{class_name}.png"
            save_np_image(img_np, os.path.join(forget_neg_dir, name), args.resize)
            saved_neg += 1

        print(f"[MIST][{class_name}] retain={saved_pos} forget={saved_pos} forget_neg={saved_neg}")


if __name__ == "__main__":
    main()
