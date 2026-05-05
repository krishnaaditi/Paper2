#!/usr/bin/env python3
import os
import re
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

import torch
from transformers import AutoProcessor


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


def whitebox_remove(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    out = img_np.copy()
    out[y1:y2, x1:x2] = 255
    return out


def get_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def load_vlm(model_path: str, dtype_name: str):
    dtype = get_dtype(dtype_name)
    last_err = None
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
        )
    except Exception as e:
        last_err = e
    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
        )
    except Exception as e:
        last_err = e
    raise RuntimeError(f"Could not load model: {last_err}")


def parse_yes_no(text: str):
    text = text.strip()
    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if m is None:
        return "unknown", text
    return m.group(1).lower(), text


@torch.no_grad()
def ask_yes_no(model, processor, image: Image.Image, class_name: str, device: str, max_new_tokens: int = 8):
    prompt = f"Is there a {class_name} in the image? Answer only yes or no."
    try:
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    except Exception:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else None
    gen_ids = output_ids[0][prompt_len:] if prompt_len is not None else output_ids[0]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return parse_yes_no(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_image_root", required=True)
    ap.add_argument("--coco_ann_json", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_per_class", type=int, default=100)
    ap.add_argument("--test_per_class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--class_names", type=str, default="")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--use_edited_view", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.output_root)
    random.seed(args.seed)

    coco = COCO(args.coco_ann_json)
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {c["name"]: c["id"] for c in cats}
    all_img_ids = coco.getImgIds()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_vlm(args.model_path, args.dtype).to(args.device).eval()

    classes = COCO80_CLASSES
    if args.class_names:
        wanted = {x.strip() for x in args.class_names.split(",") if x.strip()}
        classes = [c for c in COCO80_CLASSES if c in wanted]

    for ci, class_name in enumerate(classes):
        if class_name not in name_to_id:
            continue

        cat_id = name_to_id[class_name]
        pos_ids = list(set(coco.getImgIds(catIds=[cat_id])))
        neg_ids = [iid for iid in all_img_ids if iid not in set(pos_ids)]

        random.Random(args.seed + ci).shuffle(pos_ids)
        random.Random(args.seed + 1000 + ci).shuffle(neg_ids)

        need = args.train_per_class + args.test_per_class
        pos_ids = pos_ids[:need]
        neg_ids = neg_ids[:need]

        meta = []

        for split_name, start, end in [
            ("train", 0, args.train_per_class),
            ("test", args.train_per_class, args.train_per_class + args.test_per_class),
        ]:
            # retain/correct
            retain_yes_dir = os.path.join(args.output_root, "images", split_name, "retain", "original_yes", class_name)
            retain_no_dir  = os.path.join(args.output_root, "images", split_name, "retain", "original_no", class_name)
            # forget/failure
            forget_yes_dir = os.path.join(args.output_root, "images", split_name, "forget", "original_yes", class_name)  # gt=1,pred=0
            forget_no_dir  = os.path.join(args.output_root, "images", split_name, "forget", "original_no", class_name)   # gt=0,pred=1
            for d in [retain_yes_dir, retain_no_dir, forget_yes_dir, forget_no_dir]:
                ensure_dir(d)

            retain_n, forget_n = 0, 0

            entries = []
            for iid in pos_ids[start:end]:
                info = coco.loadImgs([iid])[0]
                img_path = os.path.join(args.coco_image_root, info["file_name"])
                anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False))
                bbox = None
                if os.path.exists(img_path):
                    w, h = Image.open(img_path).convert("RGB").size
                    bbox = union_bbox_from_anns(anns, w, h) if anns else None
                entries.append({"img_path": img_path, "gt": 1, "bbox": bbox})

            for iid in neg_ids[start:end]:
                info = coco.loadImgs([iid])[0]
                img_path = os.path.join(args.coco_image_root, info["file_name"])
                entries.append({"img_path": img_path, "gt": 0, "bbox": None})

            for item in entries:
                img_path = item["img_path"]
                gt = item["gt"]
                bbox = item["bbox"]
                if not os.path.exists(img_path):
                    continue

                image = Image.open(img_path).convert("RGB")
                query_image = image

                if args.use_edited_view and gt == 1 and bbox is not None:
                    img_np = np.array(image)
                    x1, y1, x2, y2 = bbox
                    query_image = Image.fromarray(whitebox_remove(img_np, x1, y1, x2, y2))

                pred_str, raw = ask_yes_no(model, processor, query_image, class_name, args.device, args.max_new_tokens)
                pred = 1 if pred_str == "yes" else 0

                if gt == 1 and pred == 1:
                    out_dir = retain_yes_dir
                    split_group = "retain/original_yes"
                    retain_n += 1
                elif gt == 0 and pred == 0:
                    out_dir = retain_no_dir
                    split_group = "retain/original_no"
                    retain_n += 1
                elif gt == 1 and pred == 0:
                    out_dir = forget_yes_dir
                    split_group = "forget/original_yes"
                    forget_n += 1
                else:  # gt == 0 and pred == 1
                    out_dir = forget_no_dir
                    split_group = "forget/original_no"
                    forget_n += 1

                out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{class_name}_gt{gt}_pred{pred}.png"
                query_image.save(os.path.join(out_dir, out_name))

                meta.append({
                    "split": split_name,
                    "group": split_group,
                    "file": out_name,
                    "class_name": class_name,
                    "gt": gt,
                    "pred": pred,
                    "raw_text": raw,
                    "source_path": img_path,
                })

            print(f"[PROBE][{split_name}][{class_name}] retain={retain_n} forget={forget_n}")

        with open(os.path.join(args.output_root, f"{class_name}_probe_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
