#!/usr/bin/env python3
import os
import csv
import re
import json
import random
import argparse
from typing import Dict, List

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    return x1, y1, x2, y2


def parse_bbox_from_row(row: Dict, bbox_mode: str):
    if bbox_mode == "xywh":
        x = int(float(row["x"]))
        y = int(float(row["y"]))
        w = int(float(row["w"]))
        h = int(float(row["h"]))
        return x, y, x + w, y + h
    elif bbox_mode == "xyxy":
        x1 = int(float(row["x1"]))
        y1 = int(float(row["y1"]))
        x2 = int(float(row["x2"]))
        y2 = int(float(row["y2"]))
        return x1, y1, x2, y2
    else:
        raise ValueError(f"Unknown bbox_mode={bbox_mode}")


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
    ap.add_argument("--waterbirds_image_root", required=True)
    ap.add_argument("--waterbirds_csv", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_per_class", type=int, default=100)
    ap.add_argument("--test_per_class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--use_edited_view", action="store_true")

    ap.add_argument("--wb_image_col", type=str, default="img_filename")
    ap.add_argument("--wb_label_col", type=str, default="y")
    ap.add_argument("--wb_bbox_mode", choices=["xywh", "xyxy"], default="xywh")
    ap.add_argument("--class_names", type=str, default="landbird,waterbird")
    args = ap.parse_args()

    ensure_dir(args.output_root)
    random.seed(args.seed)

    rows = read_csv_rows(args.waterbirds_csv)
    labels = sorted({str(r[args.wb_label_col]) for r in rows})
    names = [x.strip() for x in args.class_names.split(",") if x.strip()]
    label_to_name = {lab: names[i] if i < len(names) else f"class_{lab}" for i, lab in enumerate(labels)}

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_vlm(args.model_path, args.dtype).to(args.device).eval()

    meta = []

    for i, lab in enumerate(labels):
        class_name = label_to_name[lab]
        pos_rows = [r for r in rows if str(r[args.wb_label_col]) == lab]
        neg_rows = [r for r in rows if str(r[args.wb_label_col]) != lab]

        random.Random(args.seed + i).shuffle(pos_rows)
        random.Random(args.seed + 1000 + i).shuffle(neg_rows)

        need = args.train_per_class + args.test_per_class
        pos_rows = pos_rows[:need]
        neg_rows = neg_rows[:need]

        for split_name, start, end in [
            ("train", 0, args.train_per_class),
            ("test", args.train_per_class, args.train_per_class + args.test_per_class),
        ]:
            retain_yes_dir = os.path.join(args.output_root, "images", split_name, "retain", "original_yes", class_name)
            retain_no_dir  = os.path.join(args.output_root, "images", split_name, "retain", "original_no", class_name)
            forget_yes_dir = os.path.join(args.output_root, "images", split_name, "forget", "original_yes", class_name)
            forget_no_dir  = os.path.join(args.output_root, "images", split_name, "forget", "original_no", class_name)

            for d in [retain_yes_dir, retain_no_dir, forget_yes_dir, forget_no_dir]:
                ensure_dir(d)

            retain_n, forget_n = 0, 0

            entries = []
            for row in pos_rows[start:end]:
                img_path = os.path.join(args.waterbirds_image_root, row[args.wb_image_col])
                entries.append({"img_path": img_path, "gt": 1, "row": row})
            for row in neg_rows[start:end]:
                img_path = os.path.join(args.waterbirds_image_root, row[args.wb_image_col])
                entries.append({"img_path": img_path, "gt": 0, "row": row})

            for item in entries:
                img_path = item["img_path"]
                gt = item["gt"]
                row = item["row"]

                if not os.path.exists(img_path):
                    continue

                image = Image.open(img_path).convert("RGB")
                query_image = image

                if args.use_edited_view and gt == 1:
                    img_np = np.array(image)
                    h, w = img_np.shape[:2]
                    x1, y1, x2, y2 = parse_bbox_from_row(row, args.wb_bbox_mode)
                    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
                    if x2 > x1 and y2 > y1:
                        query_image = Image.fromarray(whitebox_remove(img_np, x1, y1, x2, y2))

                pred_str, raw = ask_yes_no(model, processor, query_image, class_name, args.device, args.max_new_tokens)
                pred = 1 if pred_str == "yes" else 0

                if gt == 1 and pred == 1:
                    out_dir = retain_yes_dir
                    group = "retain/original_yes"
                    retain_n += 1
                elif gt == 0 and pred == 0:
                    out_dir = retain_no_dir
                    group = "retain/original_no"
                    retain_n += 1
                elif gt == 1 and pred == 0:
                    out_dir = forget_yes_dir
                    group = "forget/original_yes"
                    forget_n += 1
                else:
                    out_dir = forget_no_dir
                    group = "forget/original_no"
                    forget_n += 1

                out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{class_name}_gt{gt}_pred{pred}.png"
                query_image.save(os.path.join(out_dir, out_name))

                meta.append({
                    "split": split_name,
                    "group": group,
                    "file": out_name,
                    "class_name": class_name,
                    "gt": gt,
                    "pred": pred,
                    "raw_text": raw,
                    "source_path": img_path,
                })

            print(f"[PROBE][{split_name}][{class_name}] retain={retain_n} forget={forget_n}")

    with open(os.path.join(args.output_root, "waterbirds_probe_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
