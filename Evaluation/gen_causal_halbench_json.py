# gen_causal_halbench_json.py
import os
import json
import re
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_vlm(model_path: str, dtype_name: str = "bf16"):
    dtype = get_dtype(dtype_name)
    errors = []

    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("[load] AutoModelForImageTextToText")
        return model
    except Exception as e:
        errors.append(f"AutoModelForImageTextToText: {repr(e)}")

    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("[load] AutoModelForVision2Seq")
        return model
    except Exception as e:
        errors.append(f"AutoModelForVision2Seq: {repr(e)}")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("[load] Qwen2_5_VLForConditionalGeneration")
        return model
    except Exception as e:
        errors.append(f"Qwen2_5_VLForConditionalGeneration: {repr(e)}")

    raise RuntimeError("Could not load model:\\n" + "\\n".join(errors))


def parse_yes_no(text: str):
    m = re.search(r"\\b(yes|no)\\b", text, flags=re.I)
    return m.group(1).lower() if m else "unknown"


def build_inputs(processor, image, prompt):
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return processor(text=[text], images=[image], return_tensors="pt")
    except Exception:
        return processor(text=[prompt], images=[image], return_tensors="pt")


@torch.no_grad()
def generate_answer(model, processor, image_path, prompt, device, max_new_tokens):
    image = Image.open(image_path).convert("RGB")
    inputs = build_inputs(processor, image, prompt)
    inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else None
    gen_ids = output_ids[0][prompt_len:] if prompt_len is not None else output_ids[0]
    text = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def resolve_image_path(image_root, image_name):
    p = os.path.join(image_root, image_name)
    if os.path.exists(p):
        return p

    # common fallback for COCO val2014 nested folder
    p2 = os.path.join(image_root, "val2014", image_name)
    if os.path.exists(p2):
        return p2

    raise FileNotFoundError(f"Image not found: {image_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_json", required=True, help="qa.json from Causal-HalBench")
    ap.add_argument("--image_root", required=True, help="root folder containing images")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    qa_rows = json.load(open(args.qa_json, "r", encoding="utf-8"))

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_vlm(args.model_path, args.dtype).to(args.device).eval()

    outputs = []

    for i, row in enumerate(qa_rows):
        image_name = row["image_name"]
        question = row["question"]
        image_path = resolve_image_path(args.image_root, image_name)

        raw = generate_answer(
            model=model,
            processor=processor,
            image_path=image_path,
            prompt=question,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )

        outputs.append({
            "image_name": image_name,
            "type": row["type"],     # target / absent
            "answer": parse_yes_no(raw),
            "tag": row["tag"],       # origin / edited / inpainted etc
            "id": row["id"],
            "raw_output": raw
        })

        if (i + 1) % 100 == 0:
            print(f"[Causal-HalBench gen] {i+1}/{len(qa_rows)}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(outputs, open(args.output_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
