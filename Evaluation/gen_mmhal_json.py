# gen_mmhal_json.py
import os
import json
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


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_jsonl", required=True,
                    help="MMHal JSONL file")
    ap.add_argument("--image_root", required=True,
                    help="folder containing MMHal images")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    rows = load_jsonl(args.questions_jsonl)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_vlm(args.model_path, args.dtype).to(args.device).eval()

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            image_name = row["image"]
            question = row["question"]
            image_path = os.path.join(args.image_root, image_name)

            pred = generate_answer(
                model=model,
                processor=processor,
                image_path=image_path,
                prompt=question,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )

            out = {
                "image": image_name,
                "question": question,
                "category": row.get("category", "other"),
                "reference": row.get("reference", ""),
                "answer": pred,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\\n")

            if (i + 1) % 50 == 0:
                print(f"[MMHal gen] {i+1}/{len(rows)}")

    print(f"saved: {args.output_jsonl}")


if __name__ == "__main__":
    main()
