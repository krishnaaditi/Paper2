import json
import re
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
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

    raise RuntimeError("Could not load model:\n" + "\n".join(errors))


def build_prompt(question: str, choices):
    labels = ["A", "B", "C", "D"]

    if isinstance(choices, dict):
        ordered = []
        for k in labels:
            ordered.append(str(choices.get(k, choices.get(k.lower(), ""))))
        choices = ordered

    if not isinstance(choices, list):
        raise ValueError(f"Unsupported choices format: {type(choices)}")

    prompt = (
        "Answer the multiple-choice visual question.\n"
        "Reply with only one letter: A, B, C, or D.\n\n"
    )
    prompt += f"Question: {question}\n"
    prompt += "Choices:\n"
    for i, c in enumerate(choices[:4]):
        prompt += f"{labels[i]}. {c}\n"
    prompt += "\nAnswer:"
    return prompt


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
def generate_answer(model, processor, image, prompt, device, max_new_tokens):
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


def extract_choice_letter(text: str):
    text = text.strip()
    m = re.search(r"\b([ABCD])\b", text, flags=re.I)
    if m:
        return m.group(1).upper()

    # fallback for outputs like "A." or "(B)"
    m = re.search(r"^[\s\(\[]*([ABCD])[\s\)\].:,-]*$", text, flags=re.I)
    if m:
        return m.group(1).upper()

    return "UNK"


def get_field(sample, keys, default=None):
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    args = ap.parse_args()

    ds = load_dataset("mmbench/MM-SpuBench", split=args.split)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_vlm(args.model_path, args.dtype).to(args.device).eval()

    outputs = []
    n = len(ds)

    for i, sample in enumerate(ds):
        image = get_field(sample, ["image"])
        question = get_field(sample, ["question", "text"])
        choices = get_field(sample, ["choices", "options"])
        answer = get_field(sample, ["answer", "label"])
        category = get_field(
            sample,
            ["type", "category", "spurious_type"],
            default="unknown"
        )

        prompt = build_prompt(question, choices)
        raw_output = generate_answer(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        pred_letter = extract_choice_letter(raw_output)

        outputs.append({
            "id": i,
            "question": question,
            "choices": choices,
            "gt": str(answer).strip(),
            "pred": pred_letter,
            "raw_output": raw_output,
            "category": category,
        })

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"[MM-SpuBench gen] {i+1}/{n}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
