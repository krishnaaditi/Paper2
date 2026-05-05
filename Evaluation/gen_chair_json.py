# gen_chair_json.py
import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor


def get_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_vlm(model_path, dtype_name):
    dtype = get_dtype(dtype_name)

    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )


def build_inputs(processor, image, prompt):
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=[image], return_tensors="pt")
    except:
        return processor(text=[prompt], images=[image], return_tensors="pt")


@torch.no_grad()
def generate(model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")

    prompt = "Describe the image in one sentence."

    inputs = build_inputs(processor, image, prompt)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]

    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--instances_json", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    coco = COCO(args.instances_json)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_vlm(args.model_path, args.dtype).to(args.device).eval()

    outputs = []
    img_ids = coco.getImgIds()

    for i, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info["file_name"]
        img_path = os.path.join(args.image_root, img_name)

        caption = generate(model, processor, img_path, args.device)

        outputs.append({
            "image_id": img_id,
            "image": img_name,
            "caption": caption
        })

        if (i + 1) % 100 == 0:
            print(f"[CHAIR gen] {i+1}/{len(img_ids)}")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(outputs, open(args.output_json, "w"), indent=2)


if __name__ == "__main__":
    main()
