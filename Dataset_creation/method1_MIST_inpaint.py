import os
import random
import argparse
import warnings
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
from pycocotools.coco import COCO
from diffusers import StableDiffusionInpaintPipeline


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_image_root",
        type=str,
        default="/data/Aditi/playground/datasets/coco/train2017",
    )
    parser.add_argument(
        "--coco_ann_json",
        type=str,
        default="/data/Aditi/playground/datasets/coco/annotations/instances_train2017.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/Aditi/playground/datasets/method_inpaint_coco80",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
    )
    parser.add_argument(
        "--start_class_idx",
        type=int,
        default=0,
        help="useful for multi-GPU split",
    )
    parser.add_argument(
        "--end_class_idx",
        type=int,
        default=-1,
        help="exclusive end index; -1 means till end",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--mask_blur",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--mask_expand",
        type=int,
        default=8,
        help="expand mask by repeated MaxFilter",
    )
    parser.add_argument(
        "--retain_prompt",
        type=str,
        default="clean realistic natural background, photorealistic, coherent scene",
    )
    parser.add_argument(
        "--forget_prompt",
        type=str,
        default="clean realistic background with no foreground object, photorealistic, coherent scene",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_pipe(model_id: str, device: str, dtype_str: str):
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # safer memory settings
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe


def resize_image_and_mask(image: Image.Image, mask: Image.Image, image_size: int):
    image = image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    mask = mask.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    return image, mask


def expand_and_blur_mask(mask: Image.Image, expand_iters: int = 8, blur_radius: float = 6.0):
    m = mask.convert("L")
    for _ in range(max(0, expand_iters)):
        m = m.filter(ImageFilter.MaxFilter(3))
    if blur_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return m


def ann_ids_for_class_in_image(coco: COCO, img_id: int, cat_id: int) -> List[int]:
    return coco.getAnnIds(imgIds=[img_id], catIds=[cat_id], iscrowd=None)


def build_union_mask(coco: COCO, anns: List[dict]) -> Optional[np.ndarray]:
    if len(anns) == 0:
        return None

    mask = None
    for ann in anns:
        try:
            m = coco.annToMask(ann)
        except Exception:
            continue
        if mask is None:
            mask = m.astype(np.uint8)
        else:
            mask = np.maximum(mask, m.astype(np.uint8))

    if mask is None:
        return None
    return (mask * 255).astype(np.uint8)


def save_image(img: Image.Image, path: str):
    img.save(path, quality=95)


def already_done(retain_path: str, forget_path: str):
    return os.path.exists(retain_path) and os.path.exists(forget_path)


@torch.inference_mode()
def run_inpaint(
    pipe: StableDiffusionInpaintPipeline,
    prompt: str,
    image: Image.Image,
    mask_image: Image.Image,
    num_inference_steps: int,
    guidance_scale: float,
):
    out = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]
    return out


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.output_root)

    print("Loading COCO...")
    coco = COCO(args.coco_ann_json)
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cats = sorted(cats, key=lambda x: x["id"])

    if args.end_class_idx == -1:
        args.end_class_idx = len(cats)

    cats = cats[args.start_class_idx:args.end_class_idx]

    print(f"Total selected classes: {len(cats)}")
    print("Loading inpainting pipeline...")
    pipe = load_pipe(args.model_id, args.device, args.dtype)

    global_count = 0

    for class_idx, cat in enumerate(cats):
        cat_id = cat["id"]
        cat_name = cat["name"]

        retain_dir = os.path.join(args.output_root, "retain", cat_name)
        forget_dir = os.path.join(args.output_root, "forget", cat_name)
        ensure_dir(retain_dir)
        ensure_dir(forget_dir)

        img_ids = coco.getImgIds(catIds=[cat_id])
        img_ids = list(img_ids)
        random.shuffle(img_ids)

        saved = 0

        print(f"\n[{class_idx + args.start_class_idx}/{args.end_class_idx - 1}] class={cat_name} total_candidates={len(img_ids)}")

        pbar = tqdm(img_ids, desc=f"{cat_name}", ncols=120)
        for img_id in pbar:
            if saved >= args.samples_per_class:
                break

            img_info = coco.loadImgs([img_id])[0]
            file_name = img_info["file_name"]
            src_path = os.path.join(args.coco_image_root, file_name)

            retain_out = os.path.join(retain_dir, file_name)
            forget_out = os.path.join(forget_dir, file_name)

            if not args.overwrite and already_done(retain_out, forget_out):
                saved += 1
                pbar.set_postfix(saved=saved)
                continue

            if not os.path.exists(src_path):
                continue

            ann_ids = ann_ids_for_class_in_image(coco, img_id, cat_id)
            anns = coco.loadAnns(ann_ids)
            if len(anns) == 0:
                continue

            mask_np = build_union_mask(coco, anns)
            if mask_np is None or mask_np.sum() == 0:
                continue

            try:
                image = Image.open(src_path).convert("RGB")
            except Exception:
                continue

            orig_w, orig_h = image.size
            if orig_w < 32 or orig_h < 32:
                continue

            object_mask = Image.fromarray(mask_np).convert("L")

            image, object_mask = resize_image_and_mask(image, object_mask, args.image_size)

            # smooth/expand boundary for more stable inpainting
            object_mask = expand_and_blur_mask(
                object_mask,
                expand_iters=args.mask_expand,
                blur_radius=args.mask_blur,
            )

            # retain: keep object, remove background
            # mask white area is inpainted, so background mask = inverse(object mask)
            obj_np = np.array(object_mask)
            bg_mask = Image.fromarray(255 - obj_np).convert("L")

            # forget: remove object, keep background
            fg_mask = object_mask

            try:
                retain_img = run_inpaint(
                    pipe=pipe,
                    prompt=args.retain_prompt,
                    image=image,
                    mask_image=bg_mask,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )

                forget_img = run_inpaint(
                    pipe=pipe,
                    prompt=args.forget_prompt,
                    image=image,
                    mask_image=fg_mask,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )

                save_image(retain_img, retain_out)
                save_image(forget_img, forget_out)

                saved += 1
                global_count += 1
                pbar.set_postfix(saved=saved, total_pairs=global_count)

            except RuntimeError as e:
                # skip OOM or bad generations and continue
                print(f"\n[warn] runtime error on {file_name}: {e}")
                if "out of memory" in str(e).lower():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue
            except Exception as e:
                print(f"\n[warn] failed on {file_name}: {e}")
                continue

        print(f"[done] {cat_name}: saved {saved}/{args.samples_per_class}")

    print("\nAll done.")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
