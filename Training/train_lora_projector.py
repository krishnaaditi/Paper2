#!/usr/bin/env python3
import os
import glob
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_valid_images(folder: str) -> List[str]:
    paths = []
    if not os.path.isdir(folder):
        return paths
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted([p for p in paths if os.path.exists(p)])


@dataclass
class Sample:
    image_path: str
    class_name: str
    question: str
    answer: str
    split_type: str


class SplitDataset(Dataset):
    def __init__(self, data_root: str, split_type: str, max_per_class: int = 100000):
        assert split_type in {"retain", "forget", "forget_neg"}
        self.samples: List[Sample] = []

        base_dir = os.path.join(data_root, split_type)
        if not os.path.isdir(base_dir):
            print(f"[dataset:{split_type}] missing: {base_dir}")
            return

        answer = "yes" if split_type == "retain" else "no"

        for root, _, _ in os.walk(base_dir):
            rel = os.path.relpath(root, base_dir)
            if rel == ".":
                continue

            imgs = list_valid_images(root)
            if not imgs:
                continue

            class_name = os.path.basename(root)
            question = f"Is there a {class_name.replace('_', ' ')} in this image? Answer only yes or no."

            for img_path in imgs[:max_per_class]:
                self.samples.append(
                    Sample(
                        image_path=img_path,
                        class_name=class_name,
                        question=question,
                        answer=answer,
                        split_type=split_type,
                    )
                )

        print(f"[dataset:{split_type}] total={len(self.samples)} label={answer}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        return {
            "image": image,
            "image_path": s.image_path,
            "class_name": s.class_name,
            "question": s.question,
            "answer": s.answer,
            "split_type": s.split_type,
        }


def single_collate(batch):
    assert len(batch) == 1
    return batch[0]


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def get_torch_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_vlm(model_path: str, dtype):
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


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.dtype = get_torch_dtype(args.dtype)

        self.processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = load_vlm(args.model_path, self.dtype)

        for p in self.model.parameters():
            p.requires_grad = False

        self._apply_lora()

        if args.train_projector:
            self._unfreeze_projector()

        self.model.to(self.device)
        self.model.train()

        if args.grad_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

        self.ref_model = None
        if args.algo in {"kl", "npo"}:
            self.ref_model = load_vlm(args.model_path, self.dtype).to(self.device)
            self.ref_model.eval()
            if hasattr(self.ref_model, "config"):
                self.ref_model.config.use_cache = False
            for p in self.ref_model.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[model] trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.4f}%)")

    def _apply_lora(self):
        lora_cfg = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            inference_mode=False,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        print("[lora] applied to q_proj/k_proj/v_proj/o_proj")

    def _unfreeze_projector(self):
        hit = 0
        keys = [
            "projector",
            "merger",
            "multi_modal_projector",
            "visual_projection",
            "vision_projector",
            "vision_proj",
        ]
        for n, p in self.model.named_parameters():
            if any(k in n.lower() for k in keys):
                p.requires_grad = True
                hit += 1
        print(f"[projector] trainable params matched: {hit}")
        if hit == 0:
            print("[projector] no projector found, continuing with LoRA-only")

    def _build_messages(self, image, question: str, answer: Optional[str] = None):
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]
        if answer is None:
            return [{"role": "user", "content": user_content}]
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

    def _prepare_batch(self, image, question: str, answer: str):
        full_msgs = self._build_messages(image, question, answer)
        prompt_msgs = self._build_messages(image, question, None)

        full = self.processor.apply_chat_template(
            full_msgs,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        full.pop("token_type_ids", None)
        prompt.pop("token_type_ids", None)

        labels = full["input_ids"].clone()
        prompt_len = prompt["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        full["labels"] = labels

        batch = {}
        for k, v in full.items():
            batch[k] = v.to(self.device) if torch.is_tensor(v) else v
        return batch

    def _forward(self, model, batch, with_labels=False):
        kwargs = {}
        for key in [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
            "image_sizes",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "cross_attention_mask",
            "labels",
        ]:
            if key in batch and (with_labels or key != "labels"):
                kwargs[key] = batch[key]
        return model(**kwargs)

    def ce_loss(self, model, image, question: str, answer: str):
        batch = self._prepare_batch(image, question, answer)
        out = self._forward(model, batch, with_labels=True)
        return out.loss

    def seq_logprob(self, model, image, question: str, answer: str):
        batch = self._prepare_batch(image, question, answer)
        out = self._forward(model, batch, with_labels=False)

        logits = out.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        mask = labels != -100

        safe_labels = labels.clone()
        safe_labels[~mask] = 0

        log_probs = F.log_softmax(logits, dim=-1)
        tok_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        seq_lp = (tok_lp * mask).sum() / mask.sum().clamp_min(1)
        return seq_lp

    def retain_kl(self, image, question: str, answer: str):
        batch = self._prepare_batch(image, question, answer)
        out_s = self._forward(self.model, batch, with_labels=False)
        with torch.no_grad():
            out_t = self._forward(self.ref_model, batch, with_labels=False)

        p = F.log_softmax(out_s.logits, dim=-1)
        q = F.softmax(out_t.logits, dim=-1)
        return F.kl_div(p, q, reduction="batchmean")

    def loss_ga(self, r, f, fn=None):
        retain_ce = self.ce_loss(self.model, r["image"], r["question"], "yes")
        forget_ce = self.ce_loss(self.model, f["image"], f["question"], "no")
        loss = retain_ce + self.args.lambda_forget * forget_ce

        logs = {
            "retain_ce": float(retain_ce.item()),
            "forget_ce": float(forget_ce.item()),
        }

        if fn is not None and self.args.use_forget_neg:
            neg_ce = self.ce_loss(self.model, fn["image"], fn["question"], "no")
            loss = loss + self.args.gamma_neg * neg_ce
            logs["forget_neg_ce"] = float(neg_ce.item())

        return loss, logs

    def loss_kl(self, r, f, fn=None):
        forget_ce = self.ce_loss(self.model, f["image"], f["question"], "no")
        keep_kl = self.retain_kl(r["image"], r["question"], "yes")
        loss = self.args.lambda_forget * forget_ce + self.args.beta_kl * keep_kl

        logs = {
            "forget_ce": float(forget_ce.item()),
            "retain_kl": float(keep_kl.item()),
        }

        if fn is not None and self.args.use_forget_neg:
            neg_ce = self.ce_loss(self.model, fn["image"], fn["question"], "no")
            loss = loss + self.args.gamma_neg * neg_ce
            logs["forget_neg_ce"] = float(neg_ce.item())

        return loss, logs

    def loss_npo(self, r, f, fn=None):
        retain_ce = self.ce_loss(self.model, r["image"], r["question"], "yes")

        lp_cur = self.seq_logprob(self.model, f["image"], f["question"], "no")
        with torch.no_grad():
            lp_ref = self.seq_logprob(self.ref_model, f["image"], f["question"], "no")

        beta = self.args.beta_npo
        npo_term = torch.log1p(torch.exp(beta * (lp_cur - lp_ref))).mean() / beta
        loss = retain_ce + self.args.lambda_forget * npo_term

        logs = {
            "retain_ce": float(retain_ce.item()),
            "npo_term": float(npo_term.item()),
        }

        if fn is not None and self.args.use_forget_neg:
            neg_ce = self.ce_loss(self.model, fn["image"], fn["question"], "no")
            loss = loss + self.args.gamma_neg * neg_ce
            logs["forget_neg_ce"] = float(neg_ce.item())

        return loss, logs

    def save_checkpoint(self, step: int):
        ckpt_dir = os.path.join(self.args.output_dir, f"step_{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.processor.save_pretrained(ckpt_dir)

        with open(os.path.join(ckpt_dir, "train_meta.json"), "w") as f:
            json.dump(
                {
                    "step": step,
                    "algo": self.args.algo,
                    "train_projector": self.args.train_projector,
                    "model_path": self.args.model_path,
                    "data_root": self.args.data_root,
                },
                f,
                indent=2,
            )
        print(f"[save] {ckpt_dir}")

    def train(self):
        retain_ds = SplitDataset(self.args.data_root, "retain", self.args.max_per_class)
        forget_ds = SplitDataset(self.args.data_root, "forget", self.args.max_per_class)
        forget_neg_ds = SplitDataset(self.args.data_root, "forget_neg", self.args.max_per_class)

        if len(retain_ds) == 0 or len(forget_ds) == 0:
            raise RuntimeError("retain and forget must both be non-empty")

        retain_loader = DataLoader(
            retain_ds, batch_size=1, shuffle=True,
            num_workers=self.args.num_workers, collate_fn=single_collate, pin_memory=True
        )
        forget_loader = DataLoader(
            forget_ds, batch_size=1, shuffle=True,
            num_workers=self.args.num_workers, collate_fn=single_collate, pin_memory=True
        )
        forget_neg_loader = None
        if len(forget_neg_ds) > 0:
            forget_neg_loader = DataLoader(
                forget_neg_ds, batch_size=1, shuffle=True,
                num_workers=self.args.num_workers, collate_fn=single_collate, pin_memory=True
            )

        retain_iter = cycle_loader(retain_loader)
        forget_iter = cycle_loader(forget_loader)
        forget_neg_iter = cycle_loader(forget_neg_loader) if forget_neg_loader is not None else None

        lora_params, proj_params = [], []
        proj_keys = [
            "projector",
            "merger",
            "multi_modal_projector",
            "visual_projection",
            "vision_projector",
            "vision_proj",
        ]

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in n:
                lora_params.append(p)
            elif any(k in n.lower() for k in proj_keys):
                proj_params.append(p)

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": self.args.lr_lora})
        if proj_params:
            param_groups.append({"params": proj_params, "lr": self.args.lr_projector})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)

        total_steps = self.args.epochs * self.args.steps_per_epoch
        os.makedirs(self.args.output_dir, exist_ok=True)

        print(f"[train] retain={len(retain_ds)} forget={len(forget_ds)} forget_neg={len(forget_neg_ds)}")
        print(f"[train] epochs={self.args.epochs} steps_per_epoch={self.args.steps_per_epoch} total_steps={total_steps}")

        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for step in range(1, total_steps + 1):
            r = next(retain_iter)
            f = next(forget_iter)
            fn = next(forget_neg_iter) if forget_neg_iter is not None else None

            if self.args.algo == "ga":
                loss, logs = self.loss_ga(r, f, fn)
            elif self.args.algo == "kl":
                loss, logs = self.loss_kl(r, f, fn)
            elif self.args.algo == "npo":
                loss, logs = self.loss_npo(r, f, fn)
            else:
                raise ValueError(self.args.algo)

            (loss / self.args.grad_accum).backward()

            if step % self.args.grad_accum == 0:
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.args.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % self.args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                step_time = elapsed / step
                eta_min = (total_steps - step) * step_time / 60.0

                msg = f"[{self.args.run_name} {step}/{total_steps}] loss={float(loss.detach().cpu()):.6f}"
                for k, v in logs.items():
                    msg += f" {k}={v:.6f}"
                msg += f" step_time={step_time:.2f}s eta={eta_min:.1f}m"
                print(msg, flush=True)

            if step % self.args.save_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(total_steps)
        print("[done]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--run_name", type=str, default="pau")

    ap.add_argument("--algo", choices=["ga", "kl", "npo"], required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_per_class", type=int, default=100000)

    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lr_lora", type=float, default=2e-5)
    ap.add_argument("--lr_projector", type=float, default=1e-5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--lambda_forget", type=float, default=1.0)
    ap.add_argument("--beta_kl", type=float, default=0.1)
    ap.add_argument("--beta_npo", type=float, default=0.1)
    ap.add_argument("--gamma_neg", type=float, default=1.0)

    ap.add_argument("--use_forget_neg", action="store_true")
    ap.add_argument("--train_projector", action="store_true")
    ap.add_argument("--grad_checkpointing", action="store_true")

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
