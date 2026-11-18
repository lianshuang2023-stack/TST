#!/usr/bin/env python3
"""Low-learning-rate UltraBERT fine-tuning with quarter-epoch validation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from run_modernbert import (
    EncodedArticles,
    collator,
    device_for,
    evaluate,
    load_records,
    load_tokenizer,
)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_bottom(model: Any, layers: int) -> dict[str, int]:
    if layers <= 0:
        return {"frozen_layers": 0, "frozen_parameters": 0, "trainable_parameters": sum(p.numel() for p in model.parameters())}
    encoder = getattr(model, "ultrabert", None)
    if encoder is None or not hasattr(encoder, "layers"):
        raise AttributeError("Expected model.ultrabert.layers for layer freezing")
    total_layers = len(encoder.layers)
    if layers >= total_layers:
        raise ValueError(f"freeze_bottom_layers must be below {total_layers}")
    for parameter in encoder.embed_tokens.parameters():
        parameter.requires_grad = False
    for layer in encoder.layers[:layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    return {
        "frozen_layers": layers,
        "total_layers": total_layers,
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def save_training_state(path: Path, optimizer: AdamW, scheduler: Any, event: dict[str, Any], seed: int) -> None:
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "event": event,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "seed": seed,
    }
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--freeze-bottom-layers", type=int, default=0)
    parser.add_argument("--quarter-patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every-steps", type=int, default=32)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = device_for(args.device)

    tokenizer = load_tokenizer(str(args.resume_from), None)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.resume_from, trust_remote_code=True
    ).to(device)
    freeze_info = freeze_bottom(model, args.freeze_bottom_layers)

    train_records = load_records(args.data_dir / "train.jsonl")
    validation_records = load_records(args.data_dir / "validation.jsonl")
    train_loader = DataLoader(
        EncodedArticles(train_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator(tokenizer),
    )
    validation_loader = DataLoader(
        EncodedArticles(validation_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator(tokenizer),
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_updates = math.ceil(len(train_loader) / args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    quarter_targets = [math.ceil(len(train_loader) * fraction / 4) for fraction in range(1, 5)]
    quarter_targets = [math.ceil(target / args.gradient_accumulation) * args.gradient_accumulation for target in quarter_targets]
    quarter_targets[-1] = len(train_loader)

    config = {
        "resume_from": str(args.resume_from.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "freeze_bottom_layers": args.freeze_bottom_layers,
        "quarter_patience": args.quarter_patience,
        "seed": args.seed,
        "device": str(device),
        "quarter_targets": quarter_targets,
        "input_policy": "cleaned headline + full body; excludes publisher, URL, author/byline, footer, topic/date/story metadata",
        **freeze_info,
    }
    save_json(args.output_dir / "config.json", config)

    baseline = evaluate(model, validation_loader, device)
    best_f1 = baseline["macro_f1"]
    best_event = {"fraction": 0.0, "step": 0, "train_loss": None, "validation": baseline}
    best_dir = args.output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    events = [best_event]
    save_json(args.output_dir / "history.json", {"evaluations": events})
    print(json.dumps({"status": "baseline", "macro_f1": best_f1, "accuracy": baseline["accuracy"]}), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    non_improving = 0
    started = time.perf_counter()
    next_quarter = 0
    stopped_early = False
    completed_step = 0

    for step, batch in enumerate(train_loader, start=1):
        completed_step = step
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        if not torch.isfinite(output.loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {output.loss.item()}")
        (output.loss / args.gradient_accumulation).backward()
        running_loss += float(output.loss.detach().cpu())
        optimizer_step = step % args.gradient_accumulation == 0 or step == len(train_loader)
        if optimizer_step:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_every_steps == 0 or step == len(train_loader):
            progress = {
                "phase": "training",
                "step": step,
                "steps_in_epoch": len(train_loader),
                "fraction": step / len(train_loader),
                "mean_loss": running_loss / step,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "best_validation_macro_f1": best_f1,
            }
            save_json(args.output_dir / "progress.json", progress)
            print(json.dumps(progress), flush=True)

        if optimizer_step and next_quarter < len(quarter_targets) and step >= quarter_targets[next_quarter]:
            validation = evaluate(model, validation_loader, device)
            event = {
                "fraction": (next_quarter + 1) / 4,
                "step": step,
                "train_loss": running_loss / step,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "validation": validation,
            }
            events.append(event)
            improved = validation["macro_f1"] > best_f1 + args.min_delta
            if improved:
                best_f1 = validation["macro_f1"]
                best_event = event
                non_improving = 0
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                save_training_state(args.output_dir / "best_training_state.pt", optimizer, scheduler, event, args.seed)
            else:
                non_improving += 1
            save_json(args.output_dir / "history.json", {"evaluations": events})
            save_json(
                args.output_dir / "progress.json",
                {
                    "phase": "validation_complete",
                    "fraction": event["fraction"],
                    "step": step,
                    "validation_macro_f1": validation["macro_f1"],
                    "validation_accuracy": validation["accuracy"],
                    "improved": improved,
                    "best_validation_macro_f1": best_f1,
                    "non_improving_checks": non_improving,
                },
            )
            print(json.dumps({"status": "quarter_validation", **event, "improved": improved}), flush=True)
            next_quarter += 1
            if non_improving >= args.quarter_patience:
                stopped_early = True
                break
            model.train()

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir, trust_remote_code=True).to(device)
    final_validation = evaluate(best_model, validation_loader, device)
    summary = {
        "config": config,
        "baseline_validation": baseline,
        "best_event": best_event,
        "best_validation": final_validation,
        "completed_step": completed_step,
        "steps_in_epoch": len(train_loader),
        "completed_fraction": completed_step / len(train_loader),
        "stopped_early": stopped_early,
    }
    save_json(args.output_dir / "summary.json", summary)
    save_json(args.output_dir / "progress.json", {"phase": "complete", **summary})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
