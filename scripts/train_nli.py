"""Fine-tune the SB9.1 NLI checkpoint on SciFact train (Week 9, SB9.3).

STATUS: DEFERRED. This script is the data-prep + Trainer scaffolding
for a SciFact fine-tune, but the actual fine-tune was not run as part
of Week 9 -- see eval/results/README.md "week9_scifact_finetune
(deferred)" for the diagnostics and rationale. tl;dr:

  * DeBERTa-v2 disentangled attention is numerically unstable on Apple
    Silicon MPS: gradients go NaN within ~20 training steps regardless
    of learning rate (5e-6 .. 2e-5) or gradient clipping.
  * CPU training of DeBERTa-v3-base at seq_len=512 measures ~4 min/step
    on M1 Pro. A full 4-epoch run on 900 train pairs (batch 4) would
    take ~60 hours.
  * The forward pass works in both modes (eval loss ~0.88), so the
    SB9.1 zero-shot eval is unaffected.

The scaffolding (label-map inversion, stratified split,
compute_metrics closure, dataset builder) is correct and
hardware-independent; tests/test_train_nli.py verifies them. Running
this script on a CUDA GPU should just work -- no code changes needed.
The reason it lives in-repo despite being unused is that re-deriving
the SciFact label-id mapping and stratified split logic on a different
machine would be the exact wrong place to introduce subtle bugs.

Builds (claim, cited_doc) training pairs with oracle premise text
(same shape as SB9.1 eval), maps SciFact labels back to the model's
MNLI label space using the same id2label table SB9.1 builds, then
runs HF Trainer for 3-5 epochs. Saves the fine-tuned checkpoint to
``data/models/scifact_nli/`` so it can be loaded by
``NLIClassifier(model_name="data/models/scifact_nli")`` with zero
caller changes.

Pre-registered success metric (set in eval/results/README.md SB9.2):
  * CONTRADICT recall on the k=5 hit subset 0.574 -> >= 0.70
  * end-to-end accuracy at k=5: 0.659 -> >= 0.72
  * SUPPORT recall on hit subset must not drop below 0.60

Usage (when GPU is available):
    python scripts/train_nli.py --epochs 4 --device cuda \
        --output-dir data/models/scifact_nli
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.scifact_eval import build_pairs, load_claims, load_corpus
from src.verification.nli_classifier import (
    CONTRADICT,
    DEFAULT_MODEL,
    NEI,
    SUPPORT,
    _build_label_map,
)

DEFAULT_TRAIN = Path("data/datasets/scifact/claims_train.json")
DEFAULT_CORPUS = Path("data/datasets/scifact/corpus.json")
DEFAULT_OUT = Path("data/models/scifact_nli")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _invert_label_map(label_map: dict[int, str]) -> dict[str, int]:
    """SciFact label -> MNLI index, the inverse of NLIClassifier.label_map."""
    return {scifact_label: idx for idx, scifact_label in label_map.items()}


def _stratified_split(
    pairs: list, val_frac: float, seed: int
) -> tuple[list, list]:
    """Stratify by gold label so the val slice is class-balanced.

    SciFact train is mildly imbalanced (CONTRADICT is ~half of SUPPORT);
    a random split risks giving the val set 0 CONTRADICT examples on a
    bad seed.
    """
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for p in pairs:
        by_label.setdefault(p.gold_label, []).append(p)
    train_out: list = []
    val_out: list = []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_frac)))
        val_out.extend(group[:n_val])
        train_out.extend(group[n_val:])
    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out


def _build_dataset(pairs, tokenizer, label_to_id, max_length: int):
    """Tokenise pairs into a HF Dataset.

    Premise is the full abstract, hypothesis is the claim -- matching
    the SB9.1 eval shape exactly so the train distribution matches the
    inference distribution.
    """
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "premise": [p.premise for p in pairs],
            "hypothesis": [p.claim for p in pairs],
            "label": [label_to_id[p.gold_label] for p in pairs],
        }
    ).map(
        lambda batch: tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )


def _compute_metrics(label_map_inv: dict[str, int]):
    """Closure that turns HF eval predictions into accuracy + per-class F1."""
    id_to_scifact = {idx: label for label, idx in label_map_inv.items()}

    def _fn(eval_pred):
        import numpy as np

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        correct = (preds == labels).sum()
        total = len(labels)
        out = {"accuracy": correct / total if total else 0.0}
        for sci_label in (SUPPORT, CONTRADICT, NEI):
            idx = label_map_inv[sci_label]
            tp = int(((preds == idx) & (labels == idx)).sum())
            fp = int(((preds == idx) & (labels != idx)).sum())
            fn = int(((preds != idx) & (labels == idx)).sum())
            if tp == 0:
                out[f"f1_{sci_label.lower()}"] = 0.0
                continue
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            out[f"f1_{sci_label.lower()}"] = (
                2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            )
        out["macro_f1"] = (
            out["f1_support"] + out["f1_contradict"] + out["f1_nei"]
        ) / 3.0
        return out

    return _fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-claims", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--base-model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, train on only the first N pairs (smoke).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help=(
            "Training device. Default cpu: DeBERTa-v2 disentangled attention is "
            "numerically unstable on MPS (gradients go NaN within ~20 steps). "
            "Inference still runs on MPS via NLIClassifier with no override needed."
        ),
    )
    args = p.parse_args()

    # HF imports gated to keep --help / unit tests light
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    print(f"[load] corpus={args.corpus}")
    corpus = load_corpus(args.corpus)
    print(f"[load] {len(corpus):,} corpus docs")

    print(f"[load] train_claims={args.train_claims}")
    train_claims = load_claims(args.train_claims)
    print(f"[load] {len(train_claims):,} train claims")

    pairs, n_missing = build_pairs(train_claims, corpus)
    print(f"[pairs] built {len(pairs):,} (claim, doc) pairs; dropped {n_missing} missing docs")
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"[pairs] --limit {args.limit} -> {len(pairs)} pairs")

    train_pairs, val_pairs = _stratified_split(pairs, args.val_frac, args.seed)
    print(f"[split] train={len(train_pairs)}  val={len(val_pairs)} (seed={args.seed})")

    print(f"[model] loading base: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    label_map = _build_label_map(model.config.id2label)  # idx -> SciFact label
    label_to_id = _invert_label_map(label_map)
    print(f"[model] label_map={label_map}")

    device = args.device
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Disable MPS path inside HF Trainer / accelerate.
        os.environ["ACCELERATE_USE_MPS_DEVICE"] = "false"
    print(f"[model] device={device}")
    model.to(device)

    train_ds = _build_dataset(train_pairs, tokenizer, label_to_id, args.max_length)
    val_ds = _build_dataset(val_pairs, tokenizer, label_to_id, args.max_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # We do *not* set load_best_model_at_end=True. The HF Trainer save+reload
    # path mangles DeBERTa-v2 LayerNorm key names (beta/gamma vs weight/bias),
    # randomising those params on reload and producing NaN eval loss. Instead
    # we eval each epoch (for the log), keep training the same model object
    # in-place, and save once at the end.
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],  # disable wandb/tensorboard auto-init
        seed=args.seed,
        fp16=False,
        dataloader_pin_memory=False,  # silences MPS warning
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics(label_to_id),
    )

    started = time.time()
    print(f"[train] starting {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    trainer.train()
    train_sec = time.time() - started
    print(f"[train] done in {train_sec/60:.1f} min")

    print(f"[save] writing best checkpoint to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    final_metrics = trainer.evaluate()
    metadata = {
        "git_commit": _git_commit(),
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "train_runtime_sec": round(train_sec, 1),
        "final_val_metrics": final_metrics,
        "label_map": {str(k): v for k, v in label_map.items()},
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[done] final val: {final_metrics}")
    print(f"[done] metadata -> {args.output_dir/'training_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
