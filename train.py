"""
train.py
Fine-tunes google/flan-t5-base on the brain-rot dataset using
HuggingFace Seq2SeqTrainer. Saves the best checkpoint to model/.

Usage:
    python train.py [--epochs 12] [--batch_size 4] [--lr 3e-4]
"""

import json
import argparse
import os
import random

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "./flan-t5-small-local"
DATA_PATH = "dataset.json"
OUTPUT_DIR = "model/"
INPUT_PREFIX = "Analyze the brain rot level from this self-description: "
MAX_INPUT_LEN = 180
MAX_TARGET_LEN = 220
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: str):
    with open(path) as f:
        raw = json.load(f)

    # Shuffle then split 80 / 10 / 10
    random.shuffle(raw)
    n = len(raw)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    train = raw[:train_end]
    val = raw[train_end:val_end]
    test = raw[val_end:]

    print(f"Split → train:{len(train)}  val:{len(val)}  test:{len(test)}")
    return (
        Dataset.from_list(train),
        Dataset.from_list(val),
        Dataset.from_list(test),
    )


def build_tokenize_fn(tokenizer):
    def tokenize(batch):
        inputs = [INPUT_PREFIX + t for t in batch["input"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["output"],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize


def main(args):
    set_seed(SEED)

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    train_ds, val_ds, test_ds = load_data(DATA_PATH)

    tokenize_fn = build_tokenize_fn(tokenizer)
    cols_to_remove = ["input", "output", "label"]

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=50,
        weight_decay=0.01,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving best model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    main(args)