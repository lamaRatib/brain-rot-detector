import json, argparse, random
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score

MODEL_NAME = "google/flan-t5-base"
DATA_PATH  = "dataset.json"
OUTPUT_DIR = "model/"
INPUT_PREFIX = "Analyze the brain rot level from this self-description: "
LABEL2ID = {"Focused": 0, "Distracted": 1, "Cooked": 2}
ID2LABEL = {0: "Focused", 1: "Distracted", 2: "Cooked"}
SEED = 42

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    random.shuffle(raw)
    n = len(raw)
    train_end, val_end = int(n*.80), int(n*.90)
    train, val, test = raw[:train_end], raw[train_end:val_end], raw[val_end:]
    print(f"Split → train:{len(train)}  val:{len(val)}  test:{len(test)}")
    return Dataset.from_list(train), Dataset.from_list(val), Dataset.from_list(test)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def main(args):
    set_seed(SEED)
    print(f"Loading tokenizer and model: distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds, val_ds, test_ds = load_data(DATA_PATH)

    def tokenize(batch):
        inputs = [INPUT_PREFIX + t for t in batch["input"]]
        result = tokenizer(inputs, max_length=128, truncation=True, padding=False)
        result["labels"] = [LABEL2ID[l] for l in batch["label"]]
        return result

    cols = ["input", "output", "label"]
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=cols)
    val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=cols)
    test_ds  = test_ds.map(tokenize,  batched=True, remove_columns=cols)

    collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training …")
    trainer.train()

    # Test evaluation
    print("\\nEvaluating on test set …")
    results = trainer.evaluate(test_ds)
    print(f"Test Accuracy: {results['eval_accuracy']*100:.1f}%")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=18)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    main(parser.parse_args())