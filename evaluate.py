"""
evaluate.py
Evaluates the fine-tuned model on the held-out test split using:
  • Classification accuracy  (label extracted from "Brain Status:" line)
  • BLEU-4                   (full output vs reference)
  • ROUGE-L                  (full output vs reference)
  • 5 qualitative samples    (input | reference | generated)

Usage:
    python evaluate.py [--model_dir model/] [--data_path dataset.json]
"""

import json
import argparse
import random

import torch
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer as rs

from model_utils import load_model, run_inference, parse_output, INPUT_PREFIX

SEED = 42
random.seed(SEED)


def load_test_split(data_path: str, test_frac: float = 0.10):
    with open(data_path) as f:
        raw = json.load(f)
    random.shuffle(raw)
    start = int(len(raw) * 0.90)
    return raw[start:]


def extract_label_from_status(status_str: str) -> str:
    """Normalise generated status to Focused / Distracted / Cooked."""
    s = status_str.lower()
    if "focused" in s:
        return "Focused"
    if "distracted" in s:
        return "Distracted"
    if "cooked" in s:
        return "Cooked"
    return "Unknown"


def main(args):
    print(f"Loading model from {args.model_dir} …")
    tokenizer, model = load_model(args.model_dir)

    test_data = load_test_split(args.data_path)
    print(f"Test examples: {len(test_data)}")

    references = []
    hypotheses = []
    correct = 0

    print("\nRunning inference …")
    for i, example in enumerate(test_data):
        generated_raw = run_inference(example["input"], tokenizer, model)
        parsed = parse_output(generated_raw)

        pred_label = parsed["status"]
        true_label = example["label"]
        if pred_label == true_label:
            correct += 1

        references.append(example["output"])
        hypotheses.append(generated_raw)

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(test_data)} done …")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    accuracy = correct / len(test_data) * 100
    print(f"\n{'='*55}")
    print(f"  Label Accuracy : {accuracy:.1f}%  ({correct}/{len(test_data)})")

    # ── BLEU-4 ────────────────────────────────────────────────────────────────
    bleu_metric = BLEU(max_ngram_order=4)
    bleu_result = bleu_metric.corpus_score(hypotheses, [references])
    print(f"  BLEU-4         : {bleu_result.score:.2f}")

    # ── ROUGE-L ───────────────────────────────────────────────────────────────
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) * 100
    print(f"  ROUGE-L        : {avg_rouge:.2f}")
    print(f"{'='*55}\n")

    # ── Qualitative samples ───────────────────────────────────────────────────
    print("── 5 Qualitative Samples ──────────────────────────────")
    sample_indices = random.sample(range(len(test_data)), min(5, len(test_data)))
    for idx in sample_indices:
        example = test_data[idx]
        generated_raw = run_inference(example["input"], tokenizer, model)
        print(f"\nINPUT     : {example['input']}")
        print(f"REFERENCE : {example['output'][:120]} …")
        print(f"GENERATED : {generated_raw[:120]} …")
        print(f"TRUE LABEL: {example['label']}")
        parsed = parse_output(generated_raw)
        print(f"PRED LABEL: {parsed['status']}")
        print("-" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model/")
    parser.add_argument("--data_path", default="dataset.json")
    args = parser.parse_args()
    main(args)