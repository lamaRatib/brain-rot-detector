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
import json, argparse, random
from model_utils import load_model, run_inference

SEED = 42
random.seed(SEED)

def load_test_split(data_path: str):
    with open(data_path) as f:
        raw = json.load(f)
    random.shuffle(raw)
    start = int(len(raw) * 0.90)
    return raw[start:]

def main(args):
    print(f"Loading model from {args.model_dir} …")
    tokenizer, model = load_model(args.model_dir)

    test_data = load_test_split(args.data_path)
    print(f"Test examples: {len(test_data)}")
    print("\\nRunning inference …")

    correct = 0
    for i, example in enumerate(test_data):
        pred = run_inference(example["input"], tokenizer, model)
        if pred == example["label"]:
            correct += 1
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(test_data)} done …")

    accuracy = correct / len(test_data) * 100
    print(f"\\n{'='*45}")
    print(f"  Label Accuracy : {accuracy:.1f}%  ({correct}/{len(test_data)})")
    print(f"{'='*45}\\n")

    print("── 5 Qualitative Samples ──────────────────")
    for example in random.sample(test_data, min(5, len(test_data))):
        pred = run_inference(example["input"], tokenizer, model)
        status = "✅" if pred == example["label"] else "❌"
        print(f"\\n{status} INPUT : {example['input'][:80]}")
        print(f"   TRUE  : {example['label']}")
        print(f"   PRED  : {pred}")
        print("-" * 45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  default="model/")
    parser.add_argument("--data_path",  default="dataset.json")
    args = parser.parse_args()
    main(args)