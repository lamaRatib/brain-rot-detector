"""
model_utils.py
Handles loading the fine-tuned FLAN-T5-base model, running inference,
and parsing the structured output into a clean dict.
"""

import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

INPUT_PREFIX = "Analyze the brain rot level from this self-description: "

# ── Label colours for Streamlit ──────────────────────────────────────────────
STATUS_META = {
    "Focused":    {"emoji": "🧘", "color": "#22c55e"},  # green
    "Distracted": {"emoji": "😵", "color": "#f59e0b"},  # amber
    "Cooked":     {"emoji": "🔥", "color": "#ef4444"},  # red
}


def load_model(model_dir: str = "model/"):
    """Load tokenizer and model from a local directory."""
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def run_inference(text: str, tokenizer, model) -> str:
    """
    Tokenize the user input, run beam-search generation,
    and return the raw decoded string.
    """
    prompt = INPUT_PREFIX + text.strip()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=180,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def parse_output(raw_text: str) -> dict:
    """
    Parse the structured model output into a Python dict.

    Expected raw_text format
    ------------------------
    Brain Status: Cooked 🔥
    Score: 82
    Issues: poor attention control, excessive phone dependency
    Advice:
    - Start with 5-minute focus sprints
    - Put your phone in another room
    Mini Challenge: Close all tabs and study for 10 minutes.

    Returns
    -------
    {
        "status":    "Cooked",
        "emoji":     "🔥",
        "score":     82,
        "color":     "#ef4444",
        "issues":    ["poor attention control", "excessive phone dependency"],
        "advice":    ["Start with 5-minute focus sprints", ...],
        "challenge": "Close all tabs and study for 10 minutes.",
        "raw":       <original string>
    }
    """
    result = {
        "status": "Unknown",
        "emoji": "❓",
        "score": 50,
        "color": "#6b7280",
        "issues": [],
        "advice": [],
        "challenge": "",
        "raw": raw_text,
    }

    # ── Brain Status ─────────────────────────────────────────────────────────
    status_match = re.search(r"Brain Status:\s*(.+?)(?:\n|$)", raw_text)
    if status_match:
        raw_status = status_match.group(1).strip()
        for key in STATUS_META:
            if key.lower() in raw_status.lower():
                result["status"] = key
                result["emoji"] = STATUS_META[key]["emoji"]
                result["color"] = STATUS_META[key]["color"]
                break

    # ── Score ─────────────────────────────────────────────────────────────────
    score_match = re.search(r"Score:\s*(\d+)", raw_text)
    if score_match:
        result["score"] = max(0, min(100, int(score_match.group(1))))

    # ── Issues ────────────────────────────────────────────────────────────────
    issues_match = re.search(r"Issues:\s*(.+?)(?:\n|$)", raw_text)
    if issues_match:
        raw_issues = issues_match.group(1).strip()
        if raw_issues.lower() != "none significant":
            result["issues"] = [i.strip() for i in raw_issues.split(",") if i.strip()]

    # ── Advice ────────────────────────────────────────────────────────────────
    advice_block = re.search(r"Advice:\n((?:- .+\n?)+)", raw_text)
    if advice_block:
        result["advice"] = [
            line.lstrip("- ").strip()
            for line in advice_block.group(1).splitlines()
            if line.strip().startswith("-")
        ]

    # ── Mini Challenge ────────────────────────────────────────────────────────
    challenge_match = re.search(r"Mini Challenge:\s*(.+?)(?:\n|$)", raw_text, re.DOTALL)
    if challenge_match:
        result["challenge"] = challenge_match.group(1).strip()

    return result


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Brain Status: Cooked 🔥\n"
        "Score: 82\n"
        "Issues: poor attention control, excessive phone dependency, task avoidance\n"
        "Advice:\n"
        "- Start with 5-minute focus sprints, not hour-long sessions\n"
        "- Put your phone in another room while studying\n"
        "- Use the Pomodoro technique: 25 min work, 5 min break\n"
        "Mini Challenge: Right now, close all tabs except one and study for exactly 10 minutes."
    )
    parsed = parse_output(sample)
    print(parsed)