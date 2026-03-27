"""
model_utils.py
Handles loading the fine-tuned FLAN-T5-base model, running inference,
and parsing the structured output into a clean dict.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT_PREFIX = "Analyze the brain rot level from this self-description: "

LABEL2ID = {"Focused": 0, "Distracted": 1, "Cooked": 2}
ID2LABEL  = {0: "Focused", 1: "Distracted", 2: "Cooked"}

STATUS_META = {
    "Focused":    {"emoji": "🧘", "color": "#22c55e"},
    "Distracted": {"emoji": "😵", "color": "#f59e0b"},
    "Cooked":     {"emoji": "🔥", "color": "#ef4444"},
}

ADVICE = {
    "Focused": [
        "Keep up your current system — consistency is your superpower",
        "Try spaced repetition to retain more of what you study",
        "Consider teaching what you learn to solidify memory",
    ],
    "Distracted": [
        "Set a clear timer before each study block",
        "Put your phone face-down in another room while studying",
        "Use the Pomodoro technique: 25 min work, 5 min break",
    ],
    "Cooked": [
        "Start with 5-minute focus sprints — not hour-long sessions",
        "Delete or hide distracting apps temporarily",
        "Put your phone in another room and study in a library",
    ],
}

CHALLENGES = {
    "Focused":    "Plan tomorrow's study session in detail tonight before you sleep.",
    "Distracted": "Enable Do Not Disturb right now and study for 20 minutes.",
    "Cooked":     "Close all tabs except one. Read for 5 minutes. That is all.",
}

SCORES = {
    "Focused": 15,
    "Distracted": 50,
    "Cooked": 82,
}


def load_model(model_dir: str = "model/"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def run_inference(text: str, tokenizer, model) -> str:
    prompt = INPUT_PREFIX + text.strip()
    inputs = tokenizer(prompt, return_tensors="pt",
                       max_length=128, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(dim=-1).item()
    return ID2LABEL[pred_id]


def parse_output(label: str) -> dict:
    meta = STATUS_META.get(label, {"emoji": "❓", "color": "#6b7280"})
    return {
        "status":    label,
        "emoji":     meta["emoji"],
        "color":     meta["color"],
        "score":     SCORES.get(label, 50),
        "issues":    [],
        "advice":    ADVICE.get(label, []),
        "challenge": CHALLENGES.get(label, ""),
        "raw":       label,
    }