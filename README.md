# 🧠 Brain Rot Test

A fun, gamified AI self-assessment app that analyzes how you describe your study habits and focus patterns, then tells you how "cooked" your brain is — with a score, a playful label, and real cognitive-science-backed advice.

> ⚠️ **Not a medical tool.** All outputs are AI-generated and for educational/entertainment purposes only.

---

## Project Structure

```
brain-rot-detector/
├── data/
│   └── dataset.json        ← 120 synthetic free-text examples (3 classes)
├── model/                  ← saved fine-tuned FLAN-T5-base weights (after training)
├── train.py                ← fine-tuning script (Seq2SeqTrainer)
├── evaluate.py             ← BLEU-4 + ROUGE-L + accuracy evaluation
├── model_utils.py          ← load model, run inference, parse output
├── app.py                  ← Streamlit app (main entry point)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Workflow

### Step 1 — Verify dataset
```bash
python -c "import json; d=json.load(open('data/dataset.json')); print(len(d), 'examples')"
# Expected: 120 examples
```

### Step 2 — Fine-tune the model
```bash
python train.py
# Optional flags:
# python train.py --epochs 12 --batch_size 4 --lr 3e-4
#
# GPU:  training takes ~10-20 minutes
# CPU:  training takes ~2-4 hours
#
# Best checkpoint saved to model/
```

### Step 3 — Evaluate
```bash
python evaluate.py
# Reports: accuracy %, BLEU-4, ROUGE-L, 5 qualitative samples
```

### Step 4 — Run the app
```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## Model

| Property | Value |
|---|---|
| Base model | `google/flan-t5-base` |
| Parameters | ~250M |
| Task | Text-to-text (seq2seq) |
| Fine-tuning | HuggingFace `Seq2SeqTrainer` |
| Input max tokens | 180 |
| Output max tokens | 220 |

### Output format

```
Brain Status: Cooked 🔥
Score: 82
Issues: poor attention control, excessive phone dependency, task avoidance
Advice:
- Start with 5-minute focus sprints, not hour-long sessions
- Put your phone in another room while studying
- Use the Pomodoro technique: 25 min work, 5 min break
Mini Challenge: Close all tabs except one and study for exactly 10 minutes.
```

---

## Dataset

120 synthetic examples across 3 classes:

| Label | Count | Score Range | Description |
|---|---|---|---|
| Focused 🧘 | ~40 | 0–30 | Long sessions, few distractions, good recall |
| Distracted 😵 | ~40 | 31–65 | Mixed habits, occasional phone, sometimes finishes |
| Cooked 🔥 | ~40 | 66–100 | Short attention, heavy phone use, task avoidance |

---

## Expected Metrics (after fine-tuning)

| Metric | Expected Range |
|---|---|
| Label Accuracy | 75–90% |
| BLEU-4 | 15–35 |
| ROUGE-L | 35–60 |

---

## Key Learning Outcomes

- Synthetic dataset design with class balance strategy
- Seq2seq fine-tuning with `Seq2SeqTrainer`
- Structured output parsing with regex
- BLEU / ROUGE evaluation
- Streamlit app with session state and dynamic UI