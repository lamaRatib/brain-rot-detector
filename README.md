# 🧠 Brain Rot Test

A fun, gamified AI self-assessment app that analyzes how you describe your study habits and focus patterns, then tells you how "cooked" your brain is — with a score, a status label, and real cognitive-science-backed advice.

> ⚠️ **Not a medical tool.** All outputs are AI-generated and for educational/entertainment purposes only.

---

## 🎯 What It Does

You type a free-text description of your study and focus habits. The app classifies you into one of three categories:

| Status | Score Range | Description |
|---|---|---|
| 🧘 Focused | 0–30 | Strong habits, consistent sessions, good retention |
| 😵 Distracted | 31–65 | Mixed habits, occasional phone use, inconsistent |
| 🔥 Cooked | 66–100 | Short attention span, heavy phone dependency, avoidance |

Then it gives you personalized advice and a mini challenge to act on immediately.

---

## 🏗️ Project Structure

```
brain-rot-detector/
├── data/
│   ├── dataset.json           ← 120 synthetic free-text examples (3 classes)
│   └── dataset_simple.json    ← simplified outputs for seq2seq experiments
├── model/                     ← saved fine-tuned DistilBERT weights (after training)
├── train.py                   ← training script (DistilBERT classifier)
├── evaluate.py                ← classification accuracy evaluation
├── model_utils.py             ← load model, run inference, build result dict
├── app.py                     ← Streamlit app (main entry point)
├── requirements.txt
└── README.md
```

---

## 🔬 Model Journey — Why We Changed Approach

### Attempt 1: FLAN-T5-base (Seq2Seq Generation)

The original plan was to fine-tune `google/flan-t5-base` to generate structured text output:

```
Brain Status: Cooked 🔥
Score: 82
Issues: poor attention control, excessive phone dependency
Advice:
- Start with 5-minute focus sprints
- Put your phone in another room
Mini Challenge: Close all tabs and study for 10 minutes.
```

**What happened:**

Training loss started at ~9.5 and barely reached ~3.6 after 25 epochs. The model collapsed — outputting "Focused, Score 16" for every single input regardless of what was typed.

Root causes:
- Task too complex for only 96 training examples
- Output format too long (89 tokens per target on average)
- Seq2seq architecture is not designed for classification tasks
- fp16 caused NaN gradients on certain GPU configurations
- Multiple version conflicts between transformers, huggingface_hub, and httpx

**Metrics after best seq2seq run:**

| Metric | Result |
|---|---|
| Label Accuracy | 41.7% |
| BLEU-4 | 39.36 |
| ROUGE-L | 45.73 |

---

### Attempt 2: DistilBERT Classifier ✅

We switched to `distilbert-base-uncased` with a sequence classification head — a model architecturally designed for exactly this task.

**Why DistilBERT:**

- Sequence classification is its native task — not a workaround
- Only 66M parameters, trains in ~3 minutes on a free Colab GPU
- Loss starts at ~1.0 (not 9.5) and converges cleanly
- No structured output parsing needed — direct label prediction
- Advice, scores, and challenges are served from curated templates per class

**Metrics after DistilBERT:**

| Metric | Result |
|---|---|
| Label Accuracy | **83.3%** |
| Training Time | ~3 min (Colab T4 GPU) |
| Final Train Loss | 0.058 (epoch 20) |

**The lesson:** Match your model architecture to your task. Seq2seq is for generation. Classification is for classification.

---

## 🚀 Setup

```bash
# 1. Clone the repo
git clone https://github.com/lamaRatib/brain-rot-detector.git
cd brain-rot-detector

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model weights
# Download model.zip from releases, unzip into project root
# You should have a model/ folder with config.json, model.safetensors etc
```

---

## 🏋️ Train Your Own Model

```bash
python train.py --epochs 20 --batch_size 16 --lr 2e-5
```

Training takes ~3 minutes on GPU, ~20 minutes on CPU.

Watch for accuracy in the logs:

```
{'eval_accuracy': 0.4167, 'epoch': 1}   ← starting
{'eval_accuracy': 0.6667, 'epoch': 7}   ← learning
{'eval_accuracy': 0.75,   'epoch': 9}   ← good
Test Accuracy: 83.3%                    ← final
```

---

## 📊 Evaluate

```bash
python evaluate.py
```

Reports label accuracy with ✅/❌ indicators per sample:

```
=============================================
  Label Accuracy : 83.3%  (10/12)
=============================================

✅ INPUT : no matter what i try i end up on my phone...
   TRUE  : Cooked
   PRED  : Cooked
```

---

## 🖥️ Run the App

```bash
streamlit run app.py
# Open http://localhost:8501
```

**Test inputs to try:**

- `"I check my phone every 2 minutes and can't finish anything"` → Cooked 🔥
- `"I do 90-minute deep work sessions with phone off"` → Focused 🧘
- `"I study for a bit but get distracted by notifications"` → Distracted 😵

---

## 📦 Dependencies

```
torch
transformers>=4.35.0
datasets
streamlit>=1.30.0
scikit-learn
accelerate
sentencepiece
```

---

## 🧱 Technical Challenges Overcome

| Challenge | Solution |
|---|---|
| WSL crashing from OOM | Moved training to Google Colab free GPU |
| HuggingFace download timeouts | Used hf-mirror.com + HF_HUB_DISABLE_XET=1 |
| fp16 NaN gradients | Disabled fp16 for affected GPU config |
| transformers API breaking changes | Replaced as_target_tokenizer and evaluation_strategy |
| Model collapsing to one class | Switched from seq2seq to DistilBERT classifier |
| Colab disk full (18GB) | Cleared checkpoint cache + pip cache between runs |
| GitHub 100MB file limit | Added model/ and weights to .gitignore |
| SSH not working in Colab | Switched remote URL to HTTPS with personal access token |

---

## 🧠 Key Learning Outcomes

| Outcome | How Achieved |
|---|---|
| Dataset design | 120 synthetic free-text examples with class balance strategy |
| Model selection | Empirically compared seq2seq vs classifier |
| Fine-tuning | DistilBERT + HuggingFace Trainer with early stopping |
| Evaluation | Accuracy metric + qualitative sample review |
| App deployment | Streamlit with session state, score bar, and dynamic UI |
| Debugging | Real-world ML debugging across 3 different failure modes |

---

## 📈 Results Summary

| Approach | Accuracy | Notes |
|---|---|---|
| FLAN-T5-base seq2seq | 41.7% | Collapsed to single output class |
| DistilBERT classifier | **83.3%** | Clean convergence, fast training |

---

*Built as a portfolio ML project exploring fine-tuning, evaluation, and Streamlit deployment.*
