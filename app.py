import streamlit as st
from datetime import datetime

from model_utils import load_model, run_inference, parse_output

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Brain Rot Test",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .big-title  { font-size: 2.4rem; font-weight: 800; margin-bottom: 0; }
    .subtitle   { color: #9ca3af; margin-top: 0; margin-bottom: 1.5rem; }
    .status-box { padding: 1rem 1.5rem; border-radius: 12px;
                  margin-bottom: 1rem; font-size: 1.1rem; font-weight: 600; }
    .issue-pill { display: inline-block; background: #1f2937;
                  border-radius: 9999px; padding: 2px 12px;
                  margin: 3px; font-size: 0.85rem; color: #e5e7eb; }
    .advice-card{ background: #111827; border-left: 4px solid;
                  padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
                  margin-bottom: 0.5rem; font-size: 0.95rem; }
    .challenge  { background: #1e3a5f; border-radius: 10px;
                  padding: 1rem 1.25rem; font-style: italic;
                  color: #93c5fd; margin-top: 0.5rem; }
    .history-item { font-size: 0.85rem; padding: 6px 0;
                    border-bottom: 1px solid #374151; }
    .disclaimer { font-size: 0.75rem; color: #6b7280; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model (cached so it only loads once) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading model … this takes a moment ⚙️")
def get_model():
    return load_model("model/")


# ── Session history ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# ── Sidebar — session history ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Session History")
    if not st.session_state["history"]:
        st.caption("No tests yet this session.")
    else:
        for entry in reversed(st.session_state["history"]):
            color = entry["color"]
            st.markdown(
                f"<div class='history-item'>"
                f"<span style='color:{color};'>●</span> "
                f"<b>{entry['time']}</b> — {entry['status']} "
                f"({entry['score']}/100)<br>"
                f"<small style='color:#9ca3af;'>{entry['snippet']}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("<div class='big-title'>🧠 Brain Rot Test</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Describe your study & focus habits honestly — "
    "get an AI-powered brain rot score, diagnosis, and real advice.</div>",
    unsafe_allow_html=True,
)

user_input = st.text_area(
    label="How would you describe your focus and study habits? Be honest 👇",
    placeholder=(
        "e.g. I can't sit still for more than 5 minutes without "
        "checking my phone. I've tried studying but I just end up scrolling TikTok…"
    ),
    height=140,
)

col1, col2 = st.columns([3, 1])
with col1:
    analyze_btn = st.button("🔬 Analyze My Brain", use_container_width=True, type="primary")
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)
    if clear_btn:
        st.rerun()

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if not user_input.strip():
        st.warning("Please describe your habits first!")
    else:
        with st.spinner("Analyzing your brain rot level … 🧬"):
            try:
                tokenizer, model = get_model()
                raw = run_inference(user_input, tokenizer, model)
                result = parse_output(raw)
            except Exception as e:
                st.error(f"Model error: {e}. Make sure the model/ directory exists.")
                st.stop()

        # Save to history
        st.session_state["history"].append(
            {
                "time": datetime.now().strftime("%H:%M"),
                "status": f"{result['status']} {result['emoji']}",
                "score": result["score"],
                "color": result["color"],
                "snippet": user_input[:60] + ("…" if len(user_input) > 60 else ""),
            }
        )

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Results")

        # Status badge
        st.markdown(
            f"<div class='status-box' style='background:{result['color']}22; "
            f"border: 2px solid {result['color']};'>"
            f"🧠 Brain Status: <span style='color:{result['color']};'>"
            f"{result['status']} {result['emoji']}</span></div>",
            unsafe_allow_html=True,
        )

        # Score bar
        score = result["score"]
        bar_color = result["color"]
        st.markdown(f"**Score: {score} / 100**")
        st.markdown(
            f"""
            <div style="background:#1f2937; border-radius:9999px;
                        height:14px; width:100%; overflow:hidden; margin-bottom:1rem;">
              <div style="background:{bar_color}; width:{score}%;
                          height:100%; border-radius:9999px;
                          transition: width 0.8s ease;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Issues
        if result["issues"]:
            st.markdown("**⚠️ Key Issues:**")
            pills = "".join(
                f"<span class='issue-pill'>{issue}</span>"
                for issue in result["issues"]
            )
            st.markdown(f"<div style='margin-bottom:1rem;'>{pills}</div>",
                        unsafe_allow_html=True)

        # Advice
        if result["advice"]:
            st.markdown("**💡 What You Should Do:**")
            for tip in result["advice"]:
                st.markdown(
                    f"<div class='advice-card' "
                    f"style='border-color:{bar_color};'>✦ {tip}</div>",
                    unsafe_allow_html=True,
                )

        # Mini challenge
        if result["challenge"]:
            st.markdown("**🎯 Mini Challenge:**")
            st.markdown(
                f"<div class='challenge'>\"{result['challenge']}\"</div>",
                unsafe_allow_html=True,
            )

        # Debug expander
        with st.expander("🔍 Raw model output"):
            st.code(result["raw"])

        # Disclaimer
        st.markdown(
            "<div class='disclaimer'>⚠️ Not a medical tool. "
            "All outputs are AI-generated and for educational/entertainment purposes only.</div>",
            unsafe_allow_html=True,
        )