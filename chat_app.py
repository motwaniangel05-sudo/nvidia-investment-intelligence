"""AI Investment Intelligence: dynamic question page.  Run: streamlit run chat_app.py"""
import streamlit as st

from core.dynamic_pipeline import run
from core.report_builder import build_report

st.set_page_config(page_title="AI Investment Intelligence", page_icon="🟩", layout="wide")

EXAMPLES = [
    "What is NVIDIA's current financial health?",
    "Should I buy NVIDIA at the current valuation?",
    "NVIDIA is trading at $500, yesterday it was $200, and my average price is $120. Should I buy more or sell?",
    "Compare NVIDIA with AMD.",
    "What are NVIDIA's biggest risks right now?",
    "What does the DCF valuation indicate?",
]


def _set_question(q):
    st.session_state["question"] = q


def _esc(text):
    return str(text).replace("$", "\\$")


def _icon(status):
    return "❌" if status == "failed" else ("⚠️" if status == "partial" else "✅")


def _analyze(q):
    status = st.status("Working...", expanded=True)

    def progress(stage, info):
        if stage == "plan":
            status.write(_esc(f"✅ Plan ready ({info['intent']}): " + ", ".join(info["agents"])))
            status.write("⏳ Running the agents in parallel...")
        elif stage == "agents":
            for r in info:
                extra = f" - {str(r['error'])[:80]}" if r["error"] else ""
                status.write(_esc(f"{_icon(r['status'])} {r['agent']}: {r['status']} "
                                  f"({r['duration_seconds']:.1f}s){extra}"))
        elif stage == "evidence":
            status.write(f"✅ Evidence built: {info['sources']} source group(s), "
                         f"{info['conflicts']} conflict(s)")

    try:
        out = run(q, progress=progress,
                  synthesizer=lambda query, context: {"answer": "", "source": "skipped"})
        status.write("⏳ Qwen is writing the short draft summary (can take up to a minute)...")
        report = build_report(q, out.get("context") or {}, out["agent_runs"], out["errors"])
        status.update(label="Done", state="complete", expanded=False)
        st.session_state["result"] = {"question": q, "out": out, "report": report}
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.error(f"The analysis failed: {exc}")


st.title("AI INVESTMENT INTELLIGENCE")
st.caption("Ask a question about NVIDIA. The system chooses the agents it needs. "
           "Numbers come from the agents' evidence, not from the language model.")

with st.sidebar:
    st.header("Example questions")
    for ex in EXAMPLES:
        st.button(ex, on_click=_set_question, args=(ex,), use_container_width=True)
    st.divider()
    st.caption("Old dropdown page: run `streamlit run app.py` in the terminal.")
    st.caption("Local Qwen (Ollama) writes only a short draft summary. "
               "Everything else is built by code. Not financial advice.")

question = st.text_area("Ask anything about NVIDIA...", key="question", height=110)
go = st.button("Analyze", type="primary")

if go:
    if question.strip():
        _analyze(question.strip())
    else:
        st.warning("Please type a question first.")

res = st.session_state.get("result")
if res:
    out, report = res["out"], res["report"]
    st.subheader("Question")
    st.write(_esc(res["question"]))

    st.subheader("Analysis plan")
    for r in out["agent_runs"]:
        st.write(_esc(f"{_icon(r['status'])} {r['agent']}: {r['status']} ({r['duration_seconds']:.1f}s)"))
    if out["errors"]:
        st.warning("Problems during this run: " + _esc("; ".join(out["errors"])))

    st.divider()
    st.markdown(_esc(report["markdown"]))

    with st.expander("Raw evidence passed to the report (JSON)"):
        st.json(out.get("context") or {})
    with st.expander("Timing and technical details"):
        st.json(out["metadata"])
