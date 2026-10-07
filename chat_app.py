"""AI Investment Intelligence: question page with the original NVIDIA look.
Run:  streamlit run chat_app.py"""
import html

import streamlit as st

from core.dynamic_pipeline import run
from core.report_builder import build_report
from core.report_view import (
    AMBER,
    ANIMATION_CSS,
    CALM_CSS,
    CHART_COLORS,
    ICON_CSS,
    NVIDIA_GREEN,
    PAGE_CSS,
    THEME_CSS,
    agent_icon,
    badge_html,
    build_charts,
    default_ticker,
    fade_in_html,
    form_header_html,
    hero_stats,
    hero_v2_html,
    icon_stat_html,
    section_icon_html,
    stat_glass_html,
)

st.set_page_config(page_title="AI Investment Intelligence", page_icon="🟩", layout="wide")
for css in (PAGE_CSS, ANIMATION_CSS, CALM_CSS, ICON_CSS, THEME_CSS):
    st.markdown(css, unsafe_allow_html=True)

EXAMPLES = [
    "What is NVIDIA's current financial health?",
    "Should I buy NVIDIA at the current valuation?",
    "NVIDIA is trading at $500, yesterday it was $200, and my average price is $120. Should I buy more or sell?",
    "Compare NVIDIA with AMD.",
    "What are NVIDIA's biggest risks right now?",
    "What does the DCF valuation indicate?",
]
CUSTOM = "(type my own question)"
RED = "#e5534b"
SECTION_ICONS = [
    ("Executive", "🎯"), ("AI Draft", "🤖"), ("What The Data", "📊"), ("Financial", "💵"),
    ("Market", "🏁"), ("Valuation", "⚖️"), ("Risks", "⚠️"), ("Recent", "📰"),
    ("Verification", "🛡️"), ("Bull", "🐂"), ("Bear", "🐻"), ("Key Uncertainties", "❓"),
    ("Final", "✅"), ("Evidence", "📚"),
]


@st.cache_data(show_spinner=False)
def cached_stats(ticker):
    return hero_stats(ticker)


@st.cache_data(show_spinner="Loading charts...")
def cached_charts(ticker):
    return build_charts(ticker)


def _esc(text):
    return str(text).replace("$", "\\$")


def _status_color(status):
    return {"success": NVIDIA_GREEN, "partial": AMBER}.get(status, RED)


def _agent_icon(name):
    try:
        return agent_icon(name)
    except Exception:
        return "🤖"


def _agent_card(r):
    color = _status_color(r["status"])
    err = f" &middot; {html.escape(str(r['error'])[:90])}" if r["error"] else ""
    return (f'<div class="card" style="border-left-color:{color}">'
            f'<h4>{_agent_icon(r["agent_name"])} {html.escape(r["agent_name"])}</h4>'
            f'{badge_html(r["status"].capitalize(), color)}'
            f'<div class="muted" style="margin-top:.5rem">{r["duration_seconds"]:.1f}s{err}</div></div>')


def _section_icon(title):
    for start, icon in SECTION_ICONS:
        if title.startswith(start):
            return icon
    return "▪️"


def _sections(markdown):
    out = []
    for part in markdown.split("\n## ")[1:]:
        title, _, body = part.partition("\n")
        out.append((title.strip(), body.strip()))
    return out


def _analyze(q, ticker):
    status = st.status("Working...", expanded=True)

    def progress(stage, info):
        if stage == "plan":
            status.write(_esc(f"✅ Plan ready ({info['intent']}): " + ", ".join(info["agents"])))
            status.write("⏳ Running the agents in parallel...")
        elif stage == "agents":
            for r in info:
                extra = f" - {str(r['error'])[:80]}" if r["error"] else ""
                icon = "❌" if r["status"] == "failed" else ("⚠️" if r["status"] == "partial" else "✅")
                status.write(_esc(f"{icon} {r['agent']}: {r['status']} ({r['duration_seconds']:.1f}s){extra}"))
        elif stage == "evidence":
            status.write(f"✅ Evidence built: {info['sources']} source group(s), "
                         f"{info['conflicts']} conflict(s)")

    try:
        out = run(q, company_ticker=ticker or None, progress=progress,
                  synthesizer=lambda query, context: {"answer": "", "source": "skipped"})
        status.write("⏳ Qwen is writing the short draft summary (can take up to a minute)...")
        report = build_report(q, out.get("context") or {}, out["agent_runs"], out["errors"])
        status.update(label="Done", state="complete", expanded=False)
        st.session_state["result"] = {"question": q, "out": out, "report": report}
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.error(f"The analysis failed: {exc}")


def _show_result(res):
    out, report = res["out"], res["report"]
    runs = out["agent_runs"]
    failed = sum(1 for r in runs if r["status"] == "failed")
    c1, c2, c3 = st.columns(3)
    c1.markdown(fade_in_html(icon_stat_html("🤖", "Agents run", len(runs), NVIDIA_GREEN), 0),
                unsafe_allow_html=True)
    c2.markdown(fade_in_html(icon_stat_html(
        "✅" if failed == 0 else "⚠️", "Agents failed", failed,
        NVIDIA_GREEN if failed == 0 else AMBER), 1), unsafe_allow_html=True)
    c3.markdown(fade_in_html(icon_stat_html(
        "🧮", "Evidence confidence", report["confidence"], NVIDIA_GREEN), 2), unsafe_allow_html=True)

    st.markdown(section_icon_html("❓", "Question"), unsafe_allow_html=True)
    st.write(_esc(res["question"]))

    st.markdown(section_icon_html("🤖", "Agents selected and run"), unsafe_allow_html=True)
    columns = st.columns(3)
    for i, r in enumerate(runs):
        columns[i % 3].markdown(fade_in_html(_agent_card(r), i), unsafe_allow_html=True)
    if out["errors"]:
        st.warning("Problems during this run: " + _esc("; ".join(out["errors"])))

    for title, body in _sections(report["markdown"]):
        st.markdown(section_icon_html(_section_icon(title), title), unsafe_allow_html=True)
        st.markdown(_esc(body))

    with st.expander("Raw evidence passed to the report (JSON)"):
        st.json(out.get("context") or {})
    with st.expander("Timing and technical details"):
        st.json(out["metadata"])


# ---- header and the three number cards ----
st.markdown(hero_v2_html(), unsafe_allow_html=True)
for column, (value, label) in zip(st.columns(3), cached_stats(default_ticker())):
    column.markdown(stat_glass_html(value, label), unsafe_allow_html=True)

st.write("")
tab_analysis, tab_charts = st.tabs(["📊 Analysis", "📈 Charts"])

with tab_analysis:
    with st.container(key="setup"):
        st.markdown(
            form_header_html("Analysis Setup", "Ask anything about NVIDIA. The system picks the agents."),
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        ticker = left.text_input("🏢 Company ticker", value=default_ticker())
        left.caption("Filing text (risks, research) exists only for NVDA.")
        example = right.selectbox("💡 Example questions", [CUSTOM] + EXAMPLES)
        question = st.text_input(
            "❓ Ask anything about NVIDIA...",
            value="" if example == CUSTOM else example,
            key=f"question_{example}",
        )
        _, middle, _ = st.columns([1, 1, 1])
        go = middle.button("🚀 Analyze", type="primary")

    if go:
        if question.strip():
            _analyze(question.strip(), ticker.strip().upper())
        else:
            st.warning("Please type a question first.")

    result = st.session_state.get("result")
    if result is None:
        st.markdown(
            '<div class="card"><h4>Ready when you are</h4>'
            '<div class="muted">Type a question above and click Analyze. '
            'The system chooses the agents it needs and shows what each one did.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        _show_result(result)

with tab_charts:
    charts = cached_charts(ticker.strip().upper() or default_ticker())
    row1 = st.columns(2)
    with row1[0]:
        st.markdown(section_icon_html("💵", "Revenue by fiscal year ($B)"), unsafe_allow_html=True)
        if charts["revenue"] is None:
            st.info("No revenue data found.")
        else:
            st.bar_chart(charts["revenue"], color=NVIDIA_GREEN, height=300)
    with row1[1]:
        st.markdown(section_icon_html("📈", "Margins over time (%)"), unsafe_allow_html=True)
        if charts["margins"] is None:
            st.info("No margin data found.")
        else:
            st.line_chart(charts["margins"], color=CHART_COLORS[: len(charts["margins"].columns)], height=300)

    st.markdown(section_icon_html("⚖️", "DCF scenarios: enterprise value ($T)"), unsafe_allow_html=True)
    if charts["valuation"] is None:
        st.info("No valuation data found.")
    else:
        st.bar_chart(charts["valuation"], color=NVIDIA_GREEN, height=300)
        st.caption("These use operating cash flow as a stand-in for free cash flow, "
                   "so the values are likely too high.")
