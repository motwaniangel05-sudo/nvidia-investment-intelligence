"""
Streamlit page for the NVIDIA Investment Intelligence project.
Run it with:  streamlit run app.py
"""

import streamlit as st

from core.report_view import (
    AMBER,
    ANIMATION_CSS,
    CALM_CSS,
    CHART_COLORS,
    EXAMPLE_QUERIES,
    ICON_CSS,
    NVIDIA_GREEN,
    PAGE_CSS,
    THEME_CSS,
    agent_chart_data,
    build_charts,
    caveat_html,
    default_ticker,
    empty_state_html,
    fade_in_html,
    finding_card_html,
    form_header_html,
    generate_report,
    headline_rows,
    hero_stats,
    hero_v2_html,
    icon_agent_card_html,
    icon_stat_html,
    section_icon_html,
    stat_glass_html,
    status_color,
    status_icon,
    status_label,
)

st.set_page_config(page_title="NVIDIA Investment Intelligence", page_icon="🟩", layout="wide")
for css in (PAGE_CSS, ANIMATION_CSS, CALM_CSS, ICON_CSS, THEME_CSS):
    st.markdown(css, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_stats(ticker):
    return hero_stats(ticker)


@st.cache_data(show_spinner="Loading charts...")
def cached_charts(ticker):
    return build_charts(ticker)


def show_results(report):
    issues = report.verification_summary["total_issues"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(fade_in_html(icon_stat_html(
        status_icon(report.overall_status), "Overall status",
        status_label(report.overall_status), status_color(report.overall_status)), 0),
        unsafe_allow_html=True)
    c2.markdown(fade_in_html(icon_stat_html(
        "🛡️", "Red-Team issues", issues, NVIDIA_GREEN if issues == 0 else AMBER), 1),
        unsafe_allow_html=True)
    c3.markdown(fade_in_html(icon_stat_html(
        "🤖", "Agents run", len(report.agents), NVIDIA_GREEN), 2),
        unsafe_allow_html=True)

    st.markdown(section_icon_html("🤖", "Agents"), unsafe_allow_html=True)
    columns = st.columns(3)
    for i, (name, agent) in enumerate(report.agents.items()):
        columns[i % 3].markdown(
            fade_in_html(icon_agent_card_html(name, agent), i), unsafe_allow_html=True
        )

    agent_data = agent_chart_data(report)
    if agent_data is not None:
        st.markdown(section_icon_html("📊", "Findings per agent"), unsafe_allow_html=True)
        st.bar_chart(agent_data, color=[NVIDIA_GREEN, AMBER], height=260)

    st.markdown(section_icon_html("🔍", "Top findings"), unsafe_allow_html=True)
    for i, row in enumerate(headline_rows(report)):
        st.markdown(fade_in_html(finding_card_html(row), i), unsafe_allow_html=True)

    st.markdown(section_icon_html("⚠️", "Warnings"), unsafe_allow_html=True)
    if report.caveats:
        for i, caveat in enumerate(report.caveats):
            st.markdown(fade_in_html(caveat_html(caveat), i), unsafe_allow_html=True)
    else:
        st.info("No warnings.")
    with st.expander("Plain-text version"):
        st.text(report.text)


# ---- header and the three number cards ----
st.markdown(hero_v2_html(), unsafe_allow_html=True)
stat_columns = st.columns(3)
for column, (value, label) in zip(stat_columns, cached_stats(default_ticker())):
    column.markdown(stat_glass_html(value, label), unsafe_allow_html=True)

st.write("")
tab_analysis, tab_charts = st.tabs(["📊 Analysis", "📈 Charts"])

with tab_analysis:
    with st.container(key="setup"):
        st.markdown(
            form_header_html("Analysis Setup", "Choose a question for the agents to answer"),
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        ticker = left.text_input("🏢 Company ticker", value=default_ticker())
        CUSTOM = "(type my own question)"
        example = right.selectbox("💡 Example questions", [CUSTOM] + EXAMPLE_QUERIES)
        query = st.text_input(
            "❓ Your question",
            value="" if example == CUSTOM else example,
            key=f"query_{example}",  # a new key makes the box refill when you pick an example
        )
        verify = st.checkbox("🛡️ Run Red-Team verification", value=True)
        _, middle, _ = st.columns([1, 1, 1])
        go = middle.button("🚀 Generate report", type="primary")

    if go:
        try:
            with st.spinner("Running the agents..."):
                new_report = generate_report(ticker, query, run_verification=verify)
        except ValueError as problem:
            st.warning(str(problem))
        except Exception as problem:  # show the problem instead of a crash screen
            st.error(f"Something went wrong: {problem}")
        else:
            st.session_state["report"] = new_report  # kept, so it survives other clicks

    report = st.session_state.get("report")
    if report is None:
        st.markdown(empty_state_html(), unsafe_allow_html=True)
    else:
        show_results(report)

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
            st.line_chart(
                charts["margins"], color=CHART_COLORS[: len(charts["margins"].columns)], height=300
            )

    st.markdown(section_icon_html("⚖️", "DCF scenarios: enterprise value ($T)"), unsafe_allow_html=True)
    if charts["valuation"] is None:
        st.info("No valuation data found.")
    else:
        st.bar_chart(charts["valuation"], color=NVIDIA_GREEN, height=300)
        st.caption(
            "These use operating cash flow as a stand-in for free cash flow, "
            "so the values are likely too high."
        )
