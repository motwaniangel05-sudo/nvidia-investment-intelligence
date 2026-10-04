"""
Helpers for the Streamlit page (app.py). All the logic lives here so it can be
tested; app.py only draws things on the screen.
"""

from typing import Dict, List

from core.config_loader import load_config
from core.orchestrator import run_query

STATUS_LABELS = {
    "complete": "Complete",
    "success": "Success",
    "partial": "Partial",
    "failed": "Failed",
}

# These example questions are known to match real agents.
EXAMPLE_QUERIES = [
    "revenue growth margin",
    "Analyze NVIDIA's financial health and valuation",
    "company overview revenue growth margin",
]


def default_ticker() -> str:
    """The company ticker set in config.yaml."""
    return load_config()["company"]["ticker"]


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status.title())


def headline_rows(report) -> List[Dict]:
    """One table row per headline finding, across all agents."""
    rows = []
    for name, agent in report.agents.items():
        for h in agent["headline_findings"]:
            rows.append({
                "Agent": name,
                "Finding": h["claim"],
                "Confidence": h["confidence"],
                "Source": h["source_form"],
                "Red-Team": ", ".join(h["issues"]) or "OK",
            })
    return rows


def agent_summaries(report) -> List[Dict]:
    """One short summary row per agent."""
    return [
        {
            "Agent": name,
            "Status": status_label(agent["status"]),
            "Findings": agent["finding_count"],
            "Flagged": agent["flagged_finding_count"],
        }
        for name, agent in report.agents.items()
    ]


def generate_report(ticker: str, query: str, run_verification: bool = True):
    """Clean up the inputs, run the full pipeline, and return the SynthesisReport."""
    ticker = ticker.strip().upper()
    query = query.strip()
    if not ticker:
        raise ValueError("Please enter a company ticker.")
    if not query:
        raise ValueError("Please type a question.")
    response = run_query(
        ticker, query,
        run_verification_step=run_verification,
        synthesize_report=True,
    )
    return response["synthesis"]


# ---- animations (CSS only) ----
ANIMATION_CSS = """
<style>
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmer{from{background-position:0% 50%}to{background-position:100% 50%}}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(118,185,0,.55)}50%{box-shadow:0 0 18px 4px rgba(118,185,0,.45)}}
.fade{opacity:0;animation:fadeUp .6s ease forwards}
.hero{background-size:200% 100%;animation:shimmer 7s ease-in-out infinite alternate}
.card{transition:transform .2s ease,box-shadow .2s ease}
.card:hover{transform:translateY(-3px);box-shadow:0 6px 22px rgba(118,185,0,.28)}
div.stButton>button{animation:glow 2.6s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){
  .fade{animation:none;opacity:1}
  .hero,div.stButton>button{animation:none}
  .card{transition:none}
}
</style>
"""


def fade_in_html(inner: str, index: int = 0) -> str:
    """Wrap a piece of HTML so it fades in. Later items start a little later."""
    delay = min(max(index, 0), 12) * 0.08
    return f'<div class="fade" style="animation-delay:{delay:.2f}s">{inner}</div>'


# ---- NVIDIA look: colors and small HTML pieces for the page ----
from html import escape as _escape

NVIDIA_GREEN = "#76B900"
AMBER = "#F5A623"
RED = "#E5484D"
GREY = "#9AA0A6"

STATUS_COLORS = {
    "complete": NVIDIA_GREEN,
    "success": NVIDIA_GREEN,
    "partial": AMBER,
    "failed": RED,
}

PAGE_CSS = """
<style>
.hero{background:linear-gradient(90deg,#76b900 0%,#4a7a00 55%,#0b0b0b 100%);
  padding:1.6rem 2rem;border-radius:14px;margin-bottom:1.2rem}
.hero h1{color:#0b0b0b;margin:0;font-size:2.1rem}
.hero p{color:#101010;margin:.3rem 0 0 0;font-weight:500}
.card{background:#1a1a1a;border:1px solid #2c2c2c;border-left:5px solid #76b900;
  border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.7rem}
.card h4{margin:0 0 .4rem 0;color:#fff}
.muted{color:#a0a0a0;font-size:.85rem}
.stat{font-size:1.8rem;font-weight:800;color:#fff}
.badge{display:inline-block;padding:.12rem .6rem;border-radius:999px;
  color:#0b0b0b;font-weight:700;font-size:.78rem}
.caveat{background:#2a2410;border-left:5px solid #f5a623;border-radius:10px;
  padding:.7rem 1rem;margin-bottom:.5rem;color:#f2e6c2}
.section{color:#76b900;font-weight:700;letter-spacing:.04em;
  text-transform:uppercase;margin:1.4rem 0 .6rem}
div.stButton>button{background:#76b900;color:#0b0b0b;border:0;font-weight:700}
div.stButton>button:hover{background:#8fd400;color:#0b0b0b}
</style>
"""


def status_color(status: str) -> str:
    return STATUS_COLORS.get(status, GREY)


def redteam_color(result_text: str) -> str:
    """OK is green, a plain FLAGGED is orange, anything worse is red."""
    if result_text == "OK":
        return NVIDIA_GREEN
    if result_text == "FLAGGED":
        return AMBER
    return RED


def badge_html(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{_escape(text)}</span>'


def hero_html() -> str:
    return (
        '<div class="hero"><h1>NVIDIA Investment Intelligence</h1>'
        "<p>Ask a question. The agents analyze the data and the Red-Team checks their work.</p></div>"
    )


def section_html(title: str) -> str:
    return f'<div class="section">{_escape(title)}</div>'


def stat_html(label: str, value, color: str = NVIDIA_GREEN) -> str:
    return (
        f'<div class="card" style="border-left-color:{color}">'
        f'<div class="muted">{_escape(str(label))}</div>'
        f'<div class="stat">{_escape(str(value))}</div></div>'
    )


def agent_card_html(name: str, agent: dict) -> str:
    color = status_color(agent["status"])
    return (
        f'<div class="card" style="border-left-color:{color}">'
        f"<h4>{_escape(name)}</h4>"
        f"{badge_html(status_label(agent['status']), color)}"
        f'<div class="muted" style="margin-top:.5rem">'
        f"{agent['finding_count']} findings &middot; "
        f"{agent['flagged_finding_count']} flagged</div></div>"
    )


def finding_card_html(row: dict) -> str:
    color = redteam_color(row["Red-Team"])
    return (
        f'<div class="card" style="border-left-color:{color}">'
        f'<div class="muted">{_escape(row["Agent"])} &middot; {_escape(row["Source"])}'
        f" &middot; confidence {_escape(str(row['Confidence']))}</div>"
        f"<div>{_escape(row['Finding'])}</div>"
        f'<div style="margin-top:.4rem">{badge_html("Red-Team: " + row["Red-Team"], color)}</div></div>'
    )


def caveat_html(text: str) -> str:
    return f'<div class="caveat">{_escape(text)}</div>'


# ---- charts, calmer look and empty state ----
import pandas as pd

from tools.financial_tools import calculate_margins, get_annual_series
from tools.valuation_tools import run_dcf_scenarios

CHART_COLORS = [NVIDIA_GREEN, AMBER, "#4DA3FF"]

MARGIN_LABELS = {
    "gross_margin": "Gross margin %",
    "operating_margin": "Operating margin %",
    "net_margin": "Net margin %",
}

CALM_CSS = """
<style>
.hero{padding:.9rem 1.4rem !important;margin-bottom:.8rem !important}
.hero h1{font-size:1.6rem !important}
div.stButton>button,div.stButton>button[kind="primary"]{
  background:#2d4a0a !important;color:#e6f4c8 !important;
  border:1px solid #5b8c00 !important;animation:none !important;
  box-shadow:none !important;font-weight:600 !important}
div.stButton>button:hover{
  background:#3a5f0e !important;border-color:#76b900 !important;color:#ffffff !important}
</style>
"""


def _safe(fn, *args):
    """Run fn; return None instead of crashing the page when data is missing."""
    try:
        return fn(*args)
    except Exception:
        return None


def revenue_chart_data(ticker: str):
    series = _safe(get_annual_series, "revenue", ticker)
    if not series:
        return None
    return pd.DataFrame(
        {"Revenue ($B)": {p[:4]: v / 1e9 for p, v in series.items()}}
    ).sort_index()


def margin_chart_data(ticker: str):
    margins = _safe(calculate_margins, ticker)
    if not margins:
        return None
    data = {
        label: {p[:4]: v for p, v in margins[key].items()}
        for key, label in MARGIN_LABELS.items()
        if margins.get(key)
    }
    if not data:
        return None
    return pd.DataFrame(data).sort_index()


def valuation_chart_data(ticker: str):
    scenarios = _safe(run_dcf_scenarios, ticker)
    if not scenarios:
        return None
    # The number in front keeps the bars in scenario order (charts sort by name).
    return pd.DataFrame({
        "Enterprise value ($T)": {
            f"{i}. {name.replace('_', ' ')}": result["enterprise_value"] / 1e12
            for i, (name, result) in enumerate(scenarios.items(), start=1)
        }
    })


def agent_chart_data(report):
    if not report.agents:
        return None
    return pd.DataFrame({
        "OK findings": {
            n: a["finding_count"] - a["flagged_finding_count"]
            for n, a in report.agents.items()
        },
        "Flagged": {n: a["flagged_finding_count"] for n, a in report.agents.items()},
    })


def build_charts(ticker: str) -> dict:
    return {
        "revenue": revenue_chart_data(ticker),
        "margins": margin_chart_data(ticker),
        "valuation": valuation_chart_data(ticker),
    }


def empty_state_html() -> str:
    return (
        '<div class="card"><h4>Ready when you are</h4>'
        '<div class="muted">Choose a question in the left panel, '
        "then press Generate report.</div></div>"
    )


# ---- icons and bars ----
AGENT_ICONS = {
    "FinancialAgent": "💰",
    "ValuationAgent": "⚖️",
    "RiskAgent": "⚠️",
    "NewsAgent": "📰",
    "MarketAgent": "🏁",
    "ResearchAgent": "🔎",
}
DEFAULT_AGENT_ICON = "🤖"

STATUS_ICONS = {"complete": "✅", "success": "✅", "partial": "⚠️", "failed": "❌"}
DEFAULT_STATUS_ICON = "ℹ️"

ICON_CSS = """
<style>
.icon{font-size:1.5rem;margin-right:.5rem}
.row{display:flex;align-items:center;gap:.9rem}
.bar{height:8px;border-radius:999px;background:#2c2c2c;overflow:hidden;margin-top:.6rem}
.bar>div{height:100%;border-radius:999px}
</style>
"""


def agent_icon(name: str) -> str:
    return AGENT_ICONS.get(name, DEFAULT_AGENT_ICON)


def status_icon(status: str) -> str:
    return STATUS_ICONS.get(status, DEFAULT_STATUS_ICON)


def bar_html(percent: float, color: str) -> str:
    """A thin progress bar. The percent is kept between 0 and 100."""
    width = max(0.0, min(100.0, percent))
    return f'<div class="bar"><div style="width:{width:.0f}%;background:{color}"></div></div>'


def ok_share(agent: dict) -> float:
    """Percent of an agent's findings that the Red-Team did not flag."""
    total = agent["finding_count"]
    if total == 0:
        return 0.0
    return (total - agent["flagged_finding_count"]) / total * 100


def section_icon_html(icon: str, title: str) -> str:
    return f'<div class="section">{icon} {_escape(title)}</div>'


def icon_stat_html(icon: str, label: str, value, color: str = NVIDIA_GREEN) -> str:
    return (
        f'<div class="card" style="border-left-color:{color}"><div class="row">'
        f'<div style="font-size:2.2rem">{icon}</div>'
        f'<div><div class="muted">{_escape(str(label))}</div>'
        f'<div class="stat">{_escape(str(value))}</div></div></div></div>'
    )


def icon_agent_card_html(name: str, agent: dict) -> str:
    color = status_color(agent["status"])
    pct = ok_share(agent)
    badge = badge_html(f"{status_icon(agent['status'])} {status_label(agent['status'])}", color)
    return (
        f'<div class="card" style="border-left-color:{color}">'
        f'<div><span class="icon">{agent_icon(name)}</span><b>{_escape(name)}</b></div>'
        f'<div style="margin-top:.4rem">{badge}</div>'
        f'<div class="muted" style="margin-top:.5rem">'
        f"{agent['finding_count']} findings &middot; {agent['flagged_finding_count']} flagged</div>"
        f"{bar_html(pct, color)}"
        f'<div class="muted">{pct:.0f}% passed Red-Team</div></div>'
    )


# ---- theme v2: black/green background, glass cards, count-up numbers ----
from core.orchestrator import AGENT_CLASSES

THEME_CSS = """
<style>
@property --num{syntax:'<integer>';initial-value:0;inherits:false}
@keyframes bgShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes floatA{0%,100%{transform:translate(0,0)}50%{transform:translate(60px,-40px)}}
@keyframes floatB{0%,100%{transform:translate(0,0)}50%{transform:translate(-50px,50px)}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes titleShine{from{background-position:0% 50%}to{background-position:200% 50%}}
@keyframes rise{from{height:14%}to{height:100%}}
@keyframes countup{from{--num:0}}
@keyframes grow{from{width:0}}
.stApp{background:linear-gradient(135deg,#000000 0%,#031207 35%,#0b2e0f 65%,#000000 100%) !important;background-size:300% 300% !important;animation:bgShift 22s ease-in-out infinite}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background:radial-gradient(circle at 15% 20%,rgba(118,185,0,.22),transparent 40%),radial-gradient(circle at 85% 75%,rgba(118,185,0,.16),transparent 45%);animation:floatA 16s ease-in-out infinite}
.stApp::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background-image:linear-gradient(rgba(118,185,0,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(118,185,0,.06) 1px,transparent 1px);background-size:46px 46px;animation:floatB 30s ease-in-out infinite}
.block-container,[data-testid="stMainBlockContainer"]{position:relative;z-index:1;max-width:1100px;padding-top:1.5rem}
header[data-testid="stHeader"]{background:transparent}
.hero2{text-align:center;padding:.4rem 0 1.2rem}
.hero2-icon{font-size:3rem;display:inline-block;animation:bob 3.2s ease-in-out infinite;filter:drop-shadow(0 0 14px rgba(118,185,0,.7))}
.hero2 h1{font-size:2.8rem;margin:.2rem 0;background:linear-gradient(90deg,#ffffff,#76b900,#ffffff);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:titleShine 6s linear infinite}
.hero2 p{color:#b9d98a;font-size:1.1rem;margin:0}
.bars{display:flex;gap:6px;justify-content:center;align-items:flex-end;height:46px;margin-top:1rem}
.bars span{width:9px;border-radius:3px 3px 0 0;background:linear-gradient(180deg,#9be15d,#3d6100);animation:rise 1.4s ease-in-out infinite alternate;animation-delay:calc(var(--i) * .13s)}
.glass{background:rgba(255,255,255,.07);backdrop-filter:blur(10px);border:1px solid rgba(118,185,0,.35);border-radius:18px;padding:1.1rem .8rem;text-align:center;transition:transform .2s ease,box-shadow .2s ease}
.glass:hover{transform:translateY(-4px);box-shadow:0 8px 26px rgba(118,185,0,.3)}
.count{--num:0;counter-reset:n var(--num);animation:countup 1.8s ease-out;font-size:2.4rem;font-weight:800;color:#ffffff}
.count::after{content:counter(n)}
.glass-label{letter-spacing:.14em;text-transform:uppercase;color:#b9d98a;font-size:.8rem;margin-top:.2rem}
div[data-baseweb="tab-list"]{gap:1rem;background:transparent}
button[data-baseweb="tab"]{flex:1;height:3.4rem;border-radius:14px;background:rgba(255,255,255,.08);border:1px solid rgba(118,185,0,.35);transition:background .2s ease,transform .2s ease}
button[data-baseweb="tab"]:hover{background:rgba(118,185,0,.18);transform:translateY(-2px)}
button[data-baseweb="tab"] p{color:#e8f5d0 !important;font-size:1.05rem;font-weight:700}
button[data-baseweb="tab"][aria-selected="true"]{background:#f4f7ee}
button[data-baseweb="tab"][aria-selected="true"] p{color:#1f3300 !important}
div[data-baseweb="tab-highlight"],div[data-baseweb="tab-border"]{display:none}
div[data-baseweb="tab-panel"]{padding-top:1.2rem}
.st-key-setup{background:rgba(255,255,255,.06);backdrop-filter:blur(12px);border:1px solid rgba(118,185,0,.35);border-radius:22px;padding:1.4rem 2rem 1.6rem;animation:fadeUp .7s ease both}
.form-title{text-align:center;font-size:1.7rem;font-weight:800;color:#ffffff;margin:0}
.form-sub{text-align:center;color:#a8c487;margin:.2rem 0 .8rem}
.form-line{height:2px;background:linear-gradient(90deg,transparent,#76b900,transparent);margin:.6rem 0 1rem}
.bar>div{animation:grow 1.2s ease-out}
</style>
"""


def hero_v2_html() -> str:
    bars = "".join(f'<span style="--i:{i}"></span>' for i in range(9))
    return (
        '<div class="hero2"><div class="hero2-icon">📈</div>'
        "<h1>NVIDIA Investment Intelligence</h1>"
        "<p>AI-Powered Multi-Agent Investment Analysis</p>"
        f'<div class="bars">{bars}</div></div>'
    )


def stat_glass_html(value: int, label: str) -> str:
    """A glass card whose number counts up from zero (pure CSS)."""
    return (
        '<div class="glass">'
        f'<div class="count" style="--num:{max(0, int(value))}"></div>'
        f'<div class="glass-label">{_escape(str(label))}</div></div>'
    )


def hero_stats(ticker: str) -> list:
    """Three real numbers for the top cards: agents, companies, years of data."""
    config = _safe(load_config) or {}
    companies = 1 + len(config.get("competitors", []))
    years = len(_safe(get_annual_series, "revenue", ticker) or {})
    return [
        (len(AGENT_CLASSES), "Agents"),
        (companies, "Companies"),
        (years, "Fiscal years"),
    ]


def form_header_html(title: str, subtitle: str) -> str:
    return (
        f'<div class="form-title">{_escape(title)}</div>'
        f'<div class="form-sub">{_escape(subtitle)}</div>'
        '<div class="form-line"></div>'
    )
