"""Tests for the helpers behind the Streamlit page."""

import py_compile
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.red_team_agent import VerificationResult
from agents.synthesis_agent import synthesize
from core import report_view as rv
from core.schemas import AgentResult, Finding


def _report():
    good = Finding("good claim", "e", "c1", "XBRL_metrics", "2026-01-25", 1.0)
    bad = Finding("bad claim", "e", "c2", "10-K", "2025-01-26", 0.5)
    results = {"FinancialAgent": AgentResult(agent_name="FinancialAgent", task="t", findings=[good, bad])}
    verification = [VerificationResult(
        claim="bad claim", evidence="e", verification_result="FLAGGED",
        confidence=0.5, problem="p", correction="c", source_agent="FinancialAgent",
    )]
    return synthesize({
        "query": "q", "company_ticker": "NVDA", "agents_activated": ["FinancialAgent"],
        "agent_results": results, "verification": verification,
    })


@pytest.mark.parametrize("status, label", [
    ("complete", "Complete"), ("success", "Success"),
    ("partial", "Partial"), ("failed", "Failed"), ("weird", "Weird"),
])
def test_status_label(status, label):
    assert rv.status_label(status) == label


def test_default_ticker_reads_the_config():
    with patch.object(rv, "load_config", return_value={"company": {"ticker": "NVDA"}}):
        assert rv.default_ticker() == "NVDA"


def test_headline_rows_show_agent_finding_and_redteam_result():
    rows = rv.headline_rows(_report())

    assert rows == [
        {"Agent": "FinancialAgent", "Finding": "good claim", "Confidence": 1.0,
         "Source": "XBRL_metrics", "Red-Team": "OK"},
        {"Agent": "FinancialAgent", "Finding": "bad claim", "Confidence": 0.5,
         "Source": "10-K", "Red-Team": "FLAGGED"},
    ]


def test_headline_rows_empty_when_no_agents():
    assert rv.headline_rows(synthesize({})) == []


def test_agent_summaries_show_status_counts_and_flags():
    assert rv.agent_summaries(_report()) == [
        {"Agent": "FinancialAgent", "Status": "Success", "Findings": 2, "Flagged": 1},
    ]


def test_generate_report_cleans_inputs_and_returns_the_synthesis():
    with patch.object(rv, "run_query", return_value={"synthesis": "REPORT"}) as mock_run:
        out = rv.generate_report(" nvda ", "  revenue growth margin ", run_verification=False)

    assert out == "REPORT"
    mock_run.assert_called_once_with(
        "NVDA", "revenue growth margin",
        run_verification_step=False, synthesize_report=True,
    )


def test_generate_report_runs_verification_by_default():
    with patch.object(rv, "run_query", return_value={"synthesis": "R"}) as mock_run:
        rv.generate_report("NVDA", "q")
    assert mock_run.call_args.kwargs["run_verification_step"] is True


@pytest.mark.parametrize("ticker, query, message", [
    ("", "revenue growth margin", "ticker"),
    ("   ", "revenue growth margin", "ticker"),
    ("NVDA", "", "question"),
    ("NVDA", "   ", "question"),
])
def test_generate_report_rejects_blank_inputs_without_running(ticker, query, message):
    with patch.object(rv, "run_query") as mock_run:
        with pytest.raises(ValueError, match=message):
            rv.generate_report(ticker, query)
    mock_run.assert_not_called()


def test_example_queries_are_not_empty():
    assert rv.EXAMPLE_QUERIES and all(q.strip() for q in rv.EXAMPLE_QUERIES)


def test_app_file_has_no_syntax_errors():
    app = Path(__file__).resolve().parent.parent / "app.py"
    py_compile.compile(str(app), doraise=True)


# ---- animations ----

def test_fade_in_wraps_content_and_starts_first_item_immediately():
    out = rv.fade_in_html("<p>hi</p>", 0)
    assert out == '<div class="fade" style="animation-delay:0.00s"><p>hi</p></div>'


def test_fade_in_delay_grows_with_position():
    assert "animation-delay:0.24s" in rv.fade_in_html("x", 3)


def test_fade_in_delay_is_capped_and_never_negative():
    assert "animation-delay:0.96s" in rv.fade_in_html("x", 100)
    assert "animation-delay:0.00s" in rv.fade_in_html("x", -5)


def test_animation_css_has_keyframes_hover_and_reduced_motion_rule():
    css = rv.ANIMATION_CSS
    assert "@keyframes fadeUp" in css
    assert "@keyframes shimmer" in css
    assert "@keyframes glow" in css
    assert ".card:hover" in css
    assert "prefers-reduced-motion" in css


# ---- NVIDIA look ----
import tomllib


def test_status_color_known_and_unknown():
    assert rv.status_color("success") == rv.NVIDIA_GREEN
    assert rv.status_color("complete") == rv.NVIDIA_GREEN
    assert rv.status_color("partial") == rv.AMBER
    assert rv.status_color("failed") == rv.RED
    assert rv.status_color("weird") == rv.GREY


@pytest.mark.parametrize("text, color", [
    ("OK", rv.NVIDIA_GREEN),
    ("FLAGGED", rv.AMBER),
    ("UNSUPPORTED", rv.RED),
    ("FLAGGED, UNSUPPORTED", rv.RED),
])
def test_redteam_color(text, color):
    assert rv.redteam_color(text) == color


def test_badge_html_escapes_text_and_sets_color():
    out = rv.badge_html("<b>x</b>", "#123456")
    assert "<b>" not in out
    assert "&lt;b&gt;" in out
    assert "background:#123456" in out


def test_agent_card_html_shows_name_status_and_counts():
    agent = _report().agents["FinancialAgent"]
    out = rv.agent_card_html("FinancialAgent", agent)

    assert "FinancialAgent" in out
    assert "Success" in out
    assert "2 findings" in out
    assert "1 flagged" in out
    assert rv.NVIDIA_GREEN in out


def test_finding_card_html_marks_ok_green_and_flagged_orange():
    rows = rv.headline_rows(_report())
    ok = rv.finding_card_html(rows[0])
    bad = rv.finding_card_html(rows[1])

    assert "Red-Team: OK" in ok and rv.NVIDIA_GREEN in ok
    assert "Red-Team: FLAGGED" in bad and rv.AMBER in bad


def test_finding_card_html_escapes_the_claim():
    row = {"Agent": "A", "Finding": "<script>alert(1)</script>", "Confidence": 1.0,
           "Source": "s", "Red-Team": "OK"}
    out = rv.finding_card_html(row)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_caveat_section_stat_and_hero_pieces():
    assert "&lt;i&gt;" in rv.caveat_html("<i>careful</i>")
    assert "Agents" in rv.section_html("Agents")
    stat = rv.stat_html("Overall status", "Complete", "#abcdef")
    assert "Overall status" in stat and "Complete" in stat and "#abcdef" in stat
    assert "NVIDIA Investment Intelligence" in rv.hero_html()


def test_page_css_uses_nvidia_green():
    assert rv.NVIDIA_GREEN.lower() in rv.PAGE_CSS.lower()


def test_streamlit_theme_uses_nvidia_green():
    path = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    theme = tomllib.loads(path.read_text())["theme"]
    assert theme["primaryColor"] == rv.NVIDIA_GREEN


# ---- charts and calm look ----

def test_safe_returns_result_or_none_on_error():
    def boom():
        raise RuntimeError("x")

    assert rv._safe(lambda x: x + 1, 1) == 2
    assert rv._safe(boom) is None


def test_revenue_chart_data_in_billions_sorted_by_year():
    series = {"2026-01-25": 216e9, "2025-01-26": 130e9}
    with patch.object(rv, "get_annual_series", return_value=series) as mock_get:
        df = rv.revenue_chart_data("NVDA")

    mock_get.assert_called_once_with("revenue", "NVDA")
    assert list(df.index) == ["2025", "2026"]
    assert list(df["Revenue ($B)"]) == pytest.approx([130.0, 216.0])


def test_revenue_chart_data_none_when_no_data_or_error():
    with patch.object(rv, "get_annual_series", return_value={}):
        assert rv.revenue_chart_data("NVDA") is None
    with patch.object(rv, "get_annual_series", side_effect=RuntimeError("db")):
        assert rv.revenue_chart_data("NVDA") is None


def test_margin_chart_data_has_one_column_per_margin_with_data():
    margins = {
        "gross_margin": {"2025-01-26": 75.0, "2026-01-25": 71.0},
        "operating_margin": {"2026-01-25": 60.0},
        "net_margin": {},
    }
    with patch.object(rv, "calculate_margins", return_value=margins):
        df = rv.margin_chart_data("NVDA")

    assert list(df.columns) == ["Gross margin %", "Operating margin %"]
    assert list(df.index) == ["2025", "2026"]
    assert df.loc["2026", "Gross margin %"] == 71.0


def test_margin_chart_data_none_when_nothing_available():
    with patch.object(rv, "calculate_margins", return_value={}):
        assert rv.margin_chart_data("NVDA") is None
    empty = {"gross_margin": {}, "operating_margin": {}, "net_margin": {}}
    with patch.object(rv, "calculate_margins", return_value=empty):
        assert rv.margin_chart_data("NVDA") is None


def test_valuation_chart_data_keeps_scenario_order_in_trillions():
    scenarios = {
        "conservative": {"enterprise_value": 1.32e12},
        "moderate": {"enterprise_value": 1.95e12},
        "historical_cagr": {"enterprise_value": 4.8e12},
    }
    with patch.object(rv, "run_dcf_scenarios", return_value=scenarios):
        df = rv.valuation_chart_data("NVDA")

    assert list(df.index) == ["1. conservative", "2. moderate", "3. historical cagr"]
    assert list(df["Enterprise value ($T)"]) == pytest.approx([1.32, 1.95, 4.8])


def test_valuation_chart_data_none_when_no_scenarios_or_error():
    with patch.object(rv, "run_dcf_scenarios", return_value={}):
        assert rv.valuation_chart_data("NVDA") is None
    with patch.object(rv, "run_dcf_scenarios", side_effect=RuntimeError("x")):
        assert rv.valuation_chart_data("NVDA") is None


def test_agent_chart_data_splits_ok_and_flagged_findings():
    df = rv.agent_chart_data(_report())
    assert df.loc["FinancialAgent", "OK findings"] == 1
    assert df.loc["FinancialAgent", "Flagged"] == 1


def test_agent_chart_data_none_when_no_agents():
    assert rv.agent_chart_data(synthesize({})) is None


def test_build_charts_collects_all_three_for_the_ticker():
    with patch.object(rv, "revenue_chart_data", return_value="R") as r, \
         patch.object(rv, "margin_chart_data", return_value="M") as m, \
         patch.object(rv, "valuation_chart_data", return_value="V") as v:
        out = rv.build_charts("NVDA")

    assert out == {"revenue": "R", "margins": "M", "valuation": "V"}
    r.assert_called_once_with("NVDA")
    m.assert_called_once_with("NVDA")
    v.assert_called_once_with("NVDA")


def test_empty_state_tells_the_user_what_to_do():
    assert "Generate report" in rv.empty_state_html()


def test_calm_css_turns_off_the_glow_and_uses_a_dark_green_button():
    assert "animation:none" in rv.CALM_CSS
    assert "glow" not in rv.CALM_CSS
    assert "#2d4a0a" in rv.CALM_CSS


# ---- icons and bars ----

def test_agent_icon_known_and_unknown():
    assert rv.agent_icon("FinancialAgent") == rv.AGENT_ICONS["FinancialAgent"]
    assert rv.agent_icon("MysteryAgent") == rv.DEFAULT_AGENT_ICON


def test_status_icon_known_and_unknown():
    assert rv.status_icon("success") == rv.STATUS_ICONS["success"]
    assert rv.status_icon("failed") == rv.STATUS_ICONS["failed"]
    assert rv.status_icon("weird") == rv.DEFAULT_STATUS_ICON


def test_bar_html_width_is_kept_between_0_and_100():
    assert "width:50%" in rv.bar_html(50, "#fff")
    assert "width:100%" in rv.bar_html(150, "#fff")
    assert "width:0%" in rv.bar_html(-5, "#fff")
    assert "background:#abcdef" in rv.bar_html(10, "#abcdef")


def test_ok_share_is_percent_of_findings_not_flagged():
    assert rv.ok_share({"finding_count": 4, "flagged_finding_count": 1}) == 75.0
    assert rv.ok_share({"finding_count": 0, "flagged_finding_count": 0}) == 0.0


def test_section_icon_html_shows_icon_and_escapes_title():
    out = rv.section_icon_html("📊", "<b>Charts</b>")
    assert "📊" in out
    assert "<b>" not in out
    assert "&lt;b&gt;Charts" in out


def test_icon_stat_html_shows_icon_label_value_and_color():
    out = rv.icon_stat_html("🛡️", "Red-Team issues", 3, "#abcdef")
    assert "🛡️" in out and "Red-Team issues" in out and ">3<" in out and "#abcdef" in out


def test_icon_agent_card_html_shows_icon_counts_and_bar():
    agent = _report().agents["FinancialAgent"]
    out = rv.icon_agent_card_html("FinancialAgent", agent)

    assert rv.AGENT_ICONS["FinancialAgent"] in out
    assert "FinancialAgent" in out
    assert "2 findings" in out and "1 flagged" in out
    assert "50% passed Red-Team" in out
    assert "width:50%" in out


def test_icon_css_defines_the_bar():
    assert ".bar" in rv.ICON_CSS


# ---- theme v2 ----

def test_hero_v2_has_title_subtitle_and_nine_animated_bars():
    out = rv.hero_v2_html()
    assert "NVIDIA Investment Intelligence" in out
    assert "AI-Powered" in out
    assert out.count('<span style="--i:') == 9


def test_stat_glass_html_sets_the_count_target_and_label():
    out = rv.stat_glass_html(6, "Agents")
    assert "--num:6" in out
    assert "Agents" in out


def test_stat_glass_html_never_goes_below_zero_and_escapes_label():
    assert "--num:0" in rv.stat_glass_html(-3, "x")
    out = rv.stat_glass_html(1, "<b>bad</b>")
    assert "<b>" not in out and "&lt;b&gt;" in out


def test_hero_stats_counts_agents_companies_and_years():
    config = {"competitors": [{"ticker": "AMD"}, {"ticker": "INTC"}]}
    with patch.object(rv, "load_config", return_value=config), \
         patch.object(rv, "get_annual_series", return_value={"2025": 1, "2026": 2}):
        stats = rv.hero_stats("NVDA")

    assert stats == [
        (len(rv.AGENT_CLASSES), "Agents"),
        (3, "Companies"),
        (2, "Fiscal years"),
    ]


def test_hero_stats_falls_back_to_zero_years_and_one_company():
    with patch.object(rv, "load_config", side_effect=RuntimeError("no config")), \
         patch.object(rv, "get_annual_series", side_effect=RuntimeError("no db")):
        stats = rv.hero_stats("NVDA")

    assert stats[1] == (1, "Companies")
    assert stats[2] == (0, "Fiscal years")


def test_form_header_html_shows_title_and_subtitle_and_escapes():
    out = rv.form_header_html("Setup", "<i>pick</i>")
    assert "Setup" in out
    assert "<i>" not in out and "&lt;i&gt;pick" in out
    assert "form-line" in out


def test_theme_css_has_background_cards_tabs_and_count_up():
    css = rv.THEME_CSS
    for piece in ("@keyframes bgShift", "@property --num", ".st-key-setup",
                  'data-baseweb="tab"', ".glass", ".hero2"):
        assert piece in css, piece
    assert "\n\n" not in css  # a blank line would break the style block in Markdown
