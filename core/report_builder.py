"""Structured final report (Prompts 8 and 9).
Code writes every part that holds facts and numbers. Qwen writes only the short
judgement parts. Confidence is computed by code, not guessed by the model."""
import json
import re
import sys
import time

import requests

from core.qwen_synthesizer import MODEL, OLLAMA_URL, parse_json, unsupported_numbers

DISCLAIMER = ("This is an educational analysis built only from the evidence above. "
              "It is not financial advice.")


def _latest(values, n=3):
    return sorted((values or {}).items())[-n:]


def _failed(runs):
    return [r for r in runs or [] if r.get("status") == "failed"]


def _conflict_text(c):
    sides = []
    for p in c.get("positions") or []:
        who = p.get("source") or p.get("agent") or "?"
        sides.append(f"{who}: {p.get('value')} {p.get('unit', '')}".strip())
    return f"{c.get('subject', 'conflict')} -> " + " vs ".join(sides)


def _data_lines(ctx):
    lines = []
    pos = ctx.get("position_calculations")
    if pos:
        lines.append("**USER-PROVIDED (not verified market data):** "
                     + ", ".join(f"{k} = {v}" for k, v in pos["inputs"].items()))
        lines.append("")
        lines.append("**CALCULATED BY CODE from your numbers:**")
        lines.append("")
        for c in pos["calculations"]:
            lines.append(f"- {c['metric'].replace('_', ' ')}: {c['value']} {c['unit']}".rstrip())
        for flag in pos.get("flags") or []:
            lines.append(f"- WARNING: {flag}")
    stored = ctx.get("market_findings") or []
    if stored:
        lines.append("")
        lines.append("**STORED MARKET DATA:**")
        lines.append("")
        for m in stored:
            lines.append(f"- {m.get('metric')}: {m.get('value')} {m.get('unit', '')} "
                         f"(date {m.get('date', 'n/a')}, source quality {m.get('quality')})")
    for c in ctx.get("conflicts") or []:
        lines.append("")
        lines.append("**CONFLICT (not resolved):** " + _conflict_text(c)
                     + " | " + str(c.get("note", ""))[:200])
    return lines or ["No data available."]


def _financial_lines(ctx):
    lines = []
    for m in ctx.get("financial_metrics") or []:
        vals = ", ".join(f"{k}: {v}" for k, v in _latest(m.get("values")))
        lines.append(f"- {m.get('metric')} ({m.get('unit', '')}) latest: {vals} "
                     f"[source quality {m.get('quality')}]")
    return lines or ["No financial metrics available."]


def _market_lines(ctx):
    lines = []
    for m in [x for x in (ctx.get("competitor_findings") or []) if x.get("metric")][:8]:
        lines.append(f"- {m.get('metric')}: {m.get('value')} {m.get('unit', '')} "
                     f"({m.get('period', 'n/a')})")
    return lines or ["No competitor findings available."]


def _market_ev(ctx):
    for v in ctx.get("valuation") or []:
        if v.get("method") == "EV/EBIT" and v.get("enterprise_value") is not None:
            return v["enterprise_value"]
    return None


def _valuation_lines(ctx):
    lines = []
    mev = _market_ev(ctx)
    for v in ctx.get("valuation") or []:
        method = str(v.get("method", "?"))
        ev = v.get("enterprise_value")
        if method == "DCF" and ev is not None:
            growth = (v.get("assumptions") or {}).get("revenue_growth_rate")
            rel = ""
            if mev:
                side = "BELOW" if ev < mev else "ABOVE"
                rel = f"; {side} the market-implied enterprise value ({ev / mev:.2f}x)"
            lines.append(f"- DCF ({v.get('scenario')}): enterprise value about "
                         f"{ev / 1e9:,.0f} billion USD (assumed revenue growth {growth}){rel}")
        elif method == "EV/EBIT" and ev is not None:
            lines.append(f"- EV/EBIT: market-implied enterprise value about {ev / 1e9:,.0f} billion USD")
        elif v.get("multiple") is not None:
            lines.append(f"- {method}: {v['multiple']}x")
        else:
            lines.append(f"- {method}: see agent output")
    if lines:
        lines.append("- P/E and the market-implied value use the STORED price, not the price you typed. DCF results depend on fixed assumptions.")
        lines.append("- Not calculated by the agents: equity value, implied share price, sensitivity.")
    return lines or ["No valuation available."]


def _risk_lines(ctx):
    lines = [f"- {r.get('risk')}: severity {r.get('severity')}" for r in ctx.get("risks") or []]
    if lines:
        lines.append("- Severity = how many filing years mention the risk. "
                     "It is not a measure of impact or probability.")
    return lines or ["No risk findings available."]


def _news_lines(ctx):
    lines = [f"- [{n.get('date', 'n/a')}] ({n.get('label')}) {str(n.get('text', ''))[:220]}"
             for n in ctx.get("news_events") or []]
    if lines:
        lines.append("")
        lines.append("_Matched to your question by keywords, so some items may not be relevant. The stored 8-K data has no earnings press releases._")
    return lines or ["No recent events available."]


def _verification_lines(ctx):
    lines = [f"- {a.get('agent')}: {a.get('status')}" for a in ctx.get("agent_status") or []]
    for f in ctx.get("verification_flags") or []:
        lines.append(f"- Red-Team: {f.get('count')} claim(s) from {f.get('agent')} marked {f.get('result')}")
        if f.get("agent") == "MarketAgent" and f.get("result") == "UNSUPPORTED":
            lines.append("  (Known limitation: the checker cannot find these numbers in the "
                         "evidence text. These are not proven errors.)")
    return lines or ["No verification data available."]


def _uncertainty_lines(ctx, runs):
    lines = [f"- {u}" for u in ctx.get("uncertainties") or []]
    lines += [f"- Missing: {m}" for m in ctx.get("missing_evidence") or []]
    lines += [f"- Agent failed: {r.get('agent')} ({str(r.get('error', ''))[:120]})"
              for r in _failed(runs)]
    return lines or ["No uncertainties recorded."]


def _source_lines(ctx):
    lines = [f"- {s.get('form')} (quality {s.get('quality')}, {s.get('count')} item(s))"
             for s in ctx.get("sources") or []]
    return lines or ["No sources recorded."]


BANNED = re.compile(
    r"undervalu|overvalu|momentum|manipulat|buy signal|sell signal|lock in|"
    r"should (buy|sell)|consider (selling|buying)|selling could|buying exposes|"
    r"priced lower|cheap|expensive|market share|-to-(one|two|three|four|five)|twice|double|triple", re.I)


def compute_confidence(ctx, runs, qwen_ok, removed):
    score = 0.8
    reasons = ["Base score 0.8 (maximum allowed is 0.9)."]
    failed = _failed(runs)
    if failed:
        score -= 0.15 * len(failed)
        reasons.append(f"-{0.15 * len(failed):.2f}: {len(failed)} agent(s) failed.")
    if ctx.get("conflicts"):
        score -= 0.1
        reasons.append("-0.10: sources or user data conflict.")
    if any(f.get("result") == "UNSUPPORTED" for f in ctx.get("verification_flags") or []):
        score -= 0.1
        reasons.append("-0.10: Red-Team marked some claims as unsupported.")
    if ctx.get("missing_evidence"):
        score -= 0.1
        reasons.append("-0.10: some evidence is missing.")
    if removed:
        score -= 0.2
        reasons.append("-0.20: model text with unsupported numbers was removed.")
    if not qwen_ok:
        score -= 0.2
        reasons.append("-0.20: Qwen was not available.")
    return round(max(0.1, min(0.9, score)), 2), reasons


def _position_sentence(ctx):
    pos = ctx.get("position_calculations")
    if not pos:
        return ""
    vals = {c["metric"]: c["value"] for c in pos["calculations"]}
    out = []
    if "gain_or_loss_vs_cost_percent" in vals:
        out.append("Using the prices you typed (unverified), your position is a "
                   f"{vals.get('position_state', '')} of {vals['gain_or_loss_vs_cost_percent']}% "
                   "versus your average cost.")
    if "daily_change_percent" in vals:
        out.append(f"The typed one-day move is {vals['daily_change_percent']}%.")
    return " ".join(out)


def _valuation_point(ctx):
    mev = _market_ev(ctx)
    if not mev:
        return ""
    ratios = [v["enterprise_value"] / mev for v in ctx.get("valuation") or []
              if v.get("method") == "DCF" and v.get("enterprise_value") is not None]
    if not ratios:
        return ""
    below = sum(1 for r in ratios if r < 1)
    return (f"Valuation: {below} of {len(ratios)} DCF scenarios are below the market-implied "
            f"enterprise value (range {min(ratios):.2f}x to {max(ratios):.2f}x). These models use "
            "fixed assumptions, so they do not show support for the market price.")


def _block(title, lines):
    return f"## {title}\n" + "\n".join(lines) + "\n"


PLAIN_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}
PLAIN_PROMPT = (
    "Using ONLY the POINTS below the QUESTION, write a plain-language summary that helps with the question, in at most 3 short sentences. If the points do not answer it, say what they do show. "
    "Add nothing: no new facts, numbers, opinions, predictions or advice. Copy numbers exactly as written: never turn them into words or ratios. "
    "Do not say buy, sell, cheap, expensive, undervalued or overvalued. "
    "Plain text only, no markdown. Reply with JSON only."
)


def ask_summary(points_text, model=MODEL, base_url=OLLAMA_URL, timeout=600):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PLAIN_PROMPT},
            {"role": "user", "content": "POINTS:\n" + points_text},
        ],
        "stream": False,
        "think": False,
        "format": PLAIN_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 400},
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = parse_json(resp.json()["message"]["content"])
    if not isinstance(data, dict) or not data.get("summary"):
        raise ValueError("Qwen did not return a summary")
    return str(data["summary"])


def _clean_summary(text, allowed):
    bad, keep = set(), []
    for sent in re.split(r"(?<=[.!?])\s+", " ".join(str(text or "").split())):
        found = unsupported_numbers(sent, allowed)
        if found:
            bad.update(found)
        elif BANNED.search(sent):
            continue
        else:
            keep.append(sent)
    return " ".join(keep[:3]), sorted(bad)


def _bull_points(ctx):
    pts = []
    for m in ctx.get("financial_metrics") or []:
        latest = _latest(m.get("values"), 1)
        if latest and m.get("metric") in ("Revenue Growth (YoY)", "Net Margin", "Return on Equity"):
            pts.append(f"{m['metric']}: {latest[0][1]}{m.get('unit', '')} "
                       f"(period {latest[0][0]}, HIGH quality filing data).")
    ps = _position_sentence(ctx)
    if ps:
        pts.append(ps)
    return pts


def _bear_points(ctx):
    pts = []
    vp = _valuation_point(ctx)
    if vp:
        pts.append(vp)
    high = [r["risk"] for r in ctx.get("risks") or [] if r.get("severity") == "HIGH"]
    if high:
        pts.append("Risk categories rated HIGH (by how many filing years mention them, "
                   "not by impact): " + ", ".join(high) + ".")
    pos = ctx.get("position_calculations") or {}
    pts += list(pos.get("flags") or [])
    for c in ctx.get("conflicts") or []:
        pts.append("Conflict, not resolved: " + _conflict_text(c) + ".")
    n = sum(f.get("count", 0) for f in ctx.get("verification_flags") or []
            if f.get("result") == "UNSUPPORTED")
    if n:
        pts.append(f"Red-Team marked {n} claim(s) as unsupported "
                   "(known checker limitation, not proven errors).")
    if any("Free Cash Flow" in str(u) for u in ctx.get("uncertainties") or []):
        pts.append("Free cash flow could not be computed (CapEx data missing).")
    return pts


def _assessment(ctx, runs):
    open_points = []
    if ctx.get("conflicts"):
        open_points.append("a price conflict between your typed price and stored data")
    if ctx.get("position_calculations"):
        open_points.append("typed prices that are unverified")
    if any(v.get("method") == "DCF" for v in ctx.get("valuation") or []):
        open_points.append("DCF models that use fixed assumptions")
    if _failed(runs):
        open_points.append(f"{len(_failed(runs))} agent(s) that failed")
    text = ("What the evidence shows: the reported financials and the calculations listed above. "
            "What it cannot show: whether the current price is fair, or whether to buy or sell.")
    if open_points:
        text += " Open points: " + "; ".join(open_points) + "."
    return text


def _focus_points(ctx):
    """Choose the points Qwen gets, based on the type of question."""
    intent = (ctx.get("intent") or "").lower()
    pts = []
    if intent == "risk":
        for r in ctx.get("risks") or []:
            pts.append(f"Risk category {r.get('risk')}: severity {r.get('severity')} "
                       "(counts how many filing years mention it, not impact).")
    elif intent == "valuation":
        vp = _valuation_point(ctx)
        if vp:
            pts.append(vp)
            pts.append("DCF results depend on fixed assumptions.")
        for v in ctx.get("valuation") or []:
            if v.get("multiple") is not None:
                pts.append(f"{v.get('method')}: {v['multiple']}x, based on the stored share price.")
    elif intent in ("competitor", "market"):
        for m in ctx.get("competitor_findings") or []:
            if m.get("metric"):
                pts.append(f"{m['metric']}: {m.get('value')} {m.get('unit', '')} "
                           f"({m.get('period', 'n/a')}).")
    elif intent == "news":
        for n in ctx.get("news_events") or []:
            pts.append(f"[{n.get('date', 'n/a')}] {str(n.get('text', ''))[:160]}")
        if pts:
            pts.append("These news items were matched by keywords and may not be relevant.")
    elif intent in ("financial", "research"):
        pts = _bull_points(ctx)
    else:
        pts = _bull_points(ctx) + _bear_points(ctx)
    return pts or (_bull_points(ctx) + _bear_points(ctx))


def build_report(query, ctx, runs=None, errors=None, ask=ask_summary):
    runs = runs or []
    bull, bear = _bull_points(ctx), _bear_points(ctx)
    ps = _position_sentence(ctx)
    exec_text = ((ps + " ") if ps else "") + (
        "This report lists the evidence found; it does not decide whether to buy or sell.")

    points = _focus_points(ctx)
    summary, removed, qwen_ok = "", [], True
    if points:
        points_text = "QUESTION: " + query + "\n" + "\n".join("- " + p for p in points)
        try:
            raw = ask(points_text)
            summary, removed = _clean_summary(raw, points_text + " " + query)
            if not summary:
                summary = "Removed: the model's summary used numbers or wording not found in the evidence."
        except Exception as exc:
            qwen_ok = False
            summary = "Not available: Qwen could not be reached (" + str(exc)[:80] + ")."
    else:
        summary = "No points to summarize."

    confidence, reasons = compute_confidence(ctx, runs, True, [])
    final = [_assessment(ctx, runs), "",
             f"**Evidence confidence (calculated by code): {confidence}**"]
    final += [f"- {r}" for r in reasons]
    final += ["", DISCLAIMER]

    md = [f"# Investment Intelligence Report\n**Question:** {query}\n"]
    md.append(_block("Executive Conclusion", [exec_text]))
    md.append(_block("AI Draft Summary (written by Qwen, may contain mistakes)", [summary, "", "Only the sections below are built by code from the evidence. If this summary differs from them, trust the sections."]))
    md.append(_block("What The Data Says", _data_lines(ctx)))
    md.append(_block("Financial Analysis", _financial_lines(ctx)))
    md.append(_block("Market Analysis", _market_lines(ctx)))
    md.append(_block("Valuation", _valuation_lines(ctx)))
    md.append(_block("Risks", _risk_lines(ctx)))
    md.append(_block("Recent Events", _news_lines(ctx)))
    md.append(_block("Verification", _verification_lines(ctx)))
    md.append(_block("Bull Case (supporting points from the evidence)",
                     [f"- {x}" for x in bull] or ["- None found in the evidence."]))
    md.append(_block("Bear Case (concerns from the evidence)",
                     [f"- {x}" for x in bear] or ["- None found in the evidence."]))
    md.append(_block("Key Uncertainties", _uncertainty_lines(ctx, runs)))
    md.append(_block("Final Assessment", final))
    md.append(_block("Evidence", _source_lines(ctx)))

    return {
        "markdown": "\n".join(md),
        "parts": {"bull_case": bull, "bear_case": bear, "summary": summary},
        "confidence": confidence,
        "confidence_reasons": reasons,
        "removed_numbers": removed,
        "qwen_ok": qwen_ok,
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('Usage: python3 -m core.report_builder "your question"')
        return 1
    from core.dynamic_pipeline import run
    started = time.time()
    out = run(argv[0], synthesizer=lambda q, c: {"answer": "", "source": "skipped"})
    report = build_report(argv[0], out.get("context") or {}, out["agent_runs"], out["errors"])
    print(report["markdown"])
    print(f"(Total time: {time.time() - started:.0f} seconds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
