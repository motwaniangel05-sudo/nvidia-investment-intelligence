import json
import sys

from core import qwen_synthesizer as qs

CTX = {
    "query": "q",
    "financial_metrics": [{"metric": "Revenue CAGR", "unit": "%", "values": {"2026": 45.8}}],
    "risks": [{"risk": "Financial"}],
    "conflicts": [{"subject": "price"}],
    "uncertainties": ["u1"],
    "missing_evidence": ["m1"],
}


def good(**over):
    base = {"answer": "Revenue CAGR is 45.8%.", "recommendation": "Watch risks.",
            "reasoning": ["a"], "supporting_evidence": ["b"], "risks": ["c"],
            "uncertainties": ["d"], "assumptions": ["e"], "confidence": 0.7}
    base.update(over)
    return base


def fake(*responses):
    it = iter(responses)
    calls = []

    def _f(messages, model, base_url):
        calls.append(messages)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    _f.calls = calls
    return _f


class FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_build_messages():
    msgs = qs.build_messages("q", {"a": 1})
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "QUESTION: q" in msgs[1]["content"]
    assert '"a":1' in msgs[1]["content"]


def test_call_ollama(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["url"], seen["json"], seen["timeout"] = url, json, timeout
        return FakeResp({"message": {"content": "hi"}})

    monkeypatch.setattr(qs.requests, "post", fake_post)
    out = qs.call_ollama([{"role": "user", "content": "x"}], model="m", base_url="http://h:1", timeout=5)
    assert out == "hi"
    assert seen["url"] == "http://h:1/api/chat"
    assert seen["json"]["think"] is False
    assert seen["json"]["model"] == "m"
    assert seen["timeout"] == 5


def test_parse_json():
    assert qs.parse_json('{"a": 1}') == {"a": 1}
    assert qs.parse_json('text {"a": 1} more') == {"a": 1}
    assert qs.parse_json("{bad json}") is None
    assert qs.parse_json("no braces") is None
    assert qs.parse_json(None) is None


def test_number_helpers():
    assert qs._norm("45.80") == "45.8"
    assert qs._norm("3.") == "3"
    assert qs._norm("1,000") == "1000"
    assert qs.unsupported_numbers("Revenue 45.8 and 99", "growth 45.8") == ["99"]
    assert qs.unsupported_numbers("only 7 items", "") == []


def test_fallback_answer_with_data():
    out = qs.fallback_answer("q", CTX, "down")
    assert out["source"] == "fallback"
    assert out["confidence"] == 0.0
    assert "down" in out["answer"]
    assert "Revenue CAGR" in out["reasoning"][0]
    assert "price" in out["supporting_evidence"][0]
    assert "u1" in " ".join(out["uncertainties"])
    assert "m1" in " ".join(out["uncertainties"])
    assert "Financial" in out["risks"][0]


def test_fallback_answer_empty_context():
    out = qs.fallback_answer("q", {}, "x")
    assert out["reasoning"] == [] and out["risks"] == [] and out["uncertainties"] == []


def test_clean_fixes_types_and_confidence():
    assert qs._clean({"confidence": 5})["confidence"] == 1.0
    assert qs._clean({"confidence": -2})["confidence"] == 0.0
    cleaned = qs._clean({"confidence": "high", "reasoning": "just text"})
    assert cleaned["confidence"] == 0.0
    assert cleaned["reasoning"] == ["just text"]


def test_shorten():
    assert qs._shorten("### Heading\nText.") == "Heading Text."
    assert qs._shorten("One. Two. Three. Four.") == "One. Two. Three."


def test_synthesize_success(monkeypatch):
    monkeypatch.setattr(qs, "call_ollama", fake(json.dumps(good())))
    out = qs.synthesize("q", CTX)
    assert out["source"] == "qwen"
    assert out["retried"] is False
    assert out["unsupported_numbers"] == []
    assert out["confidence"] == 0.7


def test_synthesize_retry_fixes_bad_number(monkeypatch):
    f = fake(json.dumps(good(answer="Price is 99.5 today.")), json.dumps(good()))
    monkeypatch.setattr(qs, "call_ollama", f)
    out = qs.synthesize("q", CTX)
    assert out["retried"] is True
    assert out["unsupported_numbers"] == []
    assert len(f.calls) == 2


def test_synthesize_removes_sentence_with_bad_number(monkeypatch):
    bad = json.dumps(good(answer="Revenue CAGR is 45.8%. Price will be 99.5 dollars.",
                          recommendation="Value 77.7 here.", risks=["Risk 88.8 now."]))
    monkeypatch.setattr(qs, "call_ollama", fake(bad, bad))
    out = qs.synthesize("q", CTX)
    assert out["answer"] == "Revenue CAGR is 45.8%."
    assert out["recommendation"] == ""
    assert out["risks"] == []
    assert out["confidence"] <= 0.3
    assert "removed" in out["uncertainties"][-1]
    assert "99.5" in out["unsupported_numbers"]


def test_synthesize_all_text_removed(monkeypatch):
    bad = json.dumps(good(answer="Price will be 99.5 dollars."))
    monkeypatch.setattr(qs, "call_ollama", fake(bad, bad))
    out = qs.synthesize("q", CTX)
    assert "removed" in out["answer"]


def test_synthesize_too_long_is_shortened(monkeypatch):
    long = json.dumps(good(answer="Sentence one. " * 100))
    monkeypatch.setattr(qs, "call_ollama", fake(long, long))
    out = qs.synthesize("q", CTX)
    assert out["answer"].count("Sentence one.") == 3
    assert out["retried"] is True


def test_synthesize_markdown_is_cleaned(monkeypatch):
    md = json.dumps(good(answer="### Heading\nText."))
    monkeypatch.setattr(qs, "call_ollama", fake(md, md))
    out = qs.synthesize("q", CTX)
    assert "#" not in out["answer"]


def test_synthesize_second_call_fails_keeps_first(monkeypatch):
    first = json.dumps(good(answer="Price is 99.5 today."))
    monkeypatch.setattr(qs, "call_ollama", fake(first, RuntimeError("boom")))
    out = qs.synthesize("q", CTX)
    assert out["source"] == "qwen"
    assert out["unsupported_numbers"] == ["99.5"]
    assert out["confidence"] <= 0.3


def test_synthesize_fallback_when_ollama_down(monkeypatch):
    monkeypatch.setattr(qs, "call_ollama", fake(RuntimeError("down")))
    out = qs.synthesize("q", CTX)
    assert out["source"] == "fallback"
    assert "down" in out["answer"]


def test_synthesize_fallback_on_invalid_json(monkeypatch):
    monkeypatch.setattr(qs, "call_ollama", fake("not json at all"))
    out = qs.synthesize("q", CTX)
    assert out["source"] == "fallback"
    assert "valid JSON" in out["answer"]


def test_main_usage(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert qs.main() == 1
    assert "Usage" in capsys.readouterr().out


def test_main_runs(monkeypatch, capsys, tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"query": "q"}))
    monkeypatch.setattr(sys, "argv", ["x", str(f)])
    monkeypatch.setattr(qs, "synthesize", lambda q, c: {"answer": "ok"})
    assert qs.main() == 0
    assert '"answer": "ok"' in capsys.readouterr().out
