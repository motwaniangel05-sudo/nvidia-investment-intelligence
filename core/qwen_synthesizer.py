"""Qwen synthesis layer (local Ollama). Facts come only from the evidence context."""
import json
import re
import sys

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3.5:2b"

SYSTEM_PROMPT = (
    "You are the synthesis layer of an investment intelligence system. "
    "Use ONLY the evidence in CONTEXT and the numbers in the question. "
    "Never invent figures, sources or news. If evidence is missing, say so. "
    "If sources conflict, state the conflict. "
    "Numbers the user typed are user-provided, not verified market data. "
    "If CONTEXT has position_calculations, copy those numbers exactly and never do your own maths. "
    "A current price above the cost basis is a GAIN, not undervaluation. "
    "This is not financial advice: give considerations and scenarios, not an order. "
    "Keep it SHORT: answer = max 3 plain sentences, no markdown. Each list has max 4 items, one short sentence each. Never write a number, price or valuation that is not in CONTEXT or the question. Reply with JSON only."
)

_LIST = {"type": "array", "items": {"type": "string"}}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "recommendation": {"type": "string"},
        "reasoning": _LIST,
        "supporting_evidence": _LIST,
        "risks": _LIST,
        "uncertainties": _LIST,
        "assumptions": _LIST,
        "confidence": {"type": "number"},
    },
    "required": ["answer", "recommendation", "reasoning", "supporting_evidence",
                 "risks", "uncertainties", "assumptions", "confidence"],
}
LIST_FIELDS = ["reasoning", "supporting_evidence", "risks", "uncertainties", "assumptions"]


def build_messages(query, context):
    ctx = json.dumps(context, separators=(",", ":"), default=str)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {query}\n\nCONTEXT (JSON):\n{ctx}"},
    ]


def call_ollama(messages, model=MODEL, base_url=OLLAMA_URL, timeout=600):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": ANSWER_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1000},
    }
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def _norm(num):
    num = num.replace(",", "").rstrip(".")
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def _numbers(text):
    return {_norm(n) for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def unsupported_numbers(answer_text, allowed_text):
    allowed = _numbers(allowed_text)
    found = set()
    for n in _numbers(answer_text):
        if n.isdigit() and len(n) <= 1:
            continue
        if n not in allowed:
            found.add(n)
    return sorted(found)


def fallback_answer(query, context, reason):
    reasoning = []
    for m in (context.get("financial_metrics") or [])[:5]:
        reasoning.append(f"{m.get('metric')} ({m.get('unit', '')}): {json.dumps(m.get('values'))}"[:200])
    conflicts = [json.dumps(c, default=str)[:200] for c in (context.get("conflicts") or [])]
    uncertainties = [str(u)[:200] for u in (context.get("uncertainties") or [])]
    uncertainties += [str(u)[:200] for u in (context.get("missing_evidence") or [])]
    return {
        "answer": f"Qwen synthesis was not available ({reason}). Showing the agent evidence only.",
        "recommendation": "No recommendation without model synthesis. This is not financial advice.",
        "reasoning": reasoning,
        "supporting_evidence": conflicts,
        "risks": [json.dumps(r, default=str)[:200] for r in (context.get("risks") or [])[:5]],
        "uncertainties": uncertainties,
        "assumptions": [],
        "confidence": 0.0,
        "unsupported_numbers": [],
        "source": "fallback",
    }


MAX_ANSWER_CHARS = 700


def _clean(data):
    data["answer"] = str(data.get("answer", ""))
    data["recommendation"] = str(data.get("recommendation", ""))
    for key in LIST_FIELDS:
        value = data.get(key) or []
        data[key] = [str(v).strip() for v in value] if isinstance(value, list) else [str(value)]
    try:
        data["confidence"] = min(1.0, max(0.0, float(data.get("confidence", 0))))
    except Exception:
        data["confidence"] = 0.0
    return data


def _problems(data, query, context):
    text = " ".join([data["answer"], data["recommendation"]] + [
        item for key in LIST_FIELDS for item in data[key]])
    allowed = json.dumps(context, default=str) + " " + query
    bad = unsupported_numbers(text, allowed)
    too_long = len(data["answer"]) > MAX_ANSWER_CHARS or "###" in data["answer"]
    return bad, too_long


def _shorten(answer):
    plain = re.sub(r"[#*]", "", answer)
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(plain.split()))
    return " ".join(sentences[:3])


def synthesize(query, context, model=MODEL, base_url=OLLAMA_URL):
    messages = build_messages(query, context)
    data = None
    bad, too_long = [], False
    attempt = 0
    for attempt in range(2):
        try:
            raw = call_ollama(messages, model, base_url)
            parsed = parse_json(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Qwen did not return valid JSON")
        except Exception as exc:
            if data is None:
                return fallback_answer(query, context, str(exc)[:150])
            break
        data = _clean(parsed)
        bad, too_long = _problems(data, query, context)
        if not bad and not too_long:
            break
        if attempt == 0:
            fixes = []
            if bad:
                fixes.append("Remove these numbers because they are not in CONTEXT: "
                             + ", ".join(bad) + ".")
            if too_long:
                fixes.append("Make answer at most 3 short plain sentences, no markdown or headings.")
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": " ".join(fixes) + " Reply with JSON only."},
            ]

    if too_long:
        data["answer"] = _shorten(data["answer"])
    data["unsupported_numbers"] = bad
    if bad:
        def _strip(text):
            parts = re.split(r"(?<=[.!?])\s+", text)
            keep = [x for x in parts if not (_numbers(x) & set(bad))]
            return " ".join(keep).strip()

        data["answer"] = _strip(data["answer"]) or (
            "The model's answer was removed because it used numbers that are not in the evidence.")
        data["recommendation"] = _strip(data["recommendation"])
        for key in LIST_FIELDS:
            data[key] = [x for x in data[key] if not (_numbers(x) & set(bad))]
        data["confidence"] = min(data["confidence"], 0.3)
        data["uncertainties"].append(
            "Sentences with numbers not found in the evidence were removed: " + ", ".join(bad))
    data["retried"] = attempt == 1
    data["source"] = "qwen"
    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m core.qwen_synthesizer /tmp/context.json")
        return 1
    with open(sys.argv[1]) as f:
        context = json.load(f)
    result = synthesize(context.get("query", ""), context)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
