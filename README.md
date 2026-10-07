![tests](https://github.com/motwaniangel05-sudo/nvidia-investment-intelligence/actions/workflows/tests.yml/badge.svg)

# AI-Powered Investment Intelligence & Financial Due-Diligence System

A multi-agent financial research system built from scratch (no cloud or paid LLM APIs; an optional local Qwen model runs through Ollama),
using NVIDIA as the initial case study. Company-agnostic by design.

**Status:** Phase 1 (environment setup) complete.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Disclaimer
Educational project. Not financial advice.
## Dynamic question mode (local Qwen)

Ask a natural-language question and the system picks the agents it needs, runs them in
parallel, verifies the results, builds an evidence package, and writes a structured report.

Rule for the language model: local Qwen (through Ollama) is used only for an optional short
draft summary. Facts, numbers, valuation reading and confidence come from code and agents.
No cloud or paid APIs are used.

### Setup and run

    ollama pull qwen3.5:2b
    python -m core.dynamic_pipeline "Should I buy NVIDIA at the current price?"
    python -m core.report_builder "Compare NVIDIA with AMD."
    streamlit run chat_app.py        # new question page
    streamlit run app.py             # old dropdown page, unchanged

### How it works

question -> query analyzer -> agent planner -> selected agents (parallel) -> Red-Team ->
evidence aggregator -> position calculator (user-typed prices) -> report builder -> optional Qwen summary

### Known limitations

- A 2B model is too weak for reliable investment reasoning, so code writes the report sections
  and Qwen only writes a short summary that is labelled as possibly wrong.
- Prices typed by the user are treated as unverified. P/E and the market-implied value use the
  stored price (last stored close), not the typed price.
- The Red-Team marks the Market agent's comparison claims as unsupported. This is a known checker
  limitation, not a proven error.
- The stored 8-K text has no earnings press releases, and news is matched by keywords, so the
  Recent Events section can contain irrelevant items.
- DCF scenarios use fixed assumptions. Equity value, implied share price and sensitivity are not calculated.
- Free cash flow cannot be computed because CapEx is missing in the XBRL data.
- Qwen can take up to a minute on an 8 GB laptop. If Ollama is not running, the report still works
  without the summary.
- The Research and Risk agents search NVIDIA's filings for any ticker, so a question about another company returns NVIDIA evidence. A ticker filter is not implemented yet.
- The Qwen summary is not tailored to the question. It mostly repeats the headline financial numbers and can use a number with the wrong meaning. Trust the code-written sections.
- This is an educational project. It is not financial advice.

## Rebuilding the data (needed after a fresh clone)

Large data files are not stored on GitHub. Without them, the Risk and Research agents fail or
return nothing. Use Python 3.11 or newer (3.13 is what the tests use). To see what is missing:

    python3 scripts/check_data.py

Then run these commands, in this order (the filing download needs an internet connection and
about 70 MB of space, so it can take several minutes):

    python -m data_acquisition.document_downloader   # downloads the 10-K / 10-Q filings
    python -m rag.chunker                            # cuts the filings into text chunks
    python -m core.memory                            # builds the knowledge database
    python -m rag.vector_store                       # builds the search index

Run `python3 scripts/check_data.py` again at the end. All four lines should say OK.
