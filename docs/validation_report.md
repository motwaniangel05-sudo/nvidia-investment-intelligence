# Validation report

Qwen server (must be local): http://localhost:11434   model: qwen3.5:2b

## Six test questions

### Test 1: What is NVIDIA's financial health?
- Selected agents (4): research, financial, risk, red_team
- Agent runs: research=success(0.1s), financial=success(0.1s), risk=success(2.8s), red_team=success(0.0s)
- Agent failures: none
- Evidence counts: {'financial_metrics': 4, 'valuation': 0, 'risks': 8, 'news_events': 0, 'competitor_findings': 0, 'sources': 4, 'conflicts': 0, 'missing_evidence': 0}
- Position math ran: False
- Context size sent to the report: 8878 characters
- Qwen reachable: True   removed numbers: []
- Confidence (code): 0.8
- Errors: none
- Qwen summary: The company shows exceptional financial performance with revenue growing 65.5%, a net margin of 55.6%, and an equity return of 76.3%. However, risk factors are rated high due to financial issues, and free cash flow canno
- Total latency: 29.1s

### Test 2: Compare NVIDIA with AMD.
- Selected agents (5): research, financial, market, risk, red_team
- Agent runs: research=success(0.4s), financial=success(0.6s), market=success(0.4s), risk=success(1.1s), red_team=success(0.0s)
- Agent failures: none
- Evidence counts: {'financial_metrics': 4, 'valuation': 0, 'risks': 8, 'news_events': 0, 'competitor_findings': 10, 'sources': 4, 'conflicts': 0, 'missing_evidence': 0}
- Position math ran: False
- Context size sent to the report: 11670 characters
- Qwen reachable: True   removed numbers: []
- Confidence (code): 0.7
- Errors: none
- Qwen summary: The company shows exceptional financial performance with revenue growing by 65.5%, a net margin of 55.6%, and an equity return of 76.3%. However, the financial risk category is rated as high due to numerous mentions in f
- Total latency: 10.4s

### Test 3: What are NVIDIA's biggest risks?
- Selected agents (4): research, financial, risk, red_team
- Agent runs: research=success(0.4s), financial=success(0.5s), risk=success(0.9s), red_team=success(0.0s)
- Agent failures: none
- Evidence counts: {'financial_metrics': 4, 'valuation': 0, 'risks': 8, 'news_events': 0, 'competitor_findings': 0, 'sources': 4, 'conflicts': 0, 'missing_evidence': 0}
- Position math ran: False
- Context size sent to the report: 8098 characters
- Qwen reachable: True   removed numbers: []
- Confidence (code): 0.8
- Errors: none
- Qwen summary: The company shows exceptional financial performance with revenue growing 65.5%, a net margin of 55.6%, and an equity return of 76.3%. However, risk factors are rated high due to financial issues, and free cash flow canno
- Total latency: 7.8s

### Test 4: What does the DCF valuation indicate?
- Selected agents (5): research, financial, market, valuation, red_team
- Agent runs: research=success(0.3s), financial=success(0.4s), market=success(0.2s), valuation=success(0.5s), red_team=success(0.0s)
- Agent failures: none
- Evidence counts: {'financial_metrics': 4, 'valuation': 6, 'risks': 0, 'news_events': 0, 'competitor_findings': 10, 'sources': 5, 'conflicts': 0, 'missing_evidence': 0}
- Position math ran: False
- Context size sent to the report: 9900 characters
- Qwen reachable: True   removed numbers: []
- Confidence (code): 0.7
- Errors: none
- Qwen summary: Despite strong revenue growth, the analysis cannot determine free cash flow due to missing capital expenditure data, and some red-team claims were flagged as unsupported by the system.
- Total latency: 10.3s

### Test 5: What happened in NVIDIA's latest earnings?
- Selected agents (5): research, financial, market, news, red_team
- Agent runs: research=success(0.4s), financial=success(3.7s), market=success(0.4s), news=success(3.4s), red_team=success(0.0s)
- Agent failures: none
- Evidence counts: {'financial_metrics': 4, 'valuation': 0, 'risks': 0, 'news_events': 9, 'competitor_findings': 10, 'sources': 5, 'conflicts': 0, 'missing_evidence': 0}
- Position math ran: False
- Context size sent to the report: 11606 characters
- Qwen reachable: True   removed numbers: []
- Confidence (code): 0.7
- Errors: none
- Qwen summary: The company shows exceptional financial performance with revenue growing by 65.5%, a net margin of 55.6%, and an equity return of 76.3%. However, red-team analysis flagged six unsupported claims due to known limitations,
- Total latency: 11.4s

### Test 6: NVIDIA is trading at $500 today, yesterday it was $200, and my average
- Selected agents (6): financial, market, news, risk, valuation, red_team
- Agent runs: financial=success(3.0s), market=success(2.9s), news=success(2.9s), risk=success(3.3s), valuation=success(3.1s), red_team=success(0.0s)
- Agent failures: none
- Evidence counts: {'financial_metrics': 4, 'valuation': 6, 'risks': 8, 'news_events': 5, 'competitor_findings': 10, 'sources': 6, 'conflicts': 1, 'missing_evidence': 0}
- Position math ran: True
- Context size sent to the report: 15123 characters
- Qwen reachable: True   removed numbers: []
- Confidence (code): 0.6
- Errors: none
- Qwen summary: The company shows exceptional financial health with a 76.3% return on equity and massive revenue growth of 65.5%. However, the current share price is significantly higher than the calculated valuation of $228.86, resulti
- Total latency: 15.0s

## Failure cases

- PASS: Ollama not reachable: report still complete, Qwen marked as unavailable
- PASS: one agent unavailable (risk): error recorded: risk: no adapter registered
- PASS: missing financial data (ticker ZZZZ): agent statuses {'research': 'success', 'financial': 'partial', 'risk': 'success', 'red_team': 'success'}
- PASS: malformed agent output: errors: ["financial: 'str' object has no attribute 'status'"]
- PASS: empty evidence: empty evidence handled, confidence 0.8
- PASS: invalid user prices: negative and zero prices handled
- NOT TESTED HERE: invalid valuation assumptions (covered by the earlier agent tests)

## Prohibited libraries loaded in this run

- none
