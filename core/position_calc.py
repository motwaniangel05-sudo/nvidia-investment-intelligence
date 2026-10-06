"""Deterministic maths for numbers the USER typed (price, previous price, cost).
Qwen never does this maths. These values are user-provided, not verified market data."""

BIG_MOVE_PCT = 20.0


def _pct(new, old):
    if old in (None, 0):
        return None
    return round((new - old) / abs(old) * 100, 1)


def calculate_position(user_prices):
    if not user_prices:
        return None
    cur = user_prices.get("current_price")
    prev = user_prices.get("previous_price")
    cost = user_prices.get("cost_basis")
    qty = user_prices.get("quantity")

    out = {
        "source": "USER_PROVIDED (not verified market data)",
        "inputs": {k: v for k, v in user_prices.items() if v is not None},
        "calculations": [],
        "flags": [],
    }
    calcs = out["calculations"]

    if cur is not None and prev is not None:
        calcs.append({"metric": "daily_change", "value": round(cur - prev, 2), "unit": "USD per share",
                      "formula": "current_price - previous_price"})
        pct = _pct(cur, prev)
        if pct is not None:
            calcs.append({"metric": "daily_change_percent", "value": pct, "unit": "%",
                          "formula": "(current_price - previous_price) / previous_price * 100"})
            if abs(pct) >= BIG_MOVE_PCT:
                out["flags"].append(
                    f"One-day move of {pct}% is unusually large. Check that the prices you typed are correct.")

    if cur is not None and cost is not None:
        calcs.append({"metric": "gain_or_loss_vs_cost", "value": round(cur - cost, 2), "unit": "USD per share",
                      "formula": "current_price - cost_basis"})
        pct = _pct(cur, cost)
        if pct is not None:
            calcs.append({"metric": "gain_or_loss_vs_cost_percent", "value": pct, "unit": "%",
                          "formula": "(current_price - cost_basis) / cost_basis * 100"})
        state = "GAIN" if cur > cost else ("LOSS" if cur < cost else "BREAK-EVEN")
        calcs.append({"metric": "position_state", "value": state, "unit": "",
                      "formula": "compare current_price with cost_basis"})
        if qty:
            calcs.append({"metric": "total_gain_or_loss", "value": round((cur - cost) * qty, 2), "unit": "USD",
                          "formula": "(current_price - cost_basis) * quantity"})

    return out if calcs else None
