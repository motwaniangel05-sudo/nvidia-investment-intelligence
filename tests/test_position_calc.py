from core.position_calc import calculate_position


def _values(out):
    return {c["metric"]: c["value"] for c in out["calculations"]}


def test_none_and_empty_input():
    assert calculate_position(None) is None
    assert calculate_position({}) is None


def test_full_example_numbers():
    out = calculate_position({"current_price": 500.0, "previous_price": 200.0, "cost_basis": 120.0})
    v = _values(out)
    assert v["daily_change"] == 300.0
    assert v["daily_change_percent"] == 150.0
    assert v["gain_or_loss_vs_cost"] == 380.0
    assert v["gain_or_loss_vs_cost_percent"] == 316.7
    assert v["position_state"] == "GAIN"
    assert out["flags"]
    assert "USER_PROVIDED" in out["source"]


def test_loss_and_break_even():
    loss = calculate_position({"current_price": 80.0, "cost_basis": 100.0})
    even = calculate_position({"current_price": 100.0, "cost_basis": 100.0})
    assert _values(loss)["position_state"] == "LOSS"
    assert _values(even)["position_state"] == "BREAK-EVEN"


def test_quantity_gives_total():
    out = calculate_position({"current_price": 150.0, "cost_basis": 100.0, "quantity": 10})
    assert _values(out)["total_gain_or_loss"] == 500.0


def test_small_move_has_no_flag():
    out = calculate_position({"current_price": 105.0, "previous_price": 100.0})
    assert out["flags"] == []


def test_zero_previous_price_skips_percent():
    v = _values(calculate_position({"current_price": 5.0, "previous_price": 0}))
    assert "daily_change" in v
    assert "daily_change_percent" not in v


def test_zero_cost_skips_percent():
    v = _values(calculate_position({"current_price": 5.0, "cost_basis": 0}))
    assert "gain_or_loss_vs_cost_percent" not in v
    assert v["position_state"] == "GAIN"


def test_no_prices_returns_none():
    assert calculate_position({"quantity": 5}) is None
