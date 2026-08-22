"""Rule-enforcing simulator + checks. run() raises if solve() breaks any challenge rule."""
from solver import START, solve


def run(case, actions):
    energy, capital = case["energy"], case["capital"]
    timeline = {int(y): {s: dict(d) for s, d in v.items()} for y, v in case["timeline"].items()}
    year, held = START, {}
    for a in actions:
        kind, x, z = a.split("-")
        if kind == "j":
            assert int(x) == year, f"jump from wrong year: {a} while at {year}"
            year = int(z)
            energy -= abs(int(x) - year)
            assert energy >= 0, "out of energy"
        elif kind == "b":
            d, q = timeline[year][x], int(z)
            assert q <= d["qty"], f"overbuy {a}"
            assert d["price"] * q <= capital, f"overspend {a}"
            d["qty"] -= q
            capital -= d["price"] * q
            held[x] = held.get(x, 0) + q
        else:
            q = int(z)
            assert held.get(x, 0) >= q, f"selling unheld {a}"
            held[x] -= q
            capital += timeline[year][x]["price"] * q
    assert year == START, "did not return to 2037"
    return capital


SAMPLE = {
    "energy": 2,
    "capital": 500,
    "timeline": {
        "2037": {"Apple": {"price": 100, "qty": 10}},
        "2036": {"Apple": {"price": 10, "qty": 50}},
    },
}


def test_sample():
    assert run(SAMPLE, solve(SAMPLE)) == 5000


def test_no_profit_stays_home():
    case = {"energy": 4, "capital": 100, "timeline": {
        "2037": {"A": {"price": 5, "qty": 10}},
        "2036": {"A": {"price": 5, "qty": 10}}}}
    assert solve(case) == []


def test_qty_limit():
    case = {"energy": 2, "capital": 500, "timeline": {
        "2037": {"A": {"price": 100, "qty": 0}},
        "2036": {"A": {"price": 10, "qty": 20}}}}
    assert run(case, solve(case)) == 500 - 200 + 2000


def test_energy_limit():
    case = {"energy": 4, "capital": 100, "timeline": {
        "2037": {"A": {"price": 50, "qty": 0}},
        "2035": {"A": {"price": 10, "qty": 100}},
        "2030": {"A": {"price": 1, "qty": 100}}}}
    acts = solve(case)
    assert "2030" not in "".join(acts)  # needs 14 energy, only 4 available
    assert run(case, acts) == 500


def test_compounding_multihop():
    case = {"energy": 10, "capital": 10, "timeline": {
        "2037": {"B": {"price": 10, "qty": 0}},
        "2036": {"A": {"price": 1, "qty": 10}},
        "2035": {"A": {"price": 5, "qty": 0}, "B": {"price": 1, "qty": 100}}}}
    # buy A@2036, sell A@2035, buy B@2035, sell B@2037: 10 -> 50 -> 500
    assert run(case, solve(case)) == 500
