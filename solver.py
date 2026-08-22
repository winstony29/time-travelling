"""Time Travelling Stonks Man solver.

Route insight: all years are <= 2037 and jump cost is linear, so any set of
visited years costs 2*(2037 - lowest) energy via one monotonic down-then-up
sweep. Only the depth is a real route decision; try every reachable listed
year as the turning point and keep the best simulated outcome.
"""
START = 2037


def solve(case):
    energy = case["energy"]
    capital = case["capital"]
    timeline = {int(y): s for y, s in case["timeline"].items()}
    reachable = sorted(y for y in timeline if y <= START and 2 * (START - y) <= energy)
    best, best_cap = [], capital
    for lo in reachable:  # each reachable year as deepest stop
        visits = [y for y in reachable if y >= lo]
        seq = visits[::-1] + visits[1:]  # down pass, then back up
        actions, cap = _greedy(seq, capital, timeline)
        if cap > best_cap:
            best, best_cap = actions, cap
    return best


def _greedy(seq, capital, timeline):
    # ponytail: per-visit greedy by return ratio; capital-indexed DP if scoring demands
    n = len(seq)
    qty_left = {(y, s): d["qty"] for y in set(seq) for s, d in timeline[y].items()}
    stocks = {s for y in set(seq) for s in timeline[y]}
    # suffix best sell per stock: best[s][j] = (price, index) over seq[j:], earliest max
    best = {s: [None] * (n + 1) for s in stocks}
    for s in stocks:
        cur = None
        for j in range(n - 1, -1, -1):
            d = timeline[seq[j]].get(s)
            if d and (cur is None or d["price"] >= cur[0]):
                cur = (d["price"], j)
            best[s][j] = cur

    scheduled = [[] for _ in range(n)]  # sells due at visit index: (stock, qty)
    acts = [[] for _ in range(n)]
    for i, y in enumerate(seq):
        for s, q in scheduled[i]:
            capital += timeline[y][s]["price"] * q
            acts[i].append(f"s-{s}-{q}")
        cands = []
        for s, d in timeline[y].items():
            if qty_left[(y, s)] > 0 and i + 1 < n:
                b = best[s][i + 1]
                if b and b[0] > d["price"]:
                    cands.append((b[0] / d["price"], d["price"], s, b[1]))
        cands.sort(key=lambda t: -t[0])
        for _, p, s, j in cands:
            q = min(qty_left[(y, s)], capital // p)
            if q > 0:
                capital -= p * q
                qty_left[(y, s)] -= q
                scheduled[j].append((s, q))
                acts[i].append(f"b-{s}-{q}")

    out, cur = [], START
    for i, y in enumerate(seq):
        if acts[i]:
            if y != cur:
                out.append(f"j-{cur}-{y}")
                cur = y
            out.extend(acts[i])
    if out and cur != START:
        out.append(f"j-{cur}-{START}")
    return out, capital
