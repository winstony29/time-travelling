"""Time Travelling Stonks Man solver.

Jumps cost linearly and every year is <= 2037, so the cheapest way to touch a
set of years is one down-then-up sweep costing 2*(2037 - deepest). Go as deep
as energy allows; visiting a year you don't trade in is free.

Walk the sweep in order. At each stop: settle any sells scheduled for now
(topping up capital), then buy greedily by return ratio -- for each stock, its
best future sell in the remaining sweep -- capped by capital and listed qty.
Sells are scheduled onto that future stop, so proceeds fund later buys.
"""
START = 2037


def solve(case):
    cap = case["capital"]
    tl = {int(y): s for y, s in case["timeline"].items()}
    deep = min((y for y in tl if 2 * (START - y) <= case["energy"]), default=START)
    ys = [y for y in tl if deep <= y <= START]
    seq = sorted(ys, reverse=True) + sorted(ys)[1:]  # down pass then up pass
    n = len(seq)

    left = {(i, s): d["qty"] for i, y in enumerate(seq) for s, d in tl[y].items()}
    due = [[] for _ in range(n)]  # sells landing at each stop: (stock, qty)
    acts = [[] for _ in range(n)]
    for i, y in enumerate(seq):
        for s, q in due[i]:
            cap += tl[y][s]["price"] * q
            acts[i].append(f"s-{s}-{q}")
        picks = []
        for s, d in tl[y].items():
            j = max((k for k in range(i + 1, n) if s in tl[seq[k]]),
                    key=lambda k: tl[seq[k]][s]["price"], default=None)
            if left[(i, s)] and j and tl[seq[j]][s]["price"] > d["price"]:
                picks.append((tl[seq[j]][s]["price"] / d["price"], d["price"], s, j))
        for _, p, s, j in sorted(picks, reverse=True):
            q = min(left[(i, s)], cap // p)
            if q:
                cap -= p * q
                left[(i, s)] -= q
                due[j].append((s, q))
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
    return out
