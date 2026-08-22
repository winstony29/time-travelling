"""Time Travelling Stonks Man solver.

Jumps cost linearly and every year is <= 2037, so the cheapest way to touch a
set of years is one down-then-up sweep costing 2*(2037 - deepest). Go as deep
as energy allows; visiting a year you don't trade in is free.

Two strategies compete and the more profitable (by simulation) wins:

* global -- treat each stock as one buy-low/sell-high trade over the window and
  pick quantities with a bounded knapsack over starting capital. Optimal when
  trades are independent; blind to reinvesting one sale into another buy.
* sweep -- walk the sweep in visit order, at each stop knapsack whatever capital
  is on hand (topped up by sells that already landed). Captures compounding
  (sell early, buy more) but can waste capital at shallow years.

Neither dominates, so we run both and return the higher-profit action list.
"""
START = 2037


def solve(case):
    tl = {int(y): s for y, s in case["timeline"].items()}
    deep = min((y for y in tl if 2 * (START - y) <= case["energy"]), default=START)
    ys = [y for y in tl if deep <= y <= START]
    seq = sorted(ys, reverse=True) + sorted(ys)[1:]  # down pass then up pass
    cap = case["capital"]

    best = []
    best_profit = 0
    for acts in (_global(tl, ys, seq, cap), _sweep(tl, seq, cap)):
        p = _profit(tl, acts, cap)
        if p > best_profit:
            best, best_profit = acts, p
    return best


def _global(tl, ys, seq, cap):
    items = []  # (buy_price, unit_value, stock, buy_year, sell_year, max_qty)
    for s in {s for y in ys for s in tl[y]}:
        buys = [(tl[y][s]["price"], y) for y in ys if s in tl[y] and tl[y][s]["qty"] > 0]
        if not buys:
            continue
        bp, by = min(buys)
        sp, sy = max((tl[y][s]["price"], y) for y in ys if s in tl[y])
        if sp > bp:
            items.append((bp, sp - bp, s, by, sy, tl[by][s]["qty"]))

    buy_at, sell_at = {}, {}
    for (_bp, _v, s, by, sy, _mq), q in _knapsack(items, cap):
        buy_at.setdefault(by, []).append((s, q))
        sell_at.setdefault(sy, []).append((s, q))

    seen, plan = set(), []
    for k, y in enumerate(seq):
        a = []
        if y not in seen:  # first (down-pass) visit: buy here
            a += [f"b-{s}-{q}" for s, q in buy_at.get(y, [])]
        seen.add(y)
        if k == max(i for i, yy in enumerate(seq) if yy == y):  # last visit: sell
            a += [f"s-{s}-{q}" for s, q in sell_at.get(y, [])]
        plan.append((y, a))
    return _emit(plan)


def _sweep(tl, seq, cap):
    n = len(seq)
    left = {(y, s): d["qty"] for y in set(seq) for s, d in tl[y].items()}
    due = [[] for _ in range(n)]  # sells landing at each stop: (stock, qty)
    plan = []
    for i, y in enumerate(seq):
        a = [f"s-{s}-{q}" for s, q in due[i]]
        for s, q in due[i]:
            cap += tl[y][s]["price"] * q
        items = []  # (buy_price, unit_value, stock, sell_stop, max_qty)
        for s, d in tl[y].items():
            j = max((k for k in range(i + 1, n) if s in tl[seq[k]]),
                    key=lambda k: tl[seq[k]][s]["price"], default=None)
            if left[(y, s)] and j and tl[seq[j]][s]["price"] > d["price"]:
                items.append((d["price"], tl[seq[j]][s]["price"] - d["price"], s,
                              j, left[(y, s)]))
        for (bp, _v, s, j, _mq), q in _knapsack(items, cap):
            cap -= bp * q
            left[(y, s)] -= q
            due[j].append((s, q))
            a.append(f"b-{s}-{q}")
        plan.append((y, a))
    return _emit(plan)


def _emit(plan):
    out, cur = [], START
    for y, a in plan:
        if not a:
            continue
        if y != cur:
            out.append(f"j-{cur}-{y}")
            cur = y
        out.extend(a)
    if out and cur != START:
        out.append(f"j-{cur}-{START}")
    return out


def _profit(tl, actions, cap):
    """Replay actions under the rules; return final-minus-start, or -1 if any
    rule is violated (so an invalid plan never wins the max)."""
    year, held = START, {}
    inv = {(y, s): d["qty"] for y, v in tl.items() for s, d in v.items()}
    start = cap
    for a in actions:
        k, x, z = a.split("-")
        if k == "j":
            if int(x) != year:
                return -1
            year = int(z)
        elif k == "b":
            q = int(z)
            if inv.get((year, x), 0) < q or tl[year][x]["price"] * q > cap:
                return -1
            inv[(year, x)] -= q
            cap -= tl[year][x]["price"] * q
            held[x] = held.get(x, 0) + q
        else:
            q = int(z)
            if held.get(x, 0) < q:
                return -1
            held[x] -= q
            cap += tl[year][x]["price"] * q
    return cap - start if year == START else -1


def _knapsack(items, cap):
    """Bounded knapsack: maximize value under budget `cap`. Each item is a tuple
    whose [0] is unit cost, [1] is unit value, [-1] is max qty. Returns
    [(item, qty)]. Budget is clamped to the max spendable so the DP width stays
    bounded even when capital dwarfs what's for sale."""
    if not items:
        return []
    budget = min(cap, sum(it[0] * it[-1] for it in items))
    if budget <= 0:
        return []
    pieces = []  # binary-decompose bounded counts into 0/1 chunks
    for idx, it in enumerate(items):
        cost, val, mq = it[0], it[1], it[-1]
        k = 1
        while mq > 0:
            take = min(k, mq)
            pieces.append((cost * take, val * take, idx, take))
            mq -= take
            k *= 2
    dp = [0] * (budget + 1)
    pick = [[] for _ in range(budget + 1)]  # (item_index, k) list to reach spend c
    for cost, val, idx, k in pieces:
        for c in range(budget, cost - 1, -1):
            if dp[c - cost] + val > dp[c]:
                dp[c] = dp[c - cost] + val
                pick[c] = pick[c - cost] + [(idx, k)]
    best = max(range(budget + 1), key=lambda c: dp[c])
    qty = {}
    for idx, k in pick[best]:
        qty[idx] = qty.get(idx, 0) + k
    return [(items[idx], q) for idx, q in qty.items()]
