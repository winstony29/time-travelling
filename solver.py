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
import time

START = 2037


def solve(case, deadline_s=0.5):
    tl = {int(y): s for y, s in case["timeline"].items()}
    deep = min((y for y in tl if 2 * (START - y) <= case["energy"]), default=START)
    ys = [y for y in tl if deep <= y <= START]
    seq = sorted(ys, reverse=True) + sorted(ys)[1:]  # down pass then up pass
    cap = case["capital"]
    deadline = time.monotonic() + deadline_s  # hard wall so a worker never hangs

    # heuristics give a strong starting bound; the exact search then only has to
    # beat it, which makes branch-and-bound prune aggressively.
    best, best_profit = [], 0
    for acts in (_global(tl, ys, seq, cap),
                 _sweep(tl, seq, cap, "far"), _sweep(tl, seq, cap, "near"),
                 _roll(tl, seq, cap)):
        p = _profit(tl, acts, cap)
        if p > best_profit:
            best, best_profit = acts, p

    if time.monotonic() < deadline:  # only try exact if there's time budget left
        exact = _exact(tl, seq, cap, best_profit, deadline=deadline)
        if exact is not None:
            ep = _profit(tl, exact, cap)
            if ep > best_profit:
                best = exact
    return best


def _exact(tl, seq, cap0, seed_profit, node_budget=400_000, deadline=None):
    """Branch-and-bound for the true optimum. Branch at each stop over which
    stocks to buy (each all-in up to capital/qty) and which held lots to sell;
    prune with an admissible bound (best remaining sell/buy ratio applied to all
    liquid value). Returns the best action list found, or None if the search
    hit its node budget without certifying optimality (caller keeps heuristic).

    Quantities aren't branched: at a stop you buy a chosen stock to the max you
    can afford/there is, and sell a held lot fully -- partial amounts never beat
    that under a single linear price per stop.
    """
    n = len(seq)
    # per stop, per stock: best price strictly later in the sweep (for the bound)
    fut_max = [{} for _ in range(n)]
    for i in range(n - 1, -1, -1):
        nxt = fut_max[i + 1] if i + 1 < n else {}
        m = dict(nxt)
        for s, d in tl[seq[i]].items():
            m[s] = max(m.get(s, 0), d["price"])
        fut_max[i] = m
    inv0 = tuple(sorted((k, d["qty"]) for k, d in
                        ((( y, s), d) for y in set(seq) for s, d in tl[y].items())))

    nodes = [0]
    best = [seed_profit, None]  # (profit, action-plan as list of (stop, action))

    # best remaining buy->sell ratio from each stop onward (admissible multiplier)
    best_ratio_from = [1.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        r = best_ratio_from[i + 1]
        for s, d in tl[seq[i]].items():
            if d["qty"] and fut_max[i + 1 if i + 1 < n else i].get(s, 0) > d["price"]:
                r = max(r, fut_max[i + 1][s] / d["price"])
        best_ratio_from[i] = r

    def bound(i, cap, held):
        # optimistic ceiling: all liquid value grows by the best remaining ratio
        liquid = cap + sum(q * tl[seq[sj]][s]["price"] for s, (q, sj) in held.items())
        return liquid * best_ratio_from[i]

    def dfs(i, cap, held, inv, path):
        nodes[0] += 1
        if nodes[0] > node_budget or (
                deadline and nodes[0] % 2048 == 0 and time.monotonic() > deadline):
            raise _Budget
        # realize any holdings whose scheduled sell stop is now (kept minimal:
        # held maps stock -> (qty, sell_stop); we settle when we pass sell_stop)
        if i == n:
            profit = cap - cap0
            if profit > best[0]:
                best[0], best[1] = profit, list(path)
            return
        if bound(i, cap, held) - cap0 <= best[0]:
            return
        y = seq[i]
        # candidate sells: any held lot listed here (sell fully -- partial never wins)
        # candidate buys: any listed stock with qty and a strictly-higher future price
        buys = []
        for s, d in tl[y].items():
            key = _idx(inv, (y, s))
            if inv[key][1] > 0 and d["price"] < fut_max[i + 1 if i + 1 < n else i].get(s, 0):
                buys.append(s)
        sells = [s for s, (q, _sj) in held.items() if s in tl[y] and q > 0]

        # branch: choose a subset of sells to realize now, then a subset of buys.
        # to bound the branching factor we cap subset enumeration; with few
        # decision-relevant stocks per stop this stays small.
        import itertools as _it
        sell_opts = list(_it.chain.from_iterable(
            _it.combinations(sells, r) for r in range(len(sells) + 1)))
        buy_opts = list(_it.chain.from_iterable(
            _it.combinations(buys, r) for r in range(len(buys) + 1)))
        for sset in sell_opts:
            c2 = cap
            h2 = dict(held)
            p2 = list(path)
            for s in sset:
                q = h2[s][0]
                c2 += tl[y][s]["price"] * q
                p2.append((i, f"s-{s}-{q}"))
                del h2[s]
            for bset in buy_opts:
                c3, h3, inv3, p3 = c2, dict(h2), list(inv), list(p2)
                for s in bset:
                    key = _idx(inv, (y, s))
                    price = tl[y][s]["price"]
                    q = min(inv3[key][1], c3 // price)
                    if q <= 0:
                        continue
                    c3 -= price * q
                    inv3[key] = (inv3[key][0], inv3[key][1] - q)
                    sj = max(range(i + 1, n), key=lambda k: tl[seq[k]][s]["price"]
                             if s in tl[seq[k]] else -1)
                    prev = h3.get(s)
                    h3[s] = (q + (prev[0] if prev else 0), sj)
                    p3.append((i, f"b-{s}-{q}"))
                dfs(i + 1, c3, h3, tuple(inv3), p3)

    try:
        dfs(0, cap0, {}, inv0, [])
    except _Budget:
        return None
    if best[1] is None:
        return []
    # rebuild ordered action list, inserting jumps
    return _emit_indexed(seq, best[1])


class _Budget(Exception):
    pass


def _idx(inv, key):
    lo, hi = 0, len(inv)
    while lo < hi:
        mid = (lo + hi) // 2
        if inv[mid][0] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _emit_indexed(seq, indexed):
    by_stop = {}
    for i, a in indexed:
        by_stop.setdefault(i, []).append(a)
    plan = [(seq[i], by_stop.get(i, [])) for i in range(len(seq))]
    return _emit(plan)


def _roll(tl, seq, cap):
    """Greedy capital-rolling for max compounding: at each stop sell everything
    held (freeing capital), then pour all capital into the single best remaining
    trade -- the (stock, later-stop) pair with the highest sell/buy ratio. Rolls
    proceeds forward hop by hop, which is what multiplies capital fastest."""
    n = len(seq)
    left = {(y, s): d["qty"] for y in set(seq) for s, d in tl[y].items()}
    held = {}  # stock -> (qty, sell_stop)
    plan = [[] for _ in range(n)]
    for i, y in enumerate(seq):
        for s, (q, sj) in list(held.items()):
            if sj == i:
                cap += tl[y][s]["price"] * q
                plan[i].append(f"s-{s}-{q}")
                del held[s]
        best = None  # (ratio, buy_price, stock, sell_stop)
        for s, d in tl[y].items():
            if not left[(y, s)] or d["price"] > cap:
                continue
            j = max((k for k in range(i + 1, n) if s in tl[seq[k]]),
                    key=lambda k: tl[seq[k]][s]["price"], default=None)
            if j and tl[seq[j]][s]["price"] > d["price"]:
                r = tl[seq[j]][s]["price"] / d["price"]
                if best is None or r > best[0]:
                    best = (r, d["price"], s, j)
        if best:
            _r, p, s, j = best
            q = min(left[(y, s)], cap // p)
            if q:
                cap -= p * q
                left[(y, s)] -= q
                held[s] = (held.get(s, (0, j))[0] + q, j)
                plan[i].append(f"b-{s}-{q}")

    out = []
    for i, y in enumerate(seq):
        out.append((y, plan[i]))
    return _emit(out)


def _global(tl, ys, seq, cap):
    items = []  # (buy_price, unit_value, stock, buy_year, sell_year, max_qty)
    for s in {s for y in ys for s in tl[y]}:
        sp, sy = max((tl[y][s]["price"], y) for y in ys if s in tl[y])
        # every year cheaper than the stock's dearest year is worth buying at,
        # to its full qty -- capital gets split across them by the knapsack
        for y in ys:
            d = tl[y].get(s)
            if d and d["qty"] > 0 and d["price"] < sp:
                items.append((d["price"], sp - d["price"], s, y, sy, d["qty"]))

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


def _sweep(tl, seq, cap, target="far"):
    """Walk the sweep, at each stop knapsack the capital on hand into buys whose
    sells land later. `target` picks each buy's sell stop:
      far  -- the dearest future stop (max total gain per lot)
      near -- the nearest future stop that still beats the buy price, so capital
              is freed sooner to compound into later buys.
    Both are heuristics; solve() keeps whichever simulates to more profit."""
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
            fut = [k for k in range(i + 1, n)
                   if s in tl[seq[k]] and tl[seq[k]][s]["price"] > d["price"]]
            if not left[(y, s)] or not fut:
                continue
            j = min(fut) if target == "near" else max(fut, key=lambda k: tl[seq[k]][s]["price"])
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


# ponytail: caps the knapsack DP width so a huge capital*qty can't OOM/timeout
# the worker. Above this, fall back to greedy-by-ratio (near-optimal for money).
# Kept small on purpose: the DP runs many times per request (4 strategies x
# stops x cases), so an exact-but-wide DP is what killed the worker at 200k.
_KNAP_MAX = 20_000


def _knapsack(items, cap):
    """Bounded knapsack: maximize value under budget `cap`. Each item is a tuple
    whose [0] is unit cost, [1] is unit value, [-1] is max qty. Returns
    [(item, qty)]. Budget is clamped to the max spendable; if that still exceeds
    _KNAP_MAX the exact DP would be too big, so use greedy-by-ratio instead."""
    if not items:
        return []
    # every item here is profitable (value>0); if we can afford them all, do so.
    total_cost = sum(it[0] * it[-1] for it in items)
    if cap >= total_cost:
        return [(it, it[-1]) for it in items]
    budget = cap
    # scale the budget into <= _KNAP_MAX buckets so a huge capital can't blow up
    # the DP. costs round up (never overspend); near-exact, and exact when scale=1.
    scale = 1
    while budget // scale > _KNAP_MAX:
        scale *= 2
    B = budget // scale
    pieces = []  # binary-decompose bounded counts into 0/1 chunks
    for idx, it in enumerate(items):
        cost, val, mq = -(-it[0] // scale), it[1], it[-1]  # ceil-divide cost
        cost = max(cost, 1)
        k = 1
        while mq > 0:
            take = min(k, mq)
            pieces.append((cost * take, val * take, idx, take))
            mq -= take
            k *= 2
    dp = [0] * (B + 1)
    par = [None] * (B + 1)  # parent pointer: (prev_c, item_index, k) to reach c
    for cost, val, idx, k in pieces:
        if cost > B:
            continue
        for c in range(B, cost - 1, -1):
            if dp[c - cost] + val > dp[c]:
                dp[c] = dp[c - cost] + val
                par[c] = (c - cost, idx, k)
    best = max(range(B + 1), key=lambda c: dp[c])
    qty = {}
    c = best
    while par[c] is not None:  # O(pieces) backtrack, no per-cell list copies
        prev, idx, k = par[c]
        qty[idx] = qty.get(idx, 0) + k
        c = prev
    # scaling rounds costs up, so the plan may nominally overspend; trim greedily
    # by real cost to fit actual capital while keeping the highest-value items.
    picked = sorted(qty.items(), key=lambda kv: items[kv[0]][1] / items[kv[0]][0])
    spent = sum(items[i][0] * q for i, q in qty.items())
    for i, q in picked:
        while q > 0 and spent > cap:
            spent -= items[i][0]
            q -= 1
            qty[i] -= 1
    return [(items[idx], q) for idx, q in qty.items() if q > 0]
