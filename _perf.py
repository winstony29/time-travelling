import time, random
from solver import solve
random.seed(0)
# large: deep time range, many stocks, big capital
big = {"energy": 200, "capital": 100000,
  "timeline": {str(y): {f"S{i}": {"price": random.randint(1,500), "qty": random.randint(0,200)}
    for i in range(40)} for y in range(1937, 2038)}}
t=time.time(); acts=solve(big); dt=time.time()-t
print(f"100 years x 40 stocks, cap 100k: {dt:.2f}s, {len(acts)} actions")

# pathological capital magnitude
big2 = dict(big); big2["capital"]=10_000_000
t=time.time(); solve(big2); print(f"cap 10M: {time.time()-t:.2f}s")
