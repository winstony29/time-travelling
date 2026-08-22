# Trust the strict simulator run(). Verify: (1) no rule violations ever,
# (2) exact >= every heuristic (never worse), across broad random cases.
import random, time
from solver import solve, _global, _sweep, _roll, _exact, _profit, START
from test_solver import run
random.seed(42)
def rc():
    e=random.randint(2,30); 
    span=random.randint(2, min(20, e//2+3))
    yrs=random.sample(range(2037-span,2038), min(span,2037-(2037-span)+1) if span<=8 else span)
    yrs=random.sample(range(2037-span,2038), span)
    return {"energy":e,"capital":random.randint(10,2000),
      "timeline":{str(y):{s:{"price":random.randint(1,100),"qty":random.randint(0,50)}
        for s in random.sample(["A","B","C","D","E"],random.randint(1,5))} for y in yrs}}
viol=0; worse=0; nexact=0; t0=time.time(); slow=0
for t in range(3000):
    c=rc()
    try:
        acts=solve(c); fin=run(c,acts)
    except AssertionError as e:
        viol+=1; import json; print("VIOL",e,json.dumps(c)); 
        if viol>5: break
        continue
    # compare vs each heuristic profit
    tl={int(y):v for y,v in c["timeline"].items()}
    deep=min((y for y in tl if 2*(START-y)<=c["energy"]),default=START)
    ys=[y for y in tl if deep<=y<=START]; seq=sorted(ys,reverse=True)+sorted(ys)[1:]
    hp=max(_profit(tl,h,c["capital"]) for h in
           [_global(tl,ys,seq,c["capital"]),_sweep(tl,seq,c["capital"],"far"),
            _sweep(tl,seq,c["capital"],"near"),_roll(tl,seq,c["capital"])])
    got=fin-c["capital"]
    if got<hp: worse+=1; print("WORSE than heuristic",got,hp)
print(f"viol={viol} worse_than_heuristic={worse} in {time.time()-t0:.1f}s /3000")
