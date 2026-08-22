import time, random
from solver import solve
random.seed(0)
# the crash shape: big capital, many stocks/years
for cap, span, ns, e in [(5000,20,10,40),(100000,30,20,60),(1000000,37,30,74)]:
    yrs=random.sample(range(2037-span,2038), span)
    c={"energy":e,"capital":cap,"timeline":{str(y):{f"S{i}":{"price":random.randint(1,500),"qty":random.randint(0,100)} for i in range(ns)} for y in yrs}}
    t=time.time(); a=solve(c); dt=time.time()-t
    print(f"cap={cap} {span}yr {ns}stk e={e}: {dt:.2f}s, {len(a)} acts")
