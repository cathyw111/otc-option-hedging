"""第 5 段：两套价格模型对照，以及各估计量的抽样区间。
用法：cd src && python seg5_robust.py
产物：out/models.csv、out/sampling.csv
"""
import numpy as np
import pandas as pd
import bs, hedge
import config as C

rng = np.random.default_rng(C.SEED + 3)
S0 = K = 4.00
ND = C.TENORS["3M"]
T0 = ND / 252
NU = C.NOTIONAL / S0
prem = float(bs.call(S0, K, T0, C.R, C.SIGMA_FALLB)) * NU
etf = hedge.Cost(C.COMMISSION, C.SPREAD_ETF, 0.0, C.LOT)

Pg = hedge.gbm_paths(S0, C.R, C.SIGMA_FALLB, ND, C.N_PATHS, rng, dt=1 / 252)
Pj = hedge.gjr_paths(S0, ND, C.N_PATHS, rng, ann_vol=C.SIGMA_FALLB)

rv_g = np.std(np.diff(np.log(Pg), axis=1), axis=1) * np.sqrt(252)
rv_j = np.std(np.diff(np.log(Pj), axis=1), axis=1) * np.sqrt(252)
print("已实现波动（年化）对照：")
print(f"  简单模型  均值 {rv_g.mean():.3f}   "
      f"5-95% 分位 {np.percentile(rv_g,5):.3f} - {np.percentile(rv_g,95):.3f}")
print(f"  聚集模型  均值 {rv_j.mean():.3f}   "
      f"5-95% 分位 {np.percentile(rv_j,5):.3f} - {np.percentile(rv_j,95):.3f}")
print("  两者均值应当接近；聚集模型的分位区间应当明显更宽\n")

RULES = [("每日1次", dict(rule="daily")),
         ("隔日1次", dict(rule="interval", interval=2)),
         ("每5日1次", dict(rule="interval", interval=5)),
         ("带宽0.02", dict(rule="band", band=0.02)),
         ("带宽0.04", dict(rule="band", band=0.04)),
         ("带宽0.06", dict(rule="band", band=0.06)),
         ("带宽0.09", dict(rule="band", band=0.09)),
         ("带宽0.13", dict(rule="band", band=0.13)),
         ("带宽0.18", dict(rule="band", band=0.18))]

rows = []
for name, kw in RULES:
    r = {"规则": name}
    for lab, P in (("简单", Pg), ("聚集", Pj)):
        o = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, etf, **kw)
        pnl = prem + o["net"]
        r[f"手续费_{lab}"] = o["cost"].mean() / C.NOTIONAL * 1e4
        r[f"标准差_{lab}"] = pnl.std()
        r[f"加点95_{lab}"] = hedge.addon_bp(pnl, C.NOTIONAL)
    r["手续费之比"] = r["手续费_聚集"] / r["手续费_简单"]
    r["加点之比"] = r["加点95_聚集"] / r["加点95_简单"]
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("../out/models.csv", index=False, float_format="%.2f")
print(df.to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
print(f"\n最优带宽：简单模型 = {df.loc[df['加点95_简单'].idxmin(),'规则']}，"
      f"聚集模型 = {df.loc[df['加点95_聚集'].idxmin(),'规则']}")

print("\n===== 抽样区间（带宽 0.06，简单模型，重抽样 2000 次）=====")
o = hedge.hedge_batch(Pg, K, T0, C.R, C.SIGMA_FALLB, NU, etf, rule="band", band=0.06)
pnl = prem + o["net"]
cost_bp = o["cost"] / C.NOTIONAL * 1e4
specs = [("平均手续费 bp", cost_bp, lambda x: float(np.mean(x))),
         ("盈亏标准差 元", pnl, lambda x: float(np.std(x))),
         ("95% 加点 bp", pnl, lambda x: hedge.addon_bp(x, C.NOTIONAL))]
out = []
for lab, arr, fn in specs:
    pt = fn(arr)
    line = {"估计量": lab, "点估计": pt}
    for n in (60, 250, 4000):
        b = np.array([fn(rng.choice(arr, n, replace=True)) for _ in range(2000)])
        lo, hi = np.percentile(b, 5), np.percentile(b, 95)
        line[f"n={n}"] = f"{lo:,.2f} – {hi:,.2f}"
        line[f"宽度%_{n}"] = (hi - lo) / 2 / pt * 100
    out.append(line)
    print(f"{lab:<14} 点估计 {pt:>10,.2f}   "
          f"n=60 {line['n=60']:<22} (±{line['宽度%_60']:.0f}%)   "
          f"n=250 {line['n=250']:<22} (±{line['宽度%_250']:.0f}%)")
pd.DataFrame(out).to_csv("../out/sampling.csv", index=False)
