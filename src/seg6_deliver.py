"""第 6 段：生成两张交付表。
用法：cd src && python seg6_deliver.py
产物：out/addon_table.csv、out/band_table.csv
"""
import numpy as np
import pandas as pd
import bs, hedge
import config as C

rng = np.random.default_rng(C.SEED + 4)
S0 = 4.00
NU = C.NOTIONAL / S0
etf = hedge.Cost(C.COMMISSION, C.SPREAD_ETF, 0.0, C.LOT)
BAND = 0.06
BOOT_N = 250          # 假想学生手上有 250 个可用窗口，用于给区间


def one_cell(nd, mk, paths_model="gbm"):
    T0 = nd / 252
    K = S0 * mk
    if paths_model == "gbm":
        P = hedge.gbm_paths(S0, C.R, C.SIGMA_FALLB, nd, C.N_PATHS, rng, dt=1 / 252)
    else:
        P = hedge.gjr_paths(S0, nd, C.N_PATHS, rng, ann_vol=C.SIGMA_FALLB)
    prem = float(bs.call(S0, K, T0, C.R, C.SIGMA_FALLB)) * NU
    o = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, etf,
                          rule="band", band=BAND)
    pnl = prem + o["net"]
    a = hedge.addon_bp(pnl, C.NOTIONAL)
    boot = np.array([hedge.addon_bp(rng.choice(pnl, BOOT_N, replace=True), C.NOTIONAL)
                     for _ in range(500)])
    return dict(加点bp=a, 区间下=np.percentile(boot, 5), 区间上=np.percentile(boot, 95),
                手续费bp=o["cost"].mean() / C.NOTIONAL * 1e4,
                权利金bp=prem / C.NOTIONAL * 1e4,
                调仓次数=o["ntr"].mean())


rows = []
for ml, mk in C.MONEYNESS.items():
    for tl, nd in C.TENORS.items():
        c = one_cell(nd, mk)
        c2 = one_cell(nd, mk, "gjr")
        rows.append(dict(行权价档=ml, 期限=tl, **c,
                         聚集模型加点bp=c2["加点bp"],
                         手续费占加点=c["手续费bp"] / c["加点bp"] * 100,
                         加点占权利金=c["加点bp"] / c["权利金bp"] * 100))
df = pd.DataFrame(rows)
df.to_csv("../out/addon_table.csv", index=False, float_format="%.2f")

print("===== 报价加点表（带宽 0.06，95% 口径，简单模型）=====")
print(f"{'档位':<6}{'期限':<5}{'加点bp':>9}{'250窗口区间':>18}{'手续费bp':>10}"
      f"{'权利金bp':>10}{'费占加点':>10}{'加点占权利金':>13}{'聚集模型加点':>13}")
for _, r in df.iterrows():
    print(f"{r['行权价档']:<6}{r['期限']:<5}{r['加点bp']:>9.1f}"
          f"{r['区间下']:>9.1f} –{r['区间上']:>7.1f}"
          f"{r['手续费bp']:>10.1f}{r['权利金bp']:>10.0f}"
          f"{r['手续费占加点']:>9.1f}%{r['加点占权利金']:>12.0f}%"
          f"{r['聚集模型加点bp']:>13.1f}")

flag = df[df["加点占权利金"] > 100]
if len(flag):
    print("\n以下格子的加点高于权利金本身，按 95% 口径无法单笔报出：")
    for _, r in flag.iterrows():
        print(f"  {r['行权价档']} {r['期限']}：加点 {r['加点bp']:.1f} bp "
              f"> 权利金 {r['权利金bp']:.0f} bp")

band_rows = []
for ml, mk in C.MONEYNESS.items():
    for tl, nd in C.TENORS.items():
        T0 = nd / 252
        K = S0 * mk
        P = hedge.gbm_paths(S0, C.R, C.SIGMA_FALLB, nd, C.N_PATHS, rng, dt=1 / 252)
        prem = float(bs.call(S0, K, T0, C.R, C.SIGMA_FALLB)) * NU
        best, best_a = None, 1e9
        for b in (0.02, 0.04, 0.06, 0.09, 0.13):
            o = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, etf,
                                  rule="band", band=b)
            a = hedge.addon_bp(prem + o["net"], C.NOTIONAL)
            if a < best_a:
                best, best_a = b, a
        band_rows.append(dict(行权价档=ml, 期限=tl, 建议带宽=best, 该带宽加点bp=best_a))
bt = pd.DataFrame(band_rows)
bt.to_csv("../out/band_table.csv", index=False, float_format="%.2f")
print("\n===== 调仓带宽建议表 =====")
print(bt.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
print("\n注：同一行内相邻带宽的加点差通常小于抽样区间宽度，"
      "本表给的是点估计最优，不代表统计上可区分。")
