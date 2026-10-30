"""第 4 段：三类调仓规则的完整对照，以及两种加点口径的排名倒挂。
用法：cd src && python seg4_rules.py
产物：out/rules_etf.csv、out/rules_stk.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import bs, hedge
import config as C

rng = np.random.default_rng(C.SEED + 2)
S0 = K = 4.00
ND = C.TENORS["3M"]
T0 = ND / 252
NU = C.NOTIONAL / S0
prem = float(bs.call(S0, K, T0, C.R, C.SIGMA_FALLB)) * NU

# 三档网格：日频、每日两次、每日四次
G = {m: hedge.gbm_paths(S0, C.R, C.SIGMA_FALLB, ND * m, C.N_PATHS, rng,
                        dt=1 / (252 * m)) for m in (1, 2, 4)}

RULES = [("每日4次", 4, dict(rule="daily")),
         ("每日2次", 2, dict(rule="daily")),
         ("每日1次", 1, dict(rule="daily")),
         ("隔日1次", 1, dict(rule="interval", interval=2)),
         ("每5日1次", 1, dict(rule="interval", interval=5)),
         ("带宽0.02", 1, dict(rule="band", band=0.02)),
         ("带宽0.04", 1, dict(rule="band", band=0.04)),
         ("带宽0.06", 1, dict(rule="band", band=0.06)),
         ("带宽0.09", 1, dict(rule="band", band=0.09)),
         ("带宽0.13", 1, dict(rule="band", band=0.13)),
         ("带宽0.18", 1, dict(rule="band", band=0.18))]

etf = hedge.Cost(C.COMMISSION, C.SPREAD_ETF, 0.0, C.LOT)
stk = hedge.Cost(C.COMMISSION, C.SPREAD_STK, C.STAMP_STK, C.LOT)


def table(cm, label, path):
    rows = []
    for name, m, kw in RULES:
        o = hedge.hedge_batch(G[m], K, T0, C.R, C.SIGMA_FALLB, NU, cm,
                              steps_per_day=m, **kw)
        pnl = prem + o["net"]
        rows.append(dict(
            规则=name,
            手续费bp=o["cost"].mean() / C.NOTIONAL * 1e4,
            占权利金=o["cost"].mean() / prem * 100,
            调仓次数=o["ntr"].mean(),
            换手倍数=o["turn"].mean() / C.NOTIONAL,
            盈亏标准差=pnl.std(),
            加点均值=hedge.addon_mean_bp(pnl, C.NOTIONAL),
            加点85=hedge.addon_bp(pnl, C.NOTIONAL, 0.85),
            加点95=hedge.addon_bp(pnl, C.NOTIONAL, 0.95),
            加点99=hedge.addon_bp(pnl, C.NOTIONAL, 0.99)))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format="%.2f")
    print(f"\n===== {label} =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    return df


df_etf = table(etf, "ETF 对冲", "../out/rules_etf.csv")
df_stk = table(stk, "个股对冲", "../out/rules_stk.csv")

print("\n===== 口径倒挂 =====")
for df, lab in ((df_etf, "ETF"), (df_stk, "个股")):
    rm = df["加点均值"].rank().astype(int)
    rq = df["加点95"].rank().astype(int)
    rho = spearmanr(df["加点均值"], df["加点95"]).statistic
    print(f"\n{lab}：Spearman = {rho:6.3f}   最大名次差 = {int((rq-rm).abs().max())}")
    print(f"  均值口径最优 = {df.loc[df['加点均值'].idxmin(),'规则']}"
          f"  ({df['加点均值'].min():.1f} bp)")
    print(f"  95%口径最优  = {df.loc[df['加点95'].idxmin(),'规则']}"
          f"  ({df['加点95'].min():.1f} bp)")
