"""第 3 段：成本层做实。ETF 与个股两组对照、印花税消融、取整下探。
用法：cd src && python seg3_cost.py
"""
import numpy as np
import bs, hedge
import config as C

rng = np.random.default_rng(C.SEED + 1)
S0 = K = 4.00
ND = C.TENORS["3M"]
T0 = ND / 252
NU = C.NOTIONAL / S0
prem = float(bs.call(S0, K, T0, C.R, C.SIGMA_FALLB)) * NU
P = hedge.gbm_paths(S0, C.R, C.SIGMA_FALLB, ND, C.N_PATHS, rng, dt=1 / 252)

etf = hedge.Cost(C.COMMISSION, C.SPREAD_ETF, 0.0, C.LOT)
stk = hedge.Cost(C.COMMISSION, C.SPREAD_STK, C.STAMP_STK, C.LOT)
stk0 = hedge.Cost(C.COMMISSION, C.SPREAD_STK, 0.0, C.LOT)   # 印花税消融组

print("A. 两组对照（每日调仓）")
print(f"{'对冲标的':<14}{'手续费bp':>10}{'占权利金':>10}{'换手倍数':>10}{'95%加点bp':>11}")
for lab, cm in (("ETF 510300", etf), ("个股", stk), ("个股·关印花税", stk0)):
    o = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, cm, rule="daily")
    pnl = prem + o["net"]
    print(f"{lab:<14}{o['cost'].mean()/C.NOTIONAL*1e4:>10.2f}"
          f"{o['cost'].mean()/prem*100:>9.2f}%{o['turn'].mean()/C.NOTIONAL:>10.2f}"
          f"{hedge.addon_bp(pnl, C.NOTIONAL):>11.1f}")
o_stk = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, stk, rule="daily")
o_stk0 = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, stk0, rule="daily")
o_etf = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, etf, rule="daily")
print(f"\n  印花税单项贡献 = "
      f"{(o_stk['cost'].mean()-o_stk0['cost'].mean())/C.NOTIONAL*1e4:.2f} bp")
print(f"  个股 / ETF 成本之比 = {o_stk['cost'].mean()/o_etf['cost'].mean():.2f}")

print("\nB. 取整下探（带宽 0.06，ETF）")
print(f"{'名义本金':>12}{'份数':>11}{'单次约':>10}{'抹掉占换手':>12}"
      f"{'手续费bp':>10}{'95%加点bp':>11}")
for nom in (50_000, 100_000, 300_000, 1_000_000, 10_000_000):
    nu = nom / S0
    pr = float(bs.call(S0, K, T0, C.R, C.SIGMA_FALLB)) * nu
    o = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, nu, etf,
                          rule="band", band=0.06)
    pnl = pr + o["net"]
    per = o["turn"].mean() / max(o["ntr"].mean(), 1) / S0
    print(f"{nom:>12,}{nu:>11,.0f}{per:>9,.0f}份"
          f"{o['rerr'].mean()*S0/max(o['turn'].mean(),1)*100:>11.2f}%"
          f"{o['cost'].mean()/nom*1e4:>10.2f}"
          f"{hedge.addon_bp(pnl, nom):>11.1f}")

print("\nC. 价差参数敏感性（每日调仓，ETF）")
print(f"{'价差率':>9}{'手续费bp':>10}{'95%加点bp':>11}")
for sp in (0.5e-4, 1.5e-4, 5.0e-4):
    cm = hedge.Cost(C.COMMISSION, sp, 0.0, C.LOT)
    o = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, cm, rule="daily")
    pnl = prem + o["net"]
    print(f"{sp*1e4:>8.1f}bp{o['cost'].mean()/C.NOTIONAL*1e4:>10.2f}"
          f"{hedge.addon_bp(pnl, C.NOTIONAL):>11.1f}")
