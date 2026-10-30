"""第 2 段：单份合约跑通，并验证现金账户计息是否接对。
用法：cd src && python seg2_baseline.py
"""
import numpy as np
import bs
import hedge
import config as C

rng = np.random.default_rng(C.SEED)
S0 = 4.00
K = 4.00
ND = C.TENORS["3M"]
T0 = ND / 252
NU = C.NOTIONAL / S0
etf = hedge.Cost(C.COMMISSION, C.SPREAD_ETF, 0.0, C.LOT)

prem = float(bs.call(S0, K, T0, C.R, C.SIGMA_FALLB)) * NU
print(f"权利金        {prem:12,.0f} 元 = 名义本金 {prem/C.NOTIONAL*1e4:6.0f} bp")
print(f"名义本金对应   {NU:12,.0f} 份 = {NU/C.LOT:,.0f} 手")
print(f"建仓日对冲比例 {float(bs.delta(np.array([S0]),K,T0,C.R,C.SIGMA_FALLB)[0]):12.4f}")
print(f"ATM vega      {float(bs.vega(S0,K,T0,C.R,C.SIGMA_FALLB))*NU:12,.0f} 元 / 每 1.0 波动率")
print()

P = hedge.gbm_paths(S0, C.R, C.SIGMA_FALLB, ND, C.N_PATHS, rng, dt=1 / 252)
o = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, etf, rule="daily")
pnl = prem + o["net"]
print("每日调仓，4000 条路径：")
print(f"  手续费均值    {o['cost'].mean():12,.0f} 元 = {o['cost'].mean()/C.NOTIONAL*1e4:6.2f} bp")
print(f"  调仓次数均值  {o['ntr'].mean():12.1f}")
print(f"  台账盈亏均值  {pnl.mean():12,.0f} 元")
print(f"  台账盈亏标准差{pnl.std():12,.0f} 元")
print()

# 自检：把计息关掉，看偏差是否等于「平均持仓 × r × T」
o0 = hedge.hedge_batch(P, K, T0, C.R, C.SIGMA_FALLB, NU, etf,
                       rule="daily", credit_interest=False)
pnl0 = prem + o0["net"]
gap = pnl0.mean() - pnl.mean()
approx = 0.54 * C.NOTIONAL * C.R * T0
print("自检：现金账户不计息时")
print(f"  台账盈亏均值  {pnl0.mean():12,.0f} 元")
print(f"  与计息版之差  {gap:12,.0f} 元")
print(f"  理论估算      {approx:12,.0f} 元  (平均持仓约 0.54M x r x T)")
print(f"  两者之比      {gap/approx:12.2f}   落在 0.8-1.2 说明计息接对了")
