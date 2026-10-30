"""对冲模拟引擎。按时间步推进，路径维度整体向量化。

一条路径 = 一份合约从建仓到到期的完整过程。
返回的 net 已包含：全部交易费用、现金账户利息、到期赔付。
台账最终盈亏 = 权利金 + net。
"""
import numpy as np
import bs


class Cost:
    """费率全部以「占成交金额的比例」计。stamp_sell 只作用于卖出方向。"""

    def __init__(self, commission, spread, stamp_sell=0.0, lot=100):
        self.commission = commission
        self.spread = spread
        self.stamp_sell = stamp_sell
        self.lot = lot


def gbm_paths(S0, mu, sigma, n_steps, n_paths, rng, dt):
    """几何布朗运动。每日独立同分布，无波动聚集。"""
    z = rng.standard_normal((n_paths, n_steps))
    lr = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    return np.concatenate([np.full((n_paths, 1), float(S0)),
                           S0 * np.exp(np.cumsum(lr, axis=1))], axis=1)


def gjr_paths(S0, n_days, n_paths, rng, ann_vol=0.18,
              alpha=0.05, beta=0.90, gamma=0.06, mu=0.0):
    """GJR-GARCH(1,1)。带波动聚集与下跌放大。
    omega 由无条件方差反解：v = omega / (1 - alpha - beta - gamma/2)。
    漏掉 gamma/2 会让实际波动系统性高于设定值。"""
    persist = alpha + beta + 0.5 * gamma
    v = (ann_vol ** 2) / 252
    omega = v * (1 - persist)
    h = np.full(n_paths, v)
    S = np.full(n_paths, float(S0))
    out = np.zeros((n_paths, n_days + 1))
    out[:, 0] = S0
    for t in range(n_days):
        eps = np.sqrt(h) * rng.standard_normal(n_paths)
        S = S * np.exp(mu / 252 - 0.5 * h + eps)
        out[:, t + 1] = S
        h = omega + (alpha + gamma * (eps < 0)) * eps ** 2 + beta * h
    return out


def block_bootstrap(returns, n_days, n_paths, rng, block=10):
    """区块自助重抽样。保留 block 长度内的时序结构，用于在真实收益上扩样本。"""
    ret = np.asarray(returns, dtype=float)
    ret = ret[np.isfinite(ret)]
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, len(ret) - block, size=(n_paths, n_blocks))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    r = ret[idx].reshape(n_paths, -1)[:, :n_days]
    return np.concatenate([np.ones((n_paths, 1)), np.exp(np.cumsum(r, axis=1))], axis=1)


def hedge_batch(S, K, T0, r, sig_hedge, n_units, cm,
                rule="band", band=0.05, interval=1, ww_gamma=1e-6,
                steps_per_day=1, credit_interest=True):
    """S 形状 (n_paths, n_steps+1)，S[:,0] 为建仓日价格。
    credit_interest=False 只用于自检：关掉计息应看到盈亏整体上移一个利息的量级。"""
    n_paths, n1 = S.shape
    n = n1 - 1
    dt = T0 / n
    disc = np.exp(r * dt) if credit_interest else 1.0
    pos = np.zeros(n_paths)      # 持有的标的份数
    cash = np.zeros(n_paths)     # 对冲账户现金，负数表示融资
    cost = np.zeros(n_paths)
    ntr = np.zeros(n_paths)
    turn = np.zeros(n_paths)
    rerr = np.zeros(n_paths)     # 取整抹掉的份数，累计
    last = np.full(n_paths, -10 ** 9)

    for t in range(n + 1):
        if t > 0:
            cash *= disc                      # 先计息，再调仓
        tau = T0 - t * dt
        St = S[:, t]
        target = n_units * bs.delta(St, K, tau, r, sig_hedge)

        if t == 0 or t == n:
            do = np.ones(n_paths, dtype=bool)   # 建仓与平仓强制执行
        elif rule == "daily":
            do = np.ones(n_paths, dtype=bool)
        elif rule == "interval":
            do = (t - last) >= interval * steps_per_day
        elif rule == "band":
            do = np.abs(target - pos) > band * n_units
        elif rule == "ww":
            g = bs.gamma(St, K, tau, r, sig_hedge)
            k = cm.commission + cm.spread
            h = np.cbrt(1.5 * k * St * g ** 2 / ww_gamma)
            do = np.abs(target - pos) > h * n_units
        else:
            raise ValueError(f"unknown rule: {rule}")

        raw = np.where(do, target - pos, 0.0)
        d = np.round(raw / cm.lot) * cm.lot if cm.lot > 1 else raw
        rerr += np.abs(raw - d)
        notion = np.abs(d) * St
        c = notion * (cm.commission + cm.spread) + \
            np.where(d < 0, notion * cm.stamp_sell, 0.0)
        cost += c
        cash -= d * St + c
        pos += d
        ntr += (np.abs(d) > 0)
        turn += notion
        last = np.where(do, t, last)

    payoff = np.maximum(S[:, -1] - K, 0.0) * n_units
    net = cash + pos * S[:, -1] - payoff
    return dict(cost=cost, net=net, ntr=ntr, turn=turn, rerr=rerr, payoff=payoff)


def addon_bp(pnl, notional, conf=0.95):
    """报价加点：使台账在 conf 置信水平上不亏的最小加成，单位 bp。"""
    q = np.percentile(pnl, (1 - conf) * 100)
    return max(-q, 0.0) / notional * 1e4


def addon_mean_bp(pnl, notional):
    return max(-float(np.mean(pnl)), 0.0) / notional * 1e4
