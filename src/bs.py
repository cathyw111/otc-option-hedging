"""Black-Scholes 定价与希腊字母。全部向量化。"""
import numpy as np
from scipy.stats import norm


def _d1(S, K, tau, r, sig):
    return (np.log(S / K) + (r + 0.5 * sig ** 2) * tau) / (sig * np.sqrt(tau))


def call(S, K, tau, r, sig):
    if np.all(np.asarray(tau) <= 1e-12):
        return np.maximum(S - K, 0.0)
    d1 = _d1(S, K, tau, r, sig)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d1 - sig * np.sqrt(tau))


def delta(S, K, tau, r, sig):
    if tau <= 1e-12:
        return (np.asarray(S) > K).astype(float)
    return norm.cdf(_d1(S, K, tau, r, sig))


def gamma(S, K, tau, r, sig):
    if tau <= 1e-12:
        return np.zeros_like(np.asarray(S, dtype=float))
    d1 = _d1(S, K, tau, r, sig)
    return norm.pdf(d1) / (S * sig * np.sqrt(tau))


def vega(S, K, tau, r, sig):
    return S * np.sqrt(tau) * norm.pdf(_d1(S, K, tau, r, sig))


def implied_vol(price, S, K, tau, r, lo=0.01, hi=2.0, tol=1e-6):
    """二分反解隐含波动率。落在无套利边界外或不收敛时返回 nan。"""
    intrinsic = max(S - K * np.exp(-r * tau), 0.0)
    if not (intrinsic < price < S):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(call(S, K, tau, r, mid)) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return 0.5 * (lo + hi)
    return float("nan")
