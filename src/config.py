"""全项目唯一的参数入口。改参数只改这里。"""
SYMBOL      = "510300"          # 沪深300ETF
START       = "20200101"
END         = "20260630"

R           = 0.018             # 无风险利率，年化
SIGMA_FALLB = 0.18              # 取不到期权数据时的降级波动率
VOL_PREMIUM = 0.02              # 降级时在已实现波动上加的溢价

NOTIONAL    = 10_000_000        # 名义本金，元
TENORS      = {"1M": 21, "3M": 63, "6M": 126}
MONEYNESS   = {"ITM": 0.95, "ATM": 1.00, "OTM": 1.05}

LOT         = 100               # 场内ETF最小申报单位，份
COMMISSION  = 2e-4              # 佣金，双向
SPREAD_ETF  = 1.5e-4            # 半价差+冲击，ETF
SPREAD_STK  = 3.0e-4            # 半价差+冲击，个股
STAMP_STK   = 5e-4              # 印花税，仅卖出，个股；ETF 为 0

N_PATHS     = 4000
SEED        = 20260820
