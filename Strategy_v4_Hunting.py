import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ==========================================
# 1. 猎杀数据引擎 (Native Data + BB + RSI)
# ==========================================
def get_hunting_data(symbol="BTC-USD", period="1y"):
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    close = df["Close"].astype(float)
    # 情绪指标：极度超卖 RSI < 25
    df["RSI"] = ta.rsi(close, length=14)
    # 基础指标：布林带 (均值回归目标)
    bb = ta.bbands(close, length=20, std=2)
    df["BB_Lower"] = bb.iloc[:, 0] # BBL_20
    df["BB_Mid"] = bb.iloc[:, 1]   # BBM_20
    return df.dropna()

# ==========================================
# 2. 猎杀回测引擎 (倍投 10%:20%:40%)
# ==========================================
def run_backtest(df, initial_balance=10000.0):
    cash = initial_balance
    total_pos = 0.0
    avg_cost = 0.0
    layer = 0
    invest_plan = [0.1, 0.2, 0.4] # 猎杀资金链：10%, 20%, 40% (留 30% 保命钱)
    
    equity_curve = []
    trades = [] # 记录所有买卖点用于绘图

    for i in range(20, len(df)):
        date = df.index[i]
        price = float(df["Close"].iloc[i])
        rsi = float(df["RSI"].iloc[i])
        lower = float(df["BB_Lower"].iloc[i])
        mid = float(df["BB_Mid"].iloc[i])

        # --- A. 猎杀出场 (回归中轨 - 鳄鱼松口) ---
        if total_pos > 0 and price >= mid:
            cash += total_pos * price
            trades.append({"type": "SELL", "price": price, "date": date})
            total_pos, avg_cost, layer = 0, 0, 0

        # --- B. 猎杀入场 (RSI < 25 且 价格跌破下轨) ---
        elif layer == 0 and price < lower and rsi < 25:
            invest = initial_balance * invest_plan[0]
            total_pos = invest / price
            avg_cost = price
            cash -= invest
            layer = 1
            trades.append({"type": "BUY", "layer": 1, "price": price, "date": date})

        # --- C. 暴力倍投 (每跌 10% 补一刀，力度翻倍) ---
        elif 0 < layer < 3:
            # 当价格比加权平均成本又跌了 10% 时，触发下一次倍投
            if price < avg_cost * 0.90:
                invest = initial_balance * invest_plan[layer]
                new_pos = invest / price
                # 更新加权平均成本: (旧量*旧价 + 新量*新价) / 总量
                avg_cost = (total_pos * avg_cost + new_pos * price) / (total_pos + new_pos)
                total_pos += new_pos
                cash -= invest
                layer += 1
                trades.append({"type": "BUY", "layer": layer, "price": price, "date": date})

        # 记录每日权益用于计算 MDD
        eq = cash + (total_pos * price)
        equity_curve.append(eq)

    equity_series = pd.Series(equity_curve, index=df.index[20:])
    final_val = equity_curve[-1] if equity_curve else initial_balance
    return final_val, trades, equity_series

# ==========================================
# 3. 统计学结算与绘图引擎 (Visualization)
# ==========================================
def analyze_and_plot(df, initial, final, trades, equity):
    # 计算 MDD
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0, np.nan)
    max_dd = drawdown.min()
    total_return_pct = ((final / initial) - 1.0) * 100.0

    print(f"--- 🏴‍☠️ 猎杀模式最终战报 ---")
    print(f"初始资金: $10,000.00")
    print(f"期末净值: ${final:.2f}")
    print(f"累计收益: {total_return_pct:.2f}%")
    print(f"最大回撤 (MDD): {max_dd * 100:.2f}%")
    print(f"总操作次数: {len([t for t in trades if t['type']=='BUY'])}")
    print("-----------------------------------")

    # 开始绘图 (使用 DejaVu Sans 避免汉字乱码问题)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    plt.subplots_adjust(hspace=0.05)

    # 上图：价格 + 布林带 + 猎杀买卖点
    ax1.plot(df.index, df["Close"], label="BTC Close", color="black", linewidth=1.0)
    ax1.plot(df.index, df["BB_Lower"], color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax1.plot(df.index, df["BB_Mid"], color="#1f77b4", linewidth=0.9, alpha=0.8, label="BB Mid")
    ax1.fill_between(df.index, df["BB_Lower"], df.iloc[:, 2], color="#eeeeee", alpha=0.5) # 填充下轨到上轨区域

    # 标记猎杀买卖点
    for t in trades:
        if t["type"] == "BUY":
            # 倍投力度越大，箭头越深
            color = "#2ca02c" if t["layer"] == 1 else "#0d4f0d" if t["layer"] == 2 else "#000000"
            ax1.scatter(t["date"], t["price"], marker="^", s=130, color=color, zorder=5)
        else:
            ax1.scatter(t["date"], t["price"], marker="v", s=130, color="#d62728", zorder=5)

    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.set_title("Strategy v4 Hunting: RSI < 25 + Pyramid Scaling (10:20:40)")

    # 中图：RSI + 极度恐惧参考线 (25)
    ax2.plot(df.index, df["RSI"], color="#9467bd", linewidth=1.0)
    ax2.axhline(25.0, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.8, label="Panic Line (25)")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right")

    # 下图：资产净值曲线 (Equity Curve)
    ax3.plot(equity.index, equity, color="#2ca02c", linewidth=1.2, label="Net Equity")
    ax3.fill_between(equity.index, equity, 10000, color="#2ca02c", alpha=0.1)
    ax3.set_ylabel("Equity ($)")
    ax3.legend(loc="upper left")
    
    # 优化日期轴格式
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax3.xaxis.get_major_locator()))

    # 保存 PNG
    save_path = Path("D:/quant/strategy_v4_hunting_trades.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图表已生成并保存至: {save_path}")

    # 自动弹出图片
    plt.show()

# ==========================================
# 4. 执行流程 (Main)
# ==========================================
if __name__ == "__main__":
    df_data = get_hunting_data()
    final_balance, trade_log, equity_series = run_backtest(df_data)
    analyze_and_plot(df_data, 10000.0, final_balance, trade_log, equity_series)