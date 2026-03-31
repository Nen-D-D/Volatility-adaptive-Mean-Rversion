"""
Strategy_v2_Pro — 布林带 + RSI 均值回归（指标：pandas_ta）
数据：yfinance BTC-USD 近一年日线
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib  # pyright: ignore[reportMissingImports]
import matplotlib.dates as mdates  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf


def _bbands_columns(bb: pd.DataFrame) -> tuple[str, str, str]:
    """从 pandas_ta bbands 结果中解析下轨 / 中轨 / 上轨列名。"""
    cols = [str(c) for c in bb.columns]
    lower = next(c for c in cols if c.startswith("BBL"))
    mid = next(c for c in cols if c.startswith("BBM"))
    upper = next(c for c in cols if c.startswith("BBU"))
    return lower, mid, upper


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """兼容 yfinance 多级列名。"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def load_data(symbol: str = "BTC-USD", period: str = "1y") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    df = _flatten_yfinance_columns(raw)
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"下载数据缺少列: {col}，实际列: {list(df.columns)}")
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is None or bb.empty:
        raise ValueError("pandas_ta.bbands 未返回有效数据")
    bbl, bbm, bbu = _bbands_columns(bb)
    df["BB_Lower"] = bb[bbl]
    df["BB_Mid"] = bb[bbm]
    df["BB_Upper"] = bb[bbu]

    df["RSI"] = ta.rsi(close, length=14)
    df["ATR"] = ta.atr(high=high, low=low, close=close, length=14)
    return df


# ---------------------------------------------------------------------------
# 回测：入场后 止损 = Entry - 2*ATR(入场日)；止盈 = 价格触及/超过布林带中轨
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, initial_balance: float = 10_000.0):
    warmup = 20
    cash = float(initial_balance)
    total_position = 0.0
    avg_entry_price = 0.0
    current_layer = 0  
    
    trades = []
    equity_values = []
    equity_dates = []

    close = df["Close"].astype(float)
    mid = df["BB_Mid"].astype(float)
    lower = df["BB_Lower"].astype(float)
    rsi = df["RSI"].astype(float)
    atr = df["ATR"].astype(float)

    for i in range(len(df)):
        date = df.index[i]
        price = float(close.iloc[i])
        
        # --- 1. 出场逻辑：只看均值回归（价格触及中轨） ---
        if total_position > 0:
            # 取消所有中间止损，坚定持有直到回归中轨
            if price >= mid.iloc[i]:
                cash = total_position * price
                trades.append({"type": "SELL", "reason": "take_profit", "price": price, "date": date})
                total_position, avg_entry_price, current_layer = 0.0, 0.0, 0
            
        # --- 2. 原始补仓逻辑 (RSI < 25) ---
        if i >= warmup:
            # 第一层：RSI 极度超卖 (20% 仓位)
            if current_layer == 0 and rsi.iloc[i] < 25.0:
                invest = initial_balance * 0.2
                total_position = invest / price
                avg_entry_price = price
                current_layer = 1
                cash -= invest
                trades.append({"type": "BUY", "layer": 1, "price": price, "date": date})

            # 第二层 & 第三层：越跌越补 (不设止损，只设补仓间距)
            elif 0 < current_layer < 3:
                # 只要价格比上次成交价又跌了 1.5 倍 ATR，就继续加码
                if price < (avg_entry_price * 0.92): # 简单暴力：跌 8% 就补
                    pct = 0.3 if current_layer == 1 else 0.5
                    invest = initial_balance * pct
                    new_pos = invest / price
                    avg_entry_price = (total_position * avg_entry_price + new_pos * price) / (total_position + new_pos)
                    total_position += new_pos
                    current_layer += 1
                    cash -= invest
                    trades.append({"type": "BUY", "layer": current_layer, "price": price, "date": date})

        eq = cash + (total_position * price)
        equity_dates.append(date)
        equity_values.append(eq)

    final_equity = cash + (total_position * close.iloc[-1])
    equity_series = pd.Series(equity_values, index=pd.Index(equity_dates))
    return final_equity, trades, equity_series


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（比例，例如 -0.25 表示 -25%）。"""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min())


def print_performance(
    initial: float,
    final: float,
    trades: list[dict],
    equity: pd.Series,
) -> None:
    total_return_pct = (final / initial - 1.0) * 100.0
    sells = [t for t in trades if t.get("type") == "SELL"]
    n_trades = len(sells)
    mdd = max_drawdown(equity)

    print("--- Strategy_v2_Pro 回测结果 ---")
    print(f"初始资金:     ${initial:,.2f}")
    print(f"期末权益:     ${final:,.2f}")
    print(f"总收益率:     {total_return_pct:.2f}%")
    print(f"交易次数:     {n_trades}（按完整卖出笔数计）")
    print(f"最大回撤:     {mdd * 100:.2f}%")
    print("--------------------------------")


def plot_trades(
    df: pd.DataFrame,
    trades: list[dict],
    *,
    title: str = "Strategy_v2_Pro — 价格 / 布林带 / 买卖点",
    save_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    """
    上图：收盘价 + 布林带 + 买入(绿▲) / 卖出止盈(橙▼) / 卖出止损(红▼)
    下图：RSI + 30 参考线
    默认保存到脚本同目录 strategy_v2_pro_trades.png
    """
    base = Path(__file__).resolve().parent
    out = Path(save_path) if save_path is not None else base / "strategy_v2_pro_trades.png"

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    ax1.plot(df.index, df["Close"], label="Close", color="black", linewidth=1.0)
    ax1.plot(df.index, df["BB_Upper"], label="BB Upper", color="#888888", linewidth=0.9, alpha=0.85)
    ax1.plot(df.index, df["BB_Mid"], label="BB Mid", color="#1f77b4", linewidth=0.9, alpha=0.9)
    ax1.plot(df.index, df["BB_Lower"], label="BB Lower", color="#888888", linewidth=0.9, alpha=0.85)

    for t in trades:
        if t.get("type") != "BUY":
            continue
        ax1.scatter(
            t["date"],
            t["price"],
            marker="^",
            s=130,
            color="#2ca02c",
            zorder=5,
            edgecolors="#0d4f0d",
            linewidths=0.6,
        )

    for t in trades:
        if t.get("type") != "SELL":
            continue
        reason = t.get("reason")
        color = "#d62728" if reason == "stop" else "#ff7f0e"
        ec = "#7a0e0e" if reason == "stop" else "#b35900"
        ax1.scatter(
            t["date"],
            t["price"],
            marker="v",
            s=130,
            color=color,
            zorder=5,
            edgecolors=ec,
            linewidths=0.6,
        )

    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax1.set_title(title)

    ax2.plot(df.index, df["RSI"], color="#9467bd", linewidth=1.0, label="RSI")
    ax2.axhline(30.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.75)
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", fontsize=8)

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    fig.align_ylabels([ax1, ax2])

    fig.savefig(out, dpi=150)
    print(f"图表已保存: {out}")
    if not show:
        plt.close(fig)
    else:
        _be = str(matplotlib.get_backend()).lower()
        if "agg" in _be or "template" in _be:
            plt.close(fig)
        else:
            plt.show()
    return out


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    data = load_data("BTC-USD", "1y")
    final_balance, trade_log, equity_curve = run_backtest(data, initial_balance=10_000.0)
    print_performance(10_000.0, final_balance, trade_log, equity_curve)
    plot_trades(data, trade_log)
