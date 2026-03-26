import time
import argparse
from datetime import datetime

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def compute_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    """Simple RSI based on rolling mean of gains/losses."""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


assets = ["BTC-USD", "ETH-USD", "SOL-USD", "NVDA", "TSLA", "AAPL"]


def bollinger_bands(
    close_series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (mid, upper, lower) Bollinger bands."""
    ma = close_series.rolling(window=window).mean()
    std = close_series.rolling(window=window).std()
    mid = ma
    upper = ma + num_std * std
    lower = ma - num_std * std
    return mid, upper, lower


def backtest_hunting_mode_single_asset(
    close_series: pd.Series,
    *,
    initial_capital: float = 10000.0,
    bb_window: int = 20,
    bb_std: float = 2.0,
    rsi_window: int = 14,
    rsi_threshold: float = 25.0,
    tranches: tuple[float, float, float] = (0.1, 0.2, 0.4),
) -> dict:
    """
    猎杀模式回测（单资产）：
    - 入场/补仓：Price < 布林带下轨 且 RSI < 25
    - 补仓倍数：10% / 20% / 40%（最多 3 次），钱不够就按剩余现金买
    - 出场：Price 回到 布林带中轨（>= mid），立刻清仓并重置
    - Max Drawdown：基于权益曲线的最大回撤（最惨浮亏）
    """
    s = close_series.dropna().copy()
    if s.empty:
        raise ValueError("close_series is empty after dropna")

    rsi = compute_rsi(s, window=rsi_window)
    boll_mid, _, boll_lower = bollinger_bands(s, window=bb_window, num_std=bb_std)

    # 避免前视偏差：用上一根K线的指标来决定今天是否交易。
    rsi_prev = rsi.shift(1)
    mid_prev = boll_mid.shift(1)
    lower_prev = boll_lower.shift(1)

    cash = float(initial_capital)
    shares = 0.0
    tranche_idx = 0  # 0 -> 10%, 1 -> 20%, 2 -> 40%

    peak_equity = float(initial_capital)
    max_drawdown = 0.0  # <= 0（负数越小代表回撤越大）
    max_drawdown_amount = 0.0  # 峰值权益到当前权益的最大差额（货币单位）

    trades_exits = 0
    equity_curve = []

    for dt, price in s.items():
        if pd.isna(price):
            continue

        # 昨日指标（上一根K线算出来的值）用于今日交易决策。
        rsi_t = rsi_prev.loc[dt] if dt in rsi_prev.index else float("nan")
        mid_t = mid_prev.loc[dt] if dt in mid_prev.index else float("nan")
        lower_t = lower_prev.loc[dt] if dt in lower_prev.index else float("nan")

        # 1) 出场：回到中轨，清仓并重置补仓进度
        if shares > 0 and pd.notna(mid_t) and float(price) >= float(mid_t):
            cash += shares * float(price)
            shares = 0.0
            tranche_idx = 0
            trades_exits += 1

        # 2) 入场/补仓：恐惧条件满足且还有档位可用
        if tranche_idx < len(tranches) and cash > 0 and pd.notna(lower_t) and pd.notna(rsi_t):
            if float(price) < float(lower_t) and float(rsi_t) < float(rsi_threshold):
                invest_target = tranches[tranche_idx] * float(initial_capital)
                invest_amount = min(invest_target, cash)
                if invest_amount > 0:
                    shares += invest_amount / float(price)
                    cash -= invest_amount
                    tranche_idx += 1

        equity = cash + shares * float(price)
        equity_curve.append((dt, equity))

        # 用“当前bar之前的峰值”算回撤，更新最大回撤。
        if peak_equity > 0:
            dd = (equity - peak_equity) / peak_equity
            if dd < max_drawdown:
                max_drawdown = dd
            dd_amount = peak_equity - equity
            if dd_amount > max_drawdown_amount:
                max_drawdown_amount = dd_amount

        if equity > peak_equity:
            peak_equity = equity

    final_equity = cash + shares * float(s.iloc[-1])
    return_pct = (final_equity / initial_capital - 1.0) * 100.0
    return {
        "final_equity": float(final_equity),
        "return_pct": float(return_pct),
        "max_drawdown_pct": float(max_drawdown * 100.0),  # 负数
        "max_drawdown_amount": float(max_drawdown_amount),
        "trades_exits": int(trades_exits),
        "equity_points": int(len(equity_curve)),
    }


def scan_once() -> tuple[pd.DataFrame, list[str]]:
    """Scan all assets once, return (summary_df, triggered_assets)."""
    data = yf.download(assets, period="1y", interval="1d", auto_adjust=False, progress=False)

    if data is None or data.empty:
        raise RuntimeError("yfinance 返回空数据")

    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    close = close.dropna(how="all")

    results = []
    triggered = []

    for asset in assets:
        if asset not in close.columns:
            results.append(
                {
                    "资产名称": asset,
                    "当前价格": float("nan"),
                    "距离下轨空间": float("nan"),
                    "RSI值": float("nan"),
                    "乖离率(%)": float("nan"),
                    "当前诊断建议": "数据缺失",
                }
            )
            continue

        s = close[asset].dropna()
        if s.empty:
            results.append(
                {
                    "资产名称": asset,
                    "当前价格": float("nan"),
                    "距离下轨空间": float("nan"),
                    "RSI值": float("nan"),
                    "乖离率(%)": float("nan"),
                    "当前诊断建议": "数据缺失",
                }
            )
            continue

        rsi14 = compute_rsi(s, window=14)
        ma20 = s.rolling(window=20).mean()
        std20 = s.rolling(window=20).std()
        boll_mid = ma20
        boll_upper = ma20 + 2 * std20
        boll_lower = ma20 - 2 * std20

        latest_price = s.iloc[-1]
        latest_rsi = rsi14.iloc[-1]
        latest_lower = boll_lower.iloc[-1]
        latest_mid = boll_mid.iloc[-1]
        latest_upper = boll_upper.iloc[-1]

        space_to_lower = latest_price - latest_lower if pd.notna(latest_lower) else float("nan")
        deviation_pct = ((latest_price - latest_mid) / latest_mid * 100) if pd.notna(latest_mid) else float("nan")

        if pd.isna(latest_rsi) or pd.isna(latest_lower) or pd.isna(latest_mid):
            advice = "数据不足"
        elif latest_price < latest_lower and latest_rsi < 25:
            advice = "抄底"
            triggered.append(asset)
        elif latest_price > latest_upper or latest_rsi > 70:
            advice = "减仓"
        else:
            advice = "观望"

        results.append(
            {
                "资产名称": asset,
                "当前价格": float(latest_price) if pd.notna(latest_price) else float("nan"),
                "距离下轨空间": float(space_to_lower) if pd.notna(space_to_lower) else float("nan"),
                "RSI值": float(latest_rsi) if pd.notna(latest_rsi) else float("nan"),
                "乖离率(%)": float(deviation_pct) if pd.notna(deviation_pct) else float("nan"),
                "当前诊断建议": advice,
            }
        )

    return pd.DataFrame(results), triggered


def fmt2(x) -> str:
    return f"{x:.2f}" if pd.notna(x) else "NaN"


def main():
    big_alert = "！" * 10000
    sleep_seconds = 300

    # 启动时只画一次：资产相关性热力图（Past 1Y 日收益相关）
    try:
        data0 = yf.download(assets, period="1y", interval="1d", auto_adjust=False, progress=False)
        close0 = data0["Close"] if isinstance(data0.columns, pd.MultiIndex) else data0[["Close"]]
        close0 = close0.dropna(how="all")
        valid_close0 = close0[assets].dropna(how="any")
        if not valid_close0.empty:
            corr_mat0 = valid_close0.pct_change().dropna().corr()
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(corr_mat0.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_mat0.columns)))
            ax.set_xticklabels(corr_mat0.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr_mat0.index)))
            ax.set_yticklabels(corr_mat0.index)
            ax.set_title("Asset Return Correlation Heatmap (Past 1Y)")
            for i in range(corr_mat0.shape[0]):
                for j in range(corr_mat0.shape[1]):
                    ax.text(j, i, f"{corr_mat0.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"启动热力图绘制失败（可忽略）：{e}")

    while True:
        print(f"\n===== 实时雷达刷新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
        try:
            summary_df, triggered_assets = scan_once()
        except Exception as e:
            print(f"刷新失败：{e}")
            time.sleep(sleep_seconds)
            continue

        # 人性化交互（每次刷新必输出）
        print("皇上，前线暂无战事，可以继续接着奏乐接着舞。")

        # 输出乖离率提示（越负，反弹动力越大）
        if not summary_df.empty:
            summary_df = summary_df.sort_values("乖离率(%)")
            show_df = summary_df.copy()
            for col in ["当前价格", "距离下轨空间", "RSI值", "乖离率(%)"]:
                show_df[col] = show_df[col].map(fmt2)
            print("\n乖离率(%) 越负 => 反弹动力越大：")
            print(show_df[["资产名称", "当前价格", "乖离率(%)", "RSI值", "当前诊断建议"]].to_string(index=False))

        # 触发警报（只要任意一个资产满足条件，就弹出巨大警告）
        if len(triggered_assets) > 0:
            assets_str = ", ".join(triggered_assets)
            print("\n【触发警报】以下资产满足：Price < 布林带下轨 且 RSI < 25")
            print(f"触发资产：{assets_str}")
            print(big_alert)

        time.sleep(sleep_seconds)


def backtest_main(args: argparse.Namespace) -> None:
    data = yf.download(
        assets,
        period=args.period,
        interval=args.interval,
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        raise RuntimeError("yfinance 返回空数据")

    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    close = close.dropna(how="all")

    results = []
    for asset in assets:
        if asset not in close.columns:
            continue
        s = close[asset].dropna()
        if s.empty:
            continue

        r = backtest_hunting_mode_single_asset(
            s,
            initial_capital=args.capital,
            bb_window=args.bb_window,
            bb_std=args.bb_std,
            rsi_window=args.rsi_window,
            rsi_threshold=args.rsi_threshold,
            tranches=(0.1, 0.2, 0.4),
        )
        results.append({"asset": asset, **r})

    if not results:
        raise RuntimeError("没有任何资产可用于回测（数据不足？）")

    df = pd.DataFrame(results).sort_values("max_drawdown_pct")  # 最负的排前
    df["max_drawdown_pct"] = df["max_drawdown_pct"].map(lambda x: f"{x:.2f}%")
    df["return_pct"] = df["return_pct"].map(lambda x: f"{x:.2f}%")
    df["max_drawdown_amount"] = df["max_drawdown_amount"].map(lambda x: f"{x:,.2f}")
    df["final_equity"] = df["final_equity"].map(lambda x: f"{x:,.2f}")

    cols = ["asset", "final_equity", "return_pct", "max_drawdown_pct", "max_drawdown_amount", "trades_exits"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["scan", "backtest"], default="scan")
        parser.add_argument("--period", type=str, default="3y")
        parser.add_argument("--interval", type=str, default="1d")
        parser.add_argument("--capital", type=float, default=10000.0)
        parser.add_argument("--bb-window", type=int, default=20)
        parser.add_argument("--bb-std", type=float, default=2.0)
        parser.add_argument("--rsi-window", type=int, default=14)
        parser.add_argument("--rsi-threshold", type=float, default=25.0)
        args = parser.parse_args()

        if args.mode == "scan":
            main()
        else:
            backtest_main(args)
    except KeyboardInterrupt:
        print("\n已手动停止。")
