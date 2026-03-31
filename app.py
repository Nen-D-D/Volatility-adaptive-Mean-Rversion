import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from Strategy_v2_Pro import load_data, max_drawdown, run_backtest


def _trades_to_df(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["date", "type", "reason", "price", "stop_loss", "entry_atr"])
    df = pd.DataFrame(trades).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ("price", "stop_loss", "entry_atr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df


def _build_price_figure(df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=1.8)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], name="BB Mid", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(width=1, dash="dot")))

    if not trades_df.empty:
        buys = trades_df[trades_df["type"] == "BUY"].copy()
        sells = trades_df[trades_df["type"] == "SELL"].copy()

        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(buys["date"]),
                    y=buys["price"],
                    mode="markers",
                    name="BUY",
                    marker=dict(symbol="triangle-up", size=10),
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(sells["date"]),
                    y=sells["price"],
                    mode="markers",
                    name="SELL",
                    marker=dict(symbol="triangle-down", size=10),
                )
            )

    fig.update_layout(
        title="BTC-USD Price + Bollinger Bands + Trades",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series",
        height=520,
    )
    return fig


def _build_equity_figure(equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(width=2)))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity", height=360)
    return fig


def main() -> None:
    st.set_page_config(page_title="Strategy v2 Pro Backtest", layout="wide")
    st.title("Strategy_v2_Pro 可视化回测")
    st.caption("Bollinger Bands + RSI 均值回归，ATR 动态止损（止损固定为入场当日 ATR）")

    with st.sidebar:
        st.header("参数")
        symbol = st.text_input("交易标的", value="BTC-USD")
        period = st.selectbox("回测区间", options=["1y", "2y", "5y"], index=0)
        initial_balance = st.number_input("初始资金", min_value=100.0, value=10000.0, step=100.0)
        run = st.button("运行回测", type="primary")

    if not run:
        st.info("在左侧调整参数后点击“运行回测”。")
        return

    with st.spinner("下载数据并运行策略中..."):
        data = load_data(symbol=symbol, period=period)
        final_equity, trades, equity_curve = run_backtest(data, initial_balance=initial_balance)

    total_return_pct = (final_equity / initial_balance - 1.0) * 100.0
    trade_count = sum(1 for t in trades if t.get("type") == "SELL")
    mdd_pct = max_drawdown(equity_curve) * 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric("最终权益", f"${final_equity:,.2f}")
    c2.metric("总收益率", f"{total_return_pct:.2f}%")
    c3.metric("最大回撤", f"{mdd_pct:.2f}%")
    st.write(f"交易次数（按卖出笔数计）: **{trade_count}**")

    trades_df = _trades_to_df(trades)
    fig_price = _build_price_figure(data, trades_df)
    fig_equity = _build_equity_figure(equity_curve)

    st.plotly_chart(fig_price, width="stretch")
    st.plotly_chart(fig_equity, width="stretch")

    with st.expander("交易明细"):
        st.dataframe(trades_df, width="stretch")

    with st.expander("指标数据（最近 20 行）"):
        view_cols = ["Close", "RSI", "ATR", "BB_Lower", "BB_Mid", "BB_Upper"]
        st.dataframe(data[view_cols].tail(20), width="stretch")


if __name__ == "__main__":
    main()
