import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 网页配置 ---
st.set_page_config(page_title="V5 Optimizer Pro + Stress Test", layout="wide")
st.title("🏴‍☠️ 贪婪猎人 V5：压力测试 & 全域寻优实验室")

# --- 2. 核心回测逻辑函数 ---
def run_backtest(df, rsi_thresh, gap_pct, init_bal=10000.0):
    cash = init_bal
    total_pos = 0.0
    avg_cost = 0.0
    layer = 0
    invest_plan = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25] 
    equity = []
    trade_log = []
    
    for i in range(len(df)):
        price = float(df["Close"].iloc[i])
        rsi = float(df["RSI"].iloc[i])
        mid = float(df["BB_Mid"].iloc[i])
        lower = float(df["BB_Lower"].iloc[i])
        date = df.index[i]
        
        if total_pos > 0 and price >= mid:
            cash += total_pos * price
            trade_log.append({"date": date, "type": "SELL", "price": price})
            total_pos, avg_cost, layer = 0, 0, 0
        elif layer == 0 and price < lower and rsi < rsi_thresh:
            invest = init_bal * invest_plan[0]
            total_pos = invest / price
            avg_cost, layer, cash = price, 1, cash - invest
            trade_log.append({"date": date, "type": "BUY", "price": price, "layer": 1})
        elif 0 < layer < len(invest_plan) and price < avg_cost * (1 - gap_pct):
            invest = init_bal * invest_plan[layer]
            new_pos = invest / price
            avg_cost = (total_pos * avg_cost + new_pos * price) / (total_pos + new_pos)
            total_pos += new_pos
            cash -= invest
            layer += 1
            trade_log.append({"date": date, "type": "BUY", "price": price, "layer": layer})
        equity.append(cash + total_pos * price)
    
    eq_ser = pd.Series(equity, index=df.index)
    rets = eq_ser.pct_change().dropna()
    total_ret = (eq_ser.iloc[-1] / init_bal - 1) * 100
    mdd = ((eq_ser - eq_ser.cummax()) / eq_ser.cummax()).min() * 100
    sharpe = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() != 0 else 0
    return total_ret, mdd, sharpe, eq_ser, trade_log

# --- 3. 侧边栏参数 ---
st.sidebar.header("🕹️ 手动调优控制台")
symbol = st.sidebar.text_input("标的", value="ETH-USD")
rsi_input = st.sidebar.slider("RSI 阈值", 10, 45, 25)
gap_input = st.sidebar.slider("补仓间距 %", 1.0, 15.0, 5.0) / 100
initial_bal = st.sidebar.number_input("初始资金", value=10000)

st.sidebar.markdown("---")
stress_test = st.sidebar.checkbox("🚨 开启 -40% 极端压力测试")

# --- 4. 加载数据 ---
@st.cache_data
def load_data(s):
    df_raw = yf.download(s, period="1y", interval="1d", progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
    return df_raw

# 原始数据
df_raw = load_data(symbol)
df = df_raw.copy()

# 【核心逻辑】：在计算指标前注入黑天鹅
if stress_test:
    st.warning("⚠️ 正在模拟末端暴跌 40% 的黑天鹅行情...")
    # 模拟最后 20 天突然暴跌
    df.iloc[-20:, df.columns.get_loc('Close')] = df.iloc[-20:]['Close'] * 0.6

# 重新计算指标
df["RSI"] = ta.rsi(df["Close"], length=14)
bb = ta.bbands(df["Close"], length=20, std=1.5)
df["BB_Lower"], df["BB_Mid"] = bb.iloc[:, 0], bb.iloc[:, 1]
df = df.dropna()

# --- 5. 执行主回测并展示 ---
tr, md, sh, equity_series, t_log = run_backtest(df, rsi_input, gap_input, initial_bal)

c1, c2, c3, c4 = st.columns(4)
c1.metric("最终资产", f"${equity_series.iloc[-1]:,.2f}")
c2.metric("累计收益率", f"{tr:.2f}%")
c3.metric("夏普比率", f"{sh:.2f}")
c4.metric("最大回撤", f"{md:.2f}%")

# --- 6. 绘图 ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="价格 (包含压测)", line=dict(color='black', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["BB_Mid"], name="中轨止盈", line=dict(dash='dash', color='blue')), row=1, col=1)

buys = [t for t in t_log if t["type"] == "BUY"]
sells = [t for t in t_log if t["type"] == "SELL"]
if buys:
    fig.add_trace(go.Scatter(x=[b["date"] for b in buys], y=[b["price"] for b in buys], mode="markers", name="买入/补仓", marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
if sells:
    fig.add_trace(go.Scatter(x=[s["date"] for s in sells], y=[s["price"] for s in sells], mode="markers", name="止盈离场", marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series, name="账户权益", fill='tozeroy', line=dict(color='green')), row=2, col=1)
fig.update_layout(height=600, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- 7. 自动化寻优模块 ---
st.sidebar.markdown("---")
if st.sidebar.button("🚀 开启全域网格寻优"):
    st.write("### 🔍 正在基于当前行情(含压测)寻优...")
    results = []
    rsi_space = range(20, 46, 5)
    gap_space = np.arange(0.03, 0.11, 0.01)
    
    prog = st.progress(0)
    steps = len(rsi_space) * len(gap_space)
    curr = 0

    for r in rsi_space:
        for g in gap_space:
            tr_opt, md_opt, sh_opt, _, _ = run_backtest(df, r, g, initial_bal) 
            results.append({"RSI": r, "间距%": round(g*100, 1), "收益率%": round(tr_opt, 2), "MDD%": round(md_opt, 2), "夏普": round(sh_opt, 2)})
            curr += 1
            prog.progress(curr/steps)

    res_df = pd.DataFrame(results)
    st.dataframe(res_df.sort_values("夏普", ascending=False).head(10), use_container_width=True)
    