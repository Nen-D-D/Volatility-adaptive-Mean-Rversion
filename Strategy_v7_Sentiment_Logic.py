import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# --- 1. 网页配置 ---
st.set_page_config(page_title="V7.2 Pro Dashboard", layout="wide")
st.title("🏴‍☠️ 贪婪猎人 V7.2：全功能逻辑加固版")

# --- 2. AI 参谋部逻辑 (锁定时间戳哈希，杜绝实验漂移) ---
def get_market_sentiment(timestamp):
    """根据日期生成唯一确定的情绪分，确保回测可复现"""
    random.seed(hash(timestamp))
    return round(random.uniform(-1.0, 1.0), 2)

def adjust_params_by_sentiment(base_rsi, base_gap, sentiment_score):
    active_rsi = base_rsi
    active_gap = base_gap
    if sentiment_score < -0.5:
        active_rsi += 10        
    elif sentiment_score > 0.5:
        active_gap *= 1.5       
    active_rsi = min(active_rsi, 45) # 理性刹车
    return active_rsi, active_gap

# --- 3. 核心回测逻辑函数 ---
def run_backtest(df_input, rsi_thresh, gap_pct, init_bal, use_ai):
    cash = float(init_bal)
    total_pos, avg_cost, layer = 0.0, 0.0, 0
    invest_plan = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25] 
    equity, trade_log = [], []
    
    # 使用 numpy 数组加速逻辑运算
    closes = df_input["Close"].values
    rsis = df_input["RSI"].values
    mids = df_input["BB_Mid"].values
    lowers = df_input["BB_Lower"].values
    times = df_input.index
    
    for i in range(len(df_input)):
        if use_ai:
            sentiment = get_market_sentiment(times[i])
            active_rsi, active_gap = adjust_params_by_sentiment(rsi_thresh, gap_pct, sentiment)
        else:
            active_rsi, active_gap = rsi_thresh, gap_pct
            
        p, r = float(closes[i]), float(rsis[i])
        m, l = float(mids[i]), float(lowers[i])
        
        # 止盈逻辑
        if total_pos > 0 and p >= m:
            cash += total_pos * p
            trade_log.append({"date": times[i], "type": "SELL", "price": p})
            total_pos, avg_cost, layer = 0.0, 0.0, 0
        # 入场逻辑
        elif layer == 0 and p < l and r < active_rsi:
            inv = init_bal * invest_plan[0]
            total_pos = inv / p
            avg_cost, layer, cash = p, 1, cash - inv
            trade_log.append({"date": times[i], "type": "BUY", "price": p, "layer": 1})
        # 补仓逻辑
        elif 0 < layer < len(invest_plan) and p < avg_cost * (1 - active_gap):
            inv = init_bal * invest_plan[layer]
            new_pos = inv / p
            avg_cost = (total_pos * avg_cost + new_pos * p) / (total_pos + new_pos)
            total_pos += new_pos
            cash -= inv
            layer += 1
            trade_log.append({"date": times[i], "type": "BUY", "price": p, "layer": layer})
        
        equity.append(cash + total_pos * p)
    
    eq_ser = pd.Series(equity, index=times)
    total_ret = (eq_ser.iloc[-1] / init_bal - 1) * 100
    mdd = ((eq_ser - eq_ser.cummax()) / eq_ser.cummax()).min() * 100
    rets = eq_ser.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(365)) if rets.std() != 0 else 0
    return total_ret, mdd, sharpe, eq_ser, trade_log

# --- 4. 侧边栏与数据流 ---
st.sidebar.header("🕹️ 控制中心")
symbol = st.sidebar.text_input("标的", value="ETH-USD")
rsi_in = st.sidebar.slider("基础RSI阈值", 10, 45, 25)
gap_in = st.sidebar.slider("基础补仓间距%", 1.0, 15.0, 5.0) / 100
bal_in = st.sidebar.number_input("起始资金", value=10000)
ai_on = st.sidebar.toggle("启用 AI 情绪联动", value=True)
crash_on = st.sidebar.checkbox("开启 -40% 极端压测")

@st.cache_data
def load_data(s):
    d = yf.download(s, period="1y", interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    return d

raw_df = load_data(symbol).copy()
if crash_on:
    raw_df.iloc[-20:, raw_df.columns.get_loc('Close')] = raw_df.iloc[-20:]['Close'] * 0.6

# 统一预计算
raw_df["RSI"] = ta.rsi(raw_df["Close"], length=14)
bb_data = ta.bbands(raw_df["Close"], length=20, std=1.5)
raw_df["BB_Lower"], raw_df["BB_Mid"] = bb_data.iloc[:, 0], bb_data.iloc[:, 1]
clean_df = raw_df.dropna().copy()

# --- 5. 执行主回测 ---
tr, md, sh, eq_ser, t_log = run_backtest(clean_df, rsi_in, gap_in, bal_in, ai_on)

# 指标展示
c1, c2, c3, c4 = st.columns(4)
c1.metric("最终资产", f"${eq_ser.iloc[-1]:,.2f}")
c2.metric("累计收益", f"{tr:.2f}%")
c3.metric("夏普比率", f"{sh:.2f}")
c4.metric("最大回撤", f"{md:.2f}%")

# --- 6. 豪华 Plotly 雷达图 (买卖点回归) ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

# 价格与买卖点
fig.add_trace(go.Scatter(x=clean_df.index, y=clean_df["Close"], name="价格", line=dict(color='black', width=1)), row=1, col=1)
buys = [t for t in t_log if t["type"] == "BUY"]
sells = [t for t in t_log if t["type"] == "SELL"]

if buys:
    fig.add_trace(go.Scatter(x=[b["date"] for b in buys], y=[b["price"] for b in buys], mode="markers", name="买入/补仓", marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
if sells:
    fig.add_trace(go.Scatter(x=[s["date"] for s in sells], y=[s["price"] for s in sells], mode="markers", name="止盈离场", marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

# 账户权益
fig.add_trace(go.Scatter(x=eq_ser.index, y=eq_ser, name="账户权益", fill='tozeroy', line=dict(color='royalblue')), row=2, col=1)

fig.update_layout(height=700, template="plotly_white", hovermode="x unified", title_text="V7.2 工业级战术看板")
st.plotly_chart(fig, use_container_width=True)

# --- 7. 寻优战报表格 ---
if st.sidebar.button("🚀 开启全域网格寻优"):
    st.write("### 🔍 寻优结果 (逻辑加固版)")
    results = []
    for r_opt in range(20, 46, 5):
        for g_opt in np.arange(0.03, 0.11, 0.01):
            t_r, m_d, s_h, _, _ = run_backtest(clean_df, r_opt, g_opt, bal_in, ai_on)
            results.append({"RSI": r_opt, "间距%": round(g_opt*100, 1), "收益率%": round(t_r, 2), "MDD%": round(m_d, 2), "夏普": round(s_h, 2)})
    st.dataframe(pd.DataFrame(results).sort_values("夏普", ascending=False), use_container_width=True)
