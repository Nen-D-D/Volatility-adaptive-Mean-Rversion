
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import datetime

# --- 1. 网页配置 ---
st.set_page_config(page_title="V8.0 Sentiment Sentinel", layout="wide")
st.title("🏴‍☠️ 贪婪猎人 V8.0：真实情绪哨兵 (API 联动版)")

# --- 2. 真实情绪 API 抓取逻辑 ---
@st.cache_data(ttl=3600)
def get_historical_fng(limit=365):
    """抓取真实 Fear & Greed Index 历史数据"""
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}"
        response = requests.get(url).json()
        data = []
        for entry in response['data']:
            date = datetime.datetime.fromtimestamp(int(entry['timestamp'])).strftime('%Y-%m-%d')
            data.append({"Date": date, "FnG": int(entry['value'])})
        fng_df = pd.DataFrame(data)
        fng_df['Date'] = pd.to_datetime(fng_df['Date'])
        return fng_df.sort_values('Date').set_index('Date')
    except Exception as e:
        st.error(f"情绪数据抓取失败: {e}")
        return pd.DataFrame()

# --- 3. 动态调参逻辑 (基于真实 FnG) ---
def adjust_params_by_fng(base_rsi, base_gap, fng_value):
    active_rsi = base_rsi
    active_gap = base_gap
    
    # 逻辑：FnG < 25 (极度恐慌) -> 提高RSI门槛抢跑
    if fng_value < 25:
        active_rsi += 10
    # 逻辑：FnG > 75 (极度贪婪) -> 拉大间距防御
    elif fng_value > 75:
        active_gap *= 1.5
        
    active_rsi = min(active_rsi, 45) # 理性刹车限制
    return active_rsi, active_gap

# --- 4. 核心回测循环 ---
def run_backtest(df_bt, rsi_thresh, gap_pct, init_bal, use_ai):
    cash = float(init_bal)
    total_pos, avg_cost, layer = 0.0, 0.0, 0
    invest_plan = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25] 
    equity, trade_log = [], []
    
    # 提取数组加速运算
    closes = df_bt["Close"].values
    rsis = df_bt["RSI"].values
    mids = df_bt["BB_Mid"].values
    lowers = df_bt["BB_Lower"].values
    fng_values = df_bt["FnG"].values if "FnG" in df_bt.columns else [50]*len(df_bt)
    times = df_bt.index
    
    for i in range(len(df_bt)):
        if use_ai:
            active_rsi, active_gap = adjust_params_by_fng(rsi_thresh, gap_pct, fng_values[i])
        else:
            active_rsi, active_gap = rsi_thresh, gap_pct
            
        p, r, m, l = float(closes[i]), float(rsis[i]), float(mids[i]), float(lowers[i])
        
        # 止盈: 价格破中轨
        if total_pos > 0 and p >= m:
            cash += total_pos * p
            trade_log.append({"date": times[i], "type": "SELL", "price": p})
            total_pos, avg_cost, layer = 0.0, 0.0, 0
        
        # 入场: 破下轨且RSI达标
        elif layer == 0 and p < l and r < active_rsi:
            inv = init_bal * invest_plan[0]
            total_pos = inv / p
            avg_cost, layer, cash = p, 1, cash - inv
            trade_log.append({"date": times[i], "type": "BUY", "price": p, "layer": 1})
            
        # 补仓: 价格跌破成本间距
        elif 0 < layer < len(invest_plan) and p < avg_cost * (1 - active_gap):
            inv = init_bal * invest_plan[layer]
            new_p_count = inv / p
            avg_cost = (total_pos * avg_cost + new_p_count * p) / (total_pos + new_p_count)
            total_pos += new_p_count
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

# --- 5. 侧边栏与数据流 ---
st.sidebar.header("🕹️ 控制中心")
symbol = st.sidebar.text_input("标的", value="ETH-USD")
rsi_in = st.sidebar.slider("基础 RSI 阈值", 10, 45, 30)
gap_in = st.sidebar.slider("基础补仓间距 %", 1.0, 15.0, 5.0) / 100
bal_in = st.sidebar.number_input("起始资金", value=10000)
ai_on = st.sidebar.toggle("启用真实 FnG 情绪联动", value=True)
crash_on = st.sidebar.checkbox("开启 -40% 极端压测")

@st.cache_data
def load_market_data(s):
    d = yf.download(s, period="1y", interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    return d

# 加载价格 + 情绪
df_price = load_market_data(symbol).copy()
df_fng = get_historical_fng(limit=400)

# 数据对齐与压测注入
if crash_on:
    df_price.iloc[-20:, df_price.columns.get_loc('Close')] = df_price.iloc[-20:]['Close'] * 0.6

# 统一预计算
df_price["RSI"] = ta.rsi(df_price["Close"], length=14)
bb = ta.bbands(df_price["Close"], length=20, std=1.5)
df_price["BB_Lower"], df_price["BB_Mid"] = bb.iloc[:, 0], bb.iloc[:, 1]

# 核心对齐: 确保价格每一行都有情绪分
full_df = df_price.join(df_fng, how='left').fillna(method='ffill').dropna()

# --- 6. 运行与展示 ---
tr, md, sh, eq_ser, t_log = run_backtest(full_df, rsi_in, gap_in, bal_in, ai_on)

c1, c2, c3, c4 = st.columns(4)
c1.metric("最终资产", f"${eq_ser.iloc[-1]:,.2f}")
c2.metric("累计收益", f"{tr:.2f}%")
c3.metric("夏普比率", f"{sh:.2f}")
c4.metric("最大回撤", f"{md:.2f}%")

# Plotly 大图逻辑
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.3])

# 主图: 价格与买卖点
fig.add_trace(go.Scatter(x=full_df.index, y=full_df["Close"], name="价格", line=dict(color='black', width=1)), row=1, col=1)
buys = [t for t in t_log if t["type"] == "BUY"]
sells = [t for t in t_log if t["type"] == "SELL"]
if buys:
    fig.add_trace(go.Scatter(x=[b["date"] for b in buys], y=[b["price"] for b in buys], mode="markers", name="买入/补仓", marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
if sells:
    fig.add_trace(go.Scatter(x=[s["date"] for s in sells], y=[s["price"] for s in sells], mode="markers", name="止盈离场", marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

# 副图1: 情绪分 (FnG)
fig.add_trace(go.Scatter(x=full_df.index, y=full_df["FnG"], name="恐慌指数", line=dict(color='orange')), row=2, col=1)
fig.add_hline(y=25, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=75, line_dash="dash", line_color="green", row=2, col=1)

# 副图2: 账户权益
fig.add_trace(go.Scatter(x=eq_ser.index, y=eq_ser, name="账户权益", fill='tozeroy', line=dict(color='royalblue')), row=3, col=1)

fig.update_layout(height=800, template="plotly_white", hovermode="x unified", title_text="V8.0 实战级情绪哨兵看板")
st.plotly_chart(fig, use_container_width=True)

# --- 7. 自动化寻优表格 ---
st.sidebar.markdown("---")
if st.sidebar.button("🚀 开启全域寻优 (基于真数据)"):
    st.write("### 🔍 寻优战报 (真实情绪联动)")
    results = []
    for r_opt in range(20, 46, 5):
        for g_opt in np.arange(0.03, 0.11, 0.01):
            t_r, m_d, s_h, _, _ = run_backtest(full_df, r_opt, g_opt, bal_in, ai_on)

            results.append({"RSI": r_opt, "间距%": round(g_opt*100, 1), "收益%": round(t_r, 2), "MDD%": round(m_d, 2), "夏普": round(s_h, 2)})
    st.dataframe(pd.DataFrame(results).sort_values("夏普", ascending=False), use_container_width=True)