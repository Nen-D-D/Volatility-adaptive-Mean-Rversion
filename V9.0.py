import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import datetime

# --- 1. 配置与 API ---
st.set_page_config(page_title="V9.1 Matrix Commander", layout="wide")
st.title("🏴‍☠️ 贪婪猎人 V9.1：矩阵指挥部 (逻辑全加固版)")

@st.cache_data(ttl=3600)
def get_historical_fng(limit=450):
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}"
        res = requests.get(url).json()
        data = [{"Date": datetime.datetime.fromtimestamp(int(e['timestamp'])).strftime('%Y-%m-%d'), "FnG": int(e['value'])} for e in res['data']]
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index('Date').sort_index()
    except: return pd.DataFrame()

@st.cache_data
def load_matrix_data(symbols):
    all_dfs = {}
    for s in symbols:
        d = yf.download(s, period="1y", interval="1d", progress=False)
        if d.empty: continue
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
        d["RSI"] = ta.rsi(d["Close"], length=14)
        bb = ta.bbands(d["Close"], length=20, std=1.5)
        d["BB_Lower"], d["BB_Mid"] = bb.iloc[:, 0], bb.iloc[:, 1]
        all_dfs[s] = d.dropna()
    return all_dfs

# --- 2. 侧边栏 ---
st.sidebar.header("🕹️ 指挥部参数")
symbols = st.sidebar.multiselect("狩猎名单", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"], default=["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"])
rsi_base = st.sidebar.slider("基础 RSI", 10, 45, 30)
gap_base = st.sidebar.slider("补仓间距 %", 1.0, 15.0, 6.0) / 100
total_bal = st.sidebar.number_input("起始子弹库 (USD)", value=10000)
ai_on = st.sidebar.toggle("启用情绪联动", value=True)
crash_on = st.sidebar.checkbox("开启末端 -40% 压测")

# --- 3. 核心加固回测引擎 ---
def run_hardened_matrix(multi_dfs, fng_df, rsi_in, gap_in, init_bal, use_ai, is_crash):
    # A. 确定共同时间轴
    if not multi_dfs: return pd.DataFrame()
    common_index = sorted(list(set.intersection(*[set(df.index) for df in multi_dfs.values()])))
    
    cash = float(init_bal)
    # 状态追踪
    states = {s: {"pos": 0.0, "cost": 0.0, "layer": 0} for s in multi_dfs}
    invest_plan = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25] # 基于总本金的比例
    history = []

    for date in common_index:
        fng_val = fng_df.loc[date, 'FnG'] if date in fng_df.index else 50
        
        # 压测逻辑
        is_last_day = (date == common_index[-1] and is_crash)
        
        # 1. 止盈阶段 (释放资金)
        for s in multi_dfs:
            if states[s]["pos"] > 0:
                p = multi_dfs[s].loc[date, "Close"] * (0.6 if is_last_day else 1.0)
                mid = multi_dfs[s].loc[date, "BB_Mid"]
                if p >= mid:
                    cash += states[s]["pos"] * p
                    states[s] = {"pos": 0.0, "cost": 0.0, "layer": 0}

        # 2. 策略执行 (按RSI排序优先最低的)
        sorted_s = sorted(multi_dfs.keys(), key=lambda x: multi_dfs[x].loc[date, "RSI"])
        
        for s in sorted_s:
            row = multi_dfs[s].loc[date]
            p = row["Close"] * (0.6 if is_last_day else 1.0)
            rsi = row["RSI"]
            lower = row["BB_Lower"]
            
            # AI 参数调整
            act_rsi = min(rsi_in + (10 if (use_ai and fng_val < 25) else 0), 45)
            act_gap = gap_in * (1.5 if (use_ai and fng_val > 75) else 1.0)

            # 开仓
            if states[s]["layer"] == 0 and p < lower and rsi < act_rsi:
                inv = init_bal * invest_plan[0]
                if cash >= inv:
                    states[s] = {"pos": inv/p, "cost": p, "layer": 1}
                    cash -= inv
            # 补仓
            elif 0 < states[s]["layer"] < len(invest_plan) and p < states[s]["cost"] * (1 - act_gap):
                inv = init_bal * invest_plan[states[s]["layer"]]
                if cash >= inv:
                    new_pos = inv/p
                    states[s]["cost"] = (states[s]["pos"] * states[s]["cost"] + inv) / (states[s]["pos"] + new_pos)
                    states[s]["pos"] += new_pos
                    states[s]["layer"] += 1
                    cash -= inv

        # 每日结算
        market_val = sum(states[s]["pos"] * (multi_dfs[s].loc[date, "Close"] * (0.6 if is_last_day else 1.0)) for s in multi_dfs)
        history.append({"Date": date, "Equity": cash + market_val, "FnG": fng_val, "Cash": cash})

    return pd.DataFrame(history).set_index("Date")

# --- 4. 结果展示 ---
if symbols:
    fng_h = get_historical_fng()
    data_map = load_matrix_data(symbols)
    
    if data_map:
        res = run_hardened_matrix(data_map, fng_h, rsi_base, gap_base, total_bal, ai_on, crash_on)
        
        if not res.empty:
            final_val = res["Equity"].iloc[-1]
            tr = (final_val / total_bal - 1) * 100
            mdd = ((res["Equity"] - res["Equity"].cummax()) / res["Equity"].cummax()).min() * 100
            sharpe = (res["Equity"].pct_change().mean() / res["Equity"].pct_change().std() * np.sqrt(365)) if res["Equity"].pct_change().std() != 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最终资产", f"${final_val:,.2f}")
            c2.metric("收益率", f"{tr:.2f}%")
            c3.metric("夏普比率", f"{sharpe:.2f}")
            c4.metric("最大回撤", f"{mdd:.2f}%")

            # 综合看板
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=res.index, y=res["Equity"], name="总权益(USD)", fill='tozeroy', line=dict(color='gold')), row=1, col=1)
            fig.add_trace(go.Scatter(x=res.index, y=res["FnG"], name="恐惧贪婪指数", line=dict(color='orange')), row=2, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="red", row=2, col=1)
            fig.update_layout(height=700, template="plotly_dark", title=f"V9.1 矩阵实战看板 ({', '.join(symbols)})")
            st.plotly_chart(fig, use_container_width=True)

            # 寻优逻辑
            if st.sidebar.button("🚀 矩阵参数寻优"):
                st.write("### 🔍 寻优报告")
                opt_data = []
                for r_opt in [25, 30, 35]:
                    for g_opt in [0.04, 0.06, 0.08]:
                        rdf = run_hardened_matrix(data_map, fng_h, r_opt, g_opt, total_bal, ai_on, False)
                        if not rdf.empty:
                            s_h = (rdf["Equity"].pct_change().mean() / rdf["Equity"].pct_change().std() * np.sqrt(365))
                            ret = (rdf["Equity"].iloc[-1] / total_bal - 1) * 100
                            opt_data.append({"RSI": r_opt, "间距%": g_opt*100, "收益%": round(ret,2), "夏普": round(s_h, 2)})
                st.table(pd.DataFrame(opt_data).sort_values("夏普", ascending=False))
