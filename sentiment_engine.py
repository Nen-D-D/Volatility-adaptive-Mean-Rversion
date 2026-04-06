import streamlit as st
import random

# --- V7 核心：AI 情绪评分模拟 ---
def get_market_sentiment():
    # 模拟从 Twitter 抓取的关键词
    scary_words = ["crash", "dead", "scam", "selling", "liquidated", "bear"]
    greedy_words = ["moon", "buy", "bull", "lfg", "ath", "pump"]
    
    # 随机生成模拟推文流
    sample_size = 10
    score = 0
    for _ in range(sample_size):
        if random.random() > 0.7: # 模拟恐慌爆发
            score -= random.uniform(0.1, 0.5)
        else:
            score += random.uniform(0.1, 0.3)
    
    return round(max(-1.0, min(1.0, score)), 2)

st.title("🏹 V7 预研：社交媒体情绪嗅探器")
current_sentiment = get_market_sentiment()

if current_sentiment < -0.3:
    st.error(f"😱 当前情绪：{current_sentiment} (市场极度恐慌！)")
    st.info("💡 系统建议：调高 RSI 阈值，准备提前抄底。")
elif current_sentiment > 0.3:
    st.success(f"🤑 当前情绪：{current_sentiment} (市场过热！)")
    st.info("💡 系统建议：收紧止盈线，防止回调。")
else:
    st.warning(f"😐 当前情绪：{current_sentiment} (情绪稳定)")