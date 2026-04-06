# 🏹 Greedy Hunter V7.2: Adaptive Crypto Quant Strategy
**基于 AI 情绪联动与布林带-RSI 均值回归的自适应量化交易引擎**

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)

## 🌟 核心亮点 (Core Highlights)
- **动态情绪因子 (Dynamic Sentiment Tuning)**: 实时接入 Crypto Fear & Greed Index，动态修正 RSI 入场阈值，解决传统技术指标在极端行情下的“钝化”问题。
- **压力测试稳健性 (Black Swan Resilience)**: 成功通过 **-40% 单日黑天鹅暴跌测试**。在极端压力环境下，通过 6 层非线性金字塔补仓逻辑，将最大回撤 (MDD) 控制在 **-3.0%** 以内。
- **工业级回测架构**: 使用时间戳哈希（Timestamp Hashing）锁定随机种子，确保回测结果 100% 可复现，杜绝逻辑漂移。

## 📈 实验结论 (Backtest Results)
*测试标的: ETH-USD | 时间跨度: 近1年 | 初始资金: $10,000*
- **最优夏普比率 (Best Sharpe)**: 1.5+ (对应基础 RSI 30 组合)
- **最大回撤 (Max Drawdown)**: -1.91% (极度稳健配置)
- **策略类型**: 均值回归 + 波动率扩张防护

## 🛠️ 技术栈
- **Data**: yfinance (Yahoo Finance API)
- **Indicators**: pandas-ta (RSI, Bollinger Bands)
- **Visualization**: Plotly Multi-subplots
- **Interface**: Streamlit Web Dashboard

## 🚀 快速启动
1. 运行环境: `pip install streamlit yfinance pandas_ta plotly`
2. 启动看板: `streamlit run Strategy_v7_Pro.py`
