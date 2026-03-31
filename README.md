# 🏴‍☠️ Greedy-Hunter-Quant: V6 Multi-Asset Strategy
基于统计学均值回归与动态仓位管理的量化交易系统

## 📈 项目概述
本项目是一个基于 Python 的全自动量化回测与监控仪表盘。
核心逻辑利用 **RSI 极端超卖回踩** 与 **布林带 (Bollinger Bands) 均值回归**，配合**六层金字塔式加仓策略**，旨在波动性市场中获取风险调整后的超额收益。

## 🚀 核心功能
- **多币种矩阵回测**：支持 BTC, ETH, SOL 等主流资产的并行模拟。
- **动态参数寻优 (Grid Search)**：自动寻找最优 RSI 阈值与补仓间距。
- **风险控制面板**：实时监控最大回撤 (MDD)、夏普比率 (Sharpe Ratio) 及资产曲线。

## 📊 阶段性战果 (2026.03)
- **ETH 策略**：在 RSI 40 / 5.6% 补仓参数下，实现 **18% 收益**，MDD 控制在 **-7.58%**。
- **资产组合**：通过多币种配置，成功将组合 MDD 压低至 **5%** 以内（开发中）。

## 🛠️ 技术栈
- **Language**: Python 3.x
- **Libraries**: Streamlit, Pandas, Pandas-TA, Plotly, YFinance
