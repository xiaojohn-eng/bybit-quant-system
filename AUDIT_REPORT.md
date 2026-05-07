# 审计报告 - Bybit量化交易系统

## P0 致命问题（必须修复，否则系统无法运行）

### 1. Signal类接口完全不一致
- strategies/base_strategy.py 的 Signal: action, confidence, strategy_name, symbol, metadata
- backtest/engine.py 的 Signal: action, price, size_pct, sl_pct, tp_pct, quality_score, confidence
- 两者完全不兼容

### 2. 策略缺少 get_parameters() 方法
- backtest/engine.py 调用 strategy.get_parameters()
- BaseStrategy 没有此方法

### 3. main.py API调用与实际模块完全不匹配
- orchestrator.run_all() 不存在 -> 实际: generate_signals()/generate_combined_signal()
- cache.store() 不存在 -> 实际: set()/set_klines()
- client.close() 不存在
- client.get_historical_klines() 不存在 -> 实际: get_klines()
- executor返回dict但main检查 result.success/result.side/result.message
- risk_mgr.check_trade_allowed() 签名完全不匹配

### 4. BybitClient 初始化签名不匹配
- main用: BybitClient(api_key=..., api_secret=..., testnet=...)
- 实际: BybitClient(config: BybitConfig)

### 5. Config 属性访问错误
- main用: getattr(config, "BYBIT_API_KEY", "")
- 实际: config.bybit.api_key

### 6. PaperTradingEngine side值大小写不一致（致命逻辑bug）
- place_order接收side="buy"/"sell"(策略输出)
- 但_check_liquidation/_calc_pnl/_settle_funding检查pos.side=="Buy"/"Sell"
- 导致所有风控检查失效

### 7. PaperTradingEngine equity计算错误
- get_equity_unlocked() = balance + _used_margin + unrealized
- 但balance已扣除margin，再加_used_margin导致double counting

### 8. RiskManager接收Config对象但期望dict
- main: RiskManager(config=config) # Config对象
- 实际: RiskManager(config: dict = None) # 期望dict

## P1 严重问题

### 9. FundingArbitrage.generate_signal签名不匹配基类
- 基类: generate_signal(self, df)
- 实际: generate_signal(self, df, funding_rate=0.0)

### 10. __init__.py全部为空
- 需要导出模块方便导入

### 11. import math位置错误
- main.py第556行在函数内import math，应在顶部
