"""交易对配置 - 合约规格、精度、限制"""

SYMBOL_CONFIGS = {
    "ETHUSDT": {"min_qty": 0.01, "qty_step": 0.01, "price_scale": 2, "max_leverage": 100, "recommended_leverage": 30},
    "BTCUSDT": {"min_qty": 0.001, "qty_step": 0.001, "price_scale": 2, "max_leverage": 100, "recommended_leverage": 30},
    "XRPUSDT": {"min_qty": 1, "qty_step": 1, "price_scale": 4, "max_leverage": 75, "recommended_leverage": 30},
    "SOLUSDT": {"min_qty": 0.1, "qty_step": 0.1, "price_scale": 2, "max_leverage": 50, "recommended_leverage": 30},
    "LINKUSDT": {"min_qty": 0.1, "qty_step": 0.1, "price_scale": 3, "max_leverage": 50, "recommended_leverage": 20},
}

CORRELATED_GROUPS = {
    "majors": ["BTCUSDT", "ETHUSDT"],
    "alts": ["XRPUSDT", "SOLUSDT", "LINKUSDT"],
}


def get_symbol_config(symbol: str) -> dict:
    return SYMBOL_CONFIGS.get(symbol, {})


def get_all_symbols() -> list:
    return list(SYMBOL_CONFIGS.keys())
