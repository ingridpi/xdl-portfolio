# List of technical indicators
TECHNICAL_INDICATORS = {
    "close_5_sma": "Simple Moving Average over 5 days (1 week)",
    # "close_10_sma": "Simple Moving Average over 10 days (2 weeks)",
    # "close_20_sma": "Simple Moving Average over 20 days (1 month)",
    "close_5_ema": "Exponential Moving Average over 5 days (1 week)",
    # "close_10_ema": "Exponential Moving Average over 10 days (2 weeks)",
    "macd": "Moving Average Convergence Divergence",
    "rsi": "Relative Strength Index over 14 days",
    # "kdjk": "KDJ indicator - K line",
    # "kdjd": "KDJ indicator - D line",
    "cci": "Commodity Channel Index",
    "boll": "Bollinger Bands",
    # "boll_ub": "Bollinger Bands Upper Band",
    # "boll_lb": "Bollinger Bands Lower Band",
    "atr": "Average True Range",
    "adx": "Average Directional Index",
    "vwma": "Volume Weighted Moving Average",
}

# List of macroeconomic indicators
MACROECONOMIC_INDICATORS_DEFAULT = {
    "^VIX": "Volatility Index (VIX)",
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "^IRX": "3-Month Treasury Yield (IRX)",
    "^FVX": "5-Year Treasury Yield (FVX)",
    "^TNX": "10-Year Treasury Yield (TNX)",
}

MACROECONOMIC_INDICATORS_DW30 = {
    "^VXD": "Volatility Index (VXD)",
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "^IRX": "3-Month Treasury Yield (IRX)",
    "^FVX": "5-Year Treasury Yield (FVX)",
    "^TNX": "10-Year Treasury Yield (TNX)",
}

MACROECONOMIC_INDICATORS_EUROSTOXX50 = {
    "^V2TX.DE": "Volatility Index (VSTOXX)",
    "^XDE": "Euro Currency Index (EXY)",
}

MACROECONOMIC_INDICATORS_FTSE100 = {
    "VFTSE.AS": "Volatility Index (FTSE 100)",
    "^XDB": "British Pound Currency Index (BXY)",
}
