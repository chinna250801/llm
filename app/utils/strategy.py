import pandas as pd
import numpy as np


def calculate_rsi(historical_data, period=14):
    if len(historical_data) < period:
        return "NEUTRAL", None  # Not enough data

    delta = historical_data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    signal = "SELL" if rsi.iloc[-1] > 70 else "BUY" if rsi.iloc[-1] < 30 else "NEUTRAL"
    return signal, rsi.iloc[-1]


def calculate_macd(historical_data, short_period=12, long_period=26, signal_period=9):
    if len(historical_data) < long_period:
        return "NEUTRAL", None  # Not enough data

    short_ema = historical_data['close'].ewm(span=short_period, adjust=False).mean()
    long_ema = historical_data['close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()

    signal = "BUY" if macd.iloc[-1] > signal_line.iloc[-1] else "SELL" if macd.iloc[-1] < signal_line.iloc[-1] else "NEUTRAL"
    return signal, macd.iloc[-1]


def calculate_sma(historical_data, period=14):
    if len(historical_data) < period:
        return "NEUTRAL", None  # Not enough data

    sma_series = historical_data['close'].rolling(window=period).mean()
    latest_price = historical_data['close'].iloc[-1]
    latest_sma = sma_series.iloc[-1]

    signal = "BUY" if latest_price > latest_sma else "SELL"
    return signal, latest_sma


def calculate_bollinger_bands(historical_data, period=14, num_std_dev=2):
    if len(historical_data) < period:
        return "NEUTRAL", (None, None)  # Not enough data

    sma = historical_data['close'].rolling(window=period).mean()
    rolling_std = historical_data['close'].rolling(window=period).std()
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)

    signal = "BUY" if historical_data['close'].iloc[-1] < lower_band.iloc[-1] else "SELL" if historical_data['close'].iloc[-1] > upper_band.iloc[-1] else "NEUTRAL"
    return signal, (upper_band.iloc[-1], lower_band.iloc[-1])


def calculate_adx(historical_data, period=14):
    """
    Calculate the Ichimoku Cloud strategy and generate a trading signal.

    Args:
        historical_data (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.

    Returns:
        tuple: (str) -> Trading signal ("BUY", "SELL", or "NEUTRAL")
    """
    if len(historical_data) < 52:  # Need at least 52 periods for full Ichimoku calculation
        return "NEUTRAL", None  # Not enough data

    # Calculate the Ichimoku lines
    high_9 = historical_data['high'].rolling(window=9).max()
    low_9 = historical_data['low'].rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2  # Conversion line

    high_26 = historical_data['high'].rolling(window=26).max()
    low_26 = historical_data['low'].rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2  # Base line

    high_52 = historical_data['high'].rolling(window=52).max()
    low_52 = historical_data['low'].rolling(window=52).min()
    senkou_span_b = (high_52 + low_52) / 2  # Senkou Span B

    # Plotting Senkou Span A (midpoint of Tenkan and Kijun)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    chikou_span = historical_data['close'].shift(-26)  # Chikou Span

    # Trading Signal Logic
    if historical_data['close'].iloc[-1] > senkou_span_a.iloc[-1] and tenkan_sen.iloc[-1] > kijun_sen.iloc[-1]:
        signal = "BUY"
    elif historical_data['close'].iloc[-1] < senkou_span_a.iloc[-1] and tenkan_sen.iloc[-1] < kijun_sen.iloc[-1]:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return signal, None




def calculate_vtr(historical_data, period=14):
    if len(historical_data) < period:
        return "NEUTRAL", None  # Not enough data

    log_returns = np.log(historical_data['close'] / historical_data['close'].shift(1))
    volatility = log_returns.rolling(window=period).std() * np.sqrt(period)

    signal = "BUY" if volatility.iloc[-1] < volatility.median() else "SELL"
    return signal, volatility.iloc[-1]


def calculate_stochastic(historical_data, period=14, smooth_k=3, smooth_d=3):
    if len(historical_data) < period:
        return "NEUTRAL", (None, None)  # Not enough data

    lowest_low = historical_data['low'].rolling(window=period).min()
    highest_high = historical_data['high'].rolling(window=period).max()

    k = 100 * (historical_data['close'] - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=smooth_d).mean()

    signal = "BUY" if k.iloc[-1] > d.iloc[-1] else "SELL" if k.iloc[-1] < d.iloc[-1] else "NEUTRAL"
    return signal, (k.iloc[-1], d.iloc[-1])


def check_risk(balance, price, risk_threshold=2):
    if balance <= 0 or price <= 0:
        return "Invalid Input", None

    risk_amount = balance * risk_threshold / 100
    position_size = risk_amount / price
    return "Position Size", position_size
