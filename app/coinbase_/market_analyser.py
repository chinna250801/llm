import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class MarketAnalyzer:
    def __init__(self, product_details, historical_data, portfolio_value, risk_percentage=0.02, loss_percentage=0.05, period=9):
        self.product_details = product_details
        self.historical_data = historical_data
        self.price = float(product_details['price'])
        self.volume = float(product_details['volume_24h'])
        self.price_percentage_change_24h = float(product_details['price_percentage_change_24h'])
        self.portfolio_value = portfolio_value
        self.risk_percentage = risk_percentage
        self.loss_percentage = loss_percentage
        self.period = period

    # 1. Calculate RSI (14-day)
    def calculate_rsi(self):
        delta = self.historical_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # 2. Calculate MACD (Moving Average Convergence Divergence)
    def calculate_macd(self, short_period=12, long_period=26, signal_period=9):
        short_ema = self.historical_data['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = self.historical_data['close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line

    # 3. Calculate SMA (14-day Simple Moving Average)
    def calculate_sma(self):
        return self.historical_data['close'].rolling(window=self.period).mean()

    # 4. Calculate Bollinger Bands (14-day)
    def calculate_bollinger_bands(self, num_std_dev=2):
        sma = self.calculate_sma()
        rolling_std = self.historical_data['close'].rolling(window=self.period).std()
        upper_band = sma + (rolling_std * num_std_dev)
        lower_band = sma - (rolling_std * num_std_dev)
        return upper_band, lower_band

    # 5. Calculate ADX (14-day Average Directional Index)
    def calculate_adx(self):
        high = self.historical_data['high']
        low = self.historical_data['low']
        close = self.historical_data['close']

        # True Range (TR) calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        print("True Range:\n", tr.tail(10))  # Debug

        atr = tr.rolling(window=self.period, min_periods=1).mean()
        atr.fillna(method='bfill', inplace=True)  # Fix NaN issue

        print("ATR Data:\n", atr.tail(10))  # Debug

        # Directional Movement (DM) calculation
        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        print("Plus DM:\n", plus_dm.tail(10))  # Debug
        print("Minus DM:\n", minus_dm.tail(10))  # Debug

        # Directional Indicators (DI)
        plus_di = 100 * (plus_dm.rolling(window=self.period, min_periods=1).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.period, min_periods=1).sum() / atr)

        print("Plus DI:\n", plus_di.tail(10))  # Debug
        print("Minus DI:\n", minus_di.tail(10))  # Debug

        # DX (Directional Index) calculation
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        dx.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values
        dx.fillna(method='bfill', inplace=True)  # Fix NaN

        print("DX:\n", dx.tail(10))  # Debug

        # ADX calculation
        adx = dx.rolling(window=self.period, min_periods=1).mean()
        adx.fillna(method='bfill', inplace=True)  # Fix NaN

        print("ADX Data:\n", adx.tail(10))  # Debug

        return adx, plus_di, minus_di

    # 6. Calculate VTR (Volatility)
    def calculate_vtr(self):
        return self.historical_data['close'].rolling(window=self.period).std()

    # Signal Evaluation
    def evaluate_signals(self, rsi, macd, signal_line, sma, upper_band, lower_band, adx, vtr, stx):
        # Remove NaN values from the series
        rsi = rsi.dropna()
        macd = macd.dropna()
        signal_line = signal_line.dropna()
        sma = sma.dropna()
        upper_band = upper_band.dropna()
        lower_band = lower_band.dropna()
        adx = adx.dropna()
        vtr = vtr.dropna()
        stx = stx.dropna()
        print("ADX Data:\n", adx.tail(10))
        # Ensure there is valid data in each series before accessing the last element
        if len(rsi) > 0:
            print("rsi", rsi.iloc[-1])
            rsi_signal = "Oversold" if rsi.iloc[-1] < 30 else "Overbought" if rsi.iloc[-1] > 70 else "Neutral"
        else:
            rsi_signal = "No Data Available"

        if len(macd) > 0 and len(signal_line) > 0:
            print("macd",macd.iloc[-1])
            macd_signal = "Bullish" if macd.iloc[-1] > signal_line.iloc[-1] else "Bearish"
        else:
            macd_signal = "No Data Available"

        if len(sma) > 1:
            print("sma", sma.iloc[-1])
            sma_signal = "Bullish" if sma.iloc[-1] > sma.iloc[-2] else "Bearish"
        else:
            sma_signal = "No Data Available"

        if len(upper_band) > 0 and len(lower_band) > 0:
            # print("bollingup"+str(upper_band)+"bolling low" +str(lower_band))
            bollinger_signal = "Buy" if self.price < lower_band.iloc[-1] else "Sell" if self.price > upper_band.iloc[
                -1] else "Neutral"
        else:
            bollinger_signal = "No Data Available"

        if len(adx) > 0:
            adx_signal = "Strong Trend" if adx.iloc[-1] > 25 else "Weak Trend"
        else:
            adx_signal = "No Data Available"

        if len(vtr) > 0:
            # print("vtr"+vtr)
            vtr_signal = "High Volatility" if vtr.iloc[-1] > 1 else "Low Volatility"
        else:
            vtr_signal = "No Data Available"
        stochastic_signal = "Overbought" if stx.iloc[-1] > 80 else "Oversold" if stx.iloc[-1] < 20 else "Neutral"
        return rsi_signal, macd_signal, sma_signal, bollinger_signal, adx_signal, vtr_signal

    def calculate_stochastic_oscillator(self):
        """
        Calculate the Stochastic Oscillator.
        """
        low_min = self.historical_data['low'].rolling(window=self.period).min()
        high_max = self.historical_data['high'].rolling(window=self.period).max()
        close_prices = self.historical_data['close']
        # print("stoch", 100 * (close_prices - low_min) / (high_max - low_min))
        return 100 * (close_prices - low_min) / (high_max - low_min)

    # Risk Management: Position size calculation
    def calculate_position_size(self):
        risk_amount = self.portfolio_value * self.risk_percentage
        position_size = risk_amount / self.price
        return position_size

    # Risk Management: Stop-loss calculation
    def calculate_stop_loss(self):
        stop_loss = self.price * (1 - self.loss_percentage)
        return stop_loss

    # Decision Making Based on Signals
    def make_decision(self, rsi_signal, macd_signal, sma_signal, bollinger_signal, adx_signal, vtr_signal):
        if rsi_signal == "Oversold" and macd_signal == "Bullish" and sma_signal == "Bullish" and bollinger_signal == "Buy":
            decision = "Buy"
        elif rsi_signal == "Overbought" and macd_signal == "Bearish" and sma_signal == "Bearish" and bollinger_signal == "Sell":
            decision = "Sell"
        else:
            decision = "Hold"

        return decision

    # Main method to run the analysis
    def analyze(self):
        # Calculate indicators
        rsi = self.calculate_rsi()
        macd, signal_line = self.calculate_macd()
        sma = self.calculate_sma()
        upper_band, lower_band = self.calculate_bollinger_bands()
        adx, plus_di, minus_di = self.calculate_adx()  # Update here
        vtr = self.calculate_vtr()
        stx = self.calculate_stochastic_oscillator()
        # Evaluate signals
        rsi_signal, macd_signal, sma_signal, bollinger_signal, adx_signal, vtr_signal, stochastic_signal = self.evaluate_signals(
            rsi, macd, signal_line, sma, upper_band, lower_band, adx, vtr, stx
        )

        # Risk management
        position_size = self.calculate_position_size()
        stop_loss = self.calculate_stop_loss()

        # Make decision based on signals
        decision = self.make_decision(rsi_signal, macd_signal, sma_signal, bollinger_signal, adx_signal, vtr_signal)
        # Plot indicators
        self.plot_indicators(rsi, macd, signal_line, sma, upper_band, lower_band, adx, vtr)

        # Return results
        return {
            'decision': decision,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'sma_signal': sma_signal,
            'bollinger_signal': bollinger_signal,
            'adx_signal': adx_signal,
            'vtr_signal': vtr_signal,
            'product_details': self.product_details  # Include the product details in the result
        }

        # Separate method for plotting

    def plot_indicators(self, rsi, macd, signal_line, sma, upper_band, lower_band, adx, vtr):
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))

        # Plot RSI
        ax[0].plot(rsi, label="RSI", color='blue')
        ax[0].axhline(70, color='red', linestyle='--', label="Overbought")
        ax[0].axhline(30, color='green', linestyle='--', label="Oversold")
        ax[0].set_title("RSI (Relative Strength Index)")
        ax[0].legend()

        # Plot MACD and Signal Line
        ax[1].plot(macd, label="MACD", color='blue')
        ax[1].plot(signal_line, label="Signal Line", color='orange')
        ax[1].set_title("MACD (Moving Average Convergence Divergence)")
        ax[1].legend()

        plt.tight_layout()
        plt.show()



