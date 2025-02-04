import pandas as pd
import logging


class WebSocketAnalyzer:
    def __init__(self, risk_threshold=2, balance=10000, period=14, price=None):
        self.risk_threshold = risk_threshold  # Risk threshold in percentage
        self.balance = balance  # User's account balance
        self.period = period  # Lookback period for indicators (e.g., 14)
        self.price = price  # Current price of the asset (for position sizing)
        self.historical_data = pd.DataFrame(columns=['close', 'high', 'low'])  # Data frame to hold historical data
        self.logger = logging.getLogger(__name__)

    def update_data(self, websocket_data):
        tickers = websocket_data.get('events', [])[0].get('tickers', [])
        if not tickers:
            self.logger.warning("No tickers found in WebSocket data.")
            return

        for ticker in tickers:
            close_price = float(ticker.get('price', 0))
            high_price = float(ticker.get('high_24_h', 0))
            low_price = float(ticker.get('low_24_h', 0))

            if close_price == 0:
                self.logger.warning("Received invalid close price.")
                continue  # Skip invalid entries

            new_row = pd.DataFrame({'close': [close_price], 'high': [high_price], 'low': [low_price]})
            self.historical_data = pd.concat([self.historical_data, new_row], ignore_index=True).dropna()

    def analyze(self, websocket_data):
        # Update historical data from WebSocket batch
        self.update_data(websocket_data)

        # Run all strategies
        strategy_results = self.run_strategies()

        # Calculate the trading signal based on strategy results
        signal = self.calculate_signal(strategy_results)

        # Perform risk analysis before taking any action
        if self.check_risk():
            self.logger.info(f"Signal determined: {signal}")
            return signal
        else:
            self.logger.warning("Risk threshold exceeded. No action taken.")
            return "No Action"

    def run_strategies(self):
        # Run all strategies and store their results
        results = []
        results.append(self.calculate_rsi())
        results.append(self.calculate_macd())
        results.append(self.calculate_sma())
        results.append(self.calculate_bollinger_bands())
        results.append(self.calculate_adx())
        results.append(self.calculate_vtr())
        results.append(self.calculate_stochastic_oscillator())

        return results

    def calculate_signal(self, strategy_results):
        # Ensure that strategy_results is a list or Series
        if isinstance(strategy_results, pd.Series):
            strategy_results = strategy_results.tolist()  # Convert to list if it's a Pandas Series

        # Use list comprehension or generator to count occurrences
        buy_count = sum(1 for result in strategy_results if str(result) == "BUY")
        sell_count = sum(1 for result in strategy_results if str(result) == "SELL")

        # Proceed with signal calculation
        if buy_count >= 5:
            return "BUY"
        elif sell_count >= 4:
            return "SELL"
        return "NEUTRAL"

    def check_risk(self):
        """
        Ensure that the position size based on risk management does not exceed the allowable risk.
        """
        position_size = self.calculate_position_size()
        risk_amount = self.balance * self.risk_threshold / 100
        # If the position size exceeds the allowable risk amount, we do not proceed
        if position_size * self.price > risk_amount:
            self.logger.warning(f"Position size {position_size} exceeds risk threshold.")
            return False
        return True

    def calculate_rsi(self):
        """
        Calculate the RSI (Relative Strength Index) based on historical data.
        """
        delta = self.historical_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        if rsi.iloc[-1] > 70:
            return "SELL"
        elif rsi.iloc[-1] < 30:
            return "BUY"
        print("RSI:", rsi.iloc[-1])
        return "NEUTRAL"

    def calculate_macd(self, short_period=12, long_period=26, signal_period=9):
        """
        Calculate the MACD (Moving Average Convergence Divergence).
        """
        short_ema = self.historical_data['close'].ewm(span=short_period, adjust=False).mean()
        long_ema = self.historical_data['close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()

        if macd.iloc[-1] > signal_line.iloc[-1]:
            return "BUY"
        elif macd.iloc[-1] < signal_line.iloc[-1]:
            return "SELL"
        print("macd:", macd.iloc[-1])
        return "NEUTRAL"

    def calculate_sma(self):
        if len(self.historical_data) < self.period:
            print(f"Not enough data for SMA. Required: {self.period}, Available: {len(self.historical_data)}")
            return "NEUTRAL", None

        sma_series = self.historical_data['close'].rolling(window=self.period).mean()

        if sma_series.isna().all():
            print("SMA calculation returned NaN. Check data integrity.")
            return "NEUTRAL", None

        latest_price = self.historical_data['close'].iloc[-1]
        latest_sma = sma_series.iloc[-1]

        print(f"Latest Price: {latest_price}, Latest SMA: {latest_sma}")

        return "BUY" if latest_price > latest_sma else "SELL", latest_sma

    def calculate_bollinger_bands(self, num_std_dev=2):
        """
        Calculate the Bollinger Bands (Upper and Lower).
        """
        sma, sma_val = self.calculate_sma()
        rolling_std = self.historical_data['close'].rolling(window=self.period).std()

        # Ensure both are numeric values
        if not isinstance(sma_val, (int, float)):
            self.logger.error(f"SMA is not a valid number: {sma}")
            return "Error"

        if sma=="Neutral" and rolling_std.isna().any():
            self.logger.error(f"Rolling standard deviation contains NaN values.")
            return "Error"

        upper_band = sma_val + (rolling_std * num_std_dev)
        lower_band = sma_val - (rolling_std * num_std_dev)

        if self.historical_data['close'].iloc[-1] < lower_band.iloc[-1]:
            return "BUY"
        elif self.historical_data['close'].iloc[-1] > upper_band.iloc[-1]:
            return "SELL"
        print("Bolinger buy case ",  lower_band.iloc[-1])
        print("Bolinger sell case ", upper_band.iloc[-1])
        return "NEUTRAL"

    def calculate_adx(self):
        """
        Calculate the ADX (Average Directional Index).
        """
        high = self.historical_data['high']
        low = self.historical_data['low']
        close = self.historical_data['close']

        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()

        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)

        plus_di = 100 * (plus_dm.rolling(window=self.period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.period).sum() / atr)

        adx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di)) * 100).rolling(window=self.period).mean()

        if adx.iloc[-1] > 25:
            return "BUY"
        print("adx", adx.iloc[-1])
        return "SELL"

    def calculate_vtr(self):
        """
        Calculate the Volatility (VTR).
        """
        print("vtr", self.historical_data['close'].rolling(window=self.period).std())
        return self.historical_data['close'].rolling(window=self.period).std()

    def calculate_stochastic_oscillator(self):
        """
        Calculate the Stochastic Oscillator.
        """
        low_min = self.historical_data['low'].rolling(window=self.period).min()
        high_max = self.historical_data['high'].rolling(window=self.period).max()
        close_prices = self.historical_data['close']
        return 100 * (close_prices - low_min) / (high_max - low_min)

    def calculate_position_size(self):
        """
        Calculate the position size for risk management.
        """
        risk_amount = self.balance * self.risk_threshold / 100
        position_size = risk_amount / self.price
        print("position", position_size)
        return position_size

