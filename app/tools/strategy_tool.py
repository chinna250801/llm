from langchain.agents import Tool

from app.utils.strategy import calculate_rsi, calculate_macd, calculate_sma, calculate_bollinger_bands, calculate_adx, \
    calculate_vtr, check_risk, calculate_stochastic

# Wrap RSI calculation as a tool
rsi_tool = Tool(
    name="RSI Tool",
    func=calculate_rsi,
    description="Calculates the Relative Strength Index (RSI) of market data"
)

# Wrap MACD calculation as a tool
macd_tool = Tool(
    name="MACD Tool",
    func=calculate_macd,
    description="Calculates the Moving Average Convergence Divergence (MACD)"
)

# Wrap SMA calculation as a tool
sma_tool = Tool(
    name="SMA Tool",
    func=calculate_sma,
    description="Calculates the Simple Moving Average (SMA) of market data"
)

# Wrap Bollinger Bands calculation as a tool
bollinger_tool = Tool(
    name="Bollinger Bands Tool",
    func=calculate_bollinger_bands,
    description="Calculates the Bollinger Bands for market data"
)

# Wrap ADX calculation as a tool
adx_tool = Tool(
    name="ADX Tool",
    func=calculate_adx,
    description="Calculates the Average Directional Index (ADX)"
)

# Wrap VTR calculation as a tool
vtr_tool = Tool(
    name="VTR Tool",
    func=calculate_vtr,
    description="Calculates Volatility (VTR) for market data"
)

# Wrap Risk calculation as a tool
risk_tool = Tool(
    name="Risk Management Tool",
    func=check_risk,
    description="Calculates the position size based on risk management"
)

# Stochastic calculation as a tool
stochastic_tool = Tool(
    name="Stochastic Oscillator Tool",
    func=calculate_stochastic,
    description="Calculates the Stochastic Oscillator (%K and %D)"
)
