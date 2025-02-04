import pandas as pd
from app.agents.data_fetcher_agent import DataFetcherAgent
from app.coinbase_.market_analyser import MarketAnalyzer
from app.config.config import settings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Assuming your settings contain the API keys
api_key = settings.coinbase_api_key
api_secret = settings.coinbase_api_secret

# Initialize the agent for data fetching
data_fetcher_agent = DataFetcherAgent(api_key=api_key, secret_api_key=api_secret)

# Test fetching data for a specific query
query = ["ETH-USD", "BTC-USD"]
result = data_fetcher_agent.fetch_data(query)

# Store analysis results
analysis_results = {}
secret_api_key = settings.open_api_key
# Initialize LangChain's ChatModel (using OpenAI)
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=secret_api_key)

# Define a prompt template to instruct LangChain’s chat model on what kind of responses you're expecting
template = """
You are a professional financial advisor specializing in technical analysis. Given the following market indicators, analyze the data and provide a precise trading recommendation (Buy, Sell, Hold). Your response must be based strictly on these signals, ensuring logical consistency and risk management considerations.

Market Signals:

RSI Signal: {rsi_signal} (Oversold, Overbought, Neutral) - Weight: 0.2
MACD Signal: {macd_signal} (Bullish, Bearish, Neutral) - Weight: 0.3
SMA Signal: {sma_signal} (Bullish, Bearish, Neutral) - Weight: 0.1
Bollinger Band Signal: {bollinger_signal} (Buy, Sell, Neutral) - Weight: 0.15
ADX Signal: {adx_signal} (Strong Trend, Weak Trend, No Data Available) - Weight: 0.15
VTR Signal: {vtr_signal} (High Volatility, Low Volatility, No Data Available) - Weight: 0.1

Decision: (Buy / Sell / Hold)
Explanation: (Justify your decision logically and concisely, considering trend strength, volatility, confirmation signals, and any missing data. Acknowledge if any signal is missing and adjust the decision accordingly.)
"""

# Create a prompt template using the above structure
prompt = PromptTemplate(input_variables=["rsi_signal", "macd_signal", "sma_signal", "bollinger_signal", "adx_signal", "vtr_signal"], template=template)

# Create the LLMChain that ties the prompt template and the chat model together
llm_chain = LLMChain(prompt=prompt, llm=chat_model)

# Iterate through each asset and perform analysis
for symbol, data in result.get("multiple_product_details").items():
    product_details = data['product_details']
    historical_data_df = pd.DataFrame(data['historical_data'])  # Historical data as DataFrame

    # Initialize MarketAnalyzer
    analyzer = MarketAnalyzer(
        historical_data=historical_data_df,
        product_details=product_details,
        portfolio_value=10000  # Example portfolio value
    )

    # Perform analysis
    analysis_result = analyzer.analyze()

    # Extract signals for LangChain
    rsi_signal = analysis_result['rsi_signal']
    macd_signal = analysis_result['macd_signal']
    sma_signal = analysis_result['sma_signal']
    bollinger_signal = analysis_result['bollinger_signal']
    adx_signal = analysis_result['adx_signal']
    vtr_signal = analysis_result['vtr_signal']

    # Get recommendation from LangChain model based on the signals
    recommendation = llm_chain.run({
        "rsi_signal": rsi_signal,
        "macd_signal": macd_signal,
        "sma_signal": sma_signal,
        "bollinger_signal": bollinger_signal,
        "adx_signal": adx_signal,
        "vtr_signal": vtr_signal
    })

    # Store the result for each asset
    analysis_results[symbol] = {
        **analysis_result,
        'langchain_recommendation': recommendation  # Adding LangChain's recommendation
    }

# Print the analysis results for each asset
for symbol, result in analysis_results.items():
    print(f"Analysis for {symbol}:")
    print(f"Decision: {result['decision']}")
    print(f"Position Size: {result['position_size']}")
    print(f"Stop-Loss: {result['stop_loss']}")
    print(f"RSI Signal: {result['rsi_signal']}")
    print(f"MACD Signal: {result['macd_signal']}")
    print(f"SMA Signal: {result['sma_signal']}")
    print(f"Bollinger Signal: {result['bollinger_signal']}")
    print(f"ADX Signal: {result['adx_signal']}")
    print(f"VTR Signal: {result['vtr_signal']}")
    print(f"LangChain Recommendation: {result['langchain_recommendation']}")
    print("\n")
