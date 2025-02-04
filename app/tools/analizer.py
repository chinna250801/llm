import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from app.tools.strategy_tool import (
    rsi_tool, macd_tool, sma_tool, bollinger_tool,
    adx_tool, vtr_tool, risk_tool, stochastic_tool
)

class AnalyzerAgent:
    def __init__(self, secret_api_key: str):
        """
        Initializes the market analyzer agent with LangChain tools.
        """
        self.secret_api_key = secret_api_key

        # List of technical analysis tools
        self.tools = [
            rsi_tool, macd_tool, sma_tool, bollinger_tool,
            adx_tool, vtr_tool, risk_tool, stochastic_tool
        ]

        # Initialize LangChain agent with LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=secret_api_key)
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        """
        Initializes the agent using the tools and LLM.
        """
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def analyze(self, websocket_data):
        """
        Analyzes market data using technical indicators and returns a trading signal.
        """
        # Convert WebSocket data to a pandas DataFrame
        historical_data = pd.DataFrame([websocket_data])

        # Run all technical indicators through LangChain
        analysis_results = {
            "RSI": rsi_tool._run(historical_data),
            "MACD": macd_tool._run(historical_data),
            "SMA": sma_tool._run(historical_data),
            "Bollinger Bands": bollinger_tool._run(historical_data),
            "ADX": adx_tool._run(historical_data),
            "VTR (Volatility)": vtr_tool._run(historical_data),
            "Risk Analysis": risk_tool._run(historical_data),
            "Stochastic": stochastic_tool._run(historical_data)
        }

        # Determine BUY/SELL signal (at least 5 indicators for BUY, 4 for SELL)
        buy_signals = sum(1 for key, value in analysis_results.items() if value == "BUY")
        sell_signals = sum(1 for key, value in analysis_results.items() if value == "SELL")

        if buy_signals >= 5:
            final_signal = "BUY"
        elif sell_signals >= 4:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        # Return the full analysis
        return {
            "final_signal": final_signal,
            "indicators": analysis_results
        }
