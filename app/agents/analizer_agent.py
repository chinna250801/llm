import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from app.config.config import settings
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
        self.llm = ChatOpenAI(openai_api_key=settings.open_api_key)
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

    def analyze(self, historical_data, account_balance, risk_threshold=2):
        """
        Analyzes market data using technical indicators and returns a trading signal.
        Integrates risk management (position size) based on account balance and risk threshold using the risk_tool.
        """
        # Convert historical data to DataFrame
        historical_df = pd.DataFrame(historical_data)

        # Ensure there's enough data for calculations
        if len(historical_df) < 14:
            return {"final_signal": "HOLD", "indicators": "Insufficient data"}

        # Run all technical indicators
        analysis_results = {
            "RSI": rsi_tool._run(historical_df, config=None),
            "MACD": macd_tool._run(historical_df, config=None),
            "SMA": sma_tool._run(historical_df, config=None),
            "Bollinger Bands": bollinger_tool._run(historical_df, config=None),
            "VTR (Volatility)": vtr_tool._run(historical_df, config=None),
            "Stochastic": stochastic_tool._run(historical_df, config=None)
        }

        # Prepare a textual summary of the analysis
        analysis_summary = "\n".join([f"{key}: {value[0]}" for key, value in analysis_results.items()])

        # Determine BUY/SELL signal based on technical analysis
        buy_signals = sum(1 for key, value in analysis_results.items() if value[0] == "BUY")
        sell_signals = sum(1 for key, value in analysis_results.items() if value[0] == "SELL")

        if buy_signals >= 3:
            analyzer_signal = "BUY"
        elif sell_signals >= 3:
            analyzer_signal = "SELL"
        else:
            analyzer_signal = "HOLD"

        # Integrate risk management using the risk_tool to determine position size
        status, position_size = risk_tool.func(account_balance, historical_df['close'].iloc[-1], risk_threshold)

        if position_size is None:
            return {"final_signal": "HOLD", "message": "Invalid input for position size calculation"}

        # Create a prompt for the LLM to evaluate and suggest a final signal based on the analysis
        prompt = f"""
               Here is the technical analysis for the market data:

               {analysis_summary}

               Based on this analysis, provide a final trading signal (BUY, SELL, or HOLD).
               Consider the following:
               - A BUY signal requires at least 3 indicators confirming.
               - A SELL signal requires at least 3 indicators confirming.
               - Any other situation should result in a HOLD signal.

               The user has an account balance of ${account_balance}, and the current price is {historical_df['close'].iloc[-1]}.
               Based on a {risk_threshold}% risk threshold, the position size is {position_size} units of the asset.

               Please provide a final recommendation:
               1. Should the user execute a BUY or SELL trade?
               2. If so, what position size is recommended based on the risk management calculation?

               The final signal should include a brief explanation of why the signal was chosen (considering both the technical analysis and risk management).
               """

        # Send the prompt to the LLM for evaluation
        llm_response = self.llm.predict(prompt)

        # Extract the final signal and explanation from the LLM's response
        llm_signal = llm_response.strip()  # Clean the response
        print(llm_signal)
        # Return both the analyzer signal and the LLM's signal with explanation
        return {
            "analyzer_signal": analyzer_signal,
            # "llm_signal": llm_signal,
            "position_size": position_size,
            "indicators": analysis_results
        }

