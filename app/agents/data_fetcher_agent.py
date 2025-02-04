import asyncio
import json
import numpy as np
import logging
from collections import deque

import websockets
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI

from app.agents.analizer_agent import AnalyzerAgent
from app.config.config import settings
from app.tools.data_fetcher_tool import DataFetcherTool


class DataFetcherAgent:
    def __init__(self, api_key: str, secret_api_key: str):
        """
        Initializes the agent with API keys for Coinbase.
        """
        self.api_key = api_key
        self.secret_api_key = secret_api_key
        self.websocket_url = "wss://advanced-trade-ws.coinbase.com"
        self.analyzer_agent = AnalyzerAgent(secret_api_key)
        self.historical_data = {}  # Store historical data for each product ID
        # Initialize the DataFetcherTool
        self.data_fetcher_tool = DataFetcherTool(
            name="coinbase_data_fetcher",
            func=None,
            description="Tool for fetching data from Coinbase"
        )

        # Initialize the LLM for agent reasoning (can be ChatOpenAI, for example)
        self.llm = ChatOpenAI(openai_api_key=settings.open_api_key)

        # Initialize agent tools (currently using only DataFetcherTool)
        self.tools = [self.data_fetcher_tool]

        # Set up a simple agent executor
        self.agent = self._initialize_agent()

    def _initialize_agent(self) -> AgentExecutor:
        """
        Initializes the agent using the tools and LLM.
        """
        # Create an agent that can use the data fetcher tool
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent

    def fetch_data(self, query: str) -> str:
        """
        Fetches data from the Coinbase API using the DataFetcherTool.
        """
        try:
            # Run the tool to fetch data for the given query
            result = self.data_fetcher_tool._run(query=query, config=None)
            return result
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return str(e)

    def run(self, input_query: str) -> str:
        """
        Executes the agent to process the query.
        """
        # You can process the input query using the agent
        try:
            response = self.agent.run(input_query)
            return response
        except Exception as e:
            logging.error(f"Error running agent: {e}")
            return str(e)

    async def stream_market_data(self, product_ids: list):
        """Subscribe to live trade updates using Coinbase WebSocket API and batch the signals."""
        batch_size = 5  # Number of products to process in each batch
        current_batch = []

        while True:
            try:
                async with websockets.connect(self.websocket_url) as ws:
                    # Subscribe to ticker updates
                    subscribe_message = {
                        "type": "subscribe",
                        "channel": "ticker",
                        "product_ids": product_ids
                    }
                    await ws.send(json.dumps(subscribe_message))
                    print(f"Subscribed to {product_ids} live feed!")

                    # Initialize historical data storage for each product ID
                    for product_id in product_ids:
                        self.historical_data[product_id] = deque(maxlen=100)  # Store up to 100 data points

                    # Receive and process messages
                    async for message in ws:
                        data = json.loads(message)
                        if data.get("channel") == "ticker":
                            tickers = data.get("events", [])[0].get("tickers", [])
                            for ticker in tickers:
                                product_id = ticker.get("product_id")
                                if product_id in self.historical_data:
                                    # Append new data to historical data
                                    self.historical_data[product_id].append({
                                        "timestamp": data.get("timestamp"),
                                        "close": float(ticker.get("price", 0)),
                                        "high": float(ticker.get("high_24_h", 0)),
                                        "low": float(ticker.get("low_24_h", 0)),
                                        "volume": float(ticker.get("volume_24_h", 0))
                                    })

                                    # Analyze the data for the current product
                                    signal = self.analyzer_agent.analyze(
                                        list(self.historical_data[product_id]), account_balance=1000, risk_threshold=2
                                    )
                                    print(signal)
                                    if signal.get("message") or signal.get('indicators') == "Insufficient data":
                                        continue

                                    # Add the signal to the current batch
                                    # current_batch.append({
                                    #     "product_id": product_id,
                                    #     "signal": signal
                                    # })
                                    #
                                    # # If the batch is full, send it to the LLM
                                    # if len(current_batch) >= batch_size:
                                    #     llm_input = self.create_batch_input(current_batch)
                                    #     llm_call_result = await self.llm_call(input_text=llm_input)
                                    #     print(f"Generated Signal for batch: {llm_call_result}")
                                    #
                                    #     # Clear the batch after processing
                                    #     current_batch.clear()

                # Handle WebSocket reconnection logic
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket disconnected. Reconnecting...")
                await asyncio.sleep(5)

    async def llm_call(self, input_text: str) -> str:
        """
        Interacts with the LLM to generate a final decision based on technical analysis for a batch of signals.
        """
        try:
            # LLM will interpret the raw signal batch and provide a comprehensive response
            prompt = f"Batch Signal Analysis:\n{input_text}\n\nBased on the analysis, provide a final recommendation (BUY, SELL, or HOLD) for each product in the batch, including position sizes, reasoning, and only respond with BUY, SELL, HOLD."

            # Run the LLM to get a detailed response
            response = await self.llm.invoke([prompt])

            # Return the response from the LLM
            return response
        except Exception as e:
            logging.error(f"Error generating LLM response: {e}")
            return "Unable to generate a recommendation at this time."

    def create_batch_input(self, batch: list) -> str:
        """
        Creates a combined input string for the batch of signals to be processed by the LLM.
        """
        batch_input = ""
        for item in batch:
            product_id = item["product_id"]
            signal = item["signal"]
            llm_input = f"Product: {product_id}, Indicators: {signal['indicators']}, Analysis Signal: {signal['analyzer_signal']}, LLM Signal: {signal['llm_signal']}, Position Size: {signal['position_size']}\n"
            batch_input += llm_input

        return batch_input

