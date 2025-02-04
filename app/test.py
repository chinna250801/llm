import asyncio

from app.agents.data_fetcher_agent import DataFetcherAgent
from app.config.config import settings

api_key = settings.coinbase_api_key
secret_api_key = settings.coinbase_api_secret

fetcher_agent = DataFetcherAgent(api_key=api_key, secret_api_key=secret_api_key)

# Start WebSocket data streaming (async operation)
product_ids = ["BTC-USD", "ETH-USD"]
asyncio.run(fetcher_agent.stream_market_data(product_ids))