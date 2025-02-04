from typing import Optional, Any, Callable

from langchain.agents import Tool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig

from app.coinbase_.data_fetcher import DataFetcher
from app.config.config import settings
from app.models.models import InputParam

class DataFetcherTool(Tool):
    def __init__(self, name: str, func: Optional[Callable], description: str, **kwargs: Any):

        super().__init__(name, func, description, **kwargs)

    def _run(
        self,
        *args: Any,
        config: Optional[RunnableConfig] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Executes the tool by calling the DataFetcher's methods.
        """
        # Call the method to fetch data from the Coinbase API
        query = kwargs.get('query', None)
        if query:
            data_fetcher = DataFetcher(api_key=settings.coinbase_api_key, api_secret=settings.coinbase_api_secret)
            if isinstance(query, str):
                product_details = data_fetcher.get_product_details(query)
                input = InputParam(product_id=str(query),granularity="ONE_DAY")
                historical_data = data_fetcher.fetch_historical_data(
                    input
                )
                return {
                    "product_details": product_details,
                    "historical_data": historical_data
                }
            elif isinstance(query, list):
                multiple_product_details = data_fetcher.get_multiple_product_details_and_history(query,
                                                                                                 granularity="ONE_DAY")
                return {"multiple_product_details": multiple_product_details}
        else:
            raise ValueError("Query parameter is required to fetch data.")


