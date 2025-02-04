import asyncio
import json
from datetime import datetime, timedelta

import pandas as pd

import requests
import websockets
from coinbase import jwt_generator

from app.coinbase_.websocket_analyser import WebSocketAnalyzer
from app.config.config import settings
from app.models.models import InputParam


class DataFetcher:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.websocket_url = "wss://advanced-trade-ws.coinbase.com"

    def generate_jwt(self, request_method: str, request_path: str) -> str:
        """Generates a JWT for authenticating requests."""
        # Format the JWT URI for the request you want to make
        jwt_uri = jwt_generator.format_jwt_uri(request_method, request_path)

        # Generate the JWT token using the API key and secret
        jwt_token = jwt_generator.build_rest_jwt(jwt_uri, self.api_key, self.api_secret)
        return jwt_token

    def get_product_details(self, product_id: str, get_tradability_status: bool = False) -> dict:
        """Fetch details of a specific product using its product ID."""
        path = f"/api/v3/brokerage/products/{product_id}"
        url = f"https://api.coinbase.com{path}"
        params = f"?get_tradability_status={str(get_tradability_status).lower()}"

        jwt_token = self.generate_jwt("GET", path)  # Generate JWT

        headers = {
            "Authorization": f"Bearer {jwt_token}"
        }

        try:
            response = requests.get(url + params, headers=headers)
            response.raise_for_status()
            return response.json()  # Return product details as a JSON object
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def get_top_us_crypto_details(self, product_ids: list) -> dict:
        """Fetch details for a list of top US-based cryptocurrencies."""
        crypto_details = {}

        for product_id in product_ids:
            details = self.get_product_details(product_id)  # Fetch product details
            crypto_details[product_id] = details  # Store details in dictionary

        return crypto_details

    def get_market_trades(self, product_id: str, limit: int = 5) -> dict:
        """Fetches recent market trades and best bid/ask for a given product."""
        url = f"https://api.coinbase.com/api/v3/brokerage/products/{product_id}/ticker?limit={limit}"
        path = f"/api/v3/brokerage/products/{product_id}/ticker"

        jwt_token = self.generate_jwt("GET", path)
        headers = {"Authorization": f"Bearer {jwt_token}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


    def fetch_historical_data(self, params: InputParam):
        """
        Fetch historical candlestick (OHLC) data from Coinbase.
        """
        # Determine the time range (start_time, end_time)
        end_time = params.end_time or datetime.utcnow()
        start_time = params.start_time or (end_time - timedelta(hours=params.limit))

        # Convert times to Unix timestamps
        end_time_unix = int(end_time.timestamp())
        start_time_unix = int(start_time.timestamp())

        # Build the API URL for historical data
        url = f"https://api.coinbase.com/api/v3/brokerage/products/{params.product_id}/candles"
        request_path = f"/api/v3/brokerage/products/{params.product_id}/candles"

        # Generate JWT for authentication
        jwt_token = self.generate_jwt("GET", request_path)

        # Set up the request headers and parameters
        headers = {"Authorization": f"Bearer {jwt_token}"}
        query_params = {
            "start": str(start_time_unix),
            "end": str(end_time_unix),
            "granularity": params.granularity,
        }

        try:
            # Send the GET request to fetch the historical data
            response = requests.get(url, headers=headers, params=query_params)
            response.raise_for_status()

            # Process and return the data
            data = response.json().get("candles")
            df = pd.DataFrame(data, columns=["low", "high", "open", "close", "volume"])

            # Ensure that the 'close' column is numeric for analysis
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["high"] = pd.to_numeric(df["high"], errors="coerce")
            df["low"] = pd.to_numeric(df["low"], errors="coerce")
            return df  # Return the processed data

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return {"error": str(e)}
    def get_all_products(self, limit: int = 10, get_all_products: bool = True) -> list:
        """Fetch the list of all available products using the Coinbase brokerage REST API."""
        url = "https://api.coinbase.com/api/v3/brokerage/products"
        path = "/api/v3/brokerage/products"
        params = f"?limit={limit}&get_all_products={get_all_products}"  # Add query params

        # Generate the JWT for the request
        jwt_token = self.generate_jwt("GET", path)

        headers = {
            "Authorization": f"Bearer {jwt_token}"  # Set JWT in Authorization header
        }

        try:
            response = requests.get(url + params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["products"]
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def get_multiple_product_details_and_history(self, product_ids: list, granularity: str = "ONE_DAY") -> dict:
        """Fetch details and historical data for multiple products."""
        all_product_details = {}

        for product_id in product_ids:
            # Fetch product details
            product_details = self.get_product_details(product_id)

            # Fetch historical data
            historical_data = self.fetch_historical_data(InputParam(product_id=product_id, granularity=granularity))

            # Store the results in the dictionary
            all_product_details[product_id] = {
                "product_details": product_details,
                "historical_data": historical_data
            }

        return all_product_details

    def get_account_balance(self) -> dict:
        """Fetch the account balance."""
        url = f"https://api.coinbase.com/api/v3/brokerage/accounts"
        path = "/api/v3/brokerage/accounts"
        jwt_token = self.generate_jwt("GET", path)

        headers = {"Authorization": f"Bearer {jwt_token}"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()  # Return account balance details
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def place_order(self, product_id: str, side: str, size: float, price: float) -> dict:
        """Place an order on Coinbase."""
        url = f"https://api.coinbase.com/api/v3/brokerage/orders"
        path = "/api/v3/brokerage/orders"

        # Prepare the order details
        order_data = {
            "product_id": product_id,
            "side": side,  # 'buy' or 'sell'
            "size": size,
            "price": price,
            "type": "limit",  # We are using a limit order type here
            "time_in_force": "GTC"  # Good 'til canceled
        }

        jwt_token = self.generate_jwt("POST", path)
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, headers=headers, json=order_data)
            response.raise_for_status()
            return response.json()  # Return the order response
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

