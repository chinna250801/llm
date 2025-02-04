from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Settings(BaseSettings):
    coinbase_api_key: str
    coinbase_api_secret: str
    open_api_key: str


# Initialize the settings instance
settings = Settings()
