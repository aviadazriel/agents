
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain_community.docstore import Wikipedia
import yfinance as yf
from langchain.tools import BaseTool
from typing import Optional, Type

from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
import requests

# Define input schema for our custom tool
class StockPriceCheckInput(BaseModel):
    symbol: str = Field(description="Stock symbol to check")

class StockPriceTool(BaseTool):
    name: str = "stock_price_checker"
    description: str = "Get the current stock price using Yahoo Finance"
    args_schema: Type[BaseModel] = StockPriceCheckInput
    def _run(self, symbol: str) -> str:
        try:
            ticker = yf.Ticker(symbol)
            # Fetch live market price
            price = ticker.info.get('regularMarketPrice')
            if price is None:
                return f"Could not fetch price for symbol '{symbol}'."
            return f"The current price of {symbol} is ${price:.2f}."
        except Exception as e:
            return f"Error fetching price for {symbol}: {e}"

    def _arun(self, symbol: str):
        raise NotImplementedError("Async not implemented")



search = GoogleSerperAPIWrapper()
# Define the tool for search
search_tool = Tool(
    name="Google Search",
    func=search.run,
    description="Use this tool when you need to search for real-time information from Google."
)


# Define input schema for weather tool
class WeatherCheckInput(BaseModel):
    city: str = Field(description="City name to get the weather for, e.g. London")

# Weather tool using wttr.in (free, no API key required)
class WeatherTool(BaseTool):
    name: str = "weather_checker"
    description: str = "Get current weather for a specified city using wttr.in free API"
    args_schema: Type[BaseModel] = WeatherCheckInput

    def _run(self, city: str) -> str:
        try:
            url = f"http://wttr.in/{city}?format=j1"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            current = data["current_condition"][0]
            temp = current["temp_C"]
            desc = current["weatherDesc"][0]["value"]
            humidity = current["humidity"]
            return (
                f"Current weather in {city}: {desc}, "
                f"temperature: {temp}°C, humidity: {humidity}%."
            )
        except Exception as e:
            return f"Error fetching weather for {city}: {e}"

    async def _arun(self, city: str):
        raise NotImplementedError("Async not implemented")


get_db_tools(dbPath, llm):
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri(dbPath) #"sqlite:///Chinook.db"

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()

    for tool in tools:
        print(f"{tool.name}: {tool.description}\n")

    return tools