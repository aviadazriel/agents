# AI Agent Assistant

An intelligent agent built with LangChain that can perform various tasks including stock price checking, weather information retrieval, and web searches.

## Features

- **Stock Price Checking**: Get real-time stock prices using Yahoo Finance
- **Weather Information**: Retrieve current weather conditions for any city
- **Web Search**: Perform Google searches using Serper API
- **Conversation Memory**: Maintains context of the conversation using a buffer window memory

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Serper API key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

## Usage

Run the main script:
```bash
python main.py
```

The agent can handle various queries such as:
- Stock price information: "What is the current price of [stock symbol]?"
- Weather updates: "What is the weather in [city]?"
- General information: The agent can search and provide detailed answers about various topics

## Project Structure

- `main.py`: Main application file with agent initialization and configuration
- `tools.py`: Custom tool implementations for stocks, weather, and search functionality
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)

## Dependencies

Key dependencies include:
- langchain
- langchain-openai
- langchain-community
- openai
- yfinance
- requests
- python-dotenv
- wikipedia
- google-serper
- pydantic

## Note

Make sure to keep your API keys secure and never commit them to version control.