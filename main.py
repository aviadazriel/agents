
import os

from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=2)


# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
os.environ["SERPER_API_KEY"]  = SERPER_API_KEY
from tools import StockPriceTool, WeatherTool, search_tool

# Initialize the language model
llm = OpenAI(temperature=0)

# Load basic tools
# tools = load_tools(["wikipedia", "llm-math"], llm=llm)
# Create an agent with custom tools
custom_tools = [
    StockPriceTool(),
WeatherTool(),
search_tool
]

agent_with_custom_tools = initialize_agent(
    custom_tools ,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
# handle_parsing_errors=True  # Allows the agent to retry when parsing fails

    memory=memory
)

# Test custom tool
# result = agent_with_custom_tools.run("What is the current price of cyber ark stock?")
# print(result)
# result = agent_with_custom_tools.run("What is the weather in each on of the 5 biggest israel cities?"
#                                      "give me the result in 5 lines for each city"
#                                      )


prompt = """
give a very details answer for any input and its must to be correct

<question>

"""

query = ("can you summary for me the latest news on nvidia from the last 3 days?")
query = ("please explain to me about langchain")
query = ("when Elon Mask born? and how old Elon Mask was in 2022?")

prompt  = prompt.replace("<question>", query)
result = agent_with_custom_tools.run(prompt)

print(result)