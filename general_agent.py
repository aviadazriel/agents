import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory

from base_agent import BaseAgent
from tools import StockPriceTool, WeatherTool, search_tool

class AIAgent(BaseAgent):
    def __init__(self, openai_api_key: Optional[str] = None, serper_api_key: Optional[str] = None):
        """
        Initialize the AI Agent.
        
        Args:
            openai_api_key: OpenAI API key (optional)
            serper_api_key: Serper API key (optional)
        """
        # Load environment variables if keys not provided
        load_dotenv()
        
        # Set API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        
        if not self.openai_api_key or not self.serper_api_key:
            raise ValueError("OpenAI API key and Serper API key are required")
        
        # Set Serper API key for the environment
        os.environ["SERPER_API_KEY"] = self.serper_api_key
        
        # Initialize LLM before calling parent
        llm = OpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key
        )
        
        # Call parent constructor
        super().__init__(llm=llm)
        
    def _initialize_components(self) -> None:
        """Initialize the LLM, tools, memory, and agent components."""
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=2
        )
        
        # Initialize tools
        self.tools = self._get_tools()
        
        # Initialize agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )
    
    def _get_tools(self) -> List[Tool]:
        """Get the list of tools available to the agent."""
        return [
            StockPriceTool(),
            WeatherTool(),
            search_tool
        ]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the general agent."""
        return """
        give a very details answer for any input and its must to be correct

        <question>
        {query}
        </question>
        """
    
    def run(self, query: str, detailed: bool = True) -> str:
        """
        Run a query through the agent.
        
        Args:
            query (str): The user's query
            detailed (bool): Whether to provide a detailed response
        
        Returns:
            str: The agent's response
        """
        if detailed:
            prompt = self._get_system_prompt().format(query=query)
        else:
            prompt = query
            
        return self.agent.run(prompt)