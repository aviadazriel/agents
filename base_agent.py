from abc import ABC, abstractmethod
from typing import List, Optional, Any

from langchain.llms.base import BaseLLM
from langchain.tools.base import BaseTool


class BaseAgent(ABC):
    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Initialize the base agent.
        
        Args:
            llm: Language model to use (optional)
        """
        self.llm = llm
        self._initialize_components()
    
    @abstractmethod
    def _initialize_components(self) -> None:
        """Initialize agent components. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_tools(self) -> List[BaseTool]:
        """Get the list of tools available to the agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """
        Run a query through the agent.
        
        Args:
            query: The user's query
            **kwargs: Additional arguments specific to the agent type
        
        Returns:
            str: The agent's response
        """
        pass 