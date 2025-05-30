from typing import List, Optional, Any
from langgraph.prebuilt import create_react_agent
from langchain.llms.base import BaseLLM
from langchain.tools.base import BaseTool
from tools import get_db_tools
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from base_agent import BaseAgent

class DBAgent(BaseAgent):
    def __init__(
        self,
        db_url: str,  # Database connection/instance
        openai_api_key: Optional[str] = None ,
        top_k: int = 5
    ):
        """
        Initialize the Database Agent.
        
        Args:
            llm: Language model to use
            db: Database connection/instance
            tools: List of tools for database operations
            top_k: Maximum number of results to return (default: 5)
        """
        self.db = db
        self.top_k = top_k
        self.db_url = db_url

        load_dotenv()

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        llm = OpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key
        )
        
        # Call parent constructor
        super().__init__(llm=llm)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the database agent."""
        return """
        You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct {dialect} query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most {top_k} results.

        You can order the results by a relevant column to return the most interesting
        examples in the database. Never query for all the columns from a specific table,
        only ask for the relevant columns given the question.

        You MUST double check your query before executing it. If you get an error while
        executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
        database.

        To start you should ALWAYS look at the tables in the database to see what you
        can query. Do NOT skip this step.

        Then you should query the schema of the most relevant tables.
        """.format(
            dialect=self.db.dialect,
            top_k=self.top_k,
        )
    
    def _get_tools(self) -> List[BaseTool]:
        """Get the list of tools available to the agent."""
        return get_db_tools(self.db_url, self.llm)
    
    def _initialize_components(self) -> None:
        """Initialize the database agent with tools and system prompt."""
        self.tools = self._get_tools()
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._get_system_prompt(),
        )
    
    def run(self, query: str) -> str:
        """
        Run a query through the database agent.
        
        Args:
            query (str): The user's query about the database
        
        Returns:
            str: The agent's response after querying the database
        """
        return self.agent.invoke({"input": query})

def create_db_agent(
    llm: BaseLLM,
    db: Any,
    tools: List[BaseTool],
    top_k: int = 5
) -> DBAgent:
    """
    Factory function to create a database agent.
    
    Args:
        llm: Language model to use
        db: Database connection/instance
        tools: List of tools for database operations
        top_k: Maximum number of results to return
    
    Returns:
        DBAgent: Initialized database agent
    """
    return DBAgent(llm=llm, db=db, tools=tools, top_k=top_k)