import os
import time
import openai
from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter
from langchain.agents import tool, AgentExecutor, initialize_agent, AgentType, load_tools
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMMathChain
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_sql_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.schema import(
    HumanMessage, 
    SystemMessage
)
from Prompts.sql_prompt_template import SQL_GENERATION_TEMPLATE
