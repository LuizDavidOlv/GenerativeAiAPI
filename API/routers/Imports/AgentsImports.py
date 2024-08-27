import os
import time

import openai
from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter
from langchain import hub
from langchain.agents import (AgentExecutor, AgentType, create_react_agent,
                              create_sql_agent, initialize_agent, load_tools,
                              tool)
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents.format_scratchpad import \
    format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.sql_database import SQLDatabase
from langchain.tools import Tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.utilities import SerpAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from Prompts.sql_prompt_template import SQL_GENERATION_TEMPLATE
