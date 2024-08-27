import asyncio
import operator
import os
from typing import Annotated, List, TypedDict

import aiosqlite
from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from IPython.display import Image
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (AnyMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

from Prompts.EssayWriterPrompts import (PLAN_PROMT, REFLECTION_PROMPT,
                                        RESEARCH_CRITIQUE_PROMPT,
                                        RESEARCH_PLAN_PROMPT, WRITER_PROMPT)
from Prompts.sql_prompt_template import SQL_GENERATION_TEMPLATE
