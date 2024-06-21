from fastapi import APIRouter
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated,List
from langgraph.graph import StateGraph, END
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from IPython.display import Image
from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import asyncio
import aiosqlite
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from fastapi.responses import StreamingResponse
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
import os
from Prompts.sql_prompt_template import SQL_GENERATION_TEMPLATE
from Prompts.EssayWriterPrompts import PLAN_PROMT, WRITER_PROMPT, REFLECTION_PROMPT, RESEARCH_PLAN_PROMPT, RESEARCH_CRITIQUE_PROMPT