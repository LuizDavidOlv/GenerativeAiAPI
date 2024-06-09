from fastapi import APIRouter
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from IPython.display import Image
from dotenv import load_dotenv, find_dotenv