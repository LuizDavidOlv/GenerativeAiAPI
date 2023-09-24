import os
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import pinecone
#from .Models.PineconeCreateIndexModel import CreateIndexModel

load_dotenv(find_dotenv(), override=True)
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV')
)

class Instructions(BaseModel):
    instruction: str

app = FastAPI(
    title="My API",
    description="This is a very fancy API",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    swagger_ui=True  # This line enables Swagger UI
)

@app.get("/version/")
def version():
    try:
        return pinecone.info.version()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 

@app.get("/list-indexes/")
def text_davinci_003():
    try:
        return pinecone.list_indexes()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 

@app.post("/describe-index/")
def describe_index(index_name: str):
    try:
        return pinecone.describe_index(index_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')

# @app.post("/create-index/")
# def create_index(data: CreateIndexModel):
#     try:
#         return pinecone.create_index(data.index_name, dimensions= data.dimension, metrix = data.metric, pods= 1, pod_type='p1.x2')
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f'Error:  {e}') 
    

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi