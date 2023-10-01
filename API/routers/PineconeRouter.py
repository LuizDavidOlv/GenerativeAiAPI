import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pinecone
#from .Models.PineconeCreateIndexModel import CreateIndexModel

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV')
)

class Instructions(BaseModel):
    instruction: str

router = APIRouter(
    prefix="/pinecone"
)

@router.get("/version/")
def version():
    try:
        return pinecone.info.version()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 

@router.get("/list-indexes/")
def text_davinci_003():
    try:
        return pinecone.list_indexes()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 

@router.post("/describe-index/")
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


