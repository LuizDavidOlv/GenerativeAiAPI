from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from langchain.llms import OpenAI


class Instructions(BaseModel):
    instruction: str

router = APIRouter(
    prefix="/openai",
    tags=["Open AI"]
)

@router.post("/text-davinci-003/")
def text_davinci_003(data: Instructions):
    try:
        llm = OpenAI(model_name='text-davinci-003',temperature=0.7, max_tokens=512)
        output = llm(data.instruction)
        return output
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 
    
@router.post("/gpt-3dot5-turbo/")
def gpt_3dot5_turbo(data: Instructions):
    try:
        llm = OpenAI(model_name='text-davinci-003',temperature=0.7, max_tokens=512)
        output = llm(data.instruction)
        return output
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 
