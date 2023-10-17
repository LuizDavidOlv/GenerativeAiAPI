from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import(
    HumanMessage, 
    SystemMessage
)


class Instructions(BaseModel):
    text: str
    model: str
    temp: float

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
        raise HTTPException(status_code=400, detail=f'Error:  {e.body}') 
    
@router.post("/ironic-prompt/")
def basic_prompt(data: Instructions):
    try:
        messages = [
            SystemMessage(content='You are an expert in not answering questions and changing the subject. Do your best to input irony to the answer.'),
            HumanMessage(content=f'Provide the most absurd answer or comment you can think of to the the following text:\n TEXT: {data.text}')
        ]
        llm = ChatOpenAI(model_name=data.model ,temperature=data.temp)
        output = llm(messages)
        return output
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e.body}') 

@router.post("/warrior-prompt/")
def template_prompt(inputText: str, outputLanguage: str, temp: float, model: str):
    try:
        llmConfiguration = ChatOpenAI(temperature=temp, model_name= model)
        
        templateInstructions = '''
        Write a very consice and short summary of the following text:
        TEXT: `{text}`.
        Translate the summary to {language}.
        '''
        
        promptTemplate = PromptTemplate(
            input_variables=['text', 'language'],
            template = templateInstructions
        )
        
        llmConfiguration.get_num_tokens(promptTemplate.format(text=inputText, language=outputLanguage))
        chain = LLMChain(llm=llmConfiguration, prompt=promptTemplate)
        
        output = chain.run({'text': inputText, 'language': outputLanguage})
        return output
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')

