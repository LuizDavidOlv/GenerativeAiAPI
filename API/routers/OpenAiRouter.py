import openai
import os
import pinecone
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
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
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 
    
@router.post("/ironic-prompt/")
def basic_prompt(data: Instructions):
    try:
        messages = [
            SystemMessage(content='You are an expert in not answering questions and changing the subject. Do your best to input irony to the answer.'),
            HumanMessage(content=f'Provide the most absurd answer or comment you can think of to the the following text:\n TEXT: {data.text}')
        ]
        llm = ChatOpenAI(model_name=data.model ,temperature=data.temp)
        output = llm(messages)
        return output.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 

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
    
@router.post("/privatDbprompt/")
def privateDB_prompt(text: str):
    embeddings = OpenAIEmbeddings()
    pinecone_index = 'my-life'
    pinecone.init(api_get=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    llmModel = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    vector_store =  Pinecone.from_existing_index(pinecone_index, embeddings)
    retrieverModel = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
    chain = RetrievalQA.from_chain_type(llm=llmModel, chain_type='stuff',retriever=retrieverModel)
    query = text
    answer = chain.run(query)
    return answer
    
@router.get("/fine-tune/list-jobs/")
def fine_tune_list_jobs():
    try:
        return openai.FineTuningJob.list(limit=10)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')


@router.post("/fine-tune/list-job/")
def fine_tune_list_job(job_id: str):
    try:
        return openai.FineTuningJob.retrieve(job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')

@router.post("/fine-tune/list-events")
def fine_tune_list_events(job_id: str):
    try:
        return openai.FineTuningJob.list_events(job_id, limit=10)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')
    
@router.post("/fine-tune/create-job/")
def fine_tune_create_job():
    try:
        file = openai.File.create(
            file=open("./fineTuneData.jsonl","rb"),
            purpose="fine_tune",       
        )
        openai.FineTunningJob.create(training_file=file.id, model="gpt-3.5-turbo")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')

@router.post("/fine-tune/use-model/")
def fine_tune_use_model(text: str):
    try:
        completion = openai.ChatCompletion.create(
            model = "ft:gpt-3.5-turbo-0613:personal::7wGbxpsA",
            messages = [
                {"role": "system", "content": "You are a very sarcastic person. that only says bad things."},
                {"role": "user", "content": f'{text}'}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')

