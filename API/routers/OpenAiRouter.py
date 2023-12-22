from pathlib import Path
from fastapi.responses import StreamingResponse
import openai
import os
import pinecone
from time import sleep
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import(
    HumanMessage, 
    SystemMessage
)
from openai import OpenAI

chat_history = []
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


@router.get("/stream-completion/")
async def stream():
    try:
        async with openai.StreamingCompletion.create(
            stop=["\n", "Human:", "AI:"],
            model="curie",
            temperature=0.9,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            best_of=1,
            n=1,
            stream=True,
        ) as stream:
            async for chunk in stream:
                yield chunk
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')
    
@router.get("/completion/")
def completion(text: str):
    return StreamingResponse(stream(text), media_type="text/event-stream")
    

@router.post("/chat-completion/")
def chat_completion(text: str):
    try:
        completion = openai.ChatCompletion.create(
            model= "text-davinci-003",
            prompt=f'{text}',
            temperature=0,
            max_tokens=100,
            stream = True,
            stop=["\n", "Human:", "AI:"],
        )
        return completion.choices[0].text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')
    

def stream(text: str):
    try:
        completion = openai.Completion.create(engine="text-davinci-003", prompt=text, stream=True)
        for line in completion:
            sleep(0.3)
            yield 'data: %s\n\n' % line.choices[0].text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')
    

@router.post("/prompt-with-memory/")
def prompt_with_memory(index_name: str, question: str):
    try:
        llm = ChatOpenAI(temperature = 1)
        vector_store = Pinecone.from_existing_index(index_name,OpenAIEmbeddings())
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        crc = ConversationalRetrievalChain.from_llm(llm, retriever)
        result = crc({'question': question, 'chat_history': chat_history})
        chat_history.append((question, result['answer']))

        return result['answer']
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')
    
@router.post("/audio-prompt/")
def audio_prompt(fileName: str = "audio1"):
    try:
        client = OpenAI()
        audio_path = Path(f'./API/Audio/{fileName}.mp3')        
        audio_file = open(audio_path, "rb")

        # with audio_path.open("rb") as audio_file:
        #     audio_data = audio_file.read()
            
           

        transcript = OpenAI().audio.transcriptions.create(
            model = "whisper-1",
            file = audio_file,
            response_format = "text",
        )
        return transcript

            

        
        # with audio_path.open("rb") as audio_file:
        #     audio_file = audio_file.read()
        #     transcript = OpenAI().audio.transcriptions.create(
        #         model = "whisper-1",
        #         file = audio_file,
        #         response_format = "text",
        #     )
        #     print("Transcript created")

        #     system_prompt = "You are a helpful assistant for the company that is trying to help the customer. The customer should never get a direct answer. Please avoid ansering it at all cost."
        #     print("System prompt created")

        #     corrected_transcript = generate_corrected_transcript(0.7, system_prompt, transcript)
        #     print("Corrected transcript created")

        #     audioResponse = OpenAI.audio.speech.create(
        #         model = "tts-1",
        #         voice= "nova",
        #         input = corrected_transcript,
        #     )
        #     print("Audio response created")

        #     audioResponse.download("../Audio/{fileName}Response.mp3")
        #     print("Audio response downloaded")

        # return True
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}')
    # except HTTPException as http_err:
    #     raise http_err
    # except FileNotFoundError:
    #     raise HTTPException(status_code=404, detail="File not found")
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def generate_corrected_transcript(temperature, system_prompt, audio_file):
    response = OpenAI().chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcribe(audio_file, "")
            }
        ]
    )
    return response['choices'][0]['message']['content']