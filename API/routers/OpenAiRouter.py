import os
from pathlib import Path
from time import sleep

import openai
import pinecone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain import PromptTemplate
from langchain.chains import (ConversationalRetrievalChain, ConversationChain,
                              LLMChain, RetrievalQA)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory,
                              ConversationSummaryBufferMemory)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Pinecone
from openai import OpenAI
from pydantic import BaseModel

chat_history = []


class Instructions(BaseModel):
    text: str
    model: str
    temp: float


router = APIRouter(prefix="/openai", tags=["Open AI"])


@router.post("/text-davinci-003/")
def text_davinci_003(data: Instructions):
    llm = OpenAI(model_name="text-davinci-003", temperature=0.7, max_tokens=512)
    output = llm(data.instruction)
    return output


@router.post("/ironic-prompt/")
def basic_prompt(data: Instructions):
    messages = [
        SystemMessage(
            content="You are an expert in not answering questions and changing the subject. Do your best to input irony to the answer."
        ),
        HumanMessage(
            content=f"Provide the most absurd answer or comment you can think of to the the following text:\n TEXT: {data.text}"
        ),
    ]
    llm = ChatOpenAI(model_name=data.model, temperature=data.temp)
    output = llm(messages)
    return output.content


@router.post("/warrior-prompt/")
def template_prompt(inputText: str, outputLanguage: str, temp: float, model: str):
    llmConfiguration = ChatOpenAI(temperature=temp, model_name=model)

    templateInstructions = """
    Write a very consice and short summary of the following text:
    TEXT: `{text}`.
    Translate the summary to {language}.
    """

    promptTemplate = PromptTemplate(
        input_variables=["text", "language"], template=templateInstructions
    )

    llmConfiguration.get_num_tokens(
        promptTemplate.format(text=inputText, language=outputLanguage)
    )
    chain = LLMChain(llm=llmConfiguration, prompt=promptTemplate)

    output = chain.run({"text": inputText, "language": outputLanguage})
    return output


@router.post("/privatDbprompt/")
def privateDB_prompt(text: str):
    embeddings = OpenAIEmbeddings()
    pinecone_index = "my-life"
    pinecone.init(
        api_get=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENV"),
    )
    llmModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    vector_store = Pinecone.from_existing_index(pinecone_index, embeddings)
    retrieverModel = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llmModel, chain_type="stuff", retriever=retrieverModel
    )
    query = text
    answer = chain.run(query)
    return answer


@router.get("/stream-completion/")
async def stream():
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


@router.get("/completion/")
def completion(text: str):
    return StreamingResponse(stream(text), media_type="text/event-stream")


@router.post("/chat-completion/")
def chat_completion(text: str):
    completion = openai.ChatCompletion.create(
        model="text-davinci-003",
        prompt=f"{text}",
        temperature=0,
        max_tokens=100,
        stream=True,
        stop=["\n", "Human:", "AI:"],
    )
    return completion.choices[0].text


def stream(text: str):
    completion = openai.Completion.create(
        engine="text-davinci-003", prompt=text, stream=True
    )
    for line in completion:
        sleep(0.3)
        yield "data: %s\n\n" % line.choices[0].text


@router.post("/prompt-with-memory/")
def prompt_with_memory(index_name: str, question: str):
    llm = ChatOpenAI(temperature=1)
    vector_store = Pinecone.from_existing_index(index_name, OpenAIEmbeddings())
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))

    return result["answer"]


@router.post("/conversation-buffer-memory/")
def prompt_with_memory(question: str):
    llmConfig = ChatOpenAI(temperature=0)
    memoryConfig = ConversationBufferMemory()
    conversation = ConversationChain(llm=llmConfig, memory=memoryConfig)

    result = conversation.predict(input=question)
    return result
