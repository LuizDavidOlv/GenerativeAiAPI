import os

from dotenv import find_dotenv, load_dotenv
from fastapi import APIRouter, HTTPException
from huggingface_hub import hf_hub_download
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline

load_dotenv(find_dotenv(), override=True)

huggingFaceTokken = os.getenv("HUGGING_FACE_KEY")

router = APIRouter(prefix="/huggingface", tags=["Hugging Face"])


@router.post("/hugging-face-llm/download/")
async def hugging_face_llms(model_id: str = "lmsys/fastchat-t5-3b-v1.0"):
    try:
        fileNames = [
            "added_tokens.json",
            "config.json",
            "generation_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "spiece.model",
        ]

        for fileName in fileNames:
            downloaded_model_path = hf_hub_download(
                repo_id=model_id, filename=fileName, token=huggingFaceTokken
            )
            print(downloaded_model_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/hugging-face-llm/use")
async def hugging_face_llm_use(
    model_id: str = "lmsys/fastchat-t5-3b-v1.0", text: str = ""
):
    llmConfig = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text2text-generation",
        model_kwargs={"temperature": 0, "max_length": 1000},
    )

    templateConfig = """
    You are a friendly chatbot assistant that responds conversationally to users' questions.
    Keep the answers short, unless specifically asked by the user to elaborate on something.

    Question: {question}

    Answer:"""

    promptConfig = PromptTemplate(template=templateConfig, input_variables=["question"])
    llm_chain = LLMChain(prompt=promptConfig, llm=llmConfig)

    return llm_chain(text)
