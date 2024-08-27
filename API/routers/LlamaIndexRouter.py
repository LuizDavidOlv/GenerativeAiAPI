from fastapi import APIRouter
from llama_index.core import (Settings, SimpleDirectoryReader, SummaryIndex,
                              VectorStoreIndex)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import FilterCondition, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from loguru import logger

router = APIRouter(prefix="/llama-index", tags=["LlamaIndex"])


@router.post("/router")
async def summary(question: str):
    llama_router = LlamaRouter()
    response = llama_router.query(question)
    return response


@router.post("/tool-calling")
async def tool_calling(question: str = "What is a summary of the paper?"):
    tool_calling = ToolCalling()
    response = tool_calling.query(question)
    return response


class LlamaRouter:
    def query(self, question: str):
        paper1 = "API/DocumentFiles/AttentionIsAllYouNeed.pdf"
        documents = SimpleDirectoryReader(input_files=[paper1]).load_data()
        logger.info("Document loaded successfully")

        splitter = SentenceSplitter(chunk_size=1024)
        logger.info("Sentence splitter initialized")
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info("Nodes extracted from document")

        Settings.llm = OpenAI(model="gpt-3.5-turbo")
        logger.info("OpenAI model loaded successfully")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
        logger.info("OpenAI Embedding model loaded successfully")

        summary_index = SummaryIndex(nodes)
        logger.info("Summary index created successfully")
        vector_index = VectorStoreIndex(nodes)
        logger.info("Vector index created successfully")

        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize", use_async=True
        )
        logger.info("Summary query engine created successfully")
        vector_query_engine = vector_index.as_query_engine()
        logger.info("Vector query engine created successfully")

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description="Useful for question summarization",
        )
        logger.info("Summary tool created successfully")
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Useful for retrieving specific content",
        )
        logger.info("Vector tool created successfully")

        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool, vector_tool],
            verbose=True,
        )
        logger.info("Router query engine created successfully")

        response = query_engine.query(question)

        return response


class ToolCalling:
    def query(self, question: str):
        paper1 = "API/DocumentFiles/AttentionIsAllYouNeed.pdf"
        documents = SimpleDirectoryReader(input_files=[paper1]).load_data()
        logger.info("Document loaded successfully")

        splitter = SentenceSplitter(chunk_size=1024)
        logger.info("Sentence splitter initialized")
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info("Nodes extracted from document")

        llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        logger.info("OpenAI model loaded successfully")

        summary_index = SummaryIndex(nodes)
        logger.info("Summary index created successfully")
        vector_index = VectorStoreIndex(nodes)
        logger.info("Vector index created successfully")

        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize", use_async=True
        )
        logger.info("Summary query engine created successfully")
        vector_query_engine = vector_index.as_query_engine(similarity_top=2)
        logger.info("Vector query engine created successfully")

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description="Useful for documents summarization",
        )
        logger.info("Summary tool created successfully")

        vector_query_tool = FunctionTool.from_defaults(
            name="vector_tool", fn=self.vector_query
        )
        logger.info("Vector query tool created successfully")

        response = llm.predict_and_call(
            [vector_query_tool, summary_tool], question, verbose=True
        )

        return response

    def vector_query(self, query: str, vector_index, page_numbers: list[str]) -> str:
        """Perform a vector search over an index

        query (str): the string query to be embedded.
        index: the vector index that contains the vectors to be searched. Always get this from VectorStoreIndex(nodes)
        page_numbers (list[str]): Filter by set of pages. Leave BLANK  if we want to perform a vector search
            over all pages.Otherwise, filter by the set of specified pages.
        """
        metadata_dicts = [{"key": "page_label", "value": page} for page in page_numbers]

        query_engine = vector_index.as_query_engine(
            similarity_top=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts, condition=FilterCondition.OR
            ),
        )

        response = query_engine.query(query)
        return response
