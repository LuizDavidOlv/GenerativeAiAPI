import datetime
import logging
import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from openai import OpenAI
import pandas as pd
import psycopg2
import pytz
from fastapi import APIRouter, HTTPException
import sqlalchemy

from API.Prompts.promp_template import SQL_GENERATION_TEMPLATE
from ../Prompts/promp_template import SQL_GENERATION_TEMPLATE
from langchain.prompts import load_prompt, PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import(
    HumanMessage, 
    SystemMessage
)



router = APIRouter(
    prefix="/querysqlserver",
    tags=["Query SQL Server"]
)

@router.post("/query-sql-server/")
async def query_sql_server(question: str):
    try:
        sqlquery = await database_lookup_internal(question,"langchain")
        messages = [
            SystemMessage(content='You are an expert on providing Dell Financial System information. Give a precise and accurate answer to the question below.Do not provide any additional information.'),
            HumanMessage(content=f'Provide the answer to the question {question} based on the following SuccessfulUserActivity value:\n TEXT: {sqlquery}')
        ]
        llm = OpenAI(model_name='text-davinci-003',temperature=1, max_tokens=512)
        output = llm(messages)
        return output.content
    except Exception as e:
        logging.error('An error occurred while querying the database: %s', str(e))
        return False


def queryDB(text):
    #TODO: Implement this method
    return "Method not implemented"
    gcp = sqlDb()
    prompt_template = load_prompt("C:\\Git\\Gen AI\\genai-api\\api\\db\\dbStructure.yaml")

    sqlStringConnection = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=" + data["host"] + ";DATABASE=" + data["dbname"] + ";UID=" + data["username"] + ";PWD=" + data["password"] 
    db =SQLDatabase.from_uri(f"sqlite:///?odbc_connect={sqlStringConnection}")
    model = Azure.chatopenai()
    agent_executor = create_sql_agent(
        llm = model,
        verbose = True,
        db = db,
        prompt = prompt_template
    )
    result = agent_executor.invoke(text)
    return result


@staticmethod
async def database_lookup_internal(query: str, vdb_collection_name: str):

    llm, sql_server_store, pg_vector_store = get_initial_config(vdb_collection_name)

    operationTypesChain = await generate_sql_chain("Get all OPERTN_NM types",1, llm, pg_vector_store)
    operationTypesQueryResult = await get_query_result(operationTypesChain, sql_server_store)


    messages = [
        SystemMessage(content='You are an expert in getting the correct operation type based on the user question. Only provide the operation type that is most similar to the one in the question itself. Do not provide any additional information.'),
        HumanMessage(content=f'Provide the most similar type name among all the following types: {operationTypesQueryResult} ,for the provided question {query}')
    ]

    operation_type = llm(messages).content
    uncheckedSqlQueryChain = await generate_sql_chain(query, 5, llm, pg_vector_store)
    
    messages = [
        SystemMessage(content='You are an expert on correcting the microsoft sql server query. Only provide the query itself. Do not provide any additional information.'),
        HumanMessage(content=f'Provide the correct query. OPERTN_NM should be equal to {operation_type} in the where clause on the following query: {uncheckedSqlQueryChain}')
    ]

    llmUpdatedQuery = llm(messages).content
    revisedSqlQueryChain = await generate_sql_chain(llmUpdatedQuery, 5, llm, pg_vector_store)
    revisedSqlQueryResult = await get_query_result(revisedSqlQueryChain, sql_server_store)

    return revisedSqlQueryResult

    
@staticmethod
async def query_sql(sql_query, aid_engine):
    try:
        with aid_engine.connect() as conn:
            dataframe: pd.DataFrame = pd.read_sql(sqlalchemy.text(sql_query), con=conn)
            if dataframe.shape[0] > 100:
                logging.warning('Generated SQL query resulted in a dataset larger than 100 rows. This dataset has been truncated.')
        if dataframe.shape[0] == 0:
            result = 'No records were found in the AID database for the given query.'
        else:
            result = dataframe.head(100).to_markdown(index=False, floatfmt=",.2f")
            return result
        
    except Exception as e:
        result = f"An error occurred while executing the SQL query: {str(e)}"
        raise Exception(result)
    
@staticmethod
async def get_query_result(sql_query, aid_engine):
    queryResult = None

    tries = 0
    while not queryResult:
        queryResult = await query_sql(sql_query, aid_engine)
        tries += 1
        if "error" in queryResult:
            queryResult = None
        else:
            return queryResult
        if tries >=5:
            return False
        
@staticmethod
async def generate_sql_chain(query, kwargs, llm, pg_vector_store):
    documents = pg_vector_store.similarity_search(
            query=query, 
            k= kwargs
    )
    sql_generation_prompt = PromptTemplate(input_variables=['task', 'context'], template=SQL_GENERATION_TEMPLATE)
    database_schema = '\n-------------------------------------------------------------------------------------------------------------------------\n'.join(  document.page_content for document in documents )
    sql_generation_chain = LLMChain(llm=llm, prompt=sql_generation_prompt, verbose=True)
    dts = datetime.datetime.now(pytz.timezone('US/Central')).strftime('%Y-%m-%d %H:%M:%S')
    chain_output = sql_generation_chain.run({'task': query, 'context': database_schema, 'datetime': dts}, metadata={'chain': 'sql_generation_chain'})
    return chain_output

@staticmethod
def get_initial_config(vdb_collection_name):
    llm = OpenAI(model_name='text-davinci-003',temperature=1, max_tokens=512)
    embeddings = HuggingFaceEmbeddings(model_name='sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja')

    pg_vector_conn_string = os.getenv("PGVECTOR_CON_STRING")
    sql_server_conn_string = os.getenv("SQL_SERVER_CON_STRING")

    sql_server_store = sqlalchemy.create_engine(sql_server_conn_string, future=True)
    pg_vector_store = PGVector(
        connection_string=pg_vector_conn_string, 
        embedding_function=embeddings,
        collection_name=vdb_collection_name
    )

    return llm, sql_server_store, pg_vector_store


