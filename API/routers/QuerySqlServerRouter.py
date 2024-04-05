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

from API.Prompts.messages import Messages
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
async def query_sql_server(question: str, collection_name: str = 'langchain'):
    try:
        llm = OpenAI(model_name='text-davinci-003',temperature=1, max_tokens=512)
        sql_query_result = await database_lookup_internal(question, collection_name)

        if not sql_query_result:
            return "Error in retriving sql server information."
        
        politeDfsEmployeeMessageConfig = Messages.get_dfs_employee_messages(question, sql_query_result)
        result = llm(politeDfsEmployeeMessageConfig)

        return result.content
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
    llm, sqlServerStore, pgVectorStore = get_initial_config(vdb_collection_name)

    operationTypesChain = await generate_sql_chain("Get all OPERTN_NM types",1, llm, pgVectorStore)
    operationTypesQueryResult = await get_query_result(operationTypesChain, sqlServerStore)

    operationTypeMessagesConfig = Messages.get_operation_type_messages(operationTypesQueryResult,query)
    operation_type = llm(operationTypeMessagesConfig).content
    uncheckedSqlQueryChain = await generate_sql_chain(query, 5, llm, pgVectorStore)
    
    countOperationMessagesConfig = Messages.get_count_operation_messages(operation_type,uncheckedSqlQueryChain)
    llmUpdatedQuery = llm(countOperationMessagesConfig).content
    revisedSqlQueryChain = await generate_sql_chain(llmUpdatedQuery, 5, llm, pgVectorStore)
    revisedSqlQueryResult = await get_query_result(revisedSqlQueryChain, sqlServerStore)

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


