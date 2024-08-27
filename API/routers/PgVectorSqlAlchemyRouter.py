# from fastapi import APIRouter
# from pgvector.sqlalchemy import Vector
# from sqlalchemy import create_engine, Text
# from sqlalchemy.orm import Session, declarative_base, mapped_column
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import TextLoader
# from dotenv import load_dotenv, find_dotenv
# from typing import List
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# import os
# import uuid
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores.pgvector import PGVector

# router = APIRouter(
#     prefix="/pg-vector",
#     tags=["PG-Vector"]
# )

# load_dotenv(find_dotenv(), override=True)

# conn_string = os.getenv("PGVECTOR2_CON_STRING")

# engine = create_engine(conn_string)
# session = Session(engine)

# @router.get("/create-table/")
# def create_table(table_name: str, vector_size: int = 768):
#     try:
#         Base = declarative_base()

#         class Item(Base):
#             __tablename__ = table_name
#             id = mapped_column(Text, primary_key=True)
#             embedding = mapped_column(Vector(vector_size))

#         Base.metadata.create_all(engine)
#         session.commit()
#         session.close()
#         return session.query(Item).all()
#     except Exception as e:
#         session.rollback()
#         raise Exception(str(e))

# @router.delete("/delete-table/")
# def delete_table(table_name: str):
#     try:
#         Base = declarative_base()

#         class Item(Base):
#             __tablename__ = table_name
#             id = mapped_column(Text, primary_key=True)
#             embedding = mapped_column(Vector(768))

#         Base.metadata.drop_all(engine)
#         session.commit()
#         session.close()
#         return session.query(Item).all()
#     except Exception as e:
#         session.rollback()
#         raise Exception(str(e))

# @router.post("/insert-into-table/")
# def insert_into_table(table_name: str, vector_size: int = 768):
#     try:
#         documentsNameList = load_documents_from_directory(Path('./API/TextDocuments/'))
#         documentsText = load_documents(documentsNameList)
#         chunkedText = chunk_text(documentsText)
#         embededChunks = embed_text(chunkedText , 'all-mpnet-base-v2')

#         Base = declarative_base()
#         class Item(Base):
#             __tablename__= table_name

#             id = mapped_column(Text, primary_key=True)
#             embedding = mapped_column(Vector(vector_size))

#         for vector in embededChunks:
#             session.add(Item(id=str(uuid.uuid4()), embedding=vector))

#         session.commit()
#         session.close()
#     except Exception as e:
#         session.rollback()
#         raise Exception(str(e))

# @router.post("/insert-into-vector-db/")
# def insert_into_vector_db(text: str, table_name: str, vector_size: int = 1536, embedding_name: str = 'sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja'):
#     try:
#         # query_md = open("C:\\Git\\Gen AI\\genai-api\\api\db\\ms_sql_query_example.md", "r")
#         # query = query_md.read()
#         conn_string = os.getenv("PGVECTOR_CON_STRING")
#         embeddings = HuggingFaceEmbeddings(model_name = embedding_name)

#         pg_vector_store = PGVector(
#             connection_string=conn_string,
#             embedding_function=embeddings
#         )

#         pg_vector_store.add_texts([text])
#         return True
#     except Exception as e:
#         raise Exception(str(e))

# def load_documents_from_directory(directory_path: str):
#     documentsPathList = []

#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)

#         if os.path.isfile(file_path) and filename.lower().endswith('.txt'):
#             documentsPathList.append(file_path)

#     return documentsPathList

# def load_documents(fileNameList: List[str]):
#     documentsText = ""
#     for file in fileNameList:
#         loader = TextLoader(file)
#         document = loader.load()

#         documentsText += document[0].page_content
#         documentsText += " "

#     return documentsText

# def chunk_text(text: str, chunkSize: int = 100, chunkOverlap: int = 10):
#     text_splitter= RecursiveCharacterTextSplitter(
#         chunk_size=chunkSize,
#         chunk_overlap=chunkOverlap,
#     )
#     chunkedDocuments = text_splitter.split_text(text)
#     return chunkedDocuments

# def embed_text(chunked_docs: List[str], model_name: str):
#     model = SentenceTransformer(model_name)
#     vectors = model.encode(chunked_docs)
#     return vectors
