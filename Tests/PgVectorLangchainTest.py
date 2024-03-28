import unittest
from unittest.mock import patch
from fastapi import Response
import requests_mock
from API.routers import PgVectorSqlAlchemyRouter
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

class TestEndpoint(unittest.TestCase):

    def test_get_method(self):
        load_dotenv(find_dotenv(), override=True)
        embeddingHuggingFace = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        embeddingOpenAi = OpenAIEmbeddings()
        conn_string = os.getenv("PGVECTOR_CON_STRING")

        self.assertEqual(300, 200)

# Run the test suite
if __name__ == '__main__':
    unittest.main()