from routers import OpenAiRouter, PineconeRouter
from fastapi import FastAPI, FastAPI
from dotenv import load_dotenv, find_dotenv
import pinecone
import os

load_dotenv(find_dotenv(), override=True)

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV')
)

app = FastAPI(
    title="My API",
    description="This is a very fancy API",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    swagger_ui=True  # This line enables Swagger UI
)

app.include_router(OpenAiRouter.router)
app.include_router(PineconeRouter.router)

# Add the Swagger middleware to the app
#app.add_middleware(SwaggerMiddleware)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}