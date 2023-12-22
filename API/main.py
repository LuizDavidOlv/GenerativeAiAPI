from fastapi.responses import HTMLResponse
from API.routers import OpenAiRouter, PineconeRouter, SpeechAndTextRouter
from fastapi import FastAPI, FastAPI
from dotenv import load_dotenv, find_dotenv
import pinecone
import os
import openai

load_dotenv(find_dotenv(), override=True)

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV')
)

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Generative AI API",
    description="LLM Generative AI Serving",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    swagger_ui=True  # This line enables Swagger UI
)


app.include_router(OpenAiRouter.router)
app.include_router(PineconeRouter.router)
app.include_router(SpeechAndTextRouter.router)

def generate_html_response():
    html_content = """
    <!DOCTYPE html>
    <html>
        <body>
        <h1>response:</h1>
        <div id="result"></div>
        <script>
        var source = new EventSource("/openai/completion?text=Tell me a joke");
        source.onmessage = function(event) {
            document.getElementById("result").innerHTML += event.data + "<br>";
        };
        </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/items/", response_class=HTMLResponse)
async def read_items():
    return generate_html_response()
