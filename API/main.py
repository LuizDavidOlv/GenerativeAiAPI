import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

# from Routers import (OpenAiRouter, SpeechAndTextRouter, FineTunningRouter, AgentsRouter, HuggingFaceRouter, QuerySqlServerRouter,
#     JwtAuthenticationRouter, LangGraphRouter, EssayWriterRouter, LlamaIndexRouter)
from Routers import routers as routers_dict
from Vault.Bootstrap import Globle

load_dotenv(find_dotenv(), override=True)

# pinecone.init(
#     api_key=os.environ.get('PINECONE_API_KEY'),
#     environment=os.environ.get('PINECONE_ENV')
# )

app = FastAPI(
    title="Generative AI API",
    description="LLM Generative AI Serving",
    version="0.1.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    swagger_ui=True,  # This line enables Swagger UI
    on_startup=[Globle.Settings],
)

for version, routers in routers_dict.items():
    for router in routers:
        app.include_router(router, prefix=f"/{version}")


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


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
