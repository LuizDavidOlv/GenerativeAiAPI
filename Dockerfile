#It instructs Docker Engine to use the Python 3.8 image as the base image
FROM python:3.8



#It creates a working directory(main) for the Docker image and container
WORKDIR /entry


#It will copy api artifacts
COPY API ./API


#It will install the required packages
#RUN apt-get update && apt-get install -y libpq-dev
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install python-dotenv
RUN pip install openai
RUN pip install langchain
RUN pip install pinecone-client



#It is the command that will start and run the FastAPI application container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"]
##############################################################################
#Commands to build and push to harbor
#docker build -t genai-sample-api:v1 .
#docker tag genai-sample-api:v1 harbor.dell.com/dfs/genai-sample-api:v1
#docker push harbor.dell.com/dfs/genai-sample-api:v1
##############################################################################