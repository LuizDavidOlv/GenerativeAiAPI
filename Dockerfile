#It instructs Docker Engine to use the Python 3.8 image as the base image
#FROM generativeaiapi-web:latest
#FROM python:3.8-slim-buster
#FROM python:3.11.8
FROM luizcunhaoliveira/python_0:latest



#It creates a working directory(main) for the Docker image and container
WORKDIR /entry


#It will copy api artifacts
COPY API ./API


#change directory to API
WORKDIR /entry/API

#It will install the required packages
#RUN apt-get update && apt-get install -y libpq-dev
# RUN pip install --upgrade pip 
# RUN apt-get update && apt-get install -y libpq-dev
# RUN pip install -r ./requirements.txt



#It is the command that will start and run the FastAPI application container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
##############################################################################
#Commands to build and push to harbor
#docker build -t genai-sample-api:v1 .
#docker tag genai-sample-api:v1 harbor.dell.com/dfs/genai-sample-api:v1
#docker push harbor.dell.com/dfs/genai-sample-api:v1
##############################################################################