import pandas as pd
from IPython.display import display
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from Prompts.ModelGradedPrompts import MODEL_GRADED_USER_MESSAGE, MODEL_GRADED_SYSTEM_MESSAGE

class ModelGradedEvaluation:

    def assistant_chain(self):
        human_template = "{question}"

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", MODEL_GRADED_SYSTEM_MESSAGE),
            ("human", human_template)
        ])

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        return chat_prompt | llm | StrOutputParser()
    
    def create_eval_prompt(self, system_message: str, human_message: str) -> ChatPromptTemplate:
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message),
        ])
        return eval_prompt

    def create_eval_chain(self, prompt: ChatPromptTemplate):
        eval_prompt = self.create_eval_prompt(MODEL_GRADED_SYSTEM_MESSAGE, MODEL_GRADED_USER_MESSAGE)
        eval_prompt = prompt

        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        return eval_prompt | llm | StrOutputParser()
    
