from langchain.prompts import ChatPromptTemplate
from Prompts.SystemMessagePrompt import SYSTEM_MESSAGE
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

class Evaluation:
    @staticmethod
    def eval_expected_words(
            system_message,
            question,
            expected_words,
            human_template="{question}",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            output_parser=StrOutputParser()
    ):
        assistant = Evaluation.assistant_chain(system_message, human_template, llm, output_parser)
        answer = assistant.invoke({"question": question})
        
        assert any(word in answer.lower() \
                    for word in expected_words), \
                        f"Expected the assistant questions to include \
                            '{expected_words}', but it did not"

    @staticmethod
    def evaluate_refusal(
            system_message,
            question,
            decline_response,
            human_template="{question}",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            output_parser=StrOutputParser()
    ):
        assistant = Evaluation.assistant_chain(human_template,system_message,llm,output_parser)
        answer = assistant.invoke({"question": question})

        assert decline_response.lower() in answer.lower(), \
            f"Expected the bot to decline with \
                '{decline_response}', got {answer}"

        

    @staticmethod
    def assistant_chain(
            system_message,
            human_template="{question}",
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            output_parser = StrOutputParser()
        ):

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template)
        ])

        return chat_prompt | llm | output_parser