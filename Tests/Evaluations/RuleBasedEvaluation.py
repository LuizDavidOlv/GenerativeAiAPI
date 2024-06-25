from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
class RuleBasedEvaluation:

    def eval_expected_words(self,
            system_message,
            question,
            expected_words,
            human_template="{question}",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            output_parser=StrOutputParser()
    ):
        assistant = self.assistant_chain(system_message, human_template, llm, output_parser)
        answer = assistant.invoke({"question": question})
        
        assert any(word in answer.lower() \
                    for word in expected_words), \
                        f"Expected the assistant questions to include \
                            '{expected_words}', but it did not"

    def evaluate_refusal(self,
            system_message,
            question,
            decline_response,
            human_template="{question}",
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            output_parser=StrOutputParser()
    ):
        assistant = self.assistant_chain(
            system_message=system_message,
            human_template = human_template,
            llm=llm,
            output_parser=output_parser
            )
        
        answer = assistant.invoke({"question": question})

        assert decline_response.lower() in answer.lower(), \
            f"Expected the bot to decline with \
                '{decline_response}', got {answer}"

        

    def assistant_chain(self,
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