import time
from Imports.AgentsImports import *
from sql_prompt_template import SQL_GENERATION_TEMPLATE

router = APIRouter(
    prefix="/agents",
    tags=["Agents"]
)

load_dotenv(find_dotenv(), override=True)

chat_history = []
chat_history_string = ""

class CalculatorInput(BaseModel):
    question: str = Field()

@router.post("/word-length/")
def word_length(text: str):
    toolsConfig= [get_word_length]
    llmConfig = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llmConfig.bind(functions=[format_tool_to_openai_function(t) for t in toolsConfig])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant, but bad at calculating lengths of words",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agentConfig = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent = agentConfig, tools = toolsConfig, verbose = True)
    result = agent_executor.invoke({"input": text})
    return result["output"]

@router.post("/current-events/")
def current_events(question: str):
    llmConfig = ChatOpenAI( temperature=0)
    search = SerpAPIWrapper()
    tools = [
        Tool.from_function(
            func = search.run,
            name = 'Search',
            description = "useful for when you need to answer questions about current envents",
        )
    ]

    agent = initialize_agent(tools, llmConfig, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
    return agent.run(question)

@router.post("/current-events-modifying-existing-tools/")
def current_events(question: str = "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"):
    llmConfig = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = load_tools(["serpapi", "llm-math"],llm = llmConfig)
    tools[0].name = "Google Search"
    agent = initialize_agent(tools, llmConfig, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
    return agent.run(question)

@router.post("/agent-with-memory/")
def prompt_with_memory( question: str):
    toolsConfig= [get_word_length]
    llmConfig = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_with_tools = llmConfig.bind(functions=[format_tool_to_openai_function(t) for t in toolsConfig])
    MEMORY_KEY = "chat_history"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your are a very powerfull assistant, but bad at calculating lengths of words",
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    #? Lambda functions are a type of High Order Function (HOF).
    #? The pipe character (|) is a custom function that leverages HOFs to achieve monadic composition. 
    #? Monadic composition is a way to chain functions together, where the output of one function is the input of the next.

    agentConfig = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agentConfig, tools=toolsConfig, verbose=True)
    preResult1 = agent_executor.invoke({"input": question, "chat_history": chat_history})

    chat_history.extend(
        [
            HumanMessage(content= question),
            AIMessage(content = preResult1["output"]),
        ]
    )

    sponsoredMessage = "Always try to sell something based on the conversation"
    preResult2 = agent_executor.invoke({"input": sponsoredMessage, "chat_history": chat_history})
    
    chat_history.extend(
        [
            HumanMessage(content = sponsoredMessage),
            AIMessage(content = preResult2["output"]),
        ]
    )

    result = {
        "conversation": chat_history,
    }
    
    return result

@router.post("/re-act-agent/")
def re_act_agent(question: str):
     # sql_server_store = sqlalchemy.create_engine(sqlServerDbConnectionString, future=True)
    tools = [TavilySearchResults(max_results=1)]
    prompt = hub.pull("hwchase17/react")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    result = agent_executor.invoke({"input": question})
    return result["output"]


#! Not working. Need to understand how get chat_history_string value
@router.post("/re-act-agent-with-chat-history/")
def re_act_agent_with_chat_history(question: str):
    # get value from chat_history_string
    toolsConfig = [TavilySearchResults(max_results=1)]
    promptConfig = hub.pull("hwchase17/react-chat")
    llmConfig = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    agent = create_react_agent(llm=llmConfig, tools=toolsConfig, prompt=promptConfig)
    agent_executor = AgentExecutor(agent=agent, tools=toolsConfig, verbose=True, handle_parsing_errors=True)

    request =  {
        "input": {question},
        # Notice that chat_history is a string, since this prompt is aimed at LLMs, not chat models
        "chat_history": {chat_history_string},
    }
    result = agent_executor.invoke({request})
    chat_history_string += f'Human: {question}\nAI: {result["output"]}\n'

    return result["output"]


@router.post("/re-act-sql-access/")
def re_act_sql_access(question: str):
    dbUri = os.getenv("SQL_CON_STRING_LOCAL")
    dbConfig = SQLDatabase.from_uri(dbUri)
    llmConfig = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    toolkitConfig = SQLDatabaseToolkit(db=dbConfig, llm=llmConfig)
    db_agent = create_sql_agent(llm=llmConfig, toolkit=toolkitConfig, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    messages = [
        SystemMessage(content=SQL_GENERATION_TEMPLATE),
        HumanMessage(content=f'Provide only the answer to the question. Remove any aditional details:\n QUESTION: {question}')
    ]
    result = db_agent.run(messages)

    return result

@tool  
def get_word_length(word: str) -> int:
    """Returns the length of a word"""
    return len(word)

if __name__ == "__main__":
    result = re_act_sql_access("Whare are all the OPERTN_NM values?")
    print(result)





