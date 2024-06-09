from Routers.Imports.LangGraphImports import *

#load_dotenv(find_dotenv(), override=True)
tool = TavilySearchResults(max_results=2)
memory = SqliteSaver.from_conn_string(":memory:")
memory_with_stream = AsyncSqliteSaver.from_conn_string(":memory:")

router = APIRouter(
    prefix="/lang-graph",
    tags=["LangGraph"]
)

@router.post("/lang-graph-agent")
async def lang_graph_agent(question: str):
    tool = TavilySearchResults(max_results=2)
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
                You are allowed to make multiple calls (either together or in sequence). \
                Only look up information when you are sure of what you want. \
                If you need to look up some information before asking a follow up question, you are allowed to do that!
                """
    model = ChatOpenAI(model="gpt-3.5-turbo")
    abot = Agent(model, [tool], system=prompt)
    messages = [HumanMessage(content=question)]
    result = abot.graph.invoke({"messages": messages})
    return result['messages'][-1].content

#! Endpoint not streaming
@router.post("/lang-graph-memory-agent")
async def lang_graph_agent(question: str, threadNumber ="1"):
    prompt = """You are a smart research assistant. Use the search engine to look up information. \
                You are allowed to make multiple calls (either together or in sequence). \
                Only look up information when you are sure of what you want. \
                If you need to look up some information before asking a follow up question, you are allowed to do that!
                """
    model = ChatOpenAI(model="gpt-3.5-turbo")
    abot = MemoryAgent(model, [tool], system=prompt, checkpointerConfig=memory_with_stream)
    messages = [HumanMessage(content=question)]
    thread = {"configurable": {"thread_id": threadNumber}}
    async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print( content+ " || ")
            
    # for event in abot.graph.stream({"messages": messages}, thread):
    #     for v in event.values():
    #         return v



# async def agent_flow_download():
#     prompt = """You are a smart research assistant. Use the search engine to look up information. \
#                 You are allowed to make multiple calls (either together or in sequence). \
#                 Only look up information when you are sure of what you want. \
#                 If you need to look up some information before asking a follow up question, you are allowed to do that!
#                 """
#     model = ChatOpenAI(model="gtp-3.5-turbo")
#     abot = Agent(model, [tool], system=prompt)
#     agentFlowPng = Image(abot.graph.get_graph().draw_png())
#     agentFlowPng.save("AgentFlow.png")
#     return True




class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile() # Will turn into a langchain runable
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
    
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


class MemoryAgent:
    def __init__(self, model, tools, checkpointerConfig, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointerConfig) # Will turn into a langchain runable
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
    
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


  