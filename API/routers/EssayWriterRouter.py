from Routers.Imports.LangGraphImports import *

router = APIRouter(
    prefix="/essay-writer",
    tags=["Essay Writer"]
)

memory = SqliteSaver.from_conn_string(":memory:")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

@router.post("/write-essay/")
def write_essay(question: str, threadNumber: str):
    thread = {"configurable": {"thread_id": threadNumber}}
    abot = Agent(model)
    result = abot.graph.invoke({
        'task': question,
        "max_revisions": 3,
        "revision_number": 1,
    }, thread)

    # for s in graph.stream({
    #     'task': "what is the difference between langchain and langsmith",
    #     "max_revisions": 2,
    #     "revision_number": 1,
    # }, thread):
    #     print(s)

    return result['content']



class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    # For generating this lists of queries to pass to Tavily, 
    # we will use function calling to ensure we get back a list of strings from the language model
    # this pydantic model represents the result we want to get back from the language model
    queries: List[str]

class Agent:
    def __init__(self, model):
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: END, "reflect": "reflect"}
        )
        builder.add_edge("planner","research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect","research_critique")
        builder.add_edge("research_critique", "generate")
        self.graph = builder.compile(checkpointer=memory)
        self.model = model

    def plan_node(self, state: AgentState):
        task = state.get('task')
        if not task:
            raise ValueError("Task is missing in state")
        messages = [
            SystemMessage(content=PLAN_PROMT),
            HumanMessage(content=state['task'])
        ]
        response = model.invoke(messages)
        return {"plan": response.content}

    def research_plan_node(self, state: AgentState):
        queries = model.with_structured_output(Queries).invoke([
                SystemMessage(content=RESEARCH_PLAN_PROMPT),
                HumanMessage(content=state['task'])
            ])
        content = state['content'] or []
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}
    
    def generation_node(self,state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
   
        messages = [
            SystemMessage(content=WRITER_PROMPT.format(content=content)),
            user_message
            ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content, 
            "revision_number": state.get("revision_number", 1) +1
            }

    def reflection_node(self,state: AgentState):
        messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=state['draft'])
        ]
        response = self.model.invoke(messages)
        return {"critique": response.content}
    
    def research_critique_node(self,state: AgentState):
        queries = model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}
    
    def should_continue(self,state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"