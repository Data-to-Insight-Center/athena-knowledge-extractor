from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from athena.agents.patra_agent import patra_agent_node
from athena.configuration_schema import Configuration
from athena.graph_state import PatraState, PatraInput, PatraOutput
from athena.agents.query_agent import cypher_generator_node
from athena.util import graph
from athena.agents.db_agent import db_agent_node
from typing import Literal


PATRA_AGENT = "patra_agent"
QUERY_AGENT = "query_agent"
DB_AGENT = "db_executor"


# def patra_node(state: PatraState) -> PatraState:
#     response = patra_executor.invoke({"messages": state.messages})
#     result = AIMessage(**response.dict(exclude={"type", "name"}), name=PATRA_AGENT_NAME)
#     state.messages.append(result)
#     state.sender = PATRA_AGENT_NAME
#     return state
#
# def cypher_generator_node(state: PatraState) -> PatraState:
#     query_question = state.messages[-1].content
#     cypher_query = query_generator.invoke({"graph_schema": state.graph_schema, "question": query_question})
#     state.messages.append(AIMessage(content=str(cypher_query), name=QUERY_AGENT_NAME))
#     return state
#

#
# def job_agent(state: PatraState) -> PatraState:
#     # Use QueryAgent and submit job
#     return {"messages": [AIMessage(content="Job submitted")]}
#
# def research_agent(state: PatraState) -> PatraState:
#     # Use GoogleScholar API and QueryAgent
#     return {"messages": [AIMessage(content="Research completed")]}
#
# def supervisor(state: PatraState) -> str:
#     # Decide which agent to invoke based on the last message
#     last_message = state["messages"][-1].content
#     if "query" in last_message.lower():
#         return "query_agent"
#     elif "job" in last_message.lower():
#         return "job_agent"
#     elif "research" in last_message.lower():
#         return "research_agent"
#     else:
#         return END
#
def router(state: PatraState, config: RunnableConfig) -> Literal["continue", "__end__"]:
    # Routing based on previous messages

    # if the patra agent has decided the work is done, end loop
    if state.answer_completed:
        return "__end__"

    # else check if the iteration count has reached the limit
    patra_config = Configuration.from_runnable_config(runnable_config)
    if state.iterations >= patra_config.max_iterations:
        return "__end__"

    # else continue the patra loop.
    return "continue"


# Setting up configurations for Patra
runnable_config = {
    "configurable": {
        "graph_schema": str(graph.get_structured_schema)
    }
}
patra_config = Configuration.from_runnable_config(runnable_config)


# Create the graph
patra = StateGraph(PatraState, input=PatraInput, output=PatraOutput, config_schema=patra_config)
patra.graph_schema = str(graph.get_structured_schema)

# Add nodes
patra.add_node(PATRA_AGENT, patra_agent_node)
patra.add_node(DB_AGENT, db_agent_node)
patra.add_node(QUERY_AGENT, cypher_generator_node)

# Set entry point
patra.set_entry_point(PATRA_AGENT)

# QueryAgent can interact with DBAgent multiple times
patra.add_edge(QUERY_AGENT, DB_AGENT)
patra.add_edge(DB_AGENT, PATRA_AGENT)

patra.add_conditional_edges(
    PATRA_AGENT,
    router,
    {"continue": QUERY_AGENT, "__end__": END},
)

# Compile the graph
app = patra.compile()


def run_patra_graph(question):
    # result = app.invoke({
    #     "messages": [HumanMessage(content=question)],
    #     "sender": "human",
    #     "graph_schema": str(graph.get_structured_schema),
    # })
    # inputs = {
    #     "messages": [HumanMessage(content=question)],
    #     "sender": "human"
    # }
    # for chunk in app.stream(inputs, stream_mode="values"):
    #     chunk["messages"][-1].pretty_print()
    user_question = PatraInput(user_question=question)
    result = app.invoke(user_question)
    return result
