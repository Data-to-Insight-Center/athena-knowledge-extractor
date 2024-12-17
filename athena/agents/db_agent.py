from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from athena.graph_state import PatraState
from athena.util import llm
from athena.tools import execute_cypher

tools = [execute_cypher]

answer_generator_template = """You are tasked with executing a given query in a neo4j graph and returning the response.
You have access to the execute_cypher tool which lets you execute the given query on the neo4j databse. 
  
Here's the cypher query: {cypher_query}  

Keep the output structured if possible.
 Don't use bold, underline or other text altering stuff. Just plain text. 
 Do not return the embeddings specially in the ModelCard information. 
 for start time and end times convert to human readable timestamps. These are EDT. 
 
 If you dont know the answer, answer failed to generate, say you don't know the answer. 
 """

db_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_generator_template),
        ("human", "Return the result to the query"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
db_agent_w_tools = create_tool_calling_agent(llm, tools, db_agent_prompt)
db_executor = AgentExecutor(agent=db_agent_w_tools, tools=tools, verbose=True)

def db_agent_node(state: PatraState):
    cypher_query = state.cypher_query
    # prompt_formatted = answer_generator_template.format(cypher_query=cypher_query)
    #
    # result = db_agent_w_tools.invoke(
    #     [SystemMessage(content=prompt_formatted),
    #      HumanMessage(content="Execute the query and return the response:")]
    # )
    # answer = result.content

    response = db_executor.invoke({"cypher_query": cypher_query})
    return {"messages": [AIMessage(content="Database: Response: {}".format(response['output']))]}