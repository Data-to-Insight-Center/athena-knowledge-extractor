from athena.agents.agent_util import create_agent
from athena.configuration_schema import Configuration
from athena.graph_state import PatraState
from athena.tools import print_hello
from athena.util import llm
import json
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

class PatraAgentResponse(BaseModel):
    """ Format the output of the response from the Patra agent"""
    question: str = Field(
        description="First question to be asked from the database in english"
    )
    thought_process: str = Field(
        description="Your thought process for asking the question"
    )
    summary_of_findings: str = Field(
        description="Summary of your findings so far pertaining to the question and responses"
    )
    answer_completed: bool = Field(
        description="Has the question from the user been answered, if yes set to True, else False"
    )
    final_answer: str = Field(
        description="Final answer to the user question if the question has been answered."
    )


patra_agent_template = """You are an helpful AI assistant that's helping to understand and query a graph database.
                Your task is to come up with a plan for a set of questions to be asked in order to answer the user question: {user_question}
                If you are satisfied with the response to the original user query, set the answer_completed to True in your response. and set the final_answer to your response to the original user query. 
                Else set the answer_completed to False and final_answer to be a summary of your findings so far. 
                
                You only have to return the first question to be asked from the database in english.  
                                    
                The database is a graph database containing models, modelcards, experiements, images, users ...etc. Do not ask the user anything. Return a single question in english.  

                You are really good at figuring out which questions to ask in which order to answer the original question. 
                Looking at the history, ask questions get the answers and solve the original question.
                
                You are only communicating with another agent who will get you information from the database. So you need to be precise with your questions. 

                If the returned result for the last question is empty, rephrase the question and try again.
                
                Here's the conversation so far with you and the database agent: {messages}.
                Schema for the database: {graph_schema}
                 """

patra_agent_llm = llm.with_structured_output(PatraAgentResponse)


def patra_agent_node(state: PatraState, config: RunnableConfig):
    # get the graph schema from the configuration dynamically
    configurable = Configuration.from_runnable_config(config)
    graph_schema = configurable.graph_schema

    user_question = state.user_question
    messages = state.messages
    iterations = state.iterations

    prompt_formatted = patra_agent_template.format(user_question=user_question, messages=messages, graph_schema=graph_schema)

    result = patra_agent_llm.invoke(
        [SystemMessage(content=prompt_formatted),
         HumanMessage(content="Generate a single question to ask the database:")]
    )
    question = result.question
    final_answer = result.summary_of_findings
    try:
        answer_completed = result.answer_completed
    except:
        answer_completed = False

    if answer_completed:
        final_answer = result.final_answer

    return {"question": question, "answer_completed": answer_completed, "iterations": iterations + 1,
            "final_answer": final_answer, "messages": [AIMessage(content="AGENT: Question asked: {}".format(question))]}