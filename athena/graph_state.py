from typing import Annotated, Sequence, TypedDict, List, ClassVar
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dataclasses import dataclass, field
import operator
from athena.util import graph

@dataclass
class PatraState:
    """
    Agent state for the Agentic workflow.
    """
    messages: Annotated[list, operator.add] = field(default_factory=list)
    # graph schema for the CKN graph
    graph_schema: str = field(default=None)
    # question for cypher generator
    question: str = field(default=None)
    # generated cypher query
    cypher_query: str = field(default=None)
    # sender of the last message
    sender: str = field(default=None)
    # input user question
    user_question: str = field(default=None)
    # final output of the agent
    final_response: str = field(default=None)


@dataclass
class PatraInput:
    """
    Input for the PatraAgent.
    """
    user_question: str = field(default=None)

@dataclass
class PatraOutput:
    """
    Output for the PatraAgent.
    """
    final_response: str = field(default=None)
