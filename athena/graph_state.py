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
    # all the messages exchanged
    messages: Annotated[list, operator.add] = field(default_factory=list)
    # question for cypher generator
    question: str = field(default=None)
    # generated cypher query
    cypher_query: str = field(default=None)
    # input user question
    user_question: str = field(default=None)
    # final output of the agent
    final_response: str = field(default=None)
    # end the cycle if this is set to true
    answer_completed: bool = field(default=False)
    # number of iterations
    iterations: int = field(default=0)

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
