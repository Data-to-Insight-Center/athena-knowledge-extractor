import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from athena.configuration_schema import Configuration
from athena.graph_state import PatraState
from athena.util import llm_json, graph

query_agent_prompt = """You are an expert in writing Cypher queries for a Neo4j database.
Your task is to write a Cypher query to answer the following question: {question}
Schema of the graph is as follows: {graph_schema}
 
Write Cypher queries that avoid using directional edges. Instead of using arrows (-> or <-) for relationships, 
use undirected relationships.
Make sure that all relationships in the queries are undirected, 
using double hyphens and square brackets to specify the relationship type.
Only return the cypher query

For example, instead of:
MATCH (u:User {{user_id: 'jstubbs'}})-[:SUBMITTED_BY]->(e:Experiment)
RETURN COUNT(e) AS NumberOfExperimentsRunByJstubbs

You should write:
MATCH (u:User {{user_id: 'jstubbs'}})-[r:SUBMITTED_BY]-(e:Experiment)
RETURN COUNT(e) AS NumberOfExperimentsRunByJstubbs

Try and convert datetime when returning. 
Here's an example:
MATCH (u:User {{user_id: 'swithana'}})-[r:SUBMITTED_BY]-(e:Experiment)
RETURN e, datetime({{epochMillis: e.start_time}}) AS start_time

To get information about the ModelCard from the experiment use:
Match (exp:Experiment {{experiment_id: `d2i-exp-3442334`}})-[r:USED]-(m:Model)-[r2:USED]-(mc:ModelCard)
return mc

Models have a '-model' in their id. Do not drop it when querying the database. 

To get information about a modelCard from a model you can use:
MATCH (m:Model {{model_id: '33232113' }})-[r2:USED]-(mc:ModelCard) return mc

Only use the relationship [:VERSION_OF] to get similar ModelCards. 

You can compare test accuracy and other model attributes across model cards using this:
MATCH 
  (mc1:ModelCard {{external_id: 'example_1'}})-[r:USED]-(m1:Model), 
  (mc2:ModelCard {{external_id: 'example_2'}})-[r2:USED]-(m2:Model) 
WITH 
  m1, m2, 
  CASE 
    WHEN m1.test_accuracy > m2.test_accuracy THEN m1 
    ELSE m2 
  END AS highest_accuracy_model
RETURN highest_accuracy_model

To get information about images executed in an experiement on a device, use this as
an example:

MATCH (img:RawImage)-[r2:PROCESSED_BY]-(e:Experiment)-[r:EXECUTED_ON]-(d:EdgeDevice) 
RETURN img

To get deployment information about model deployments you can use deployment node.
For deployment location, you can use the location field in EdgeDevice. 
Example: to find models deployed in raspi-3 device types in ohio-zoo location you can use:
match (m:Model)-[:HAS_DEPLOYMENT]-(depl:Deployment)-[:DEPLOYED_IN]-(device:EdgeDevice {{location:'ohio-zoo', device_type: 'raspi3'}})
 return m, depl.device_type, depl.average_accuracy, device.location, device.device_id

available model types for Model are [convolutional neural network, large language model, foundational model]. These are case sensitive.

external_id in certain nodes refer to the node ids. These must be returned with results. 

Return your response as a JSON object:
{{
    "cypher_query": "string",
    "context": "string"
}}
"""


def cypher_generator_node(state: PatraState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    graph_schema = configurable.graph_schema
    query_question = state.user_question
    # graph_schema = state.graph_schema

    prompt_formatted = query_agent_prompt.format(question=query_question, graph_schema=graph_schema)

    result = llm_json.invoke(
        [SystemMessage(content=prompt_formatted),
         HumanMessage(content="Generate the Cypher Query:")]
    )
    cypher_query = json.loads(result.content)["cypher_query"]
    return {"cypher_query": cypher_query, "final_response": cypher_query}