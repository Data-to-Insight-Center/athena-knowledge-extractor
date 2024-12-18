import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

# get the env variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PWD")
LLAMA_URI = os.getenv("HOSTED_LLAMA_URI")

# load the graph and the LLM
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PWD)
# gpt4
# llm = ChatOpenAI(temperature=0.2, model="gpt-4o")
# #
# llm = ChatOpenAI(
#     base_url = LLAMA_URI,
#     api_key='ollama',
#     model="llama3.1:70b",
#     max_tokens=10000,
# )

# llama 3.1 70b
llm_json = ChatOllama(
    model="llama3.3:70b",
    temperature=0.2,
    base_url=LLAMA_URI,
    format="json",
)

llm = ChatOllama(
    model="llama3.3:70b",
    temperature=0.2,
    base_url=LLAMA_URI,
)

top_k_results = 10