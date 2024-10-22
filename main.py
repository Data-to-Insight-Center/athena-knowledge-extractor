from dotenv import load_dotenv
from athena.agents.query_agent import query_generator

from athena.util import graph

load_dotenv()
def main():
    question = ("using the neo4j graph schema, generate a cypher query to answer: how many models are in the system?")
    # result = run_patra_graph(question)
    result = query_generator.invoke({"graph_schema": graph.get_structured_schema, "question": question})

    print(result)


if __name__ == '__main__':
    main()