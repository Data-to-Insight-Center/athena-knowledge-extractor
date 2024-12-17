from dotenv import load_dotenv
from athena.graph import run_patra_graph

from athena.util import graph

load_dotenv()
def main():
    question = ("how many models in the system?")
    result = run_patra_graph(question)
    # result = query_generator.invoke({"graph_schema": graph.get_structured_schema, "question": question})

    print(result)


if __name__ == '__main__':
    main()