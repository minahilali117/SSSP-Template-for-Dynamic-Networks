def read_graph(filename="dataset/full_dense.txt"):
    graph = {}

    with open(filename, "r") as file:
        for line in file:
            if line.strip():
                u, v, w = map(int, line.strip().split())

                if u not in graph:
                    graph[u] = []
                graph[u].append((v, w))  

                if v not in graph:
                    graph[v] = []

    return graph
