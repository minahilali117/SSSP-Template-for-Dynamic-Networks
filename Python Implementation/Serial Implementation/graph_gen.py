import random

def generate_graph(num_nodes, num_edges, max_weight=100000, filename="full_dense.txt"):
    edges = set()
    while len(edges) < num_edges:
        u = random.randint(1, num_nodes)
        v = random.randint(1, num_nodes)
        if u != v:
            weight = random.randint(1, max_weight)
            edges.add((u, v, weight))
    
    with open(filename, "w") as f:
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")

# Generate and save the graph
generate_graph(10000, 100000)
print("Graph saved to full_dense.txt")
