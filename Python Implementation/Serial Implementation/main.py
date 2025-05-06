from graph_maker import read_graph
from dijkstra_SSSP import dijkstra, batch_update_sssp
from sequential_version import single_change
import time
import random

print("Creating the graph...")
graph = read_graph()

print("Creating the SSSP tree...")
start_time = time.perf_counter()
dist, parent = dijkstra(graph, 1)
end_time = time.perf_counter()
print(f"Dijkstra's algorithm took {end_time - start_time:.6f} seconds.")

# print("Initial SSSP from source node 1")
# print("Node\tDistance\tParent")
# for node in sorted(graph):
#     d = dist[node]
#     p = parent[node]
#     print(f"{node}\t{d if d != float('inf') else '∞'}\t\t{p if p is not None else '-'}")
# print("." * 50, "\n")

# print("Updating the SSSP tree...")
# start_time = time.perf_counter()
# dist, parent = single_change(4825, 10000, 2, "insert", graph, dist, parent)
# end_time = time.perf_counter()
# print(f"Sequential SSSP algorithm took {end_time - start_time:.6f} seconds.")

# print("SSSP tree after inserting edge (1, 966) with weight 5")
# print("Node\tDistance\tParent")
# for node in sorted(graph):
#     d = dist[node]
#     p = parent[node]
#     print(f"{node}\t{d if d != float('inf') else '∞'}\t\t{p if p is not None else '-'}")
# print("." * 50, "\n")


# dist, parent = single_change(1, 1000, 5, "delete", graph, dist, parent, 1)

# print("SSSP tree after deleting edge (1, 30) with weight 5")
# print("Node\tDistance\tParent")
# for node in sorted(graph):
#     d = dist[node]
#     p = parent[node]
#     print(f"{node}\t{d if d != float('inf') else '∞'}\t\t{p if p is not None else '-'}")
# print("."*50, "\n")
# print("SSSP tree after inserting edge (1, 5) with weight 2 and deleting edge (3, 4) and (5, 6)")
# print("Node\tDistance\tParent")
# for node in sorted(graph):
#     d = dist[node]
#     p = parent[node]
#     print(f"{node}\t{d if d != float('inf') else '∞'}\t\t{p if p is not None else '-'}")
# print("."*50, "\n")

# Generate a large number of random insertions and deletions
random.seed(42)  # For reproducibility
num_insertions = 1000
num_deletions = 1000

# Generate random insertions (source, target, weight)
insertions = []
for _ in range(num_insertions):
    source = random.randint(1, 10000)
    target = random.randint(1, 10000)
    weight = random.randint(1, 100)
    insertions.append((source, target, weight))

# Generate random deletions (source, target)
deletions = []
for _ in range(num_deletions):
    source = random.randint(1, 10000)
    target = random.randint(1, 10000)
    deletions.append((source, target))

start_time = time.perf_counter()
dist, parent = batch_update_sssp(graph, dist, parent, insertions, deletions, 1)
end_time = time.perf_counter()
print(f"SSSP algorithm took {end_time - start_time:.6f} seconds.")

# print("SSSP tree after inserting edge (1, 5) with weight 2 and deleting edge (3, 4) and (5, 6)")
# print("Node\tDistance\tParent")
# for node in sorted(graph):
#     d = dist[node]
#     p = parent[node]
#     print(f"{node}\t{d if d != float('inf') else '∞'}\t\t{p if p is not None else '-'}")
# print("."*50, "\n")
