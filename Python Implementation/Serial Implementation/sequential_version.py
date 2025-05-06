import heapq

def single_change(u, v, weight, operation, graph, dist, parent):
    # Step 1: Update the graph
    if operation == "insert":
        graph.setdefault(u, []).append((v, weight))
    elif operation == "delete":
        if u in graph:
            graph[u] = [(n, w) for n, w in graph[u] if n != v]

    # Step 2: Determine affected vertex x and y
    if dist.get(u, float("inf")) > dist.get(v, float("inf")):
        x, y = u, v
    else:
        x, y = v, u

    pq = []

    # Step 3: Update distance of x and enqueue it
    if operation == "insert":
        alt = dist[y] + weight
        if alt < dist.get(x, float("inf")):
            dist[x] = alt
            parent[x] = y
            heapq.heappush(pq, (dist[x], x))

    elif operation == "delete":
        # Only act if the deleted edge was part of the current SSSP tree
        if parent.get(x) == y:
            dist[x] = float("inf")
            parent[x] = None
            heapq.heappush(pq, (dist[x], x))

    # Step 4: Propagate updates only if a change occurred
    while pq:
        _, z = heapq.heappop(pq)
        updated = False
        for neighbor, w in graph.get(z, []):
            alt = dist[z] + w
            if alt < dist.get(neighbor, float("inf")):
                dist[neighbor] = alt
                parent[neighbor] = z
                heapq.heappush(pq, (alt, neighbor))
                updated = True
        # Enqueue neighbors only if an update occurred
        if updated:
            for neighbor, _ in graph.get(z, []):
                heapq.heappush(pq, (dist.get(neighbor, float("inf")), neighbor))

    return dist, parent
