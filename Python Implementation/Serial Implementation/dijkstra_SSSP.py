import heapq

def dijkstra(graph, source):    
    dist = {node: float('inf') for node in graph}
    parent = {node: None for node in graph}
    dist[source] = 0

    pq = [(0, source)]  # priority queue as (distance, node)

    while pq:
        current_dist, u = heapq.heappop(pq)

        # If we already found a better path, skip
        if current_dist > dist[u]:
            continue

        for v, weight in graph[u]:
            if dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))

    
    return dist, parent

def batch_update_sssp(graph, dist, parent, insertions, deletions, source):
    affected = set()
    print(len(deletions))
    print(len(insertions))
    # 1. Handle deletions
    for u, v in deletions:
        # Remove edge u->v
        graph[u] = [(nbr, w) for nbr, w in graph[u] if nbr != v]
        # If edge was in SSSP, mark v and all its descendants
        if parent[v] == u:
            stack = [v]
            while stack:
                node = stack.pop()
                dist[node] = float('inf')
                parent[node] = None
                affected.add(node)
                # Add children in SSSP tree
                stack.extend([n for n, p in parent.items() if p == node])

    # 2. Handle insertions
    for u, v, w in insertions:
        # Add edge u->v if not present
        if not any(nbr == v for nbr, _ in graph[u]):
            graph[u].append((v, w))
        # If this edge provides a better path, update v and propagate
        if dist[v] > dist[u] + w:
            dist[v] = dist[u] + w
            parent[v] = u
            affected.add(v)

    # 3. Propagate improvements (BFS/DFS from affected nodes)
    queue = list(affected)
    while queue:
        u = queue.pop(0)
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                parent[v] = u
                queue.append(v)

    # 4. Reconnect unreachable nodes (run Dijkstra only for inf nodes)
    # Optionally, you can run a full Dijkstra from the source for all inf nodes
    # or just leave them as disconnected

    return dist, parent
