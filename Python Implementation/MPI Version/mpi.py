from mpi4py import MPI
import numpy as np
from graph_maker import read_graph
from dijkstra_SSSP import dijkstra
import heapq
import time
import random

# Hybrid asynchronous boundary-driven dynamic SSSP with MPI and NumPy buffers

def partition_graph(graph, num_parts):
    nodes = sorted(graph.keys())
    idx_map = {node: i for i, node in enumerate(nodes)}
    inv_map = {i: node for node, i in idx_map.items()}

    import pymetis
    adjacency = [[idx_map[v] for v, _ in graph[u]] for u in nodes]
    _, membership = pymetis.part_graph(num_parts, adjacency=adjacency)

    subgraphs = [{} for _ in range(num_parts)]
    node_to_worker = {}
    for i, part in enumerate(membership):
        u = inv_map[i]
        subgraphs[part][u] = list(graph[u])
        node_to_worker[u] = part + 1
    return subgraphs, node_to_worker

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size != 5:
    if rank == 0:
        print("Run with 5 ranks: 1 master + 4 workers")
    MPI.Finalize()
    exit()
source = 1

# Master: read graph, initial SSSP, generate updates

def master_phase():
    graph = read_graph("dataset/full_dense.txt")
    graph = {u: [(v,w) for v,w in nbrs if v in graph] for u,nbrs in graph.items()}
    subgraphs, owner = partition_graph(graph, 4)
    dist0, par0 = dijkstra(graph, source)
    N = max(graph)

    # build children list
    children = {u: [] for u in graph}
    for v,p in par0.items():
        if p is not None:
            children[p].append(v)

    # numpy buffers
    dist = np.full(N+1, np.inf)
    par  = np.full(N+1, -1, dtype=int)
    for u,d in dist0.items(): dist[u] = d
    for u,p in par0.items(): par[u] = -1 if p is None else p
    # Generate 1000 random edge insertions and deletions
    import random
    
    # For deletions, select from existing edges in graph
    existing_edges = []
    for u in graph:
        for v, _ in graph[u]:
            existing_edges.append((u,v))
    
    # Randomly sample 1000 edges to delete
    dels = random.sample(existing_edges, 1000)
    
    # For insertions, generate random new edges between existing nodes
    nodes = list(graph.keys())
    ins = []
    while len(ins) < 1000:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and not any(n==v for n,_ in graph[u]):
            w = random.randint(1, 100000) # Random weight
            ins.append((u,v,w))
    # updates (example)
    # ins = [(1,2, 10), (2, 50, 5)]
    # dels = [(9, 10)]

    # collect initial seeds
    pq = []
    # deletions: subtree resets
    for u,v in dels:
        if par[v] == u:
            graph[u] = [(n,w) for n,w in graph[u] if n!=v]
            queue = [v]
            while queue:
                x = queue.pop(0)
                dist[x] = np.inf
                par[x] = -1
                heapq.heappush(pq,(dist[x], x))
                for c in children[x]: queue.append(c)
                children[x] = []
    # insertions: immediate seeds
    for u,v,w in ins:
        if u in graph and v in graph:
            if not any(n==v for n,_ in graph[u]): graph[u].append((v,w))
            x,y = (u,v) if dist[u]>dist[v] else (v,u)
            alt = dist[y] + w
            if alt < dist[x]:
                dist[x] = alt
                par[x] = y
                heapq.heappush(pq,(alt,x))

    # distribute
    init_lists = [[] for _ in range(4)]
    while pq:
        _, u = heapq.heappop(pq)
        w = owner.get(u,1) - 1
        init_lists[w].append(u)
    return graph, subgraphs, owner, dist, par, init_lists, N

# helper

def parent_match(par, u, v):
    return par[v] == u

# def print_tree(dist, par, N):
    print("Node\tDist\tPar")
    for u in range(1, N+1):
        d = dist[u]
        p = par[u]
        print(f"{u}\t{d if np.isfinite(d) else '∞'}\t{p if p>=0 else '-'}")

if rank == 0:
    graph, subs, owner, dist, par, init_lists, N = master_phase()
else:
    subs = owner = dist = par = init_lists = N = None

# bcast and scatter
if rank == 0:
    to_scatter_subs = [{}] + subs
    to_scatter_init = [[]] + init_lists
else:
    to_scatter_subs = None
    to_scatter_init = None
owner = comm.bcast(owner, root=0)
dist  = comm.bcast(dist,  root=0)
par   = comm.bcast(par,   root=0)
N     = comm.bcast(N,     root=0)
my_sub = comm.scatter(to_scatter_subs, root=0)
my_aff = set(comm.scatter(to_scatter_init, root=0))

# event-driven loop
t0 = time.perf_counter() if rank == 0 else None
if rank == 0:
    done = False
    workers_done = set()
    
    while not done:
        # collect from workers
        all_msgs = {w: comm.recv(source=w) for w in range(1,5)}
        
        # Check for worker completion messages
        for w, msg in all_msgs.items():
            if msg == "WORKER_DONE":
                if w not in workers_done:
                    workers_done.add(w)
                    print(f"Worker {w} has completed")
        
        # convergence?
        if len(workers_done) == 4:
            done = True
            # All workers are done, notify them
            for w in range(1,5): 
                comm.send(None, dest=w)
            break
            
        # route - only process messages from workers that aren't done
        worker_msgs = {i: [] for i in range(1, 5)}
        
        for w, msg in all_msgs.items():
            if w in workers_done or msg == "WORKER_DONE":
                continue
                
            if len(msg) == 0:
                # Worker is signaling completion
                if w not in workers_done:
                    workers_done.add(w)
                    print(f"Worker {w} has completed")
                continue
                
            for u, nd, np_ in msg:
                tgt = owner.get(u, 1)  # Default to worker 1 if not found
                if tgt not in workers_done:
                    worker_msgs[tgt].append((u, nd, np_))
        
        # Send messages to each worker (empty list if no updates)
        for worker in range(1, 5):
            if worker not in workers_done:
                comm.send(worker_msgs[worker], dest=worker)
            else:
                # Ensure workers that are done still receive a message to prevent deadlock
                comm.send([], dest=worker)
            
    dt = time.perf_counter() - t0
    print(f"Time: {dt:.4f}s")
    print("Final tree:")
    # print_tree(dist, par, N)
else:
    done = False
    sent_done = False
    
    while not done:
        pq = [(dist[u], u) for u in my_aff]
        heapq.heapify(pq)
        my_aff.clear()
        out = []
        
        while pq:
            du, u = heapq.heappop(pq)
            if du != dist[u]:
                continue
            for v, w in my_sub.get(u, []):
                alt = du + w
                if alt < dist[v]:
                    dist[v] = alt
                    par[v] = u
                    if v in my_sub:
                        heapq.heappush(pq, (alt, v))
                    else:
                        out.append((v, alt, u))
        
        # Send updates or completion message, but only send completion once
        if not out and not sent_done:
            comm.send("WORKER_DONE", dest=0)
            print(f"Worker {rank} sending completion signal")
            sent_done = True
        else:
            comm.send(out if out else "WORKER_DONE", dest=0)
            
        msg = comm.recv(source=0)
        if msg is None:
            done = True
            print(f"Worker {rank} terminating")
            break
        
        # No need to process messages if we've already signaled completion
        if sent_done:
            continue
            
        # Process new updates for the next iteration
        my_updates = False
        for u, nd, np_ in msg:
            if nd < dist[u]:
                dist[u] = nd
                par[u] = np_
                my_aff.add(u)
                my_updates = True
        
        # If no more updates and empty queue, signal completion
        if not my_updates and not my_aff and not sent_done:
            comm.send("WORKER_DONE", dest=0)
            print(f"Worker {rank} sending completion signal")
            sent_done = True