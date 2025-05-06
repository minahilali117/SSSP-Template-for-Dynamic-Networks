#include <mpi.h>
#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <map>
#include <queue>
#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <cstring>
#include <set>
#include <omp.h>
#include <atomic>

// Structure to represent an edge
struct Edge
{
    int node;
    int weight;
    Edge(int n, int w) : node(n), weight(w) {}
    // For sorting edges
    bool operator<(const Edge &other) const
    {
        return node < other.node;
    }
};

// Structure for the priority queue
struct PQNode
{
    double dist;
    int node;
    PQNode(double d, int n) : dist(d), node(n) {}
    // For priority queue comparison (min-heap)
    bool operator>(const PQNode &other) const
    {
        return dist > other.dist;
    }
};

// Structure for messages between processes
struct Message
{
    int node;
    double dist;
    int parent;
    Message(int n, double d, int p) : node(n), dist(d), parent(p) {}
};

// Improve cache locality with better data structures
struct NodeData
{
    double distance;
    int parent;
};

// Function to read the graph from a file
std::unordered_map<int, std::vector<Edge>> read_graph(const std::string &filename)
{
    std::unordered_map<int, std::vector<Edge>> graph;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error opening file " << filename << std::endl;
        return graph;
    }

    int u, v, w;
    while (file >> u >> v >> w)
    {
        graph[u].push_back(Edge(v, w));
    }

    return graph;
}

// Function to partition the graph using METIS
std::pair<std::vector<std::unordered_map<int, std::vector<Edge>>>, std::unordered_map<int, int>>
partition_graph(const std::unordered_map<int, std::vector<Edge>> &graph, int num_parts)
{
    std::vector<int> nodes;
    for (const auto &pair : graph)
    {
        nodes.push_back(pair.first);
    }
    std::sort(nodes.begin(), nodes.end());

    // Create index map
    std::unordered_map<int, int> idx_map;
    std::unordered_map<int, int> inv_map;
    for (size_t i = 0; i < nodes.size(); i++)
    {
        idx_map[nodes[i]] = i;
        inv_map[i] = nodes[i];
    }

    // Convert graph to CSR format for METIS
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    xadj.push_back(0);

    for (int node : nodes)
    {
        for (const Edge &edge : graph.at(node))
        {
            // Only include edges to nodes that exist in the graph
            if (idx_map.find(edge.node) != idx_map.end())
            {
                adjncy.push_back(idx_map[edge.node]);
            }
        }
        xadj.push_back(adjncy.size());
    }

    // Setup METIS parameters
    idx_t nvtxs = nodes.size();
    idx_t ncon = 1; // Number of constraints/balancing constraints
    idx_t nparts = num_parts;
    idx_t objval;                   // Edge cut or communication volume
    std::vector<idx_t> part(nvtxs); // Stores the result of the partitioning

    // Call METIS partitioning function
    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                  nullptr, nullptr, nullptr, &nparts,
                                  nullptr, nullptr, nullptr, &objval, part.data());

    if (ret != METIS_OK)
    {
        std::cerr << "METIS partitioning failed with error: " << ret << std::endl;
    }

    // Build subgraphs and node-to-worker map
    std::vector<std::unordered_map<int, std::vector<Edge>>> subgraphs(num_parts);
    std::unordered_map<int, int> node_to_worker;

    for (size_t i = 0; i < nvtxs; i++)
    {
        int u = inv_map[i];
        int part_num = part[i];

        // Add the node and its edges to the appropriate subgraph
        subgraphs[part_num][u] = graph.at(u);

        // Map the node to its worker (1-indexed workers)
        node_to_worker[u] = part_num + 1;
    }

    return {subgraphs, node_to_worker};
}

// Dijkstra's algorithm for initial SSSP
std::pair<std::unordered_map<int, double>, std::unordered_map<int, int>>
dijkstra(const std::unordered_map<int, std::vector<Edge>> &graph, int source)
{
    std::unordered_map<int, double> dist;
    std::unordered_map<int, int> parent;

    // Initialize distance and parent maps
    for (const auto &pair : graph)
    {
        int node = pair.first;
        dist[node] = std::numeric_limits<double>::infinity();
        parent[node] = -1;
    }

    // Set source distance to 0
    dist[source] = 0;

    // Priority queue for the algorithm
    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;
    pq.push(PQNode(0, source));

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();

        // Skip if we've found a better path already
        if (d > dist[u])
            continue;

        // Process all neighbors
        for (const Edge &edge : graph.at(u))
        {
            int v = edge.node;
            double weight = edge.weight;

            // Only process valid edges (to nodes in the graph)
            if (dist.find(v) != dist.end())
            {
                double alt = dist[u] + weight;
                if (alt < dist[v])
                {
                    dist[v] = alt;
                    parent[v] = u;
                    pq.push(PQNode(alt, v));
                }
            }
        }
    }

    return {dist, parent};
}

// Function to check if a node is a parent of another
bool parent_match(const std::vector<int> &par, int u, int v)
{
    return par[v] == u;
}

// Master process function
void master_process()
{
    // Read the graph
    auto graph = read_graph("dataset/full_dense.txt");

    // Clean graph to ensure all edges point to valid nodes
    for (auto &pair : graph)
    {
        auto &edges = pair.second;
        edges.erase(
            std::remove_if(edges.begin(), edges.end(),
                           [&graph](const Edge &e)
                           { return graph.find(e.node) == graph.end(); }),
            edges.end());
    }

    // Partition the graph
    auto [subgraphs, owner] = partition_graph(graph, 4);

    // Run initial Dijkstra
    int source = 1;
    auto [dist0, par0] = dijkstra(graph, source);

    // Find maximum node ID for array sizing
    int N = 0;
    for (const auto &pair : graph)
    {
        N = std::max(N, pair.first);
    }

    // Build children list
    std::unordered_map<int, std::vector<int>> children;
    for (const auto &pair : graph)
    {
        children[pair.first] = std::vector<int>();
    }

    for (const auto &pair : par0)
    {
        int v = pair.first;
        int p = pair.second;
        if (p != -1)
        {
            children[p].push_back(v);
        }
    }

    // Create distance and parent arrays
    std::vector<double> dist(N + 1, std::numeric_limits<double>::infinity());
    std::vector<int> par(N + 1, -1);

    // Fill arrays with initial values
    for (const auto &pair : dist0)
    {
        dist[pair.first] = pair.second;
    }

    for (const auto &pair : par0)
    {
        par[pair.first] = pair.second;
    }

    // Generate random edge deletions and insertions
    std::vector<std::pair<int, int>> existing_edges;
    for (const auto &pair : graph)
    {
        int u = pair.first;
        for (const Edge &edge : pair.second)
        {
            existing_edges.push_back({u, edge.node});
        }
    }

    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());

    // Delete 1000 random edges
    std::vector<std::pair<int, int>> dels;
    // if (existing_edges.size() > 1000)
    // {
    //     std::shuffle(existing_edges.begin(), existing_edges.end(), gen);
    //     dels.assign(existing_edges.begin(), existing_edges.begin() + 1000);
    // }
    // else
    // {
    //     dels = existing_edges;
    // }

    // Generate 1000 random edge insertions
    std::vector<std::tuple<int, int, int>> ins;
    // std::vector<int> nodes;
    // for (const auto &pair : graph)
    // {
    //     nodes.push_back(pair.first);
    // }

    // std::uniform_int_distribution<> weight_dist(1, 100000);
    // std::uniform_int_distribution<> node_dist(0, nodes.size() - 1);

    // while (ins.size() < 1000)
    // {
    //     int u_idx = node_dist(gen);
    //     int v_idx = node_dist(gen);

    //     int u = nodes[u_idx];
    //     int v = nodes[v_idx];

    //     if (u == v)
    //         continue;

    //     // Check if edge already exists
    //     bool edge_exists = false;
    //     for (const Edge &edge : graph.at(u))
    //     {
    //         if (edge.node == v)
    //         {
    //             edge_exists = true;
    //             break;
    //         }
    //     }

    //     if (!edge_exists)
    //     {
    //         int w = weight_dist(gen);
    //         ins.push_back({u, v, w});
    //     }
    // }

    // Process deletions and insertions to generate initial seeds
    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;

    // Handle deletions: subtree resets
    for (const auto &[u, v] : dels)
    {
        if (par[v] == u)
        {
            // Remove edge from graph
            auto &edges = graph.at(u);
            edges.erase(
                std::remove_if(edges.begin(), edges.end(),
                               [v](const Edge &e)
                               { return e.node == v; }),
                edges.end());

            // Reset the subtree
            std::queue<int> q;
            q.push(v);

            while (!q.empty())
            {
                int x = q.front();
                q.pop();

                dist[x] = std::numeric_limits<double>::infinity();
                par[x] = -1;
                pq.push(PQNode(dist[x], x));

                for (int c : children[x])
                {
                    q.push(c);
                }

                children[x].clear();
            }
        }
    }

    // Handle insertions: immediate seeds
    for (const auto &[u, v, w] : ins)
    {
        if (graph.find(u) != graph.end() && graph.find(v) != graph.end())
        {
            // Check if edge already exists
            bool edge_exists = false;
            for (const Edge &edge : graph.at(u))
            {
                if (edge.node == v)
                {
                    edge_exists = true;
                    break;
                }
            }

            if (!edge_exists)
            {
                graph[u].push_back(Edge(v, w));
            }

            int x, y;
            if (dist[u] > dist[v])
            {
                x = u;
                y = v;
            }
            else
            {
                x = v;
                y = u;
            }

            double alt = dist[y] + w;
            if (alt < dist[x])
            {
                dist[x] = alt;
                par[x] = y;
                pq.push(PQNode(alt, x));
            }
        }
    }

    // Distribute initial nodes to workers
    std::vector<std::vector<int>> init_lists(4);

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();

        // Find the worker for this node, default to worker 1 (index 0)
        int w = 0;
        auto it = owner.find(u);
        if (it != owner.end())
        {
            w = it->second - 1;
        }

        init_lists[w].push_back(u);
    }

    // Broadcast and scatter data to workers

    // Prepare subgraphs for scatter
    std::vector<std::unordered_map<int, std::vector<Edge>>> to_scatter_subs;
    to_scatter_subs.push_back({}); // Empty map for master
    for (const auto &sub : subgraphs)
    {
        to_scatter_subs.push_back(sub);
    }

    // Prepare initial lists for scatter
    std::vector<std::vector<int>> to_scatter_init;
    to_scatter_init.push_back({}); // Empty list for master
    for (const auto &init : init_lists)
    {
        to_scatter_init.push_back(init);
    }

    // Broadcast owner map
    int owner_size = owner.size();
    MPI_Bcast(&owner_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> owner_keys(owner_size);
    std::vector<int> owner_vals(owner_size);
    int i = 0;
    for (const auto &pair : owner)
    {
        owner_keys[i] = pair.first;
        owner_vals[i] = pair.second;
        i++;
    }

    MPI_Bcast(owner_keys.data(), owner_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(owner_vals.data(), owner_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast dist and par arrays
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(dist.data(), N + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(par.data(), N + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter subgraphs and initial lists
    // This requires custom serialization - we'll implement it below
    for (int w = 1; w <= 4; w++)
    {
        // Serialize the subgraph
        std::vector<int> sub_data;
        const auto &sub = subgraphs[w - 1];

        // Format: [node_count, node1, edge_count1, edge1_node, edge1_weight, ...]
        sub_data.push_back(sub.size());

        for (const auto &[node, edges] : sub)
        {
            sub_data.push_back(node);
            sub_data.push_back(edges.size());

            for (const Edge &edge : edges)
            {
                sub_data.push_back(edge.node);
                sub_data.push_back(edge.weight);
            }
        }

        // Send the size first, then the data
        int sub_size = sub_data.size();
        MPI_Send(&sub_size, 1, MPI_INT, w, 0, MPI_COMM_WORLD);
        MPI_Send(sub_data.data(), sub_size, MPI_INT, w, 1, MPI_COMM_WORLD);

        // Send initial list
        const auto &init = init_lists[w - 1];
        int init_size = init.size();
        MPI_Send(&init_size, 1, MPI_INT, w, 2, MPI_COMM_WORLD);

        if (init_size > 0)
        {
            MPI_Send(init.data(), init_size, MPI_INT, w, 3, MPI_COMM_WORLD);
        }
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Event-driven loop
    bool done = false;
    std::set<int> workers_done;

    while (!done)
    {
        // Collect messages from all workers
        std::unordered_map<int, std::vector<Message>> all_msgs;

        for (int w = 1; w <= 4; w++)
        {
            // Receive the type of message first
            int msg_type;
            MPI_Status status;
            MPI_Recv(&msg_type, 1, MPI_INT, w, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (msg_type == 0)
            { // WORKER_DONE message
                if (workers_done.find(w) == workers_done.end())
                {
                    workers_done.insert(w);
                    std::cout << "Worker " << w << " has completed" << std::endl;
                }
            }
            else if (msg_type == 1)
            { // Updates message
                int msg_count;
                MPI_Recv(&msg_count, 1, MPI_INT, w, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                std::vector<Message> messages;
                if (msg_count > 0)
                {
                    std::vector<int> nodes(msg_count);
                    std::vector<double> distances(msg_count);
                    std::vector<int> parents(msg_count);

                    MPI_Recv(nodes.data(), msg_count, MPI_INT, w, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    MPI_Recv(distances.data(), msg_count, MPI_DOUBLE, w, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    MPI_Recv(parents.data(), msg_count, MPI_INT, w, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                    for (int i = 0; i < msg_count; i++)
                    {
                        messages.emplace_back(nodes[i], distances[i], parents[i]);
                    }
                }

                all_msgs[w] = std::move(messages);
            }
        }

        // Check for convergence
        if (workers_done.size() == 4)
        {
            done = true;
            // Notify all workers to terminate
            for (int w = 1; w <= 4; w++)
            {
                int terminate = -1; // Special signal for termination
                MPI_Send(&terminate, 1, MPI_INT, w, 0, MPI_COMM_WORLD);
            }
            break;
        }

        // Route messages to appropriate workers
        std::unordered_map<int, std::vector<Message>> worker_msgs;
        for (int w = 1; w <= 4; w++)
        {
            worker_msgs[w] = std::vector<Message>();
        }

        for (const auto &[w, msgs] : all_msgs)
        {
            if (workers_done.find(w) != workers_done.end())
            {
                continue;
            }

            for (const Message &msg : msgs)
            {
                int tgt = 1; // Default to worker 1
                auto it = owner.find(msg.node);
                if (it != owner.end())
                {
                    tgt = it->second;
                }

                // Only send to workers that aren't done
                if (workers_done.find(tgt) == workers_done.end())
                {
                    worker_msgs[tgt].push_back(msg);
                }
            }
        }

        // Send messages to each worker
        for (int w = 1; w <= 4; w++)
        {
            if (workers_done.find(w) != workers_done.end())
            {
                // Send empty message to workers that are done
                int empty = 0;
                MPI_Send(&empty, 1, MPI_INT, w, 0, MPI_COMM_WORLD);
            }
            else
            {
                const auto &msgs = worker_msgs[w];
                int msg_count = msgs.size();
                MPI_Send(&msg_count, 1, MPI_INT, w, 0, MPI_COMM_WORLD);

                if (msg_count > 0)
                {
                    std::vector<int> nodes(msg_count);
                    std::vector<double> distances(msg_count);
                    std::vector<int> parents(msg_count);

                    for (int i = 0; i < msg_count; i++)
                    {
                        const auto &msg = msgs[i];
                        nodes[i] = msg.node;
                        distances[i] = msg.dist;
                        parents[i] = msg.parent;
                    }

                    MPI_Send(nodes.data(), msg_count, MPI_INT, w, 0, MPI_COMM_WORLD);
                    MPI_Send(distances.data(), msg_count, MPI_DOUBLE, w, 0, MPI_COMM_WORLD);
                    MPI_Send(parents.data(), msg_count, MPI_INT, w, 0, MPI_COMM_WORLD);
                }
            }
        }
    }

    // Measure elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "Final tree:" << std::endl;

    // Can uncomment to print the final tree
    // std::cout << "Node\tDist\tPar" << std::endl;
    // for (int u = 1; u <= N; u++) {
    //     std::string d_str = std::isinf(dist[u]) ? "∞" : std::to_string(dist[u]);
    //     std::string p_str = (par[u] < 0) ? "-" : std::to_string(par[u]);
    //     std::cout << u << "\t" << d_str << "\t" << p_str << std::endl;
    // }
}

// Worker process function
void worker_process(int rank)
{
    // Receive owner map
    int owner_size;
    MPI_Bcast(&owner_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> owner_keys(owner_size);
    std::vector<int> owner_vals(owner_size);

    MPI_Bcast(owner_keys.data(), owner_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(owner_vals.data(), owner_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::unordered_map<int, int> owner;
    for (int i = 0; i < owner_size; i++)
    {
        owner[owner_keys[i]] = owner_vals[i];
    }

    // Receive N and dist/par arrays
    int N;
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> dist(N + 1, std::numeric_limits<double>::infinity());
    std::vector<int> par(N + 1, -1);

    MPI_Bcast(dist.data(), N + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(par.data(), N + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Receive subgraph
    int sub_size;
    MPI_Status status;
    MPI_Recv(&sub_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    std::vector<int> sub_data(sub_size);
    MPI_Recv(sub_data.data(), sub_size, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

    // Deserialize the subgraph
    std::unordered_map<int, std::vector<Edge>> my_sub;
    int pos = 0;
    int node_count = sub_data[pos++];

    for (int i = 0; i < node_count; i++)
    {
        int node = sub_data[pos++];
        int edge_count = sub_data[pos++];

        std::vector<Edge> edges;
        for (int j = 0; j < edge_count; j++)
        {
            int edge_node = sub_data[pos++];
            int edge_weight = sub_data[pos++];
            edges.emplace_back(edge_node, edge_weight);
        }

        my_sub[node] = std::move(edges);
    }

    // Receive initial list
    int init_size;
    MPI_Recv(&init_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);

    std::set<int> my_aff;
    if (init_size > 0)
    {
        std::vector<int> init_data(init_size);
        MPI_Recv(init_data.data(), init_size, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);

        for (int node : init_data)
        {
            my_aff.insert(node);
        }
    }

    // Event-driven loop
    bool done = false;
    bool sent_done = false;

    while (!done)
    {
        // Process local updates
        std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;
        for (int u : my_aff)
        {
            pq.push(PQNode(dist[u], u));
        }
        my_aff.clear();

        std::vector<Message> out;

        while (!pq.empty())
        {
            auto [du, u] = pq.top();
            pq.pop();

            // Skip if we've found a better path already
            if (du != dist[u])
                continue;

            // Process all neighbors in my subgraph
            auto it = my_sub.find(u);
            if (it != my_sub.end())
            {
                for (const Edge &edge : it->second)
                {
                    int v = edge.node;
                    int w = edge.weight;

                    double alt = du + w;
                    if (alt < dist[v])
                    {
                        dist[v] = alt;
                        par[v] = u;

                        // If the node is in my subgraph, process it locally
                        if (my_sub.find(v) != my_sub.end())
                        {
                            pq.push(PQNode(alt, v));
                        }
                        else
                        {
                            // Otherwise, send it to the master for routing
                            out.emplace_back(v, alt, u);
                        }
                    }
                }
            }
        }

        // Send updates or completion message
        if (out.empty() && !sent_done)
        {
            // Send WORKER_DONE message
            int msg_type = 0; // WORKER_DONE
            MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            std::cout << "Worker " << rank << " sending completion signal" << std::endl;
            sent_done = true;
        }
        else
        {
            int msg_type = 1; // Updates
            MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            int msg_count = out.size();
            MPI_Send(&msg_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            if (msg_count > 0)
            {
                std::vector<int> nodes(msg_count);
                std::vector<double> distances(msg_count);
                std::vector<int> parents(msg_count);

                for (int i = 0; i < msg_count; i++)
                {
                    const auto &msg = out[i];
                    nodes[i] = msg.node;
                    distances[i] = msg.dist;
                    parents[i] = msg.parent;
                }

                MPI_Send(nodes.data(), msg_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(distances.data(), msg_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                MPI_Send(parents.data(), msg_count, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }

        // Receive new messages from master
        int msg_count;
        MPI_Recv(&msg_count, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // Check for termination signal
        if (msg_count == -1)
        {
            done = true;
            std::cout << "Worker " << rank << " terminating" << std::endl;
            break;
        }

        // Skip processing updates if we've already signaled completion
        if (sent_done)
        {
            continue;
        }

        // Process new updates
        bool my_updates = false;

        if (msg_count > 0)
        {
            std::vector<int> nodes(msg_count);
            std::vector<double> distances(msg_count);
            std::vector<int> parents(msg_count);

            MPI_Recv(nodes.data(), msg_count, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(distances.data(), msg_count, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(parents.data(), msg_count, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            for (int i = 0; i < msg_count; i++)
            {
                int u = nodes[i];
                double nd = distances[i];
                int np = parents[i];

                if (nd < dist[u])
                {
                    dist[u] = nd;
                    par[u] = np;
                    my_aff.insert(u);
                    my_updates = true;
                }
            }
        }

        // If no more updates and empty queue, signal completion
        if (!my_updates && my_aff.empty() && !sent_done)
        {
            int msg_type = 0; // WORKER_DONE
            MPI_Send(&msg_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            std::cout << "Worker " << rank << " sending completion signal" << std::endl;
            sent_done = true;
        }
    }
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check that we have 5 processes (1 master + 4 workers)
    if (size != 5)
    {
        if (rank == 0)
        {
            std::cout << "Run with 5 ranks: 1 master + 4 workers" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    // Run the appropriate function based on rank
    if (rank == 0)
    {
        master_process();
    }
    else
    {
        worker_process(rank);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}