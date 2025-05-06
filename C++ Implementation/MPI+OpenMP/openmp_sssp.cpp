#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
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
#include <mutex>

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

// Parallel Dijkstra's algorithm using OpenMP
std::pair<std::vector<double>, std::vector<int>>
parallel_dijkstra(const std::unordered_map<int, std::vector<Edge>> &graph, int source, int num_threads)
{
    // Find max node id to size our arrays
    int N = 0;
    for (const auto &pair : graph)
    {
        N = std::max(N, pair.first);
    }

    std::vector<double> dist(N + 1, std::numeric_limits<double>::infinity());
    std::vector<int> parent(N + 1, -1);
    std::vector<std::mutex> node_mutexes(N + 1); // Mutex per node for synchronization

    // Set source distance to 0
    dist[source] = 0;

    // Priority queue for the algorithm, needs to be thread-safe
    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;
    std::mutex pq_mutex;

    pq.push(PQNode(0, source));

    // Set number of threads
    omp_set_num_threads(num_threads);

    while (!pq.empty())
    {
        // Extract min - this is a sequential part
        pq_mutex.lock();
        if (pq.empty())
        {
            pq_mutex.unlock();
            break;
        }
        PQNode current = pq.top();
        pq.pop();
        pq_mutex.unlock();

        int u = current.node;
        double du = current.dist;

        // Skip if we've found a better path already
        if (du > dist[u])
            continue;

        // Process all neighbors - can be parallelized
        auto it = graph.find(u);
        if (it != graph.end())
        {
            const std::vector<Edge> &edges = it->second;

// Parallel section - process edges in parallel
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < edges.size(); i++)
            {
                const Edge &edge = edges[i];
                int v = edge.node;
                double weight = edge.weight;

                double alt = dist[u] + weight;
                if (alt < dist[v])
                {
                    node_mutexes[v].lock();
                    if (alt < dist[v])
                    { // Check again inside the lock
                        dist[v] = alt;
                        parent[v] = u;

                        // Add to priority queue with lock
                        pq_mutex.lock();
                        pq.push(PQNode(alt, v));
                        pq_mutex.unlock();
                    }
                    node_mutexes[v].unlock();
                }
            }
        }
    }

    return {dist, parent};
}

// Function to build the children tree from the parent array in parallel
std::unordered_map<int, std::vector<int>>
build_children_tree_parallel(const std::unordered_map<int, std::vector<Edge>> &graph,
                             const std::vector<int> &parent, int num_threads)
{
    std::unordered_map<int, std::vector<int>> children;
    std::vector<std::mutex> child_mutexes(graph.size());

    // Initialize empty vectors for all nodes
    for (const auto &pair : graph)
    {
        children[pair.first] = std::vector<int>();
    }

    // Build children relationships in parallel
    omp_set_num_threads(num_threads);

    // Process nodes in chunks
    std::vector<int> nodes;
    for (size_t v = 1; v < parent.size(); v++)
    {
        nodes.push_back(v);
    }

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < nodes.size(); i++)
        {
            int v = nodes[i];
            int p = parent[v];
            if (p != -1)
            {
// This needs synchronization
#pragma omp critical
                {
                    children[p].push_back(v);
                }
            }
        }
    }

    return children;
}

// Function to handle edge deletions with parallel subtree processing
void handle_deletions_parallel(std::unordered_map<int, std::vector<Edge>> &graph,
                               std::vector<double> &dist,
                               std::vector<int> &parent,
                               std::unordered_map<int, std::vector<int>> &children,
                               const std::vector<std::pair<int, int>> &dels,
                               std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> &pq,
                               int num_threads)
{

    std::mutex pq_mutex;

    omp_set_num_threads(num_threads);

    // Process deletions
    for (const auto &[u, v] : dels)
    {
        if (parent[v] == u)
        {
            // Remove edge from graph
            auto &edges = graph.at(u);
            edges.erase(
                std::remove_if(edges.begin(), edges.end(),
                               [v](const Edge &e)
                               { return e.node == v; }),
                edges.end());

            // Reset the subtree rooted at v
            std::vector<int> subtree_nodes;

            // Collect all nodes in the subtree first
            std::queue<int> q;
            q.push(v);

            while (!q.empty())
            {
                int x = q.front();
                q.pop();
                subtree_nodes.push_back(x);

                for (int c : children[x])
                {
                    q.push(c);
                }
            }

// Now process these nodes in parallel
#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < subtree_nodes.size(); i++)
            {
                int x = subtree_nodes[i];

                dist[x] = std::numeric_limits<double>::infinity();
                parent[x] = -1;

// Add to priority queue with synchronization
#pragma omp critical
                {
                    pq.push(PQNode(dist[x], x));
                }
            }

            // Clear children vectors
            for (int x : subtree_nodes)
            {
                children[x].clear();
            }
        }
    }
}

// Function to handle edge insertions with parallel processing
void handle_insertions_parallel(std::unordered_map<int, std::vector<Edge>> &graph,
                                std::vector<double> &dist,
                                std::vector<int> &parent,
                                const std::vector<std::tuple<int, int, int>> &ins,
                                std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> &pq,
                                int num_threads)
{

    std::mutex pq_mutex;
    std::vector<std::mutex> node_mutexes(dist.size());

    omp_set_num_threads(num_threads);

    // Prepare a vector for parallel processing
    std::vector<std::tuple<int, int, int>> valid_ins;

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
                valid_ins.push_back({u, v, w});
            }
        }
    }

// Process valid insertions in parallel
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < valid_ins.size(); i++)
    {
        auto [u, v, w] = valid_ins[i];

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
            node_mutexes[x].lock();
            if (alt < dist[x])
            {
                dist[x] = alt;
                parent[x] = y;

                // Add to priority queue with synchronization
                pq_mutex.lock();
                pq.push(PQNode(alt, x));
                pq_mutex.unlock();
            }
            node_mutexes[x].unlock();
        }
    }
}

// Parallel process for handling updates after edge changes
void process_updates_parallel(std::unordered_map<int, std::vector<Edge>> &graph,
                              std::vector<double> &dist,
                              std::vector<int> &parent,
                              std::unordered_map<int, std::vector<int>> &children,
                              std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> &pq,
                              int num_threads)
{

    std::vector<std::mutex> node_mutexes(dist.size());
    std::mutex pq_mutex;

    omp_set_num_threads(num_threads);

    while (!pq.empty())
    {
        // Extract min - this is a sequential part
        pq_mutex.lock();
        if (pq.empty())
        {
            pq_mutex.unlock();
            break;
        }
        PQNode current = pq.top();
        pq.pop();
        pq_mutex.unlock();

        int u = current.node;
        double du = current.dist;

        // Skip if we've found a better path already
        if (du != dist[u])
            continue;

        // Process all neighbors in parallel
        auto it = graph.find(u);
        if (it != graph.end())
        {
            const std::vector<Edge> &edges = it->second;

#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < edges.size(); i++)
            {
                const Edge &edge = edges[i];
                int v = edge.node;
                int w = edge.weight;

                double alt = du + w;
                if (alt < dist[v])
                {
                    node_mutexes[v].lock();
                    if (alt < dist[v])
                    {
                        dist[v] = alt;

                        // Update parent safely
                        parent[v] = u;

                        // Update the children tree - need to remove v from its old parent's children list
                        // and add it to u's children list
                        // This is complex and would require more synchronization - simplifying here

                        // Add to priority queue with synchronization
                        pq_mutex.lock();
                        pq.push(PQNode(alt, v));
                        pq_mutex.unlock();
                    }
                    node_mutexes[v].unlock();
                }
            }
        }
    }

    // Rebuild the children tree after all updates (simpler than maintaining it incrementally)
    children.clear();
    for (const auto &pair : graph)
    {
        children[pair.first] = std::vector<int>();
    }

    for (size_t v = 0; v < parent.size(); v++)
    {
        int p = parent[v];
        if (p != -1)
        {
            children[p].push_back(v);
        }
    }
}

// Function to print the shortest path tree
void print_tree(const std::vector<double> &dist, const std::vector<int> &parent)
{
    std::cout << "Node\tDist\tParent" << std::endl;
    for (size_t i = 1; i < dist.size(); i++)
    {
        std::string d_str = std::isinf(dist[i]) ? "∞" : std::to_string(dist[i]);
        std::string p_str = (parent[i] < 0) ? "-" : std::to_string(parent[i]);
        std::cout << i << "\t" << d_str << "\t" << p_str << std::endl;
    }
}

// Main incremental SSSP processing function with OpenMP
void parallel_incremental_sssp(std::unordered_map<int, std::vector<Edge>> &graph,
                               int source,
                               const std::vector<std::pair<int, int>> &dels,
                               const std::vector<std::tuple<int, int, int>> &ins,
                               int num_threads)
{

    std::cout << "Running with " << num_threads << " threads" << std::endl;

    // Run initial Dijkstra with OpenMP (not included in timing)
    std::cout << "Running initial Dijkstra..." << std::endl;
    auto [dist, parent] = parallel_dijkstra(graph, source, num_threads);

    // Build the children tree in parallel (not included in timing)
    std::cout << "Building children tree..." << std::endl;
    auto children = build_children_tree_parallel(graph, parent, num_threads);

    // Create priority queue for processing updates
    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;

    // Start timing for the incremental update operations
    auto start_time = std::chrono::high_resolution_clock::now();

    // Process deletions in parallel
    auto start_deletions = std::chrono::high_resolution_clock::now();
    handle_deletions_parallel(graph, dist, parent, children, dels, pq, num_threads);
    auto end_deletions = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deletions_time = end_deletions - start_deletions;

    // Process insertions in parallel
    auto start_insertions = std::chrono::high_resolution_clock::now();
    handle_insertions_parallel(graph, dist, parent, ins, pq, num_threads);
    auto end_insertions = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> insertions_time = end_insertions - start_insertions;

    // Process all queued updates in parallel (stabilization)
    auto start_stabilization = std::chrono::high_resolution_clock::now();
    process_updates_parallel(graph, dist, parent, children, pq, num_threads);
    auto end_stabilization = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> stabilization_time = end_stabilization - start_stabilization;

    // End timing for the incremental update operations
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_update_time = end_time - start_time;

    // Report individual timings
    std::cout << "============= PERFORMANCE RESULTS =============" << std::endl;
    std::cout << "Edge Deletions:   " << deletions_time.count() << " seconds" << std::endl;
    std::cout << "Edge Insertions:  " << insertions_time.count() << " seconds" << std::endl;
    std::cout << "Stabilization:    " << stabilization_time.count() << " seconds" << std::endl;
    std::cout << "Total Update Time:" << total_update_time.count() << " seconds" << std::endl;
    std::cout << "=============================================" << std::endl;

    // Print the final tree (optional - can be expensive for large graphs)
    if (dist.size() <= 100)
    { // Only print for small graphs
        std::cout << "Final shortest path tree:" << std::endl;
        print_tree(dist, parent);
    }
    else
    {
        std::cout << "Graph too large to print. Tree has " << dist.size() - 1 << " nodes." << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // Read the graph
    std::string filename = "dataset/full_dense.txt";
    if (argc > 1)
    {
        filename = argv[1];
    }

    // Source node for SSSP
    int source = 1;
    if (argc > 2)
    {
        source = std::stoi(argv[2]);
    }

    // Number of threads (default to hardware concurrency)
    int num_threads = omp_get_max_threads();
    if (argc > 3)
    {
        num_threads = std::stoi(argv[3]);
    }

    std::cout << "Reading graph from " << filename << std::endl;
    auto graph = read_graph(filename);
    std::cout << "Graph loaded with " << graph.size() << " nodes" << std::endl;

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

    // Delete random edges (or fewer if the graph is small)
    int num_deletions = std::min(static_cast<size_t>(1000), existing_edges.size() / 2);
    std::vector<std::pair<int, int>> dels;
    if (existing_edges.size() > num_deletions)
    {
        std::shuffle(existing_edges.begin(), existing_edges.end(), gen);
        dels.assign(existing_edges.begin(), existing_edges.begin() + num_deletions);
    }
    else
    {
        dels = existing_edges;
    }

    // Generate random edge insertions
    int num_insertions = 1000;
    std::vector<std::tuple<int, int, int>> ins;
    std::vector<int> nodes;
    for (const auto &pair : graph)
    {
        nodes.push_back(pair.first);
    }

    std::uniform_int_distribution<> weight_dist(1, 100);
    std::uniform_int_distribution<> node_dist(0, nodes.size() - 1);

    while (ins.size() < num_insertions && nodes.size() > 1)
    {
        int u_idx = node_dist(gen);
        int v_idx = node_dist(gen);

        int u = nodes[u_idx];
        int v = nodes[v_idx];

        if (u == v)
            continue;

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
            int w = weight_dist(gen);
            ins.push_back({u, v, w});
        }
    }

    std::cout << "Running incremental SSSP from source node " << source << std::endl;
    std::cout << "Processing " << dels.size() << " deletions and " << ins.size() << " insertions" << std::endl;

    // Run the parallel incremental SSSP algorithm
    parallel_incremental_sssp(graph, source, dels, ins, num_threads);

    return 0;
}