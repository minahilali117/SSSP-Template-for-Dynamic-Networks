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

// Dijkstra's algorithm for SSSP
std::pair<std::vector<double>, std::vector<int>>
dijkstra(const std::unordered_map<int, std::vector<Edge>> &graph, int source)
{
    // Find max node id to size our arrays
    int N = 0;
    for (const auto &pair : graph)
    {
        N = std::max(N, pair.first);
    }

    std::vector<double> dist(N + 1, std::numeric_limits<double>::infinity());
    std::vector<int> parent(N + 1, -1);

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
        auto it = graph.find(u);
        if (it != graph.end())
        {
            for (const Edge &edge : it->second)
            {
                int v = edge.node;
                double weight = edge.weight;

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

// Function to build the children tree from the parent array
std::unordered_map<int, std::vector<int>> build_children_tree(const std::unordered_map<int, std::vector<Edge>> &graph, const std::vector<int> &parent)
{
    std::unordered_map<int, std::vector<int>> children;

    // Initialize empty vectors for all nodes
    for (const auto &pair : graph)
    {
        children[pair.first] = std::vector<int>();
    }

    // Build children relationships
    for (size_t v = 0; v < parent.size(); v++)
    {
        int p = parent[v];
        if (p != -1)
        { // Skip the source node and unreachable nodes
            children[p].push_back(v);
        }
    }

    return children;
}

// Function to handle edge deletions
void handle_deletions(std::unordered_map<int, std::vector<Edge>> &graph,
                      std::vector<double> &dist,
                      std::vector<int> &parent,
                      std::unordered_map<int, std::vector<int>> &children,
                      const std::vector<std::pair<int, int>> &dels,
                      std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> &pq)
{

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
            std::queue<int> q;
            q.push(v);

            while (!q.empty())
            {
                int x = q.front();
                q.pop();

                dist[x] = std::numeric_limits<double>::infinity();
                parent[x] = -1;
                pq.push(PQNode(dist[x], x));

                for (int c : children[x])
                {
                    q.push(c);
                }

                children[x].clear();
            }
        }
    }
}

// Function to handle edge insertions
void handle_insertions(std::unordered_map<int, std::vector<Edge>> &graph,
                       std::vector<double> &dist,
                       std::vector<int> &parent,
                       const std::vector<std::tuple<int, int, int>> &ins,
                       std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> &pq)
{

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
                parent[x] = y;
                pq.push(PQNode(alt, x));
            }
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

// Main incremental SSSP processing function
void incremental_sssp(std::unordered_map<int, std::vector<Edge>> &graph,
                      int source,
                      const std::vector<std::pair<int, int>> &dels,
                      const std::vector<std::tuple<int, int, int>> &ins)
{

    // Run initial Dijkstra - not included in timing
    std::cout << "Running initial Dijkstra..." << std::endl;
    auto [dist, parent] = dijkstra(graph, source);

    // Build the children tree - not included in timing
    std::cout << "Building children tree..." << std::endl;
    auto children = build_children_tree(graph, parent);

    // Create priority queue for processing updates
    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;

    // Start timing for incremental operations
    auto start_time = std::chrono::high_resolution_clock::now();

    // Process deletions
    auto start_deletions = std::chrono::high_resolution_clock::now();
    handle_deletions(graph, dist, parent, children, dels, pq);
    auto end_deletions = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deletions_time = end_deletions - start_deletions;

    // Process insertions
    auto start_insertions = std::chrono::high_resolution_clock::now();
    handle_insertions(graph, dist, parent, ins, pq);
    auto end_insertions = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> insertions_time = end_insertions - start_insertions;

    // Process all queued updates (stabilization phase)
    auto start_stabilization = std::chrono::high_resolution_clock::now();
    while (!pq.empty())
    {
        auto [du, u] = pq.top();
        pq.pop();

        // Skip if we've found a better path already
        if (du != dist[u])
            continue;

        // Process all neighbors
        auto it = graph.find(u);
        if (it != graph.end())
        {
            for (const Edge &edge : it->second)
            {
                int v = edge.node;
                int w = edge.weight;

                double alt = du + w;
                if (alt < dist[v])
                {
                    dist[v] = alt;
                    parent[v] = u;

                    // Update children tree
                    for (auto &children_vec : children)
                    {
                        auto &vec = children_vec.second;
                        auto it = std::find(vec.begin(), vec.end(), v);
                        if (it != vec.end())
                        {
                            vec.erase(it);
                            break;
                        }
                    }
                    children[u].push_back(v);

                    pq.push(PQNode(alt, v));
                }
            }
        }
    }
    auto end_stabilization = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> stabilization_time = end_stabilization - start_stabilization;

    // End timing for incremental operations
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_update_time = end_time - start_time;

    // Report timings
    std::cout << "============= PERFORMANCE RESULTS =============" << std::endl;
    std::cout << "Edge Deletions:   " << deletions_time.count() << " seconds" << std::endl;
    std::cout << "Edge Insertions:  " << insertions_time.count() << " seconds" << std::endl;
    std::cout << "Stabilization:    " << stabilization_time.count() << " seconds" << std::endl;
    std::cout << "Total Update Time:" << total_update_time.count() << " seconds" << std::endl;
    std::cout << "=============================================" << std::endl;

    // Print the final tree
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

    // Source node for SSSP
    int source = 1;
    if (argc > 2)
    {
        source = std::stoi(argv[2]);
    }

    std::cout << "Running incremental SSSP from source node " << source << std::endl;
    std::cout << "Processing " << dels.size() << " deletions and " << ins.size() << " insertions" << std::endl;

    // Run the incremental SSSP algorithm
    incremental_sssp(graph, source, dels, ins);

    return 0;
}