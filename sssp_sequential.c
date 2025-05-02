#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

// Define edge structure
typedef struct Edge {
    int to;
    int weight;
    struct Edge* next;
} Edge;

// Define graph structure
typedef struct Graph {
    int V;              // Number of vertices
    Edge** adj;         // Adjacency list
} Graph;

// Define a simple queue for BFS traversal
typedef struct QueueNode {
    int data;
    struct QueueNode* next;
} QueueNode;

typedef struct Queue {
    QueueNode* front;
    QueueNode* rear;
} Queue;

// Define a min-heap (priority queue) for Dijkstra's algorithm
typedef struct {
    int vertex;
    int dist;
} HeapNode;

typedef struct {
    int capacity;
    int size;
    int* pos;       // Position of vertices in the heap
    HeapNode** array;
} MinHeap;

// Define a structure for edge changes
typedef struct {
    int u;
    int v;
    int weight;
    bool isInsert;  // true for insert, false for delete
} EdgeChange;

typedef struct {
    int size;
    EdgeChange* changes;
} ChangeList;

// ==================== Queue Functions ====================

// Create a new queue
Queue* createQueue() {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    q->front = q->rear = NULL;
    return q;
}

// Check if queue is empty
bool isQueueEmpty(Queue* q) {
    return q->front == NULL;
}

// Enqueue a vertex
void enqueue(Queue* q, int x) {
    QueueNode* temp = (QueueNode*)malloc(sizeof(QueueNode));
    temp->data = x;
    temp->next = NULL;
    
    if (q->rear == NULL) {
        q->front = q->rear = temp;
        return;
    }
    
    q->rear->next = temp;
    q->rear = temp;
}

// Dequeue a vertex
int dequeue(Queue* q) {
    if (isQueueEmpty(q))
        return -1;
    
    QueueNode* temp = q->front;
    int data = temp->data;
    
    q->front = q->front->next;
    
    if (q->front == NULL)
        q->rear = NULL;
    
    free(temp);
    return data;
}

// Free the queue
void freeQueue(Queue* q) {
    while (!isQueueEmpty(q)) {
        dequeue(q);
    }
    free(q);
}

// ==================== MinHeap Functions ====================

// Create a new heap node
HeapNode* newHeapNode(int v, int dist) {
    HeapNode* node = (HeapNode*)malloc(sizeof(HeapNode));
    node->vertex = v;
    node->dist = dist;
    return node;
}

// Create a min heap
MinHeap* createMinHeap(int capacity) {
    MinHeap* minHeap = (MinHeap*)malloc(sizeof(MinHeap));
    minHeap->pos = (int*)malloc(capacity * sizeof(int));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array = (HeapNode**)malloc(capacity * sizeof(HeapNode*));
    return minHeap;
}

// Swap two nodes in the min heap
void swapHeapNodes(HeapNode** a, HeapNode** b) {
    HeapNode* t = *a;
    *a = *b;
    *b = t;
}

// Heapify at given index
void minHeapify(MinHeap* minHeap, int idx) {
    int smallest, left, right;
    smallest = idx;
    left = 2 * idx + 1;
    right = 2 * idx + 2;
    
    if (left < minHeap->size && 
        minHeap->array[left]->dist < minHeap->array[smallest]->dist)
        smallest = left;
    
    if (right < minHeap->size && 
        minHeap->array[right]->dist < minHeap->array[smallest]->dist)
        smallest = right;
    
    if (smallest != idx) {
        // Update position of vertices in position array
        HeapNode* smallestNode = minHeap->array[smallest];
        HeapNode* idxNode = minHeap->array[idx];
        
        minHeap->pos[smallestNode->vertex] = idx;
        minHeap->pos[idxNode->vertex] = smallest;
        
        // Swap nodes
        swapHeapNodes(&minHeap->array[smallest], &minHeap->array[idx]);
        
        minHeapify(minHeap, smallest);
    }
}

// Check if heap is empty
int isHeapEmpty(MinHeap* minHeap) {
    return minHeap->size == 0;
}

// Extract the minimum node
HeapNode* extractMin(MinHeap* minHeap) {
    if (isHeapEmpty(minHeap))
        return NULL;
    
    // Store the root node
    HeapNode* root = minHeap->array[0];
    
    // Replace root with last node
    HeapNode* lastNode = minHeap->array[minHeap->size - 1];
    minHeap->array[0] = lastNode;
    
    // Update position of the last node
    minHeap->pos[root->vertex] = minHeap->size - 1;
    minHeap->pos[lastNode->vertex] = 0;
    
    // Reduce heap size and heapify root
    --minHeap->size;
    minHeapify(minHeap, 0);
    
    return root;
}

// Decrease key value of a given vertex
void decreaseKey(MinHeap* minHeap, int v, int dist) {
    // Get the index of v in heap array
    int i = minHeap->pos[v];
    
    // Update the distance value
    minHeap->array[i]->dist = dist;
    
    // Heapify up: keep swapping until reaching correct position
    while (i && minHeap->array[i]->dist < minHeap->array[(i - 1) / 2]->dist) {
        // Swap with parent
        minHeap->pos[minHeap->array[i]->vertex] = (i - 1) / 2;
        minHeap->pos[minHeap->array[(i - 1) / 2]->vertex] = i;
        swapHeapNodes(&minHeap->array[i], &minHeap->array[(i - 1) / 2]);
        
        // Move to parent index
        i = (i - 1) / 2;
    }
}

// Check if vertex is in heap
bool isInMinHeap(MinHeap* minHeap, int v) {
    if (minHeap->pos[v] < minHeap->size)
        return true;
    return false;
}

// Free the min heap
void freeMinHeap(MinHeap* minHeap) {
    for (int i = 0; i < minHeap->size; i++) {
        free(minHeap->array[i]);
    }
    free(minHeap->array);
    free(minHeap->pos);
    free(minHeap);
}

// ==================== Graph Functions ====================

// Create a new graph
Graph* createGraph(int V) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->adj = (Edge**)malloc(V * sizeof(Edge*));
    
    for (int i = 0; i < V; i++) {
        graph->adj[i] = NULL;
    }
    
    return graph;
}

// Add an edge to the graph
void addEdge(Graph* graph, int src, int dest, int weight) {
    // Add edge from src to dest
    Edge* newEdge = (Edge*)malloc(sizeof(Edge));
    newEdge->to = dest;
    newEdge->weight = weight;
    newEdge->next = graph->adj[src];
    graph->adj[src] = newEdge;
}

// Remove an edge from the graph
bool removeEdge(Graph* graph, int src, int dest) {
    Edge* curr = graph->adj[src];
    Edge* prev = NULL;
    
    while (curr != NULL) {
        if (curr->to == dest) {
            if (prev == NULL) {
                // First edge in list
                graph->adj[src] = curr->next;
            } else {
                prev->next = curr->next;
            }
            free(curr);
            return true;
        }
        prev = curr;
        curr = curr->next;
    }
    
    return false;  // Edge not found
}

// Free the graph
void freeGraph(Graph* graph) {
    for (int i = 0; i < graph->V; i++) {
        Edge* current = graph->adj[i];
        while (current != NULL) {
            Edge* next = current->next;
            free(current);
            current = next;
        }
    }
    free(graph->adj);
    free(graph);
}

// ==================== SSSP Functions ====================

// Compute initial SSSP using Dijkstra's algorithm
void computeInitialSSSP(Graph* graph, int source, int* dist, int* parent) {
    int V = graph->V;
    bool* inHeap = (bool*)malloc(V * sizeof(bool));
    
    // Initialize distances and parent pointers
    for (int v = 0; v < V; v++) {
        dist[v] = INT_MAX;
        parent[v] = -1;
        inHeap[v] = true;
    }
    
    // Distance of source vertex from itself is 0
    dist[source] = 0;
    
    // Create a min heap
    MinHeap* minHeap = createMinHeap(V);
    
    // Add all vertices to the min heap
    for (int v = 0; v < V; v++) {
        minHeap->array[v] = newHeapNode(v, dist[v]);
        minHeap->pos[v] = v;
    }
    
    // Set size of min heap
    minHeap->size = V;
    
    // Decrease key value of source to 0
    decreaseKey(minHeap, source, dist[source]);
    
    // Process all vertices
    while (!isHeapEmpty(minHeap)) {
        // Extract the vertex with minimum distance
        HeapNode* minNode = extractMin(minHeap);
        int u = minNode->vertex;
        
        // Mark vertex as processed
        inHeap[u] = false;
        
        // Relax all adjacent vertices
        Edge* edge = graph->adj[u];
        while (edge != NULL) {
            int v = edge->to;
            
            // If v is in heap and distance update is needed
            if (inHeap[v] && dist[u] != INT_MAX && 
                edge->weight + dist[u] < dist[v]) {
                // Update distance
                dist[v] = dist[u] + edge->weight;
                // Update parent
                parent[v] = u;
                // Update heap
                decreaseKey(minHeap, v, dist[v]);
            }
            
            edge = edge->next;
        }
        
        free(minNode);
    }
    
    freeMinHeap(minHeap);
    free(inHeap);
}

// Traverse subtree and mark vertices as affected
void markSubtreeAffected(Graph* graph, int node, int* parent, bool* affected, int* dist) {
    Queue* q = createQueue();
    enqueue(q, node);
    
    while (!isQueueEmpty(q)) {
        int u = dequeue(q);
        affected[u] = true;
        dist[u] = INT_MAX;
        
        // Find all children of u in the SSSP tree
        for (int v = 0; v < graph->V; v++) {
            if (parent[v] == u) {
                enqueue(q, v);
            }
        }
    }
    
    freeQueue(q);
}

// Process edge insertion
void processInsertion(Graph* graph, int u, int v, int weight, int* dist, int* parent, bool* affected) {
    // Check if the new edge provides a shorter path
    if (dist[u] != INT_MAX && dist[v] > dist[u] + weight) {
        dist[v] = dist[u] + weight;
        parent[v] = u;
        affected[v] = true;
    }
}

// Process edge deletion
void processDeletion(Graph* graph, int u, int v, int* dist, int* parent, bool* affected) {
    // Check if the deleted edge was part of the SSSP tree
    if (parent[v] == u) {
        // Disconnect v and its subtree
        parent[v] = -1;
        // Mark v and its subtree as affected
        markSubtreeAffected(graph, v, parent, affected, dist);
    }
}

// Propagate updates through the graph
void propagateUpdates(Graph* graph, bool* affected, int* dist, int* parent) {
    Queue* q = createQueue();
    bool* inQueue = (bool*)malloc(graph->V * sizeof(bool));
    
    // Initialize inQueue array
    for (int i = 0; i < graph->V; i++) {
        inQueue[i] = false;
    }
    
    // Add all affected vertices to the queue
    for (int i = 0; i < graph->V; i++) {
        if (affected[i]) {
            enqueue(q, i);
            inQueue[i] = true;
        }
    }
    
    // Process queue until empty
    while (!isQueueEmpty(q)) {
        int u = dequeue(q);
        inQueue[u] = false;
        affected[u] = false;
        
        // Check all edges from u
        Edge* edge = graph->adj[u];
        while (edge != NULL) {
            int v = edge->to;
            int weight = edge->weight;
            
            // Relaxation: if a shorter path to v is found
            if (dist[u] != INT_MAX && dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                
                // Add v to queue if not already in
                if (!inQueue[v]) {
                    enqueue(q, v);
                    inQueue[v] = true;
                    affected[v] = true;
                }
            }
            
            edge = edge->next;
        }
    }
    
    freeQueue(q);
    free(inQueue);
}

// Update SSSP for a batch of edge changes
void updateSSSP(Graph* graph, ChangeList* changes, int* dist, int* parent) {
    bool* affected = (bool*)malloc(graph->V * sizeof(bool));
    
    // Initialize affected array
    for (int i = 0; i < graph->V; i++) {
        affected[i] = false;
    }
    
    // Process each edge change
    for (int i = 0; i < changes->size; i++) {
        EdgeChange change = changes->changes[i];
        
        if (change.isInsert) {
            // Add edge to graph
            addEdge(graph, change.u, change.v, change.weight);
            // Process insertion
            processInsertion(graph, change.u, change.v, change.weight, dist, parent, affected);
        } else {
            // Process deletion before removing
            processDeletion(graph, change.u, change.v, dist, parent, affected);
            // Remove edge from graph
            removeEdge(graph, change.u, change.v);
        }
    }
    
    // Propagate updates
    propagateUpdates(graph, affected, dist, parent);
    
    free(affected);
}

// ==================== Main Function ====================

int main() {
    // Example usage
    int V = 6;  // Number of vertices
    Graph* graph = createGraph(V);
    
    // Add edges to create initial graph
    addEdge(graph, 0, 1, 5);
    addEdge(graph, 0, 2, 3);
    addEdge(graph, 1, 3, 6);
    addEdge(graph, 1, 2, 2);
    addEdge(graph, 2, 4, 4);
    addEdge(graph, 2, 3, 7);
    addEdge(graph, 3, 4, 1);
    addEdge(graph, 3, 5, 8);
    addEdge(graph, 4, 5, 2);
    
    // Initialize SSSP arrays
    int* dist = (int*)malloc(V * sizeof(int));
    int* parent = (int*)malloc(V * sizeof(int));
    
    // Compute initial SSSP
    int source = 0;
    computeInitialSSSP(graph, source, dist, parent);
    
    // Print initial distances
    printf("Initial shortest distances from source %d:\n", source);
    for (int i = 0; i < V; i++) {
        printf("Vertex %d: Distance = %d, Parent = %d\n", 
               i, dist[i], parent[i]);
    }
    
    // Create a batch of edge changes
    EdgeChange changes[2];
    changes[0].u = 1;
    changes[0].v = 4;
    changes[0].weight = 2;
    changes[0].isInsert = true;
    
    changes[1].u = 3;
    changes[1].v = 4;
    changes[1].isInsert = false;
    
    ChangeList changeList;
    changeList.size = 2;
    changeList.changes = changes;
    
    // Update SSSP
    updateSSSP(graph, &changeList, dist, parent);
    
    // Print updated distances
    printf("\nUpdated shortest distances after edge changes:\n");
    for (int i = 0; i < V; i++) {
        printf("Vertex %d: Distance = %d, Parent = %d\n", 
               i, dist[i], parent[i]);
    }
    
    // Free memory
    free(dist);
    free(parent);
    freeGraph(graph);
    
    return 0;
}