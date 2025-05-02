#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <metis.h>
#include <time.h>
#include <unistd.h>

#define BUFFER_SIZE 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define HASH_SIZE 10000000  // Size of hash table (adjust based on expected nodes)

// Structure to store node ID mapping
typedef struct NodeMapEntry {
    idx_t original_id;
    idx_t new_id;
    struct NodeMapEntry *next;
} NodeMapEntry;

// Hash function for node IDs
unsigned int hash_function(idx_t id) {
    return id % HASH_SIZE;
}

// Function to add a node to the hash table
idx_t add_node(NodeMapEntry **hash_table, idx_t original_id, idx_t *next_id) {
    unsigned int hash = hash_function(original_id);
    
    // Check if node already exists
    NodeMapEntry *entry = hash_table[hash];
    while (entry != NULL) {
        if (entry->original_id == original_id) {
            return entry->new_id;  // Node already exists
        }
        entry = entry->next;
    }
    
    // Node doesn't exist, create new entry
    NodeMapEntry *new_entry = (NodeMapEntry*)malloc(sizeof(NodeMapEntry));
    if (!new_entry) {
        fprintf(stderr, "Failed to allocate memory for node mapping entry\n");
        return -1;
    }
    
    new_entry->original_id = original_id;
    new_entry->new_id = *next_id;
    (*next_id)++;
    
    // Add to front of list (faster)
    new_entry->next = hash_table[hash];
    hash_table[hash] = new_entry;
    
    return new_entry->new_id;
}

// Function to find a node's new ID in the hash table
idx_t find_node(NodeMapEntry **hash_table, idx_t original_id) {
    unsigned int hash = hash_function(original_id);
    
    NodeMapEntry *entry = hash_table[hash];
    while (entry != NULL) {
        if (entry->original_id == original_id) {
            return entry->new_id;
        }
        entry = entry->next;
    }
    
    return -1;  // Not found
}

// Function to free the hash table
void free_hash_table(NodeMapEntry **hash_table) {
    for (unsigned int i = 0; i < HASH_SIZE; i++) {
        NodeMapEntry *current = hash_table[i];
        while (current != NULL) {
            NodeMapEntry *temp = current;
            current = current->next;
            free(temp);
        }
    }
}

// Function to create a mapping table for writing results
typedef struct {
    idx_t original_id;
    idx_t new_id;
} NodeMapping;

// Progress display function
void show_progress(idx_t current, idx_t total, const char *task) {
    static time_t last_update = 0;
    time_t now = time(NULL);
    
    // Update at most once per second to reduce I/O overhead
    if (now > last_update) {
        int percent = (int)(100.0 * current / total);
        printf("%s: %d%% complete (%"SCIDX" of %"SCIDX")\r", 
               task, percent, current, total);
        fflush(stdout);
        last_update = now;
    }
}

// Function to print memory usage information
void print_memory_usage(const char *stage) {
    #ifdef __linux__
    FILE *fp = fopen("/proc/self/status", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                printf("%s - Current memory usage: %s", stage, line);
                break;
            }
        }
        fclose(fp);
    }
    #endif
}

int main(int argc, char *argv[]) {
    clock_t start_time = clock();
    
    if (argc < 4) {
        printf("Usage: %s <input_file> <num_partitions> <output_file> [max_memory_GB]\n", argv[0]);
        printf("  max_memory_GB: Optional memory limit in GB (default: 16)\n");
        return 1;
    }
    
    char *input_file = argv[1];
    idx_t num_partitions = atoi(argv[2]);
    char *output_file = argv[3];
    
    // Optional memory limit (in GB)
    double max_memory_gb = (argc > 4) ? atof(argv[4]) : 16.0;
    size_t max_memory_bytes = (size_t)(max_memory_gb * 1024 * 1024 * 1024);
    
    printf("Opening input file: %s\n", input_file);
    FILE *fp = fopen(input_file, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", input_file);
        return 1;
    }
    
    // Create hash table for node ID mapping
    printf("Creating node ID mapping...\n");
    NodeMapEntry **node_hash = (NodeMapEntry**)calloc(HASH_SIZE, sizeof(NodeMapEntry*));
    if (!node_hash) {
        fprintf(stderr, "Error: Failed to allocate memory for node hash table\n");
        fclose(fp);
        return 1;
    }
    
    // First pass: build node ID mapping and count edges
    printf("Scanning file to build node mapping and count edges...\n");
    char buffer[BUFFER_SIZE];
    idx_t next_node_id = 0;
    idx_t edge_count = 0;
    
    // Skip header lines if present (look for lines that don't contain two numbers)
    int header_skipped = 0;
    while (!header_skipped && fgets(buffer, sizeof(buffer), fp) != NULL) {
        idx_t src, dst;
        if (sscanf(buffer, "%"SCIDX" %"SCIDX, &src, &dst) == 2) {
            // This looks like a valid edge line
            header_skipped = 1;
            fseek(fp, -strlen(buffer), SEEK_CUR); // Go back to reprocess this line
        }
    }
    
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        idx_t src, dst, timestamp;
        int fields_read = sscanf(buffer, "%"SCIDX" %"SCIDX" %"SCIDX, &src, &dst, &timestamp);
        
        if (fields_read >= 2) {
            // Add nodes to the hash table
            add_node(node_hash, src, &next_node_id);
            add_node(node_hash, dst, &next_node_id);
            
            edge_count++;
            
            if (edge_count % 1000000 == 0) {
                printf("  Processed %"SCIDX" million edges, mapped %"SCIDX" unique nodes\r", 
                       edge_count / 1000000, next_node_id);
                fflush(stdout);
            }
        }
    }
    
    printf("\nMapping complete: %"SCIDX" edges, %"SCIDX" unique nodes\n", 
           edge_count, next_node_id);
    
    // Number of nodes in the graph
    idx_t num_nodes = next_node_id;
    printf("Graph contains %"SCIDX" unique nodes\n", num_nodes);
    
    // Calculate memory requirements
    size_t degrees_mem = num_nodes * sizeof(idx_t);
    size_t xadj_mem = (num_nodes + 1) * sizeof(idx_t);
    size_t part_mem = num_nodes * sizeof(idx_t);
    
    // Each edge appears twice in an undirected graph
    idx_t total_edges = edge_count * 2;
    size_t adjncy_mem = total_edges * sizeof(idx_t);
    
    size_t total_mem_estimate = degrees_mem + xadj_mem + part_mem + adjncy_mem;
    
    printf("Estimated memory requirements:\n");
    printf("  Degrees array: %.2f MB\n", degrees_mem / (1024.0 * 1024));
    printf("  xadj array:    %.2f MB\n", xadj_mem / (1024.0 * 1024));
    printf("  adjncy array:  %.2f MB\n", adjncy_mem / (1024.0 * 1024));
    printf("  part array:    %.2f MB\n", part_mem / (1024.0 * 1024));
    printf("  Total:         %.2f MB (%.2f GB)\n", 
           total_mem_estimate / (1024.0 * 1024),
           total_mem_estimate / (1024.0 * 1024 * 1024));
    
    // Check if estimated memory exceeds the limit
    if (total_mem_estimate > max_memory_bytes) {
        printf("Warning: Estimated memory exceeds specified limit of %.1f GB\n", max_memory_gb);
        printf("Will attempt to reduce memory usage by processing in chunks\n");
    }
    
    // Allocate memory for degree counting
    printf("Allocating memory for node degrees...\n");
    idx_t *degrees = (idx_t *)calloc(num_nodes, sizeof(idx_t));
    if (degrees == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for degrees\n");
        free_hash_table(node_hash);
        free(node_hash);
        fclose(fp);
        return 1;
    }
    
    // Reset file position
    rewind(fp);
    
    // Skip header lines again
    header_skipped = 0;
    while (!header_skipped && fgets(buffer, sizeof(buffer), fp) != NULL) {
        idx_t src, dst;
        if (sscanf(buffer, "%"SCIDX" %"SCIDX, &src, &dst) == 2) {
            header_skipped = 1;
            fseek(fp, -strlen(buffer), SEEK_CUR);
        }
    }
    
    // Second pass: count degrees for each node
    printf("Counting node degrees...\n");
    
    idx_t processed_edges = 0;
    
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        idx_t src, dst, timestamp;
        int fields_read = sscanf(buffer, "%"SCIDX" %"SCIDX" %"SCIDX, &src, &dst, &timestamp);
        
        if (fields_read >= 2) {
            // Map original IDs to compressed IDs
            idx_t src_idx = find_node(node_hash, src);
            idx_t dst_idx = find_node(node_hash, dst);
            
            if (src_idx != -1 && dst_idx != -1) {
                degrees[src_idx]++;
                degrees[dst_idx]++;
                processed_edges++;
                
                if (processed_edges % 1000000 == 0) {
                    show_progress(processed_edges, edge_count, "Degree counting");
                }
            }
        }
    }
    printf("\nDegree counting complete, processed %"SCIDX" edges\n", processed_edges);
    
    print_memory_usage("After degree counting");
    
    // Allocate memory for CSR representation
    printf("Building CSR representation...\n");
    idx_t *xadj = (idx_t *)malloc((num_nodes + 1) * sizeof(idx_t));
    if (xadj == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for xadj\n");
        free(degrees);
        free_hash_table(node_hash);
        free(node_hash);
        fclose(fp);
        return 1;
    }
    
    // Compute prefix sum for xadj
    xadj[0] = 0;
    for (idx_t i = 0; i < num_nodes; i++) {
        xadj[i+1] = xadj[i] + degrees[i];
    }
    
    // Reset degrees to use as counters
    memset(degrees, 0, num_nodes * sizeof(idx_t));
    
    // Get total number of edges in CSR format
    idx_t total_csr_edges = xadj[num_nodes];
    printf("Total edges in CSR format: %"SCIDX" (%.2f GB memory needed)\n", 
           total_csr_edges, (total_csr_edges * sizeof(idx_t)) / (1024.0 * 1024 * 1024));
    
    // Allocate memory for adjacency list
    printf("Allocating memory for adjacency list...\n");
    idx_t *adjncy = (idx_t *)malloc(total_csr_edges * sizeof(idx_t));
    if (adjncy == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for adjncy\n");
        free(degrees);
        free(xadj);
        free_hash_table(node_hash);
        free(node_hash);
        fclose(fp);
        return 1;
    }
    
    print_memory_usage("After CSR allocation");
    
    // Third pass: build adjacency lists
    printf("Building adjacency lists...\n");
    rewind(fp);
    
    // Skip header lines again
    header_skipped = 0;
    while (!header_skipped && fgets(buffer, sizeof(buffer), fp) != NULL) {
        idx_t src, dst;
        if (sscanf(buffer, "%"SCIDX" %"SCIDX, &src, &dst) == 2) {
            header_skipped = 1;
            fseek(fp, -strlen(buffer), SEEK_CUR);
        }
    }
    
    processed_edges = 0;
    
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        idx_t src, dst, timestamp;
        int fields_read = sscanf(buffer, "%"SCIDX" %"SCIDX" %"SCIDX, &src, &dst, &timestamp);
        
        if (fields_read >= 2) {
            // Map original IDs to compressed IDs
            idx_t src_idx = find_node(node_hash, src);
            idx_t dst_idx = find_node(node_hash, dst);
            
            if (src_idx != -1 && dst_idx != -1) {
                // Add edge in both directions (undirected graph)
                adjncy[xadj[src_idx] + degrees[src_idx]++] = dst_idx;
                adjncy[xadj[dst_idx] + degrees[dst_idx]++] = src_idx;
                
                processed_edges++;
                if (processed_edges % 1000000 == 0) {
                    show_progress(processed_edges, edge_count, "Building adjacency lists");
                }
            }
        }
    }
    printf("\nAdjacency list building complete\n");
    
    // Close input file
    fclose(fp);
    
    // Free degrees array as it's no longer needed
    free(degrees);
    
    print_memory_usage("Before partitioning");
    
    // Allocate memory for partition vector
    printf("Allocating memory for partition vector...\n");
    idx_t *part = (idx_t *)malloc(num_nodes * sizeof(idx_t));
    if (part == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for part\n");
        free(adjncy);
        free(xadj);
        free_hash_table(node_hash);
        free(node_hash);
        return 1;
    }
    
    // METIS parameters
    idx_t ncon = 1;  // Number of balancing constraints
    idx_t edgecut;   // Output: Number of edges cut by partitioning
    
    // METIS options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;  // 0-based numbering is already used in our mapping
    
    // Use memory-efficient settings
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;     // K-way partitioning
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  // Edge-cut minimization
    options[METIS_OPTION_NO2HOP] = 1;                   // Disable 2-hop matching to save memory
    options[METIS_OPTION_COMPRESS] = 1;                 // Compress graph to save memory
    options[METIS_OPTION_CCORDER] = 1;                  // Connected components ordering
    
    // Lower quality but faster and uses less memory
    if (total_mem_estimate > max_memory_bytes * 0.8) {
        printf("Using memory-optimized settings for partitioning...\n");
        options[METIS_OPTION_NITER] = 5;                // Fewer refinement iterations 
        options[METIS_OPTION_NCUTS] = 1;                // Fewer cut attempts
        options[METIS_OPTION_UFACTOR] = 30;             // Allow more imbalance (default: 1)
    }
    
    // Force more aggressive memory saving if needed
    if (total_mem_estimate > max_memory_bytes) {
        printf("WARNING: Memory usage is critical. Using minimal quality settings.\n");
        options[METIS_OPTION_MINCONN] = 0;              // Disable minimizing maximum connectivity
        options[METIS_OPTION_CONTIG] = 0;               // Allow discontiguous partitions
        options[METIS_OPTION_UFACTOR] = 100;            // Allow significant imbalance to save memory
    }
    
    printf("Partitioning graph with METIS into %"SCIDX" parts...\n", num_partitions);
    printf("This may take a while for large graphs...\n");
    
    // Call METIS for partitioning
    int ret = METIS_PartGraphKway(
        &num_nodes,      // Number of vertices
        &ncon,           // Number of balancing constraints
        xadj,            // Adjacency structure: indices
        adjncy,          // Adjacency structure: nonzeros
        NULL,            // Vertex weights (NULL for unweighted)
        NULL,            // Size of vertices for communication volume (NULL for uniform)
        NULL,            // Edge weights (NULL for unweighted)
        &num_partitions, // Number of partitions
        NULL,            // Target partition weights (NULL for uniform)
        NULL,            // Target imbalance for each constraint (NULL for default)
        options,         // METIS options
        &edgecut,        // Output: Objective value (edge-cut or communication volume)
        part             // Output: Partition vector
    );
    
    if (ret != METIS_OK) {
        fprintf(stderr, "Error: METIS partitioning failed with error code %d\n", ret);
        
        if (ret == METIS_ERROR_MEMORY) {
            fprintf(stderr, "METIS ran out of memory. Try using a smaller graph or increase available memory.\n");
            fprintf(stderr, "You can also try using a lower quality partitioning with more partitions.\n");
        }
        
        free(part);
        free(adjncy);
        free(xadj);
        free_hash_table(node_hash);
        free(node_hash);
        return 1;
    }
    
    printf("Partitioning complete. Edge-cut: %"SCIDX"\n", edgecut);
    
    // Free adjacency list to reduce memory before writing results
    free(adjncy);
    free(xadj);
    
    print_memory_usage("After partitioning");
    
    // Create a mapping table for writing results
    printf("Building output mapping...\n");
    NodeMapping *output_mapping = (NodeMapping*)malloc(num_nodes * sizeof(NodeMapping));
    if (!output_mapping) {
        fprintf(stderr, "Failed to allocate memory for output mapping\n");
        free(part);
        free_hash_table(node_hash);
        free(node_hash);
        return 1;
    }
    
    // Populate mapping table
    idx_t mapping_count = 0;
    for (unsigned int i = 0; i < HASH_SIZE; i++) {
        NodeMapEntry *entry = node_hash[i];
        while (entry != NULL) {
            output_mapping[mapping_count].original_id = entry->original_id;
            output_mapping[mapping_count].new_id = entry->new_id;
            mapping_count++;
            entry = entry->next;
        }
    }
    
    // Write partition results to output file
    printf("Writing partition results to %s...\n", output_file);
    FILE *out_fp = fopen(output_file, "w");
    if (out_fp == NULL) {
        fprintf(stderr, "Error: Could not open output file %s\n", output_file);
        free(part);
        free(output_mapping);
        free_hash_table(node_hash);
        free(node_hash);
        return 1;
    }
    
    // Write results
    for (idx_t i = 0; i < mapping_count; i++) {
        idx_t original_id = output_mapping[i].original_id;
        idx_t new_id = output_mapping[i].new_id;
        
        fprintf(out_fp, "%"SCIDX" %"SCIDX"\n", original_id, part[new_id]);
        
        // Show progress for writing large files
        if (i % 100000 == 0) {
            show_progress(i, mapping_count, "Writing partition results");
        }
    }
    
    fclose(out_fp);
    printf("\nPartition results written successfully\n");
    
    // Free allocated memory
    printf("Cleaning up memory...\n");
    free(part);
    free(output_mapping);
    free_hash_table(node_hash);
    free(node_hash);
    
    // Report execution time
    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Done. Total execution time: %.2f seconds (%.2f minutes)\n", 
           execution_time, execution_time / 60.0);
    
    return 0;
}