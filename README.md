# Parallel SSSP on StackOverflow Network

## Group Members
- **Afnan Hassan** — 22i-0991  
- **Ali Ahmed** — 22i-1055  
- **Minahil Ali** — 22i-0849  

## Dataset

We use the **StackOverflow network** dataset from Stanford SNAP:

 **Download**:  
https://snap.stanford.edu/data/sx-stackoverflow.html

 **File**: `sx-stackoverflow.txt.gz`  


**Compilation Instructions (Ubuntu/WSL)**
Sequential Version

Compile the sequential SSSP implementation:

gcc sssp_sequential.c -o sssp
./sssp

Partitioning with METIS

Install METIS:

sudo apt update
sudo apt install libmetis-dev

Compile METIS-based partitioning:

gcc -o a partition_metis2.c -lmetis -lm

Run partitioning:

./a sx-stackoverflow.txt 8 sx_partitions.txt

    sx-stackoverflow.txt: Input dataset
    8: Number of partitions
    sx_partitions.txt: Output file
