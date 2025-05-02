Group members:
Afnan Hassan: 22i-0991
Ali Ahmed: 22i-1055
Minahil Ali: 22i-0849

COMMANDS TO COMPILE (UBUNTU/WSL):
  sequential:
gcc sssp_sequential.c -o sssp

  METIS:
(first make sure to download METIS)
gcc -o a partition_metis2.c -lmetis -lm
./a sx-stackoverflow.txt 8 sx_partitions.txt
(where sx-stackoverflow.txt is the dataset, 8 is number of partitions, sx_partitions.txt is the output file)
