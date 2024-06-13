# GPU Computing: Homework 2

This repository contains the code and results for Homework 2 of the GPU Computing course. The homework focuses on analyzing the performance of a matrix transposition algorithm using CUDA.

## Makefile 
Use the Makefile to compile and run the program with specific options. Here's how you can use it

```bash
make [DATA_TYPE=float] [BANDWIDTH_PERFORMANCE=O3] [UNROLL_FLAG=true] [TILE_DIM=32] [BLOCK_ROWS=8]

[DATA_TYPE=float]: Specifies the data type used in the computation. Default is float.
[BANDWIDTH_PERFORMANCE=O3]: Sets the optimization level. Default is O3.
[UNROLL_FLAG=true]: Enables loop unrolling if set to true. Default is false.
[TILE_DIM=32]: Defines the dimensions of the CUDA tile. Default is 32.
[BLOCK_ROWS=8]: Specifies the number of rows per block. Default is 8
```

## Example to use bash
To run the program using bash. The  kernel  will be executed the number of runs specified in 5 trials and the average will be recorded.
```
sbatch batch.sh <matrix_size> <number_of_runs>

<matrix_size>: Specify the size of the square matrices to be transposed. It will be the power of two, e.g. 3 will be a matrix of 8x8
<number_of_runs>: Specify the number of runs the kernel should execute per run. Default is 100.
```

To run the program with a matrix size of 16x16 and execute 10 times the kernel per trial (in total 5 trials are ran):

```bash
sbatch batch.sh  4 10
```