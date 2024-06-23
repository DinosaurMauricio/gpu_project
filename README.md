# GPU Computing: Dense Matrix Transposition

This repository contains the code and results for the project made for the GPU Computing course. This work focuses on analyzing the performance of a Dense Matrix Transposition algorithm using CUDA using techniques such as Cooperative Groups and Dynamic Parallelism.

## Makefile 
Use the Makefile to compile and run the program with specific options. Here's how you can use it

```bash
make [DATA_TYPE=float] [TILE_DIM=32] [PRINT_TRANSPOSED_MATRIX=false]

[DATA_TYPE=float]: Specifies the data type used in the computation. Default is float.
[TILE_DIM=32]: Defines the dimensions of the CUDA tile. Default is 32.
[PRINT_TRANSPOSED_MATRIX=false]: To visualize the transposed matrix for each method. In case 
```
NOTE: Because cuBLAS doesn't not include a specific function to tranpose int matrices only float and double data types are supported.

## Example to use bash
To run the program using bash.
```
sbatch batch.sh <matrix_size>

<matrix_size>: Specify the size of the square matrices to be transposed. It will be the power of two, e.g. 3 will be a matrix of 8x8
```

To run the program with a matrix size of 16x16

```bash
sbatch batch.sh  4
```