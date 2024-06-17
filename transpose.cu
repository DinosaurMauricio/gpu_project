#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C" {
#include "my_library.h"
}

#ifndef TILE_DIM
#define TILE_DIM 2
#endif

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 4
#endif

__global__ void transposeWithTiledPartition(DATA_TYPE *odata, const DATA_TYPE *idata, int width)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> tile32 = cg::tiled_partition<TILE_DIM>(block);
    
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + tile32.thread_rank();
    int y = blockIdx.y * TILE_DIM + tile32.meta_group_rank();

    // Load data into shared memory
    for (int j = 0; j < TILE_DIM; j += TILE_DIM)
    {
        if (x < width && (y + j) < width)
        {
            tile[tile32.meta_group_rank() + j][tile32.thread_rank()] = idata[(y + j) * width + x];
        }
    }

    block.sync();

    x = blockIdx.y * TILE_DIM + tile32.thread_rank();
    y = blockIdx.x * TILE_DIM + tile32.meta_group_rank();

    // Store transposed data from shared memory to global memory
    for (int j = 0; j < TILE_DIM; j += TILE_DIM)
    {
        if (x < width && (y + j) < width)
        {
            odata[(y + j) * width + x] = tile[tile32.thread_rank()][tile32.meta_group_rank() + j];
        }
    }
}

int main()
{
    const int size = MATRIX_SIZE;
    const int bytes = size * size * sizeof(DATA_TYPE);

    int devID = 0;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&deviceProp, devID);

    if (2 * memory_size > deviceProp.totalGlobalMem) {
        printf("Input matrix size is larger than the available device memory!\n");
        printf("Please choose a smaller size matrix\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Input matrix size is smaller than the available device memory\n");
        printf("Global memory size: %llu\n", (unsigned long long)deviceProp.totalGlobalMem);
        printf("Using memory for matrix: %zu\n", 2 * memory_size);
        printf("Matrix Size %d x %d \n", size,size);
        printf("Tile Dimension %d\n", TILE_DIM);
    }

    DATA_TYPE *h_idata, *h_odata;
    h_idata = (DATA_TYPE*)malloc(bytes);
    h_odata = (DATA_TYPE*)malloc(bytes);

    initializeMatrixValues(h_idata, size);

    printf("Original \n ");
    printMatrix(h_idata, size);
    printf("\n");

    DATA_TYPE *d_idata, *d_odata;
    cudaMalloc(&d_idata, bytes);
    cudaMalloc(&d_odata, bytes);
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    dim3 grid(size / TILE_DIM, size / TILE_DIM, 1);
    dim3 threads(TILE_DIM, TILE_DIM, 1);

    printf("dimGrid: %d %d %d. dimThreads: %d %d %d\n",
           grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);

    transposeWithTiledPartition<<<grid, threads>>>(d_odata, d_idata, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);

    printf("Transposed \n ");
    printMatrix(h_odata, size);
    printf("\n");

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
