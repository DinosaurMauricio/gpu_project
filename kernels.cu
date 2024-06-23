#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

__global__ void transposeWithTiledPartition(DATA_TYPE *odata, const DATA_TYPE *idata, int matrixSize)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> tile32 = cg::tiled_partition<TILE_DIM>(block);
    
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + tile32.thread_rank();
    int y = blockIdx.y * TILE_DIM + tile32.meta_group_rank();

    // Load data into shared memory
    tile[tile32.meta_group_rank()][tile32.thread_rank()] = idata[y * matrixSize + x];


    block.sync();

    x = blockIdx.y * TILE_DIM + tile32.thread_rank();
    y = blockIdx.x * TILE_DIM + tile32.meta_group_rank();

    // Store transposed data from shared memory to global memory
    odata[y * matrixSize + x] = tile[tile32.thread_rank()][tile32.meta_group_rank()];
}

__global__ void transposeTileKernelChild(DATA_TYPE *odata, const DATA_TYPE *idata, int xOffset, int yOffset, int matrixSize) {
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];

    int x = xOffset + threadIdx.x;
    int y = yOffset + threadIdx.y;

    if (x < matrixSize && y < matrixSize) {
        tile[threadIdx.y][threadIdx.x] = idata[y * matrixSize + x];
    }

    __syncthreads();

    x = yOffset + threadIdx.x;
    y = xOffset + threadIdx.y;

    if (x < matrixSize && y < matrixSize) {
        odata[y * matrixSize + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transposeKernelParent(DATA_TYPE *odata, const DATA_TYPE *idata, int matrixSize) {
    int xTile = blockIdx.x * TILE_DIM;
    int yTile = blockIdx.y * TILE_DIM;

    if (xTile < matrixSize && yTile < matrixSize) {
        transposeTileKernelChild<<<1, dim3(TILE_DIM, TILE_DIM)>>>(odata, idata, xTile, yTile, matrixSize);
    }
}

