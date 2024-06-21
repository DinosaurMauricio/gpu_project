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


__global__ void transposeWithThreadCoarsening(DATA_TYPE *odata, const DATA_TYPE *idata, int matrixSize) {
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Coarsened load
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += blockDim.x) {
        if (x < matrixSize && (y + i) < matrixSize) {
            tile[threadIdx.y + i][threadIdx.x] = idata[(y + i) * matrixSize + x];
        }
    }

    __syncthreads();

    // Coarsened store
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += blockDim.x) {
        int transposedX = blockIdx.y * TILE_DIM + threadIdx.x;
        int transposedY = blockIdx.x * TILE_DIM + threadIdx.y + i;
        if (transposedX < matrixSize && transposedY < matrixSize) {
            odata[transposedY * matrixSize + transposedX] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

__global__ void transposeWithWarpShuffle(DATA_TYPE *odata, const DATA_TYPE *idata, int matrixSize) {
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];

    // Calculate global indexes
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load data into shared memory
    if (x < matrixSize && y < matrixSize) {
        tile[threadIdx.y][threadIdx.x] = idata[y * matrixSize + x];
    }

    __syncthreads();

    // Transpose using warp shuffle
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> tile32 = cg::tiled_partition<TILE_DIM>(block);

    int lane = tile32.thread_rank();
    int value = tile[threadIdx.x][threadIdx.y];
    for (int i = TILE_DIM / 2; i > 0; i /= 2) {
        value += tile32.shfl_down(value, i);
    }

    // Calculate transposed indexes
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Store transposed data into global memory
    if (x < matrixSize && y < matrixSize) {
        odata[y * matrixSize + x] = value;
    }
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

