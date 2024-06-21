#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 1024
#endif

__global__ void transposeWithTiledPartition(DATA_TYPE *odata, const DATA_TYPE *idata)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> tile32 = cg::tiled_partition<TILE_DIM>(block);
    
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + tile32.thread_rank();
    int y = blockIdx.y * TILE_DIM + tile32.meta_group_rank();

    // Load data into shared memory
    tile[tile32.meta_group_rank()][tile32.thread_rank()] = idata[y * MATRIX_SIZE + x];


    block.sync();

    x = blockIdx.y * TILE_DIM + tile32.thread_rank();
    y = blockIdx.x * TILE_DIM + tile32.meta_group_rank();

    // Store transposed data from shared memory to global memory
    odata[y * MATRIX_SIZE + x] = tile[tile32.thread_rank()][tile32.meta_group_rank()];
}


__global__ void transposeWithThreadCoarsening(DATA_TYPE *odata, const DATA_TYPE *idata) {
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Coarsened load
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += blockDim.x) {
        if (x < MATRIX_SIZE && (y + i) < MATRIX_SIZE) {
            tile[threadIdx.y + i][threadIdx.x] = idata[(y + i) * MATRIX_SIZE + x];
        }
    }

    __syncthreads();

    // Coarsened store
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += blockDim.x) {
        int transposedX = blockIdx.y * TILE_DIM + threadIdx.x;
        int transposedY = blockIdx.x * TILE_DIM + threadIdx.y + i;
        if (transposedX < MATRIX_SIZE && transposedY < MATRIX_SIZE) {
            odata[transposedY * MATRIX_SIZE + transposedX] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

__global__ void transposeWithWarpShuffle(DATA_TYPE *odata, const DATA_TYPE *idata) {
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];

    // Calculate global indexes
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load data into shared memory
    if (x < MATRIX_SIZE && y < MATRIX_SIZE) {
        tile[threadIdx.y][threadIdx.x] = idata[y * MATRIX_SIZE + x];
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
    if (x < MATRIX_SIZE && y < MATRIX_SIZE) {
        odata[y * MATRIX_SIZE + x] = value;
    }
}


__global__ void transposeTileKernelChild(DATA_TYPE *odata, const DATA_TYPE *idata, int xOffset, int yOffset) {
    __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM + 1];

    int x = xOffset + threadIdx.x;
    int y = yOffset + threadIdx.y;

    if (x < MATRIX_SIZE && y < MATRIX_SIZE) {
        tile[threadIdx.y][threadIdx.x] = idata[y * MATRIX_SIZE + x];
    }

    __syncthreads();

    x = yOffset + threadIdx.x;
    y = xOffset + threadIdx.y;

    if (x < MATRIX_SIZE && y < MATRIX_SIZE) {
        odata[y * MATRIX_SIZE + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transposeKernelParent(DATA_TYPE *odata, const DATA_TYPE *idata) {
    int xTile = blockIdx.x * TILE_DIM;
    int yTile = blockIdx.y * TILE_DIM;

    if (xTile < MATRIX_SIZE && yTile < MATRIX_SIZE) {
        transposeTileKernelChild<<<1, dim3(TILE_DIM, TILE_DIM)>>>(odata, idata, xTile, yTile);
    }
}

