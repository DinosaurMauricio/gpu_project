#include <iostream>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
extern "C" {
#include "my_library.h"
}

#define BLOCK_SIZE 16
#define MATRIX_SIZE 64

__global__ void transpose_block_2d_matrix(float* mat, size_t sx, size_t sy)
{
    constexpr size_t block_size = BLOCK_SIZE;
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<block_size>(block);
    size_t ix = tile.thread_rank();

    // Calculate global indices
    size_t block_x = blockIdx.x * block_size;
    size_t block_y = blockIdx.y * block_size;

    // Define shared memory to hold the tile
    __shared__ float tile_shared[block_size][block_size];

    // Load data into shared memory
    if (block_x + ix < sx)
    {
        for (size_t iy = 0; iy < block_size; ++iy)
        {
            if (block_y + iy < sy)
            {
                tile_shared[ix][iy] = mat[(block_x + ix) + (block_y + iy) * sx];
            }
        }
    }

    block.sync();

    // Transpose data in shared memory and write back to global memory
    if (block_y + ix < sy)
    {
        for (size_t iy = 0; iy < block_size; ++iy)
        {
            if (block_x + iy < sx)
            {
                mat[(block_y + ix) + (block_x + iy) * sx] = tile_shared[iy][ix];
            }
        }
    }
}

void print_mat(float* mat, size_t sx, size_t sy)
{
    printf("{\n");
    for (size_t iy = 0; iy < sy; ++iy)
    {
        printf("\t{ ");
        for (size_t ix = 0; ix < sx; ++ix)
            printf("%6.1f, ", mat[ix + iy * sx]);
        printf("},\n");
    }
    printf("}\n");
}

int main()
{
    cudaEvent_t startEvent, stopEvent;

    // events for timing
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    constexpr size_t sx = MATRIX_SIZE; // Size of the matrix
    constexpr size_t sy = MATRIX_SIZE;
    float* mat;
    int numberOfTests = 100;

    cudaMallocManaged(&mat, sx * sy * sizeof(float));

    for (size_t iy = 0; iy < sy; ++iy)
        for (size_t ix = 0; ix < sx; ++ix)
            mat[ix + sx * iy] = ix + sx * iy;

    print_mat(mat, sx, sy);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((sx + block_size.x - 1) / block_size.x, (sy + block_size.y - 1) / block_size.y);

    printf("Block size: %d %d\n", block_size.x, block_size.y);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z);

    double total_bw = 0;
    float total_ms = 0;
    const int repeat = 5;

    for (int i = 0; i < repeat; i++) {
        unsigned long long effective_bw;
        float ms;

        // Warm up
        transpose_block_2d_matrix<<<grid_size, block_size>>>(mat, sx, sy);
        cudaEventRecord(startEvent, 0);
        for (int i = 0; i < numberOfTests; i++)
            transpose_block_2d_matrix<<<grid_size, block_size>>>(mat, sx, sy);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&ms, startEvent, stopEvent);

        effective_bw = calculate_effective_bandwidth(sx * sy, numberOfTests, ms);

        printf("%20llu %20.2f ms\n", effective_bw, ms);
        total_bw += effective_bw;
        total_ms += ms;
    }

    double avg_bw = total_bw / repeat;
    double avg_ms = total_ms / repeat;

    printf("Average Bandwidth (GB/s): %.2f\n", avg_bw);
    printf("Average Time (ms): %.2f\n", avg_ms);

    print_mat(mat, sx, sy);

    cudaFree(mat);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
