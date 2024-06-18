#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

extern "C" {
#include "my_library.h"
}

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 32768
#endif

template <typename KernelFunc>
void runKernelAndMeasure(const char* kernelName, KernelFunc kernel, dim3 dimGrid, dim3 dimBlock, 
                         DATA_TYPE* d_odata, const DATA_TYPE* d_idata,
                         size_t memory_size, int numberOfTests, cudaEvent_t startEvent, 
                         cudaEvent_t stopEvent, double &effective_bw, float &ms) 
{
    printf("%25s", kernelName);
    cudaMemset(d_odata, 0, memory_size);

    // Warm up
    kernel<<<dimGrid, dimBlock>>>(d_odata, d_idata);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < numberOfTests; i++)
        kernel<<<dimGrid, dimBlock>>>(d_odata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    //cudaMemcpy(h_cdata, d_cdata, memory_size, cudaMemcpyDeviceToHost);
    effective_bw = calculate_effective_bandwidth(MATRIX_SIZE * MATRIX_SIZE, numberOfTests, ms);

    printf("%20.2f %20.2f ms\n", effective_bw, ms);
}

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


__global__ void transposeTileKernelChild(float *odata, const float *idata, int xOffset, int yOffset) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

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

__global__ void transposeKernelParent(float *odata, const float *idata) {
    int xTile = blockIdx.x * TILE_DIM;
    int yTile = blockIdx.y * TILE_DIM;

    
    float ms_dp;
    cudaEvent_t startEvent_dp, stopEvent_dp;
    cudaEventCreate(&startEvent_dp);
    cudaEventCreate(&stopEvent_dp);

    cudaEventRecord(startEvent_dp, 0);
    if (xTile < MATRIX_SIZE && yTile < MATRIX_SIZE) {
        transposeTileKernelChild<<<1, dim3(TILE_DIM, TILE_DIM)>>>(odata, idata, xTile, yTile);
    }
    cudaEventRecord(stopEvent_dp, 0);
    cudaEventSynchronize(stopEvent_dp);
    cudaEventElapsedTime(&ms_dp, startEvent_dp, stopEvent_dp);

    double effective_bw_dp = calculate_effective_bandwidth(MATRIX_SIZE * MATRIX_SIZE, 1, ms_dp);
    printf("Child kernel: %20.2f %20.2f ms\n", effective_bw_dp, ms_dp);
}


int main()
{
    int numberOfTests = 100;
    const unsigned long long memory_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE);

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
        printf("Matrix Size %d x %d \n", MATRIX_SIZE, MATRIX_SIZE);
        printf("Tile Dimension %d\n", TILE_DIM);
    }

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    DATA_TYPE *h_idata, *h_odata;
    h_idata = (DATA_TYPE*)malloc(memory_size);
    h_odata = (DATA_TYPE*)malloc(memory_size);

    initializeMatrixValues(h_idata, MATRIX_SIZE);

    printf("Original \n ");
    printMatrix(h_idata, MATRIX_SIZE);
    printf("\n");

    DATA_TYPE *d_idata, *d_odata;
    cudaMalloc(&d_idata, memory_size);
    cudaMalloc(&d_odata, memory_size);
    cudaMemcpy(d_idata, h_idata, memory_size, cudaMemcpyHostToDevice);

    dim3 grid(MATRIX_SIZE / TILE_DIM, MATRIX_SIZE / TILE_DIM, 1);
    dim3 threads(TILE_DIM, TILE_DIM, 1);

    printf("dimGrid: %d %d %d. dimThreads: %d %d %d\n",
           grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);

    printf("%25s%25s%25s\n", "Routine", "Bandwidth (GB/s)", "Time(ms)");
    
    double total_bw = 0;
    float total_ms = 0;
    const int repeat = 5;

    // Run the kernel multiple times
    for (int i = 0; i < repeat; i++) {
        double effective_bw;
        float ms;
        /*runKernelAndMeasure("transposeWithTiledPartition", transposeWithTiledPartition, grid, threads, 
                            d_odata, d_idata, memory_size, numberOfTests, startEvent, stopEvent, 
                            effective_bw, ms);*/

        runKernelAndMeasure("transposeKernelParent", transposeKernelParent, grid, threads, 
                            d_odata, d_idata, memory_size, numberOfTests, startEvent, stopEvent, 
                            effective_bw, ms);
        total_bw += effective_bw;
        total_ms += ms;
    }

     //transposeKernelParent<<<grid, 1>>>(d_odata, h_idata);

    double avg_bw = total_bw / repeat;
    double avg_ms = total_ms / repeat;

    cudaDeviceSynchronize();

    cudaMemcpy(h_odata, d_odata, memory_size, cudaMemcpyDeviceToHost);

    printf("Transposed \n ");
    printMatrix(h_odata, MATRIX_SIZE);
    printf("\n");

    printf("\nAverage Bandwidth (GB/s): %.2f\n", avg_bw);
    printf("Average Time (ms): %.2f\n", avg_ms);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
