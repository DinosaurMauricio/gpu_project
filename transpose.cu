#include <stdio.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

extern "C" {
#include "my_library.h"
}

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 8
#endif

__global__ void transposeNoBankConflicts(DATA_TYPE *odata, const DATA_TYPE *idata)
{
  __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  // Get the current thread block
  cg::thread_block cta = cg::this_thread_block();

  #pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  // Synchronize to ensure all data is loaded into shared memory
  cta.sync();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  #pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

template <typename KernelFunc>
void runKernelAndMeasure(const char* kernelName, KernelFunc kernel, dim3 dimGrid, dim3 dimBlock, 
                         DATA_TYPE* d_cdata, const DATA_TYPE* d_idata, DATA_TYPE* h_cdata, 
                         size_t memory_size, size_t size, int numberOfTests, cudaEvent_t startEvent, 
                         cudaEvent_t stopEvent, double &effective_bw, float &ms) 
{
    printf("%25s", kernelName);
    checkCuda(cudaMemset(d_cdata, 0, memory_size));

    // Warm up
    kernel<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < numberOfTests; i++)
        kernel<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, memory_size, cudaMemcpyDeviceToHost));
    effective_bw = calculate_effective_bandwidth(size * size, numberOfTests, ms);

    printf("%20.2f %20.2f ms\n", effective_bw, ms);
}

int main(int argc, char **argv)
{
    int N = -1;
    int numberOfTests = -1;

    if (argc < 2)
    {
        printf("No matrix size or number of tries was provided. Defaulting to 10. \n");
        N = 10;
        numberOfTests = 100;
    }
    else
    {
        N = atoi(argv[1]);
        numberOfTests = atoi(argv[2]);
    }

    unsigned long long size = 1 << N;
    printf("The size of the matrix is %dx%d \n", size, size);

    if (size % BLOCK_ROWS != 0) 
    {
        printf("Block size must be a multiple of the matrix size.\n");
        exit(1);
    }

    if (TILE_DIM % BLOCK_ROWS) {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        exit(1);
    }

    const size_t memory_size = size * size * sizeof(DATA_TYPE);

    dim3 dimGrid(size / TILE_DIM, size / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

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
        printf("Matrix size: %d x %d\n", size, size);
        printf("Global memory size: %llu\n", (unsigned long long)deviceProp.totalGlobalMem);
        printf("Using memory for matrix: %zu\n", 2 * memory_size);
    }

    printf("Block size: %d %d, Tile size: %d %d\n", TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    DATA_TYPE *h_idata = (DATA_TYPE*)malloc(memory_size);
    DATA_TYPE *h_cdata = (DATA_TYPE*)malloc(memory_size);
    DATA_TYPE *d_idata, *d_cdata;

    cudaEvent_t startEvent, stopEvent;

    checkCuda(cudaMalloc(&d_idata, memory_size));
    checkCuda(cudaMalloc(&d_cdata, memory_size));

    initializeMatrixValues(h_idata, size);

    // events for timing
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    // device
    checkCuda(cudaMemcpy(d_idata, h_idata, memory_size, cudaMemcpyHostToDevice));

    printf("%25s%25s%25s\n", "Routine", "Bandwidth (GB/s)", "Time(ms)");

    double total_bw = 0;
    float total_ms = 0;
    const int repeat = 5;

    // Run the kernel multiple times
    for (int i = 0; i < repeat; i++) {
        double effective_bw;
        float ms;
        runKernelAndMeasure("transposeNoBankConflicts", transposeNoBankConflicts, dimGrid, dimBlock, 
                            d_cdata, d_idata, h_cdata, memory_size, size, numberOfTests, startEvent, stopEvent, 
                            effective_bw, ms);
        total_bw += effective_bw;
        total_ms += ms;
    }

    double avg_bw = total_bw / repeat;
    double avg_ms = total_ms / repeat;

    printf("Average Bandwidth (GB/s): %.2f\n", avg_bw);
    printf("Average Time (ms): %.2f\n", avg_ms);

    // cleanup
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    checkCuda(cudaFree(d_cdata));
    checkCuda(cudaFree(d_idata));
    free(h_idata);
    free(h_cdata);

    return 0;
}
