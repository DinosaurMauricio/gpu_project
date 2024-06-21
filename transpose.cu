#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
namespace cg = cooperative_groups;



#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 1024 // 1024
#endif

#define NUMBER_OF_TESTS 100

#ifdef DATA_TYPE_FLOAT
    #define CUBLAS_Geam cublasSgeam
#elif defined(DATA_TYPE_DOUBLE)
    #define CUBLAS_Geam cublasDgeam
#else
    #error "Define DATA_TYPE_FLOAT or DATA_TYPE_DOUBLE"
#endif
#include "kernels.cu"
#include "cuda_utils.cu"


int main()
{
    const unsigned long long memory_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE);

    int devID = 0;
    cudaDeviceProp deviceProp;

    getDeviceProperties(devID, deviceProp);

    if (!checkMemorySize(memory_size, deviceProp)) {
        exit(EXIT_FAILURE);
    }

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    DATA_TYPE *h_idata;
    h_idata = (DATA_TYPE*)malloc(memory_size);

    initializeMatrixValues(h_idata, MATRIX_SIZE);

    DATA_TYPE *d_idata, *d_odata;
    cudaMalloc(&d_idata, memory_size);
    cudaMalloc(&d_odata, memory_size);
    cudaMemcpy(d_idata, h_idata, memory_size, cudaMemcpyHostToDevice);

    dim3 grid(MATRIX_SIZE / TILE_DIM, MATRIX_SIZE / TILE_DIM, 1);
    dim3 threads(TILE_DIM, TILE_DIM, 1);

    printf("dimGrid: %d %d %d. dimThreads: %d %d %d\n",
           grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);

    printf("*****************************************************************************\n");
    printf("%25s", "transposeWithTiledPartition\n");
    runTransposeKernel(transposeWithTiledPartition, grid, threads, d_odata, d_idata, memory_size, NUMBER_OF_TESTS, startEvent, stopEvent);

    printf("*****************************************************************************\n");
    printf("%25s", "transposeKernelParent\n");
    runTransposeKernel(transposeKernelParent, grid, threads, d_odata, d_idata, memory_size, NUMBER_OF_TESTS, startEvent, stopEvent);

     // CUBLAS operations
    printf("*****************************************************************************\n");
    printf("%25s", "cuBLAS\n");
    runCUBLASOperations(d_idata, d_odata, NUMBER_OF_TESTS, startEvent, stopEvent);
    
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);

    return 0;
}
