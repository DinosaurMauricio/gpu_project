#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
namespace cg = cooperative_groups;

#ifndef TILE_DIM
#define TILE_DIM 32
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


int main(int argc, char **argv)
{
    int N = -1;

    if (argc < 1)
    {
        printf("No matrix size or number of tries was provided. Defaulting to 10. \n");
        N = 10;
    }
    else
    {
        N = atoi(argv[1]);
    }

    int matrix_size = 1 << N;
    printf("Matrix Size: %dx%d \n", matrix_size, matrix_size);

    if (matrix_size % TILE_DIM != 0) 
    {
        printf("TILE DIMENSION must be a multiple of the matrix size.\n");
        exit(1);
    }

    const unsigned long long memory_size = matrix_size * matrix_size * sizeof(DATA_TYPE);

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

    initializeMatrixValues(h_idata, matrix_size);

    DATA_TYPE *d_idata, *d_odata;
    cudaMalloc(&d_idata, memory_size);
    cudaMalloc(&d_odata, memory_size);
    cudaMemcpy(d_idata, h_idata, memory_size, cudaMemcpyHostToDevice);

    dim3 grid(matrix_size / TILE_DIM, matrix_size / TILE_DIM, 1);
    dim3 threads(TILE_DIM, TILE_DIM, 1);

    printf("dimGrid: %d %d %d. dimThreads: %d %d %d\n",
           grid.x, grid.y, grid.z, threads.x, threads.y, threads.z);

    printf("*****************************************************************************\n");
    printf("%25s", "transposeWithTiledPartition\n");
    runTransposeKernel(transposeWithTiledPartition, grid, threads, d_odata, d_idata, memory_size, NUMBER_OF_TESTS, matrix_size, startEvent, stopEvent);

    printf("*****************************************************************************\n");
    printf("%25s", "transposeKernelParent\n");
    runTransposeKernel(transposeKernelParent, grid, threads, d_odata, d_idata, memory_size, NUMBER_OF_TESTS, matrix_size, startEvent, stopEvent);

     // CUBLAS operations
    printf("*****************************************************************************\n");
    printf("%25s", "cuBLAS\n");
    runCUBLASOperations(d_idata, d_odata, NUMBER_OF_TESTS, matrix_size, memory_size, startEvent, stopEvent);
    
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);

    return 0;
}
