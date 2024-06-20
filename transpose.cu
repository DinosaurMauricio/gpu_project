#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
namespace cg = cooperative_groups;

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 1024
#endif

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
    printf("%25s", "transposeWithTiledPartition\n");
    for (int i = 0; i < repeat; i++) {
        double effective_bw;
        float ms;
        runKernelAndMeasure(transposeWithTiledPartition, grid, threads, 
                            d_odata, d_idata, memory_size, numberOfTests, startEvent, stopEvent, 
                            effective_bw, ms);
        total_bw += effective_bw;
        total_ms += ms;
    }

    double avg_bw = total_bw / repeat;
    double avg_ms = total_ms / repeat;

    cudaDeviceSynchronize();

    cudaMemcpy(h_odata, d_odata, memory_size, cudaMemcpyDeviceToHost);

    printf("Transposed \n ");
    printMatrix(h_odata, MATRIX_SIZE);
    printf("\n");

    printf("\nAverage Bandwidth (GB/s): %.2f\n", avg_bw);
    printf("Average Time (ms): %.2f\n", avg_ms);

/////////////////////
    total_bw = 0;
    total_ms = 0;

        // Run the kernel multiple times
    printf("%25s", "transposeKernelParent\n");
    for (int i = 0; i < repeat; i++) {
        double effective_bw;
        float ms;
        runKernelAndMeasure(transposeKernelParent, grid, threads, 
                                d_odata, d_idata, memory_size, numberOfTests, startEvent, stopEvent, 
                                effective_bw, ms);
        total_bw += effective_bw;
        total_ms += ms;
    }

    avg_bw = total_bw / repeat;
    avg_ms = total_ms / repeat;

    cudaDeviceSynchronize();

    cudaMemcpy(h_odata, d_odata, memory_size, cudaMemcpyDeviceToHost);

    printf("Transposed \n ");
    printMatrix(h_odata, MATRIX_SIZE);
    printf("\n");

    printf("\nAverage Bandwidth (GB/s): %.2f\n", avg_bw);
    printf("Average Time (ms): %.2f\n", avg_ms);


    DATA_TYPE *d_A, *d_B;
    DATA_TYPE *h_A = (DATA_TYPE *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE));

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_A[i] = static_cast<DATA_TYPE>(rand()) / RAND_MAX;
    }

    cudaMalloc((void **)&d_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE));
    cudaMalloc((void **)&d_B, MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE));
    cudaMemcpy(d_A, h_A, MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    DATA_TYPE alpha = 1.0f;
    DATA_TYPE beta = 0.0f;

    //cudaEvent_t start, stop;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Warm up
    CUBLAS_Geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, &alpha, d_A, MATRIX_SIZE, &beta, d_B, MATRIX_SIZE, d_B, MATRIX_SIZE);

    // Timing loop
    cudaEventRecord(startEvent);
    for (int i = 0; i < numberOfTests; i++) {
        CUBLAS_Geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, &alpha, d_A, MATRIX_SIZE, &beta, d_B, MATRIX_SIZE, d_B, MATRIX_SIZE);
    }
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    // Calculate average time per iteration
    float avgMilliseconds = milliseconds / numberOfTests;

    // Total data moved: 2 * SIZE * SIZE * sizeof(float) bytes
    float totalDataGB = 2 * MATRIX_SIZE * MATRIX_SIZE * sizeof(DATA_TYPE) / 1e9; // in GB
    float timeSeconds = avgMilliseconds / 1000; // convert ms to seconds
    float bandwidth = totalDataGB / timeSeconds; // in GB/s

    printf("CUBLAS\n");
    printf("\nAverage Bandwidth (GB/s): %f\n", avgMilliseconds);
    printf("Average Time (ms): %f\n", bandwidth);

    cublasDestroy(handle);
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}
