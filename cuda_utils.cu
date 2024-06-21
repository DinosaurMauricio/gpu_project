#include "cuda_utils.h"

void printMatrix(DATA_TYPE *array, int size) 
{
    int MAX_PRINT_SIZE = size;
    if (MAX_PRINT_SIZE > 8)
    {
        MAX_PRINT_SIZE = 8;
        printf("Matrix is too large to print, printing only the first 8x8 elements\n");
    }

    printf("\n Transposed Matrix: \n");

    for (int i = 0; i < MAX_PRINT_SIZE; ++i)
    {
        for (int j = 0; j < MAX_PRINT_SIZE; ++j)
        {
            printf(FORMAT_SPECIFIER" ", array[i * size + j]);
        }
        printf("\n");
    }
}

template <typename KernelFunc>
void runKernelAndMeasure(KernelFunc kernel, dim3 dimGrid, dim3 dimBlock, 
                         DATA_TYPE* d_odata, const DATA_TYPE* d_idata,
                         size_t memory_size, int numberOfTests, cudaEvent_t startEvent, 
                         cudaEvent_t stopEvent, double &effective_bw, float &ms) 
{
    cudaMemset(d_odata, 0, memory_size);

    // Warm up
    kernel<<<dimGrid, dimBlock>>>(d_odata, d_idata);
    cudaEventRecord(startEvent, 0);
    for (int i = 0; i < numberOfTests; i++)
        kernel<<<dimGrid, dimBlock>>>(d_odata, d_idata);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    effective_bw = calculate_effective_bandwidth(MATRIX_SIZE * MATRIX_SIZE, numberOfTests, ms);
}

template <typename KernelFunc>
void runTransposeKernel(KernelFunc kernel,const dim3 &grid, const dim3 &threads, DATA_TYPE* d_odata, const DATA_TYPE* d_idata, size_t memory_size, int numberOfTests, cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    double total_bw = 0;
    float total_ms = 0;
    const int repeat = 5;

    for (int i = 0; i < repeat; i++) {
        double effective_bw;
        float ms;
        runKernelAndMeasure(kernel, grid, threads, d_odata, d_idata, memory_size, numberOfTests, startEvent, stopEvent, effective_bw, ms);
        total_bw += effective_bw;
        total_ms += ms;
    }

    double avg_bw = total_bw / repeat;
    double avg_ms = total_ms / repeat;

    cudaDeviceSynchronize();

    if(PRINT_TRANSPOSED_MATRIX)
    {
        DATA_TYPE* h_odata = (DATA_TYPE*)malloc(memory_size); 
        cudaMemcpy(h_odata, d_odata, memory_size, cudaMemcpyDeviceToHost);
        printMatrix(h_odata, MATRIX_SIZE);
        free(h_odata);
    }

    printf("\nAverage Bandwidth (GB/s): %.2f\n", avg_bw);
    printf("Average Time (ms): %.2f\n", avg_ms);
}

void initializeMatrixValues(DATA_TYPE *matrix, int size)
{
    if (matrix == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < size * size; ++i)
    {
        matrix[i] = i;
    }
}

double calculate_effective_bandwidth(int size, int number_of_repetitions, float time_ms) {
    const int GB_SIZE = 1024.0 * 1024 * 1024; // Bytes in a gigabyte
    const double time_sec = time_ms / 1000.0f; // Convert time from milliseconds to seconds

    // Calculate total data transferred in bytes (load + store)
    unsigned long long total_data_size_bytes = 2LL * size * sizeof(DATA_TYPE); 

    // Calculate effective bandwidth in bytes per second
    double effective_bandwidth_bytes_per_sec = total_data_size_bytes / time_sec;

    // Calculate effective bandwidth in GB/s
    double effective_bandwidth_gb_per_sec = (effective_bandwidth_bytes_per_sec*number_of_repetitions) / GB_SIZE;

    return effective_bandwidth_gb_per_sec;
}

void getDeviceProperties(int &devID, cudaDeviceProp &deviceProp) {
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&deviceProp, devID);
}

bool checkMemorySize(size_t memory_size, const cudaDeviceProp &deviceProp) {
    if (2 * memory_size > deviceProp.totalGlobalMem) {
        printf("Input matrix size is larger than the available device memory!\n");
        printf("Please choose a smaller size matrix\n");
        return false;
    } else {
        printf("Global memory size: %llu\n", (unsigned long long)deviceProp.totalGlobalMem);
        printf("Using memory for matrix: %zu\n", 2 * memory_size);
        printf("Matrix Size %d x %d \n", MATRIX_SIZE, MATRIX_SIZE);
        printf("Tile Dimension %d\n", TILE_DIM);
        return true;
    }
}

void runCUBLASOperations(const DATA_TYPE* d_A, DATA_TYPE* d_B, const int numberOfTests, cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    DATA_TYPE alpha = 1.0f;
    DATA_TYPE beta = 0.0f;

    cudaMemset(d_B, 0, MATRIX_SIZE*MATRIX_SIZE*sizeof(DATA_TYPE));

    CUBLAS_Geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, &alpha, d_A, MATRIX_SIZE, &beta, d_B, MATRIX_SIZE, d_B, MATRIX_SIZE);

    cudaEventRecord(startEvent);
    for (int i = 0; i < numberOfTests; i++) {
        CUBLAS_Geam(handle, CUBLAS_OP_T, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, &alpha, d_A, MATRIX_SIZE, &beta, d_B, MATRIX_SIZE, d_B, MATRIX_SIZE);
    }
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    double bw = calculate_effective_bandwidth(MATRIX_SIZE * MATRIX_SIZE, numberOfTests, milliseconds);

    if(PRINT_TRANSPOSED_MATRIX)
    {
        DATA_TYPE* h_odata = (DATA_TYPE*)malloc(MATRIX_SIZE*MATRIX_SIZE*sizeof(DATA_TYPE)); 
        cudaMemcpy(h_odata, d_B, MATRIX_SIZE*MATRIX_SIZE*sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
        printMatrix(h_odata, MATRIX_SIZE);
        free(h_odata);
    }

    printf("\nEffective Bandwidth (GB/s): %f\n", bw);
    printf("Average Time (ms): %f\n", milliseconds);

    cublasDestroy(handle);
}


