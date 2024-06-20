#include "cuda_utils.h"


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

void printMatrix(DATA_TYPE *array, int size) 
{
    int MAX_PRINT_SIZE = size;
    if (MAX_PRINT_SIZE > 8)
    {
        MAX_PRINT_SIZE = 8;
        printf("Matrix is too large to print, printing only the first 8x8 elements\n");
    }

    for (int i = 0; i < MAX_PRINT_SIZE; ++i)
    {
        for (int j = 0; j < MAX_PRINT_SIZE; ++j)
        {
            printf(FORMAT_SPECIFIER" ", array[i * size + j]);
        }
        printf("\n");
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