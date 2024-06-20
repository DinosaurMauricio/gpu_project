#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "my_library.h"

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

unsigned long long calculate_effective_bandwidth(int size, int number_of_repetitions, float time_ms) {
    const int GB_SIZE = 1024.0 * 1024 * 1024; // Bytes in a gigabyte
    const double time_sec = time_ms / 1000.0f; // Convert time from milliseconds to seconds

    // Calculate total data transferred in bytes (load + store)
    unsigned long long total_data_size_bytes = 2LL * size * sizeof(DATA_TYPE); // Assuming size is in bytes

    // Calculate effective bandwidth in bytes per second
    double effective_bandwidth_bytes_per_sec = total_data_size_bytes / time_sec;

    // Calculate effective bandwidth in GB/s
    double effective_bandwidth_gb_per_sec = (effective_bandwidth_bytes_per_sec*number_of_repetitions) / GB_SIZE;

    // Convert to unsigned long long (if needed for return value)
    unsigned long long effective_bandwidth_gb_per_sec_ull = (unsigned long long)effective_bandwidth_gb_per_sec;


    // print all the values for debugging including the parameters and arguments
    printf("Matrix size: %d\n", size);
    printf("Number of repetitions: %d\n", number_of_repetitions);
    printf("Time taken: %f ms\n", time_ms);
    printf("Time taken: %f s\n", time_sec);
    printf("Total data size: %llu bytes\n", total_data_size_bytes);
    printf("Effective bandwidth: %f GB/s\n", effective_bandwidth_gb_per_sec);
    printf("Effective bandwidth (unsigned long long): %llu GB/s\n", effective_bandwidth_gb_per_sec_ull);



    return effective_bandwidth_gb_per_sec_ull;
}