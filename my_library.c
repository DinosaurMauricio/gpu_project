#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "my_library.h"

void printMatrix(DATA_TYPE *array, int size, const char *message) 
{
    if(size < 10)
    {
        printf(message);
        for (int j = 0; j < size; j++) 
        {
            for (int i = 0; i < size; i++) 
            {
                printf(FORMAT_SPECIFIER" ", array[j * size + i]);
            }
        printf("\n");
        }
    }
    else
    {
        printf("Matrix size is to big, skipping print \n");
    }
}

void initializeMatrixValues(DATA_TYPE *matrix, int size)
{
    if (matrix == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            matrix[j*size + i] = rand() % 10 + 1;
        }
    }
}

double calculate_effective_bandwidth(int size, int number_of_repetitions,float time)
{
    const int GB_SIZE = 1000000000;

    // time is in miliseconds, we divide by 1000 to convert it to seconds
    // multiply by 2 for once for loading the matrix and once for storing
    double effective_bandwidth = (2*(size) * sizeof(DATA_TYPE))/ (time/1000);
    double effectve_bandwidth_gb_per_second = effective_bandwidth*number_of_repetitions/GB_SIZE;

    return effectve_bandwidth_gb_per_second;
}

bool validate_matrix_tranpose(const DATA_TYPE *ref, const DATA_TYPE *res, int size)
{
    bool passed = true;
    for (int i = 0; i < size; i++)
    {
        if (res[i] != ref[i]) 
        {
            printf("%25s\n", "*** FAILED ***");
            printf("%d "FORMAT_SPECIFIER" "FORMAT_SPECIFIER"\n", i, res[i], ref[i]);
            passed = false;
            break;
        }
    }

    return passed;
}