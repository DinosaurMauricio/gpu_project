#include <stdio.h>
#ifndef DATA_TYPE
#define DATA_TYPE int
#endif

#ifndef FORMAT_SPECIFIER
#define FORMAT_SPECIFIER "%d"
#endif

void printMatrix(DATA_TYPE *array, int size, const char *message);
void initializeMatrixValues(DATA_TYPE *matrix, int size);
double calculate_effective_bandwidth(int size, int number_of_repetitions,float time);