#ifndef DATA_TYPE
#define DATA_TYPE int
#endif

#ifndef FORMAT_SPECIFIER
#define FORMAT_SPECIFIER "%d"
#endif

#ifndef PRINT_TRANSPOSED_MATRIX
#define PRINT_TRANSPOSED_MATRIX false
#endif

template <typename KernelFunc>
void runKernelAndMeasure(KernelFunc kernel, dim3 dimGrid, dim3 dimBlock, 
                         DATA_TYPE* d_odata, const DATA_TYPE* d_idata,
                         size_t memory_size, int numberOfTests, cudaEvent_t startEvent, 
                         cudaEvent_t stopEvent, double &effective_bw, float &ms);

void initializeMatrixValues(DATA_TYPE* matrix, int size);
void printMatrix(const DATA_TYPE* matrix, int size);
double calculate_effective_bandwidth(int size, int number_of_repetitions, float time_ms);
void getDeviceProperties(int &devID, cudaDeviceProp &deviceProp);
bool checkMemorySize(size_t memory_size, const cudaDeviceProp &deviceProp);