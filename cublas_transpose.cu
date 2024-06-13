#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <sys/time.h>

#define uS_PER_SEC 1000000
#define uS_PER_mS 1000

void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    timeval t1, t2;
    cublasHandle_t handle;
    cublasCreate(&handle);

    int N = 15;
    int matrix_dimension = 1 << N;
    size_t memory_size = matrix_dimension * matrix_dimension * sizeof(float);

    float* h_A = (float*)malloc(memory_size);


    // Initialize the host matrix
    for(int i = 0; i < matrix_dimension; ++i) {
        for(int j = 0; j < matrix_dimension; ++j) {
            h_A[i * matrix_dimension + j] = static_cast<float>(i * matrix_dimension + j);
        }
    }

    std::cout << "Original Matrix:" << std::endl;
    //printMatrix(h_A, matrix_dimension, matrix_dimension);

    float* d_A;
    float* d_AT;
    cudaMalloc(&d_A, memory_size);
    cudaMalloc(&d_AT, memory_size);

    cudaMemcpy(d_A, h_A, memory_size, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    gettimeofday(&t1, NULL);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, matrix_dimension, matrix_dimension, &alpha, d_A, matrix_dimension, &beta, d_AT, matrix_dimension, d_AT, matrix_dimension);
    gettimeofday(&t2, NULL);

    float et2 = (((t2.tv_sec*uS_PER_SEC)+t2.tv_usec) - ((t1.tv_sec*uS_PER_SEC)+t1.tv_usec))/(float)uS_PER_mS;
    printf("GPU time = %fms\n", et2);

    float* h_AT = (float*)malloc(memory_size);
    cudaMemcpy(h_AT, d_AT, memory_size, cudaMemcpyDeviceToHost);

    std::cout << "Transposed Matrix:" << std::endl;
    //printMatrix(h_AT, matrix_dimension, matrix_dimension);

    cudaFree(d_A);
    cudaFree(d_AT);
    cublasDestroy(handle);

    return 0;
}
