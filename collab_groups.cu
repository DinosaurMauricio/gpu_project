#include <cooperative_groups.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace cooperative_groups;

__global__ void transposeNoBankConflicts(DATA_TYPE *odata, const DATA_TYPE *idata)
{
  __shared__ DATA_TYPE tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  #if UNROLL_FLAG
  #pragma unroll
  #endif
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  #if UNROLL_FLAG
  #pragma unroll
  #endif
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}



__device__ int reduce_sum(thread_group g, int *temp, int val)
{
    int lane = g.thread_rank();


    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        //printf("i is %d and lane is %d\n",i, lane);

        if(lane < i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__global__ void sum_kernel(int *input, int *output, int n)
{
    extern __shared__ int temp[];
    int val = 0;

    // Each thread loads one element from global to local shared memory
    if (threadIdx.x < n)
    {
        val = input[threadIdx.x];
    }

    thread_block g = this_thread_block();
    val = reduce_sum(g, temp, val);

    // Thread 0 writes the result
    if (g.thread_rank() == 0)
    {
        output[blockIdx.x] = val;
    }
}

int main()
{
    const int ARRAY_SIZE = 1024;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // Generate input array on host
    int h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = 1; // Initialize with 1s for simplicity
    }

    // Allocate memory on device
    int *d_in;
    int *d_out;
    cudaMalloc((void **)&d_in, ARRAY_BYTES);
    cudaMalloc((void **)&d_out, sizeof(int));

    // Copy input array to device
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // Launch kernel
    int THREADS_PER_BLOCK = 1024;
    sum_kernel<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_in, d_out, ARRAY_SIZE);

    // Copy result back to host
    int h_out;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Sum is " << h_out << std::endl;

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
