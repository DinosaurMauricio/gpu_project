#include <iostream>
#include <cooperative_groups.h>
using std::cout;
using std::endl;
namespace cg = cooperative_groups;

__global__ void globalSync(int *a_1, int *a_2, int N)
{

    printf("hello");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    auto g = cg::this_grid();

    for(int i= tid; i < N; i += gridDim.x * blockDim.x)
    {
        a_1[i]=i;
    }

    // without g.sync we have a race condition because some threads could be done
    // and otheres not
    g.sync();

    for(int i= tid; i < N; i += gridDim.x * blockDim.x)
    {
        int temp = 0;
        for(int j =0; j < N; j++)
        {
            temp +=a_1[j];
        }
        a_2[i] = temp;
        printf(" value %d \n", a_2[i]);
    }
    
}

int main(){

    int N = 1 << 10;
    size_t bytes = N * sizeof(int);

    int *a_1, *a_2;
    cudaMallocManaged(&a_1,bytes);
    cudaMallocManaged(&a_2,bytes);

    int THREADS;
    int BLOCKS;
    cudaOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, globalSync, 0,0);

    // deadlock because i call a global sync but some threads never got into the gpu
    // and its aiting for therads that never got into it so they cannot sync
    globalSync<<<BLOCKS, 100>>>(a_1, a_2, N);
    cudaDeviceSynchronize();
    cout << BLOCKS << endl;
    cout << THREADS << endl;
}