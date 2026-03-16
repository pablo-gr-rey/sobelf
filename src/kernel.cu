#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void vecAddVector(int* res, int* a, int* b, int size)
{
    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;

    int i = blockNumInGrid * threadsPerBlock + threadNumInBlock;

    if (i < size)
        res[i] = a[i] + b[i];
}

__global__ void vecBlur(int* pixels, int* sum, int threshold)
{
    int threadNum = threadIdx.x + blockDim.x * threadIdx.y;

    // we distribute the images
}

extern "C"
void run_kernel()
{
    // const int size = atoi(argv[1]), seed = atoi(argv[2]);
    const int size = 1000000, seed = 42;
    const int maxval = 1000000;     // to avoid overflows

    srand(seed);

    int* a = (int*) malloc(size * sizeof(int)), * b = (int*) malloc(size * sizeof(int)), * res = (int*) malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        a[i] = rand() % maxval;
        b[i] = rand() % maxval;
    }

    struct timeval t1, t2, t3, t4;

    int* d_a, * d_b, * d_res;

    cudaMalloc((void**) &d_a, size * sizeof(int));
    cudaMalloc((void**) &d_b, size * sizeof(int));
    cudaMalloc((void**) &d_res, size * sizeof(int));

    gettimeofday(&t1, NULL);

    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    gettimeofday(&t2, NULL);

    // at first, my code didn't work because I didn't remember that the maximum number of threads per block was 1024

    // maybe surprisingly (at least for me), this setting is slower than the following one
    // maybe it would be faster if we transfer only the relevant part of the array into each block's private cache

    // dim3 grid(100, 10, 1);
    // dim3 threads(100, 10, 1);

    dim3 grid(size, 1, 1);
    dim3 threads(1, 1, 1);
    vecAddVector << <grid, threads >> > (d_res, d_a, d_b, size);

    gettimeofday(&t3, NULL);

    cudaMemcpy(res, d_res, size * sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&t4, NULL);

    bool ok = true;
    for (int i = 0; i < size; i++)
        if (res[i] != a[i] + b[i])
        {
            printf("at index %d: res[i] = %d a[i] + b[i] = %d\n", i, res[i], a[i] + b[i]);
            ok = false;
            break;
        }

    if (ok)
        printf("PASSED\n");
    else
        printf("FAILED\n");

    printf("time spent on memory transfers: %lf\n", (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6) + (t4.tv_sec - t3.tv_sec) + ((t4.tv_usec - t3.tv_usec) / 1e6));
    printf("time spent in kernel: %lf\n", (t3.tv_sec - t2.tv_sec) + ((t3.tv_usec - t2.tv_usec) / 1e6));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    free(a);
    free(b);
    free(res);
}
