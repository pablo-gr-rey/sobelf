#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define DURATION(t1, t2) ((t2).tv_sec - (t1).tv_sec) + (((t2).tv_usec - (t1).tv_usec) / 1e6)
#define CONV(l,c,nb_c) (l)*(nb_c)+(c)

__global__ void vecAddVector(int* res, int* a, int* b, int size)
{
    int threadsPerBlock = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid = blockIdx.x + gridDim.x * blockIdx.y;

    int i = blockNumInGrid * threadsPerBlock + threadNumInBlock;

    if (i < size)
        res[i] = a[i] + b[i];
}

__global__ void vecBlur(int* pixels, int* sum, int width, int height, int size, int threshold)
{
    // we assume 1D work distribution, with only 1 block
    int num = threadIdx.x;
    int blockSize = blockDim.x;

    int minRow, maxRow;

    // we distribute the work
    // first loop: compute sum over rows

    // we can use one variable shared in the block
    // concurrent write may not execute atomatically, but it is guaranteed that at least one will go through
    // we always write 1 before the first sync, and 0 after the first sync so we can do concurrent writes since the final value does not matter
    __shared__ int end;

    do {
        __syncthreads();

        end = 1;

        // first half of thread will compute the top part, second half, bottom part
        // if statements are fine because all threads of a warp should go to the same condition
        if (num < blockSize / 2)
        {
            // distribute rows from 0 to height / 10 + size - 1 over blockSize/2 threads
            minRow = num * (height / 10 + size - 1) / (blockSize / 2);
            maxRow = (num + 1) * (height / 10 + size - 1) / (blockSize / 2);
        }
        else
        {
            // distribute rows from height * 0.9 - size to height over blockSize/2 threads
            minRow = height * 0.9 - size + (num - blockSize / 2) * (height - (height * 0.9 - size)) / (blockSize / 2);
            maxRow = height * 0.9 - size + (num + 1 - blockSize / 2) * (height - (height * 0.9 - size)) / (blockSize / 2);
        }

        for (int j = minRow; j < maxRow; j++)
        {
            int index_deb = j * width;

            // index 0 is the sum from 0 to size
            sum[index_deb] = 0;
            for (int k = 0; k <= size; k++)
                sum[index_deb] += pixels[index_deb + k];

            // from 1 to size, we add a new value
            for (int k = 1; k <= size; k++)
                sum[index_deb + k] = sum[index_deb + k - 1] + pixels[index_deb + k + size];

            // from size+1 to width-size-1, we add a new value and remove one
            for (int k = size + 1; k <= width - size - 1; k++)
                sum[index_deb + k] = sum[index_deb + k - 1] + pixels[index_deb + k + size] - pixels[index_deb + k - size - 1];

            // from width-size to width-1, we remove one value
            for (int k = width - size; k <= width - 1; k++)
                sum[index_deb + k] = sum[index_deb + k - 1] - pixels[index_deb + k - size - 1];
        }

        // wait for the first pass to be finished
        __syncthreads();

        // update pixels value (over columns)
        int minCol = size + num * (width - 2 * size) / blockSize;
        int maxCol = size + (num + 1) * (width - 2 * size) / blockSize;
        
        for (int k = minCol; k < maxCol; k++)
        {
            // top part of the image
            int total_sum = 0;
            for (int j = 0; j <= 2 * size; j++)
                total_sum += sum[CONV(j, k, width)];

            for (int j = size; j < height / 10 - size; j++)
            {
                int new_val = total_sum / ((2 * size + 1) * (2 * size + 1));
                if (pixels[CONV(j, k, width)] > new_val + threshold || pixels[CONV(j, k, width)] < new_val - threshold)
                    end = 0;

                pixels[CONV(j, k, width)] = new_val;

                total_sum += sum[CONV(j + size + 1, k, width)] - sum[CONV(j - size, k, width)];
            }

            // bottom part of the image
            total_sum = 0;
            for (int j = height - 1; j >= height - 2 * size - 1; j--)
                total_sum += sum[CONV(j, k, width)];

            for (int j = height - 1 - size; j >= height * 0.9 + size; j--)
            {
                int new_val = total_sum / ((2 * size + 1) * (2 * size + 1));
                if (pixels[CONV(j, k, width)] > new_val + threshold || pixels[CONV(j, k, width)] < new_val - threshold)
                    end = 0;

                pixels[CONV(j, k, width)] = new_val;

                total_sum += sum[CONV(j - size - 1, k, width)] - sum[CONV(j + size, k, width)];
            }
        }

        // if (num == 0)
        //     printf("new iter, hi\n");
    } while (!end);
}

extern "C"
void run_kernel(int* pixels, int width, int height, int size, int threshold)
{
    struct timeval t1, t2, t3, t4;

    gettimeofday(&t1, NULL);

    int* d_pixels, * d_sum;

    cudaMalloc((void**) &d_pixels, width * height * sizeof(int));
    cudaMalloc((void**) &d_sum, width * height * sizeof(int));


    cudaMemcpy(d_pixels, pixels, width * height * sizeof(int), cudaMemcpyHostToDevice);

    gettimeofday(&t2, NULL);

    dim3 grid(1);
    dim3 threads(1024);
    vecBlur << <grid, threads >> > (d_pixels, d_sum, width, height, size, threshold);

    gettimeofday(&t3, NULL);

    // cudaMemcpy(sum, d_sum, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pixels, d_pixels, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    bool ok = true;

    // for (int j = 0; j < height; j == height / 10 ? j = height * 0.9 : j++)
    //     for (int k = 0; k < width; k++)
    //     {
    //         int ss = 0;
    //         for (int l = k >= size ? k - size : 0; l <= k + size && l < width; l++)
    //             ss += pixels[CONV(j, l, width)];
    //         if (ss != sum[CONV(j, k, width)])
    //         {
    //             printf("BAD SUM at j %d k %d (expected %d got %d)\n", j, k, ss, sum[CONV(j, k, width)]);
    //             ok = false;
    //         }
    //     }

    // if (ok)
    //     printf("PASSED\n");
    // else
    //     printf("FAILED\n");

    cudaFree(d_pixels);
    cudaFree(d_sum);

    // free(sum);

    gettimeofday(&t4, NULL);

    printf("time spent on memory transfers: %lf\n", (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6) + (t4.tv_sec - t3.tv_sec) + ((t4.tv_usec - t3.tv_usec) / 1e6));
    printf("time spent in kernel: %lf\n", (t3.tv_sec - t2.tv_sec) + ((t3.tv_usec - t2.tv_usec) / 1e6));
}
