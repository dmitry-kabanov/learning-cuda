#include <stdio.h>

__global__ void vectorMultiplyBy2(float *v, float *w, size_t n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    w[i] = v[i] * 2;
}

int main() {
    size_t N = 1024 * 1024 * 1024;
    size_t size = N * sizeof(float);
    float *a = (float *) malloc(size);
    float *b = (float *) malloc(size);
    float *b_check = (float *) malloc(size);

    for (int i = 0; i < N; i++) {
        a[i] = i;
    }

    for (int i = 0; i < N; i++) {
        b_check[i] = a[i] * 2;
    }

    float *ha;
    cudaMalloc((void **) &ha, size);
    float *hb;
    cudaMalloc((void **) &hb, size);

    cudaMemcpy(ha, a, size, cudaMemcpyHostToDevice);

    int tInB = 1024;
    dim3 threadsInBlock(tInB);
    int numberOfBlocks = 32768;
    printf("Number of blocks is %d\n", numberOfBlocks);
    dim3 nBlocks(numberOfBlocks, 32768);

    vectorMultiplyBy2<<<nBlocks, threadsInBlock>>>(ha, hb, N); 

    cudaMemcpy(b, hb, size, cudaMemcpyDeviceToHost);

    int cmp = memcmp(b, b_check, size);

    if (cmp == 0) {
        printf("Arrays are equal.\n");
    } else {
        printf("Arrays are not equal.\n");
    }

    return 0;
}

