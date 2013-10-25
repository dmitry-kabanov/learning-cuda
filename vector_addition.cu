#include <stdio.h>

__global__ void vecAdd(float *a, float *b, float *c, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int N = 1024 * 1024;
    size_t size = N * sizeof(float);

    float *ha = (float *) malloc(size);
    float *hb = (float *) malloc(size);
    float *hc = (float *) malloc(size);
    float *hc_check = (float *) malloc(size);

    for (int i = 0; i < N; i++) {
        ha[i] = i;
        hb[i] = i + 1;
        hc_check[i] = ha[i] + hb[i];
    }

    float *da;
    cudaMalloc(&da, size);
    float *db;
    cudaMalloc(&db, size);
    float *dc;
    cudaMalloc(&dc, size);

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, N);

    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

    int cmp = memcmp(hc_check, hc, size);

    if (cmp == 0) { 
        printf("Arrays are equal.\n");
    } else {
        printf("Arrays are not equal.\n");
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(ha);
    free(hb);
    free(hc);
    
    return 0;
}
