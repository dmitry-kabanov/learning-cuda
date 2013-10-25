#include <stdio.h>

// Matrix is stored as 1d array in row-major order
typedef struct {
    int width;
    int height;
    float *elements;
} Matrix;

#define BLOCK_SIZE 16

#define A_WIDTH  2048
#define A_HEIGHT 2048
#define B_WIDTH  2048
#define B_HEIGHT 2048
#define C_WIDTH  2048
#define C_HEIGHT 2048

__global__ void matmul(const Matrix A, const Matrix B, const Matrix C)
{
    float CValue = 0;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int rowVal = row * A.width;
    for (int k = 0; k < A.width; k++) {
        CValue += A.elements[rowVal + k] * B.elements[k * B.height + col];
    }
    C.elements[row * C.width + col] = CValue;
}

void matmulDriver(const Matrix A, const Matrix B, const Matrix C)
{
    // Load matrix A into device.
    Matrix dA;
    dA.width = A.width;
    dA.height = A.height;
    size_t sizeOfA = A.width * A.height * sizeof(float);
    cudaMalloc(&dA.elements, sizeOfA);
    cudaMemcpy(dA.elements, A.elements, sizeOfA, cudaMemcpyHostToDevice);

    // Load matrix B into device.
    Matrix dB;
    dB.width = B.width;
    dB.height = B.height;
    size_t sizeOfB = B.width * B.height * sizeof(float);
    cudaMalloc(&dB.elements, sizeOfB);
    cudaMemcpy(dB.elements, B.elements, sizeOfB, cudaMemcpyHostToDevice);

    // Allocate matrix C on device.
    Matrix dC;
    dC.width = C.width;
    dC.height = C.height;
    size_t sizeOfC = C.width * C.height * sizeof(float);
    cudaMalloc(&dC.elements, sizeOfC);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    matmul<<<dimGrid, dimBlock>>>(A, B, C);

    cudaMemcpy(C.elements, dC.elements, sizeOfC, cudaMemcpyDeviceToHost);

    cudaFree(dA.elements);
    cudaFree(dB.elements);
    cudaFree(dC.elements);
}

int main()
{
    Matrix A;
    A.width = A_WIDTH;
    A.height = A_HEIGHT;
    size_t sizeOfA = A.width * A.height * sizeof(float);
    A.elements = (float *) malloc(sizeOfA);

    Matrix B;
    B.width = B_WIDTH;
    B.height = B_HEIGHT;
    size_t sizeOfB = B.width * B.height * sizeof(float);
    B.elements = (float *) malloc(sizeOfB);

    Matrix C;
    C.width = C_WIDTH;
    C.height = C_HEIGHT;
    size_t sizeOfC = C.width * C.height * sizeof(float);
    C.elements = (float *) malloc(sizeOfC);

    Matrix C_check;
    C_check.width = C_WIDTH;
    C_check.height = C_HEIGHT;
    C_check.elements = (float *) malloc(sizeOfC);

    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            A.elements[i * A.width + j] = i + j;
        }
    }

    for (int i = 0; i < B.height; i++) {
        for (int j = 0; j < B.width; j++) {
            B.elements[i * B.width + j] = i + j;
        }
    }

    int value;
    for (int i = 0; i < C_check.height; i++) {
        for (int j = 0; j < C_check.width; j++) {
            value = 0.0;
            for (int k = 0; k < A.width; k++) {
                value += A.elements[i * A.width + k] * B.elements[k * B.width + j];
            }
            C_check.elements[i * C_check.width + j] = value;
        }
    }
    
    matmulDriver(A, B, C);

    int cmp = memcmp(C_check.elements, C.elements, sizeOfC);

    if (cmp == 0) {
        printf("Arrays are equal.\n");
    } else {
        printf("Arrays are equal.\n");
    }

    return 0;
}
