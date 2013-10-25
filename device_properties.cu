#include <stdio.h>

int main() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    printf("%24s: %s\n", "Name", props.name);
    printf("%24s: %d\n", "Total global memory", props.totalGlobalMem);
    printf("%24s: %d\n", "Shared memory per block", props.sharedMemPerBlock);
    printf("%24s: %d\n", "Registers per block", props.regsPerBlock);
    printf("%24s: %d\n", "Warp size", props.warpSize);
    printf("%24s: %d\n", "memPitch", props.memPitch);
    printf("%24s: %d\n", "Max threads per block", props.maxThreadsPerBlock);
    printf("%24s: %dx%dx%d\n", "Max threads dimensions", 
            props.maxThreadsDim[0], 
            props.maxThreadsDim[1], 
            props.maxThreadsDim[2]);
    printf("%24s: %dx%dx%d\n", "Max grid size", 
            props.maxGridSize[0], 
            props.maxGridSize[1], 
            props.maxGridSize[2]);
    printf("%24s: %d kHz\n", "Clock rate", props.clockRate);
    printf("%24s: %d\n", "Total const memory", props.totalConstMem);
    printf("%24s: %d\n", "Major", props.major);
    printf("%24s: %d\n", "Minor", props.minor);
    printf("%24s: %d\n", "Texture alignment", props.textureAlignment);

    return 0;
}
