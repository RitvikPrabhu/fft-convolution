#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define N (8192)

int main() {
    cufftComplex* h_input  = (cufftComplex*) malloc(sizeof(cufftComplex)*N);
    cufftComplex* h_output = (cufftComplex*) malloc(sizeof(cufftComplex)*N);

    for (int i = 0; i < N; i++) {
        h_input[i].x = (float)(i + 1); // real
        h_input[i].y = 0.0f;           // imag
    }

    cufftComplex* d_data = NULL;
    cudaMalloc((void**)&d_data, sizeof(cufftComplex)*N);

    cudaMemcpy(d_data, h_input, sizeof(cufftComplex)*N, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event.
    cudaEventRecord(start, 0);

    // Execute the FFT kernel.
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

    // Record the stop event.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute the elapsed time in milliseconds.
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total kernel execution time: %0.3f ms\n", elapsedTime);

    // Cleanup the CUDA events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_output, d_data, sizeof(cufftComplex)*N, cudaMemcpyDeviceToHost);

    printf("\nForward FFT:\n");
    for (int i = 0; i < 8; i++) {
        printf("  freq[%d] = (%f, %f)\n", i, h_output[i].x, h_output[i].y);
    }

    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_input);
    free(h_output);

    printf("\nDone.\n");
    return 0;
}
