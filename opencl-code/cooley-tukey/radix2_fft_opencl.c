#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static const int N_VALUE = 8192;
#define CHECK_CL_ERR(err, msg) \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s\n", (int)err, msg); \
        exit(EXIT_FAILURE);     \
    }

static void print_data(const char* title, cl_float2* data, int N) {
    printf("%s:\n", title);
    for (int i = 0; i < N; i++) {
        printf("  [%d] => (%f, %f)\n", i, data[i].s[0], data[i].s[1]);
    }
}

static int get_log2(int N) {
    int logN = 0;
    int temp = N;
    while (temp > 1) {
        if (temp % 2 != 0) return -1;
        temp >>= 1;
        logN++;
    }
    return logN;
}

int main(int argc, char* argv[])
{
	cl_device_type desiredDeviceType = CL_DEVICE_TYPE_GPU;
	if (argc > 1 && strcmp(argv[1], "CPU") == 0) {
		desiredDeviceType = CL_DEVICE_TYPE_CPU;
	}
    int N = N_VALUE; 
    int inverseFFT = 0;

    int logN = get_log2(N);
    if (logN < 0) {
        fprintf(stderr, "Error: N=%d is not a power of two.\n", N);
        return 1;
    }

    cl_float2* hostInput  = (cl_float2*)malloc(sizeof(cl_float2)*N);
    cl_float2* hostOutput = (cl_float2*)malloc(sizeof(cl_float2)*N);

    for (int i = 0; i < N; i++) {
        hostInput[i].s[0] = (float)(i+1); // real
        hostInput[i].s[1] = 0.0f;         // imag
    }

    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_CL_ERR(err, "clGetPlatformIDs");
    if (numPlatforms == 0) {
        fprintf(stderr, "No OpenCL platform found.\n");
        return 1;
    }
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL_ERR(err, "clGetPlatformIDs(1)");

    cl_device_id device;
    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, desiredDeviceType, 1, &device, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        printf("No device of the requested type found. Falling back to CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &numDevices);
        CHECK_CL_ERR(err, "clGetDeviceIDs(CPU)");
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERR(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL_ERR(err, "clCreateCommandQueue");

    FILE* fp = fopen("batched_fft.cl", "r");
    if (!fp) {
        fprintf(stderr, "Could not open batched_fft_multi_pass.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    char* kernelSrc = (char*)malloc(fsize+1);
    fread(kernelSrc, 1, fsize, fp);
    kernelSrc[fsize] = '\0';
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, 
                                (const char**)&kernelSrc, NULL, &err);
    CHECK_CL_ERR(err, "clCreateProgramWithSource");
    free(kernelSrc);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize=0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* buildLog = (char*)malloc(logSize+1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
        buildLog[logSize] = '\0';
        fprintf(stderr, "Build Log:\n%s\n", buildLog);
        free(buildLog);
        CHECK_CL_ERR(err, "clBuildProgram");
    }

    cl_kernel krnBitReverse  = clCreateKernel(program, "bit_reverse_kernel", &err);
    CHECK_CL_ERR(err, "clCreateKernel(bit_reverse_kernel)");
    cl_kernel krnFFTStage    = clCreateKernel(program, "fft_stage_kernel", &err);
    CHECK_CL_ERR(err, "clCreateKernel(fft_stage_kernel)");
    cl_kernel krnFinalScale  = clCreateKernel(program, "final_scale_kernel", &err);
    CHECK_CL_ERR(err, "clCreateKernel(final_scale_kernel)");

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(cl_float2)*N, NULL, &err);
    CHECK_CL_ERR(err, "clCreateBuffer(bufA)");
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(cl_float2)*N, NULL, &err);
    CHECK_CL_ERR(err, "clCreateBuffer(bufB)");

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
                               sizeof(cl_float2)*N,
                               hostInput, 0, NULL, NULL);
    CHECK_CL_ERR(err, "clEnqueueWriteBuffer(bufA)");

    {
        err  = clSetKernelArg(krnBitReverse, 0, sizeof(cl_mem), &bufA);
        err |= clSetKernelArg(krnBitReverse, 1, sizeof(cl_mem), &bufB);
        err |= clSetKernelArg(krnBitReverse, 2, sizeof(int),    &N);
        CHECK_CL_ERR(err, "clSetKernelArg(bit_reverse_kernel)");

        size_t globalSize = N;
        err = clEnqueueNDRangeKernel(queue, krnBitReverse,
                                     1, NULL,
                                     &globalSize, NULL,
                                     0, NULL, NULL);
        CHECK_CL_ERR(err, "clEnqueueNDRangeKernel(bit_reverse_kernel)");
    }

    cl_mem readBuf  = bufB;  
    cl_mem writeBuf = bufA; 

    for (int s = 0; s < logN; s++) {
        err  = clSetKernelArg(krnFFTStage, 0, sizeof(cl_mem), &readBuf);
        err |= clSetKernelArg(krnFFTStage, 1, sizeof(cl_mem), &writeBuf);
        err |= clSetKernelArg(krnFFTStage, 2, sizeof(int),     &N);
        err |= clSetKernelArg(krnFFTStage, 3, sizeof(int),     &s);
        err |= clSetKernelArg(krnFFTStage, 4, sizeof(int),     &inverseFFT);
        CHECK_CL_ERR(err, "clSetKernelArg(fft_stage_kernel)");

        size_t globalSize = (size_t)(N/2);

        err = clEnqueueNDRangeKernel(queue, krnFFTStage,
                                     1, NULL, &globalSize, NULL,
                                     0, NULL, NULL);
        CHECK_CL_ERR(err, "clEnqueueNDRangeKernel(fft_stage_kernel)");

        cl_mem temp = readBuf;
        readBuf  = writeBuf;
        writeBuf = temp;
    }

    if (inverseFFT) {
        err  = clSetKernelArg(krnFinalScale, 0, sizeof(cl_mem), &readBuf);
        err |= clSetKernelArg(krnFinalScale, 1, sizeof(int),     &N);
        CHECK_CL_ERR(err, "clSetKernelArg(final_scale_kernel)");

        size_t globalSize = N;
        err = clEnqueueNDRangeKernel(queue, krnFinalScale,
                                     1, NULL, &globalSize, NULL,
                                     0, NULL, NULL);
        CHECK_CL_ERR(err, "clEnqueueNDRangeKernel(final_scale_kernel)");
    }

    err = clEnqueueReadBuffer(queue, readBuf, CL_TRUE, 0,
                              sizeof(cl_float2)*N,
                              hostOutput, 0, NULL, NULL);
    CHECK_CL_ERR(err, "clEnqueueReadBuffer(finalResult)");

    clFinish(queue);

    printf("\nFFT Result (first 8 samples) [inverse=%d, N=%d]:\n", inverseFFT, N);
    for(int i=0; i<8; i++) {
        printf("  idx=%d => (%f, %f)\n", i,
               hostOutput[i].s[0],
               hostOutput[i].s[1]);
    }

    free(hostInput);
    free(hostOutput);

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);

    clReleaseKernel(krnBitReverse);
    clReleaseKernel(krnFFTStage);
    clReleaseKernel(krnFinalScale);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("\nDone.\n");
    return 0;
}

