#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static const int N_VALUE = 8192;

#define CHECK_CL_ERR(err, msg) \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s\n", (int)(err), (msg)); \
        exit(EXIT_FAILURE);     \
    }

static int get_log2(int N) {
    int logN = 0;
    int tmp = N;
    while (tmp > 1) {
        if (tmp % 2 != 0) return -1; 
        tmp >>= 1;
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

    // Host buffers
    cl_float2* hostInput  = (cl_float2*)malloc(sizeof(cl_float2)*N);
    cl_float2* hostOutput = (cl_float2*)malloc(sizeof(cl_float2)*N);

    for (int i = 0; i < N; i++) {
        hostInput[i].s[0] = (float)(i+1); // real
        hostInput[i].s[1] = 0.0f;         // imag
    }

    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_CL_ERR(err, "clGetPlatformIDs(count)");
    if (numPlatforms == 0) {
        fprintf(stderr, "No OpenCL platform found.\n");
        return 1;
    }
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL_ERR(err, "clGetPlatformIDs(1)");

    cl_device_id device;
    cl_uint numDevices=0;
    err = clGetDeviceIDs(platform, desiredDeviceType, 1, &device, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        printf("No device of the requested type found. Falling back to CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &numDevices);
        CHECK_CL_ERR(err, "clGetDeviceIDs(CPU fallback)");
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERR(err, "clCreateContext");

    // Create a profiling-enabled command queue.
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL_ERR(err, "clCreateCommandQueue");

    int halfN = N/2;
    cl_float2* hostTwiddles = (cl_float2*)malloc(sizeof(cl_float2)*halfN);
    for (int k = 0; k < halfN; k++) {
        float angle = 2.0f * (float)M_PI * (float)k / (float)N;
        hostTwiddles[k].s[0] = cosf(angle);
        hostTwiddles[k].s[1] = sinf(angle);
    }

    cl_mem twiddleBuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(cl_float2)*halfN, hostTwiddles, &err);
    CHECK_CL_ERR(err, "clCreateBuffer(twiddleBuf)");
    free(hostTwiddles);

    FILE* fp = fopen("batched_fft.cl", "r");
    if (!fp) {
        fprintf(stderr, "Could not open batched_fft.cl\n");
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    char* kernelSrc = (char*)malloc(fsize+1);
    fread(kernelSrc, 1, fsize, fp);
    kernelSrc[fsize] = '\0';
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSrc, NULL, &err);
    CHECK_CL_ERR(err, "clCreateProgramWithSource");
    free(kernelSrc);

    const char* buildOptions = "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(program, 1, &device, buildOptions, NULL, NULL);
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

    cl_kernel krnBitReverse = clCreateKernel(program, "bit_reverse_kernel", &err);
    CHECK_CL_ERR(err, "clCreateKernel(bit_reverse_kernel)");

    cl_kernel krnTwoStages = clCreateKernel(program, "two_stages_fft_kernel", &err);
    CHECK_CL_ERR(err, "clCreateKernel(two_stages_fft_kernel)");

    cl_kernel krnFinalScale = clCreateKernel(program, "final_scale_kernel", &err);
    CHECK_CL_ERR(err, "clCreateKernel(final_scale_kernel)");

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_float2)*N, hostInput, &err);
    CHECK_CL_ERR(err, "clCreateBuffer(bufA)");

    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(cl_float2)*N, NULL, &err);
    CHECK_CL_ERR(err, "clCreateBuffer(bufB)");

    // Variable to accumulate total kernel execution time (in ms)
    double total_kernel_time = 0.0;

    // bit_reverse_kernel invocation
    {
        err  = clSetKernelArg(krnBitReverse, 0, sizeof(cl_mem), &bufA);
        err |= clSetKernelArg(krnBitReverse, 1, sizeof(cl_mem), &bufB);
        err |= clSetKernelArg(krnBitReverse, 2, sizeof(int),    &N);
        CHECK_CL_ERR(err, "clSetKernelArg(bit_reverse_kernel)");

        size_t globalSize = N;
        size_t localSize = (N < 128) ? N : 128;
        cl_event event_bit_reverse;
        err = clEnqueueNDRangeKernel(queue, krnBitReverse,
                                     1, NULL,
                                     &globalSize, &localSize,
                                     0, NULL, &event_bit_reverse);
        CHECK_CL_ERR(err, "clEnqueueNDRangeKernel(bit_reverse_kernel)");
        clFinish(queue);
        cl_ulong start, end;
        err = clGetEventProfilingInfo(event_bit_reverse, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        CHECK_CL_ERR(err, "clGetEventProfilingInfo(bit_reverse start)");
        err = clGetEventProfilingInfo(event_bit_reverse, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        CHECK_CL_ERR(err, "clGetEventProfilingInfo(bit_reverse end)");
        total_kernel_time += (double)(end - start) / 1e6;
        clReleaseEvent(event_bit_reverse);
    }

    cl_mem readBuf = bufB;
    cl_mem writeBuf = bufA;

    // two_stages_fft_kernel invocations
    for (int s = 0; s < logN; s += 2) {
        int numStages = 2;
        if (s + 1 >= logN) {
            numStages = 1;
        }

        err  = clSetKernelArg(krnTwoStages, 0, sizeof(cl_mem), &readBuf);
        err |= clSetKernelArg(krnTwoStages, 1, sizeof(cl_mem), &writeBuf);
        err |= clSetKernelArg(krnTwoStages, 2, sizeof(int),     &N);
        err |= clSetKernelArg(krnTwoStages, 3, sizeof(int),     &s);            
        err |= clSetKernelArg(krnTwoStages, 4, sizeof(int),     &numStages);   
        err |= clSetKernelArg(krnTwoStages, 5, sizeof(int),     &inverseFFT);
        err |= clSetKernelArg(krnTwoStages, 6, sizeof(cl_mem),  &twiddleBuf);
        CHECK_CL_ERR(err, "clSetKernelArg(two_stages_fft_kernel)");

        size_t globalSizeStage = N/2;
        size_t localSizeStage = (globalSizeStage < 128) ? globalSizeStage : 128;
        cl_event event_two_stages;
        err = clEnqueueNDRangeKernel(queue, krnTwoStages,
                                     1, NULL,
                                     &globalSizeStage, &localSizeStage,
                                     0, NULL, &event_two_stages);
        CHECK_CL_ERR(err, "clEnqueueNDRangeKernel(two_stages_fft_kernel)");
        clFinish(queue);
        cl_ulong start, end;
        err = clGetEventProfilingInfo(event_two_stages, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        CHECK_CL_ERR(err, "clGetEventProfilingInfo(two_stages start)");
        err = clGetEventProfilingInfo(event_two_stages, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        CHECK_CL_ERR(err, "clGetEventProfilingInfo(two_stages end)");
        total_kernel_time += (double)(end - start) / 1e6;
        clReleaseEvent(event_two_stages);

        cl_mem tmp = readBuf;
        readBuf = writeBuf;
        writeBuf = tmp;
    }

    // final_scale_kernel invocation (if inverseFFT)
    if (inverseFFT) {
        err  = clSetKernelArg(krnFinalScale, 0, sizeof(cl_mem), &readBuf);
        err |= clSetKernelArg(krnFinalScale, 1, sizeof(int),     &N);
        CHECK_CL_ERR(err, "clSetKernelArg(final_scale_kernel)");

        size_t finalGlobalSize = N;
        cl_event event_final_scale;
        err = clEnqueueNDRangeKernel(queue, krnFinalScale,
                                     1, NULL,
                                     &finalGlobalSize, NULL,
                                     0, NULL, &event_final_scale);
        CHECK_CL_ERR(err, "clEnqueueNDRangeKernel(final_scale_kernel)");
        clFinish(queue);
        cl_ulong start, end;
        err = clGetEventProfilingInfo(event_final_scale, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        CHECK_CL_ERR(err, "clGetEventProfilingInfo(final_scale start)");
        err = clGetEventProfilingInfo(event_final_scale, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        CHECK_CL_ERR(err, "clGetEventProfilingInfo(final_scale end)");
        total_kernel_time += (double)(end - start) / 1e6;
        clReleaseEvent(event_final_scale);
    }

    err = clEnqueueReadBuffer(queue, readBuf, CL_TRUE, 0,
                              sizeof(cl_float2)*N, hostOutput,
                              0, NULL, NULL);
    CHECK_CL_ERR(err, "clEnqueueReadBuffer(final)");

    clFinish(queue);

    printf("\nFFT Result (first 8) [N=%d, inverse=%d]:\n", N, inverseFFT);
    for(int i = 0; i < 8; i++){
        printf("  idx=%d => (%f, %f)\n", i, hostOutput[i].s[0], hostOutput[i].s[1]);
    }

    // Print the total kernel execution time (in ms)
    printf("Total kernel execution time: %0.3f ms\n", total_kernel_time);

    // Cleanup
    free(hostInput);
    free(hostOutput);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(twiddleBuf);
    clReleaseKernel(krnBitReverse);
    clReleaseKernel(krnTwoStages);
    clReleaseKernel(krnFinalScale);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("\nDone.\n");
    return 0;
}
