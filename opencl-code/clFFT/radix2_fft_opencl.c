#include <CL/cl.h>
#include <clFFT.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_CL_ERR(err, msg) \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s\n", (int)err, msg); \
        exit(EXIT_FAILURE); \
    }

#define N_VALUE 8192

int main(int argc, char* argv[]) {
    cl_device_type desiredDeviceType = CL_DEVICE_TYPE_GPU;
    if (argc > 1 && strcmp(argv[1], "CPU") == 0) {
        desiredDeviceType = CL_DEVICE_TYPE_CPU;
    }

    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_CL_ERR(err, "clGetPlatformIDs");
    if (numPlatforms == 0) {
        fprintf(stderr, "No OpenCL platform found.\n");
        return EXIT_FAILURE;
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
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL_ERR(err, "clCreateCommandQueue");

    cl_float2* hostInput  = (cl_float2*)malloc(sizeof(cl_float2) * N_VALUE);
    cl_float2* hostOutput = (cl_float2*)malloc(sizeof(cl_float2) * N_VALUE);
    for (int i = 0; i < N_VALUE; i++) {
        hostInput[i].s[0] = (float)(i + 1); // real part
        hostInput[i].s[1] = 0.0f;            // imaginary part
    }

    cl_mem bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(cl_float2) * N_VALUE, NULL, &err);
    CHECK_CL_ERR(err, "clCreateBuffer(bufInput)");
    cl_mem bufOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float2) * N_VALUE, NULL, &err);
    CHECK_CL_ERR(err, "clCreateBuffer(bufOutput)");

    err = clEnqueueWriteBuffer(queue, bufInput, CL_TRUE, 0, sizeof(cl_float2) * N_VALUE, hostInput, 0, NULL, NULL);
    CHECK_CL_ERR(err, "clEnqueueWriteBuffer(bufInput)");

    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);
    CHECK_CL_ERR(err, "clfftSetup");

    clfftPlanHandle plan;
    size_t clLengths[1] = { N_VALUE };
    err = clfftCreateDefaultPlan(&plan, context, CLFFT_1D, clLengths);
    CHECK_CL_ERR(err, "clfftCreateDefaultPlan");

    err = clfftSetPlanPrecision(plan, CLFFT_SINGLE);
    CHECK_CL_ERR(err, "clfftSetPlanPrecision");
    err = clfftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    CHECK_CL_ERR(err, "clfftSetLayout");
    
    err = clfftSetResultLocation(plan, CLFFT_OUTOFPLACE);
    CHECK_CL_ERR(err, "clfftSetResultLocation");

    err = clfftBakePlan(plan, 1, &queue, NULL, NULL);
    CHECK_CL_ERR(err, "clfftBakePlan");

    cl_event event = NULL;
    err = clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue, 0, NULL, &event, &bufInput, &bufOutput, NULL);
    CHECK_CL_ERR(err, "clfftEnqueueTransform");
    clWaitForEvents(1, &event);

    cl_ulong start, end;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    CHECK_CL_ERR(err, "clGetEventProfilingInfo(start)");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    CHECK_CL_ERR(err, "clGetEventProfilingInfo(end)");
    double kernelTimeMs = (double)(end - start) / 1e6;
    printf("clFFT kernel execution time: %0.3f ms\n", kernelTimeMs);

    clFinish(queue);

    err = clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, sizeof(cl_float2) * N_VALUE, hostOutput, 0, NULL, NULL);
    CHECK_CL_ERR(err, "clEnqueueReadBuffer(bufOutput)");

    printf("\nFFT Result (first 8 samples) [Forward FFT, N=%d]:\n", N_VALUE);
    for (int i = 0; i < 8; i++) {
        printf("  idx=%d => (%f, %f)\n", i, hostOutput[i].s[0], hostOutput[i].s[1]);
    }

    err = clfftDestroyPlan(&plan);
    CHECK_CL_ERR(err, "clfftDestroyPlan");
    clfftTeardown();

    clReleaseMemObject(bufInput);
    clReleaseMemObject(bufOutput);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(hostInput);
    free(hostOutput);

    printf("\nDone.\n");
    return 0;
}
