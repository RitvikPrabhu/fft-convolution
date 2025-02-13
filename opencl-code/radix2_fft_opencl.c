/*  - For large N, you must check device constraints.
 *  - Only works when N is a power of two.
 *  - The code here is minimal and omits some error checks for brevity.*/

#include <OpenCL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int N = 8;
static const int BATCH_COUNT = 1;

#define CHECK_ERROR(err, msg)                                                  \
  if ((err) != CL_SUCCESS) {                                                   \
    fprintf(stderr, "OpenCL Error %d at %s\n", (err), (msg));                  \
    exit(EXIT_FAILURE);                                                        \
  }

int main(int argc, char *argv[]) {

  cl_device_type desiredDeviceType = CL_DEVICE_TYPE_GPU;
  if (argc > 1 && strcmp(argv[1], "CPU") == 0) {
    desiredDeviceType = CL_DEVICE_TYPE_CPU;
  }

  int n = N;
  int batchCount = BATCH_COUNT;
  size_t totalSamples = (size_t)N * (size_t)BATCH_COUNT;
  cl_float2 *hostInput = (cl_float2 *)malloc(sizeof(cl_float2) * totalSamples);
  cl_float2 *hostOutput = (cl_float2 *)malloc(sizeof(cl_float2) * totalSamples);

  for (size_t i = 0; i < totalSamples; i++) {
    hostInput[i].s[0] = (float)(i + 1);
    hostInput[i].s[1] = 0.0f;
  }

  int inverseFFT = 0;

  cl_int err = 0;
  cl_uint numPlatforms = 0;
  cl_uint numDevices = 0;

  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_ERROR(err, "clGetPlatformIDs (count)");
  if (numPlatforms == 0) {
    fprintf(stderr, "No OpenCL platform found!\n");
    return 1;
  }
  cl_platform_id *platforms =
      (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platforms, NULL);
  CHECK_ERROR(err, "clGetPlatformIDs (list)");

  cl_platform_id chosenPlatform = platforms[0];
  free(platforms);

  err = clGetDeviceIDs(chosenPlatform, desiredDeviceType, 0, NULL, &numDevices);
  if (err != CL_SUCCESS || numDevices == 0) {
    fprintf(stderr,
            "No device of the requested type found. Falling back to CPU...\n");
    desiredDeviceType = CL_DEVICE_TYPE_CPU;
    err =
        clGetDeviceIDs(chosenPlatform, desiredDeviceType, 0, NULL, &numDevices);
    CHECK_ERROR(err, "clGetDeviceIDs (CPU fallback)");
  }

  cl_device_id *devices =
      (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
  err = clGetDeviceIDs(chosenPlatform, CL_DEVICE_TYPE_ALL, numDevices, devices,
                       NULL);
  CHECK_ERROR(err, "clGetDeviceIDs (list)");
  cl_device_id chosenDevice = devices[0];
  free(devices);

  cl_context context =
      clCreateContext(NULL, 1, &chosenDevice, NULL, NULL, &err);
  CHECK_ERROR(err, "clCreateContext");

  cl_command_queue queue = clCreateCommandQueue(context, chosenDevice, 0, &err);
  CHECK_ERROR(err, "clCreateCommandQueue");

  FILE *fp = fopen("batched_fft.cl", "r");
  if (!fp) {
    fprintf(stderr, "Failed to open batched_fft.cl\n");
    return 1;
  }
  fseek(fp, 0, SEEK_END);
  long fileSize = ftell(fp);
  rewind(fp);
  char *kernelSource = (char *)malloc(fileSize + 1);
  fread(kernelSource, 1, fileSize, fp);
  kernelSource[fileSize] = '\0';
  fclose(fp);

  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&kernelSource, NULL, &err);
  CHECK_ERROR(err, "clCreateProgramWithSource");

  err = clBuildProgram(program, 1, &chosenDevice, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program, chosenDevice, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &logSize);
    char *buildLog = (char *)malloc(logSize);
    clGetProgramBuildInfo(program, chosenDevice, CL_PROGRAM_BUILD_LOG, logSize,
                          buildLog, NULL);
    fprintf(stderr, "Build Log:\n%s\n", buildLog);
    free(buildLog);
    CHECK_ERROR(err, "clBuildProgram");
  }

  cl_kernel kernel = clCreateKernel(program, "batched_radix2_fft", &err);
  CHECK_ERROR(err, "clCreateKernel");

  cl_mem inputBuffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_float2) * totalSamples, hostInput, &err);
  CHECK_ERROR(err, "clCreateBuffer (input)");

  cl_mem outputBuffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(cl_float2) * totalSamples, NULL, &err);
  CHECK_ERROR(err, "clCreateBuffer (output)");

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
  err |= clSetKernelArg(kernel, 2, sizeof(int), &n);
  err |= clSetKernelArg(kernel, 3, sizeof(int), &batchCount);
  err |= clSetKernelArg(kernel, 4, sizeof(int), &inverseFFT);
  CHECK_ERROR(err, "clSetKernelArg");

  size_t globalSize = (size_t)BATCH_COUNT * (size_t)N;
  size_t localSize = (size_t)N;

  size_t maxWorkGroupSize;
  err = clGetDeviceInfo(chosenDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(size_t), &maxWorkGroupSize, NULL);
  CHECK_ERROR(err, "clGetDeviceInfo (CL_DEVICE_MAX_WORK_GROUP_SIZE)");
  if (localSize > maxWorkGroupSize) {
    fprintf(stderr, "ERROR: Chosen localSize %zu > device's max %zu.\n",
            localSize, maxWorkGroupSize);
    fprintf(stderr,
            "Either reduce N or use a more advanced tiling approach.\n");
    exit(EXIT_FAILURE);
  }

  err = clEnqueueNDRangeKernel(queue, kernel,
                               1,    // 1-dimensional range
                               NULL, // global offset
                               &globalSize, &localSize, 0, NULL, NULL);
  CHECK_ERROR(err, "clEnqueueNDRangeKernel");

  clFinish(queue);

  err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0,
                            sizeof(cl_float2) * totalSamples, hostOutput, 0,
                            NULL, NULL);
  CHECK_ERROR(err, "clEnqueueReadBuffer");

  for (int b = 0; b < BATCH_COUNT; b++) {
    printf("=== Batch %d ===\n", b);
    for (int i = 0; i < N; i++) {
      int index = b * N + i;
      printf("  idx=%d => (%f, %f)\n", i,
             hostOutput[index].s[0],  // real
             hostOutput[index].s[1]); // imag
    }
  }

  free(hostInput);
  free(hostOutput);

  clReleaseMemObject(inputBuffer);
  clReleaseMemObject(outputBuffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  printf("Done.\n");
  return 0;
}
