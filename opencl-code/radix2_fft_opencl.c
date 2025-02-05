#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error: %d\n", err);                                         \
    exit(1);                                                                   \
  }

char *readKernelSource(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel file: %s\n", filename);
    exit(1);
  }
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  rewind(fp);
  char *source = (char *)malloc(size + 1);
  fread(source, 1, size, fp);
  source[size] = '\0';
  fclose(fp);
  return source;
}

void computeBitReversalIndices(int *bit_rev, int n) {
  int num_bits = 0;
  int temp = n;
  while (temp > 1) {
    num_bits++;
    temp >>= 1;
  }
  for (int i = 0; i < n; i++) {
    int j = 0;
    for (int k = 0; k < num_bits; k++) {
      if (i & (1 << k))
        j |= 1 << (num_bits - 1 - k);
    }
    bit_rev[i] = j;
  }
}

int main(void) {
  const int n = 8;

  double data[2 * n] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0,
                        5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0};

  int *bit_rev = (int *)malloc(n * sizeof(int));
  computeBitReversalIndices(bit_rev, n);

  cl_int err;
  cl_platform_id platform;
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  cl_device_id device;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  CHECK_ERROR(err);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  char *source = readKernelSource("fft_kernel.cl");

  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  CHECK_ERROR(err);
  err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2 -cl-khr-fp64", NULL,
                       NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    printf("Build Log:\n%s\n", log);
    free(log);
    exit(1);
  }

  cl_kernel kernelBitRev = clCreateKernel(program, "bit_reverse", &err);
  CHECK_ERROR(err);
  cl_kernel kernelFftStage = clCreateKernel(program, "fft_stage", &err);
  CHECK_ERROR(err);

  cl_mem bufData =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(double) * 2 * n, data, &err);
  CHECK_ERROR(err);
  cl_mem bufBitRev =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * n, bit_rev, &err);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernelBitRev, 0, sizeof(cl_mem), &bufData);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernelBitRev, 1, sizeof(cl_mem), &bufBitRev);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernelBitRev, 2, sizeof(int), &n);
  CHECK_ERROR(err);

  size_t globalSize = n;
  err = clEnqueueNDRangeKernel(queue, kernelBitRev, 1, NULL, &globalSize, NULL,
                               0, NULL, NULL);
  CHECK_ERROR(err);
  clFinish(queue);

  for (int m = 2; m <= n; m *= 2) {
    int inverse = 0;
    err = clSetKernelArg(kernelFftStage, 0, sizeof(cl_mem), &bufData);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernelFftStage, 1, sizeof(int), &m);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernelFftStage, 2, sizeof(int), &inverse);
    CHECK_ERROR(err);

    globalSize = n / 2;
    err = clEnqueueNDRangeKernel(queue, kernelFftStage, 1, NULL, &globalSize,
                                 NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
    clFinish(queue);
  }

  double result[2 * n];
  err = clEnqueueReadBuffer(queue, bufData, CL_TRUE, 0, sizeof(double) * 2 * n,
                            result, 0, NULL, NULL);
  CHECK_ERROR(err);

  printf("FFT Result:\n");
  for (int i = 0; i < n; i++) {
    printf("[%d]: %f + %fi\n", i, result[2 * i], result[2 * i + 1]);
  }

  clReleaseMemObject(bufData);
  clReleaseMemObject(bufBitRev);
  clReleaseKernel(kernelBitRev);
  clReleaseKernel(kernelFftStage);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(source);
  free(bit_rev);

  return 0;
}
