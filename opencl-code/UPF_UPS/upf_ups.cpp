#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <clFFT.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------
static constexpr size_t BLOCK_LEN = 1024;                 // B
static constexpr size_t TRANSFORM_LEN = BLOCK_LEN * 2;    // K = 2B (power‑of‑two)

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static std::vector<float> readBinary(const char* fn)
{
    std::ifstream f(fn, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + fn);
    std::streamsize sz = f.tellg();
    if (sz % sizeof(float) != 0)
        throw std::runtime_error("Size of " + std::string(fn) + " not multiple of 4");
    std::vector<float> data(sz / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

static void writeBinary(const char* fn, const std::vector<float>& data)
{
    std::ofstream f(fn, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

#define CLCHK(x) do { cl_int err_ = (x); if (err_ != CL_SUCCESS) { \
    std::cerr << "OpenCL error " << err_ << " at " << __LINE__ << std::endl; std::exit(EXIT_FAILURE);} } while(0)

// -----------------------------------------------------------------------------
// OpenCL helpers – pick default platform/device, build kernels
// -----------------------------------------------------------------------------
static cl_context createContext(cl_device_id& dev)
{
    cl_uint numPlatforms = 0;
    CLCHK(clGetPlatformIDs(0, nullptr, &numPlatforms));
    std::vector<cl_platform_id> plats(numPlatforms);
    CLCHK(clGetPlatformIDs(numPlatforms, plats.data(), nullptr));
    for (auto p : plats)
    {
        cl_uint numDevs = 0;
        CLCHK(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevs));
        if (!numDevs) continue;
        std::vector<cl_device_id> devs(numDevs);
        CLCHK(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, numDevs, devs.data(), nullptr));
        dev = devs[0];
        cl_int err;
        cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
        CLCHK(err);
        return ctx;
    }
    throw std::runtime_error("No GPU device found");
}

static cl_command_queue createQueue(cl_context ctx, cl_device_id dev)
{
#if defined(CL_VERSION_2_0)
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, nullptr);
#else
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, nullptr);
#endif
    return q;
}

static const char* kernelSrc = R"CLC(
// Complex numbers are stored as float2 (x = real, y = imag)
__kernel void cmul_acc(__global const float2* __restrict X,
                       __global const float2* __restrict H,
                       __global float2* Y,
                       int first)
{
    const int gid = get_global_id(0);
    float2 a = X[gid];
    float2 b = H[gid];
    float2 prod = (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
    if (first)
        Y[gid] = prod;
    else
        Y[gid] += prod;
}
)CLC";

static cl_program buildProgram(cl_context ctx, cl_device_id dev, const char* src)
{
    size_t len = std::strlen(src);
    const char* buf = src;
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &buf, &len, &err);
    CLCHK(err);
    err = clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t logsz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::string log(logsz, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << log << std::endl;
        throw std::runtime_error("Failed to build program");
    }
    return prog;
}

// -----------------------------------------------------------------------------
// Main processing – Uniform Partitioned Overlap‑Save convolution
// -----------------------------------------------------------------------------
int main()
{
    try {
        std::vector<float> x = readBinary("input.bin");
        std::vector<float> h = readBinary("fir.bin");
        const size_t M = x.size();
        const size_t N = h.size();
        const size_t P = (N + BLOCK_LEN - 1) / BLOCK_LEN;   // partitions
        const size_t totalBlocks = (M + BLOCK_LEN - 1) / BLOCK_LEN;

        // ------------------------------------------------------------------
        // OpenCL setup
        // ------------------------------------------------------------------
        cl_device_id dev = nullptr;
        cl_context ctx = createContext(dev);
        cl_command_queue q = createQueue(ctx, dev);
        cl_program prog = buildProgram(ctx, dev, kernelSrc);
        cl_kernel k_cmul_acc = clCreateKernel(prog, "cmul_acc", nullptr);

        // ------------------------------------------------------------------
        // clFFT setup (single‑precision, in‑place complex‑interleaved)
        // ------------------------------------------------------------------
        clfftSetupData fftSetup;
        clfftInitSetupData(&fftSetup);
        clfftSetup(&fftSetup);

        clfftPlanHandle planFwd, planInv;
        size_t lengths[1] = { TRANSFORM_LEN };
        clfftCreateDefaultPlan(&planFwd, ctx, CLFFT_1D, lengths);
        clfftSetPlanPrecision(planFwd, CLFFT_SINGLE);
        clfftSetLayout(planFwd, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
        clfftSetResultLocation(planFwd, CLFFT_OUTOFPLACE);
        clfftBakePlan(planFwd, 1, &q, nullptr, nullptr);

        clfftCreateDefaultPlan(&planInv, ctx, CLFFT_1D, lengths);
        clfftSetPlanPrecision(planInv, CLFFT_SINGLE);
        clfftSetLayout(planInv, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
        clfftSetResultLocation(planInv, CLFFT_OUTOFPLACE);
        float invScale = 1.0f / static_cast<float>(TRANSFORM_LEN);
        clfftSetPlanScale(planInv, CLFFT_BACKWARD, invScale);
        clfftBakePlan(planInv, 1, &q, nullptr, nullptr);

        // ------------------------------------------------------------------
        // Device buffers
        // ------------------------------------------------------------------
        size_t spectralElems = TRANSFORM_LEN / 2 + 1;  // real→hermitian length
        size_t spectralBytes = spectralElems * sizeof(cl_float2);

        // Filter spectra (P × spectralElems)
        std::vector<cl_mem> d_H(P);
        // Host temporary buffers
        std::vector<float> hostTD(TRANSFORM_LEN, 0.0f);
        std::vector<float> hostTD_out(TRANSFORM_LEN);

        // Create accumulator spectrum buffer
        cl_mem d_Y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, spectralBytes, nullptr, nullptr);

        // Input spectrum FDL (P slots)
        std::vector<cl_mem> d_Xfdl(P);
        for (size_t i = 0; i < P; ++i)
        {
            d_Xfdl[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, spectralBytes, nullptr, nullptr);
        }

        // Temporary in/out buffers for clFFT (each transform allocates its own)
        cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE, TRANSFORM_LEN * sizeof(float), nullptr, nullptr);
        cl_mem d_spec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, spectralBytes, nullptr, nullptr);
        cl_mem d_td   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, TRANSFORM_LEN * sizeof(float), nullptr, nullptr);

        // ------------------------------------------------------------------
        // Pre‑compute filter partition spectra
        // ------------------------------------------------------------------
        for (size_t p = 0; p < P; ++p)
        {
            std::fill(hostTD.begin(), hostTD.end(), 0.0f);
            size_t base = p * BLOCK_LEN;
            size_t copy = std::min(BLOCK_LEN, N - base);
            std::memcpy(hostTD.data(), h.data() + base, copy * sizeof(float));
            // Upload real data
            CLCHK(clEnqueueWriteBuffer(q, d_in, CL_TRUE, 0, TRANSFORM_LEN * sizeof(float), hostTD.data(), 0, nullptr, nullptr));
            // Fwd FFT (real→Hermitian)
            cl_mem inbufs[1]  = { d_in };
            cl_mem outbufs[1] = { d_spec };
            CLCHK(clfftEnqueueTransform(planFwd, CLFFT_FORWARD, 1, &q, 0, nullptr, nullptr, inbufs, outbufs, nullptr));
            CLCHK(clFinish(q));
            // Allocate buffer and copy spectrum into permanent storage
            d_H[p] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, spectralBytes, nullptr, nullptr);
            CLCHK(clEnqueueCopyBuffer(q, d_spec, d_H[p], 0, 0, spectralBytes, 0, nullptr, nullptr));
        }

        // ------------------------------------------------------------------
        // Streaming convolution loop (overlap‑save)
        // ------------------------------------------------------------------
        size_t outLen = M + N - 1;
        std::vector<float> y(outLen, 0.0f);
        size_t writePos = 0;
        size_t ringHead = 0;  // index of newest spectrum in FDL

        size_t blocksToProcess = totalBlocks + P;  // flush tail with zeros
        size_t srcPos = 0;
        for (size_t blk = 0; blk < blocksToProcess; ++blk)
        {
            // Prepare time‑domain window of K samples (sliding)
            // Shift left by BLOCK_LEN
            std::move(hostTD.begin() + BLOCK_LEN, hostTD.end(), hostTD.begin());
            // Fill rightmost B with new samples or zeros if finished
            size_t samplesLeft = (srcPos < M) ? std::min(BLOCK_LEN, M - srcPos) : 0;
            if (samplesLeft)
                std::memcpy(hostTD.data() + (TRANSFORM_LEN - BLOCK_LEN), x.data() + srcPos, samplesLeft * sizeof(float));
            std::fill(hostTD.begin() + (TRANSFORM_LEN - BLOCK_LEN) + samplesLeft, hostTD.end(), 0.0f);
            srcPos += samplesLeft;

            // Upload window
            CLCHK(clEnqueueWriteBuffer(q, d_in, CL_TRUE, 0, TRANSFORM_LEN * sizeof(float), hostTD.data(), 0, nullptr, nullptr));
            // Forward FFT
            cl_mem inbufs[1]  = { d_in };
            cl_mem outbufs[1] = { d_spec };
            CLCHK(clfftEnqueueTransform(planFwd, CLFFT_FORWARD, 1, &q, 0, nullptr, nullptr, inbufs, outbufs, nullptr));
            CLCHK(clFinish(q));

            // Store spectrum in FDL slot ringHead (overwrite old)
            CLCHK(clEnqueueCopyBuffer(q, d_spec, d_Xfdl[ringHead], 0, 0, spectralBytes, 0, nullptr, nullptr));

            // Zero accumulator Y in device (lazy – first=1 will overwrite anyway)

            // Loop over filter partitions
            for (size_t p = 0; p < P; ++p)
            {
                size_t delayBlocks = p;
                size_t slot = (ringHead + P - delayBlocks) % P; // find spectrum delayed by p blocks
                int first = (p == 0) ? 1 : 0;

                // Set kernel args
                CLCHK(clSetKernelArg(k_cmul_acc, 0, sizeof(cl_mem), &d_Xfdl[slot]));
                CLCHK(clSetKernelArg(k_cmul_acc, 1, sizeof(cl_mem), &d_H[p]));
                CLCHK(clSetKernelArg(k_cmul_acc, 2, sizeof(cl_mem), &d_Y));
                CLCHK(clSetKernelArg(k_cmul_acc, 3, sizeof(int), &first));

                size_t gsz = spectralElems;
                CLCHK(clEnqueueNDRangeKernel(q, k_cmul_acc, 1, nullptr, &gsz, nullptr, 0, nullptr, nullptr));
            }
            CLCHK(clFinish(q));

            // Inverse FFT (Hermitian→real)
            cl_mem inv_in[1]  = { d_Y };
            cl_mem inv_out[1] = { d_td }; // length K real out
            CLCHK(clfftEnqueueTransform(planInv, CLFFT_BACKWARD, 1, &q, 0, nullptr, nullptr, inv_in, inv_out, nullptr));
            CLCHK(clFinish(q));

            // Download and copy last B samples to output
            CLCHK(clEnqueueReadBuffer(q, d_td, CL_TRUE, 0, TRANSFORM_LEN * sizeof(float), hostTD_out.data(), 0, nullptr, nullptr));

            if (writePos < y.size())
            {
                size_t canWrite = std::min(BLOCK_LEN, y.size() - writePos);
                std::memcpy(y.data() + writePos, hostTD_out.data() + (TRANSFORM_LEN - BLOCK_LEN), canWrite * sizeof(float));
                writePos += canWrite;
            }

            // Advance ring head pointer
            ringHead = (ringHead + P - 1) % P; // move backwards so that slot mapping works
        }

        // ------------------------------------------------------------------
        // Write to gpu.bin for the Python checker
        // ------------------------------------------------------------------
        writeBinary("gpu.bin", y);

        // Cleanup
        clfftDestroyPlan(&planFwd);
        clfftDestroyPlan(&planInv);
        clfftTeardown();
        clReleaseKernel(k_cmul_acc);
        clReleaseProgram(prog);
        for (auto b : d_H) clReleaseMemObject(b);
        for (auto b : d_Xfdl) clReleaseMemObject(b);
        clReleaseMemObject(d_Y);
        clReleaseMemObject(d_in);
        clReleaseMemObject(d_spec);
        clReleaseMemObject(d_td);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);

        std::cout << "GPU convolution written to gpu.bin (" << y.size() << " samples)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
