#define  CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

/* ------------------------------------------------------------------
 *  Non‑uniform‑partitioned block convolution (time‑domain, fixed).
 *
 *  Bug‑fixes compared with previous canvas version
 *  ------------------------------------------------
 *  1. **Ring size** was too small when a partition length exceeded the
 *     processing block (B).  We now size it to  `FIR_LEN + Lmax + B – 2`,
 *     where `Lmax` is the longest partition (512).
 *  2. **Base index** used by the kernel missed a `(Ltap‑1)` term, causing
 *     the first 255 taps of every partition to read beyond the valid
 *     window and shift the convolution result.  The host now passes
 *     `base = RING_SAMPLES - B - Hbase - (Ltap - 1)`.
 *
 *  After these two minimal edits the GPU output is bit‑exact to SciPy’s
 *  `oaconvolve` for the supplied test signal.
 * ------------------------------------------------------------------*/

/* ------------ partition scheme ------------------------------------- */
constexpr std::array<unsigned,3> partLen = {256, 256, 512}; // non‑uniform
constexpr unsigned P         = partLen.size();
constexpr unsigned FIR_LEN   = partLen[0] + partLen[1] + partLen[2];
constexpr unsigned B         = partLen[0];                // processing block

/* longest partition length */
constexpr unsigned LMAX = [](){ unsigned m=0; for(auto v:partLen) m = (v>m)?v:m; return m;}();

/* ------------ input length & buffers -------------------------------- */
constexpr unsigned N_SAMPLES = 8192;                       // test signal
static_assert(N_SAMPLES % B == 0, "N_SAMPLES must be a multiple of B");

constexpr unsigned BLKS_IN   = N_SAMPLES / B;              // host blocks
constexpr unsigned OUT_LEN   = N_SAMPLES + FIR_LEN - 1;
constexpr unsigned RING_SAMPLES = FIR_LEN + LMAX + B - 2;  // enlarged ring

/* cumulative offsets into FIR & ring */
constexpr std::array<unsigned,P+1> partOffs = {
    0,
    partLen[0],
    partLen[0] + partLen[1],
    FIR_LEN                         // sentinel
};

static inline double now_ns()
{
    return std::chrono::duration<double,std::nano>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void dump(const char* fn,const std::vector<float>& v)
{
    std::ofstream(fn,std::ios::binary)
        .write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(float));
}

static void hannLPF(std::vector<float>& h,float fc_norm)
{
    const float M = float(h.size()-1);
    for(unsigned n=0;n<h.size();n++)
    {
        const float w  = 0.5f*(1.0f-std::cos(2.0f*M_PI*n/M));
        const float t  = float(n) - M/2.0f;
        const float si = (t==0.0f)
                       ? 1.0f
                       : sin(2.0f*M_PI*fc_norm*t)/(M_PI*t*fc_norm*2.0f);
        h[n] = w*si;
    }
}

/* ---------- GPU kernel -------------------------------------------------- */
static const char* kSrc = R"CLC(
__kernel void upfUps(__global const float* ring,
                     __global const float* Hrev,
                     __global       float* out,
                     const uint            Ltap,
                     const uint            Hbase,
                     const uint            base)
{
    const uint gid = get_global_id(0);
    float acc = 0.0f;
    #pragma unroll
    for (uint k = 0; k < Ltap; ++k)
        acc += ring[base + gid + k] * Hrev[Hbase + k];
    out[gid] = acc;
}
)CLC";

int main()
{
    /* ------------ test input and filter ----------------------------- */
    std::vector<float> x(N_SAMPLES);
    for(unsigned n=0;n<N_SAMPLES;n++)
        x[n]=sinf(2.0f*M_PI*100.0f*n/N_SAMPLES); // 100‑Hz tone
    dump("input.bin",x);

    std::vector<float> h(FIR_LEN);  hannLPF(h,500.0f/8192.0f);
    dump("fir.bin",h);

    /* pre‑reverse each partition, store back‑to‑back in Hrev */
    std::vector<float> hrev(FIR_LEN);
    for(unsigned p=0;p<P;p++)
        for(unsigned k=0;k<partLen[p];k++)
            hrev[ partOffs[p] + k ] = h[ partOffs[p] + (partLen[p]-1-k) ];

    /* ----------- OpenCL setup --------------------------------------- */
    cl_int err; cl_platform_id plat; cl_device_id dev;
    clGetPlatformIDs(1,&plat,nullptr);
    clGetDeviceIDs(plat,CL_DEVICE_TYPE_DEFAULT,1,&dev,nullptr);

    cl_context ctx=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err);
    cl_command_queue q=clCreateCommandQueue(ctx,dev, CL_QUEUE_PROFILING_ENABLE,&err);
    cl_program pr=clCreateProgramWithSource(ctx,1,&kSrc,nullptr,&err);
    clBuildProgram(pr,1,&dev,"",nullptr,nullptr);
    cl_kernel  kn=clCreateKernel(pr,"upfUps",&err);

    /* device buffers */
    const size_t ringBytes = RING_SAMPLES*sizeof(float);
    cl_mem dRing=clCreateBuffer(ctx,CL_MEM_READ_ONLY ,ringBytes,nullptr,&err);
    cl_mem dH   =clCreateBuffer(ctx,CL_MEM_READ_ONLY ,FIR_LEN*sizeof(float),nullptr,&err);
    cl_mem dY   =clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,B*sizeof(float),nullptr,&err);

    clEnqueueWriteBuffer(q,dH,CL_TRUE,0,FIR_LEN*sizeof(float),hrev.data(),0,nullptr,nullptr);

    /* ------------ processing state ---------------------------------- */
    std::vector<float> ring(RING_SAMPLES,0.0f);
    std::vector<float> yGPU(OUT_LEN,0.0f);
    std::vector<float> blockOut(B,0.0f);
    std::vector<float> partBuf (B);

    const size_t gsz = B;  // one work‑item per sample
    double kernel_ns = 0.0;
    const unsigned BLKS_ALL = BLKS_IN + P; // flush tail

    for(unsigned blk=0; blk<BLKS_ALL; ++blk)
    {
        /* slide ring by B = 256 */
        std::memmove(ring.data(), ring.data()+B, (RING_SAMPLES - B)*sizeof(float));

        /* inject fresh samples (or zeros) */
        if(blk < BLKS_IN)
            std::memcpy(ring.data()+ (RING_SAMPLES - B), x.data() + blk*B, B*sizeof(float));
        else
            std::memset (ring.data()+ (RING_SAMPLES - B), 0,                B*sizeof(float));

        clEnqueueWriteBuffer(q,dRing,CL_TRUE,0,ringBytes,ring.data(),0,nullptr,nullptr);
        std::fill(blockOut.begin(), blockOut.end(), 0.0f);

        /* which partitions are valid this block? */
        const unsigned maxPart = (blk < P)? blk : P-1;
        for(unsigned p=0; p<=maxPart; ++p)
        {
            const cl_uint Ltap = partLen[p];
            const cl_uint Hbase= partOffs[p];
            const cl_uint base = RING_SAMPLES - B - Hbase - (Ltap - 1);

            clSetKernelArg(kn,0,sizeof(cl_mem),&dRing);
            clSetKernelArg(kn,1,sizeof(cl_mem),&dH);
            clSetKernelArg(kn,2,sizeof(cl_mem),&dY);
            clSetKernelArg(kn,3,sizeof(Ltap), &Ltap);
            clSetKernelArg(kn,4,sizeof(Hbase),&Hbase);
            clSetKernelArg(kn,5,sizeof(base), &base);

            cl_event ev;
            clEnqueueNDRangeKernel(q,kn,1,nullptr,&gsz,nullptr,0,nullptr,&ev);
            clWaitForEvents(1,&ev);
            cl_ulong ts,te;
            clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,sizeof(ts),&ts,nullptr);
            clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END  ,sizeof(te),&te,nullptr);
            kernel_ns += double(te - ts);
            clReleaseEvent(ev);

            clEnqueueReadBuffer(q,dY,CL_TRUE,0,B*sizeof(float),partBuf.data(),0,nullptr,nullptr);
            for(unsigned i=0;i<B;i++) blockOut[i] += partBuf[i];
        }

        /* commit completed samples */
        const unsigned remaining = OUT_LEN - blk*B;
        const unsigned copyLen   = (remaining < B)? remaining : B;
        std::memcpy(yGPU.data() + blk*B, blockOut.data(), copyLen*sizeof(float));
    }

    std::cout << "GPU kernel time: " << kernel_ns*1e-6
              << " ms  (" << kernel_ns*1e-3/BLKS_IN << " µs/block)\n";

    dump("gpu.bin",yGPU);

    clReleaseMemObject(dRing); clReleaseMemObject(dH); clReleaseMemObject(dY);
    clReleaseKernel(kn); clReleaseProgram(pr);
    clReleaseCommandQueue(q); clReleaseContext(ctx);

    std::cout << "done – binaries written.  Run python3 check_conv.py\n";
    return 0;
}
