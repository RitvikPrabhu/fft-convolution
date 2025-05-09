/*  upf_ups.cpp  – minimal‑correct UPS on UPF reference

    compile:  g++ -std=c++17 upf_ups.cpp -lOpenCL -o upf_ups
*/
#define  CL_TARGET_OPENCL_VERSION 120
#define  _POSIX_C_SOURCE 199309L
#include <CL/cl.h>

#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <complex>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// problem parameters  (unchanged B & L, FIR extended so that P = 4)
// ---------------------------------------------------------------------------
constexpr unsigned N_SAMPLES = 8192;      // input length
constexpr unsigned B         = 1024;      // block length   (signal partition)
constexpr unsigned L         = 256;       // sub‑filter len (filter partition)
constexpr unsigned FIR_LEN   = 1024;      // full FIR taps   →  P = 4
constexpr unsigned K         = 2*B;       // FFT size (power‑of‑two ≥ B+L‑1)

static_assert(FIR_LEN % L   == 0);
static_assert((K & (K-1))   == 0);        // power of two

constexpr unsigned P        = FIR_LEN / L;          // # filter partitions
constexpr unsigned Kc       = K/2 + 1;               // # unique complex bins
constexpr unsigned BLKS_IN  = (N_SAMPLES + B - 1)/B; // = 8
constexpr unsigned OUT_LEN  = N_SAMPLES + FIR_LEN - 1;

// ---------------------------------------------------------------------------
//  tiny header‑only radix‑2 FFT  (recursive, O(N log N), single‑precision)
// ---------------------------------------------------------------------------
using cpx = std::complex<float>;

void cpuFFT(std::vector<cpx>& a, bool inverse = false)
{
    const size_t n = a.size();
    if (n == 1) return;

    std::vector<cpx> a0(n/2), a1(n/2);
    for (size_t i = 0; i < n/2; ++i) {
        a0[i] = a[2*i];
        a1[i] = a[2*i+1];
    }
    cpuFFT(a0, inverse);
    cpuFFT(a1, inverse);

    const float ang = 2.0f*M_PI/n * (inverse ? -1.0f : 1.0f);
    cpx w(1.0f,0.0f), wn(std::cos(ang), std::sin(ang));
    for (size_t k = 0; k < n/2; ++k) {
        cpx t = w * a1[k];
        a[k]        = a0[k] + t;
        a[k + n/2]  = a0[k] - t;
        w *= wn;
    }
    if (inverse)          // normalise
        for (auto& v : a) v /= 2.0f;
}
// helper that takes/returns separate real array -----------------------------
void fftRealToSpec(const float* xin, std::vector<cpx>& X)
{
    X.assign(K, cpx(0,0));
    for (size_t i=0;i<K && i<B;++i) X[i] = xin[i];     // copy & zero‑pad
    cpuFFT(X,false);
}
void ifftSpecToReal(std::vector<cpx>& X, float* xout)
{
    cpuFFT(X,true);                                   // inverse
    for (size_t i=0;i<B;++i) xout[i] = X[i+B].real(); // keep right half
}

// ---------------------------------------------------------------------------
//  helper I/O
// ---------------------------------------------------------------------------
static void dump(const char* fn,const std::vector<float>& v)
{
    std::ofstream(fn,std::ios::binary)
        .write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(float));
}

// ---------------------------------------------------------------------------
//  OpenCL kernel:  sum_{p=0}^{P-1}   FDL[(head+p)%P][k] * H[p][k]
//                  one work‑item per k   (0 … Kc-1)
// ---------------------------------------------------------------------------
static const char* kSrc = R"CLC(
typedef struct{float x; float y;} cpx;

inline cpx  mul(const cpx a, const cpx b)
{
    return (cpx)(a.x*b.x - a.y*b.y,  a.x*b.y + a.y*b.x);
}
__kernel
void accumulate(__global const cpx*  FDL,     // P * Kc
                __global const cpx*  H,       // P * Kc
                __global       cpx*  Y,       // Kc
                const uint P,
                const uint Kc,
                const uint head)
{
    uint k = get_global_id(0);
    cpx sum = (cpx)(0.0f,0.0f);
    uint idx = head;
    for(uint p=0; p<P; ++p){
        cpx a = FDL[idx*Kc + k];
        cpx b = H  [p  *Kc + k];
        sum += mul(a,b);
        idx = (idx + 1) % P;       // next (older) spectrum
    }
    Y[k] = sum;
}
)CLC";

// ---------------------------------------------------------------------------

int main()
{
    // -----------------------------------------------------------------------
    // 1.  create signal and FIR
    // -----------------------------------------------------------------------
    std::vector<float> x(N_SAMPLES);
    for(unsigned n=0;n<N_SAMPLES;++n)
        x[n] = std::sin(2.0f*M_PI*100.0f*n/float(N_SAMPLES));

    std::vector<float> h(FIR_LEN);
    {   // Hann‑windowed sinc LPF  (fc = 500 Hz @ 8 kHz  → 0.0625 f_Nyq)
        const float fc = 500.0f/8000.0f;                // norm. cut‑off
        const float M  = FIR_LEN - 1.0f;
        for(unsigned n=0;n<FIR_LEN;++n){
            float w  = 0.5f*(1.0f-std::cos(2.0f*M_PI*n/M));
            float t  = n - M/2.0f;
            float si = (t==0.0f) ? 1.0f
                                 : std::sin(2.0f*M_PI*fc*t)/(M_PI*t);
            h[n] = w*si;
        }
    }
    dump("input.bin", x);  dump("fir.bin", h);

    // -----------------------------------------------------------------------
    // 2.  pre‑compute sub‑filter spectra   H[p][k]   (host)
    // -----------------------------------------------------------------------
    std::vector<cpx> Hspec(P*Kc);
    for (unsigned p=0; p<P; ++p) {
        std::vector<cpx> tmp(K, cpx(0,0));
        std::copy(h.begin()+p*L, h.begin()+p*L+L, tmp.begin()); // zero‑pad
        cpuFFT(tmp,false);
        for (unsigned k=0;k<Kc;++k)
            Hspec[p*Kc + k] = tmp[k];
    }

    // -----------------------------------------------------------------------
    // 3.  OpenCL set‑up
    // -----------------------------------------------------------------------
    cl_int err; cl_platform_id plat; cl_device_id dev;
    clGetPlatformIDs(1,&plat,nullptr);
    clGetDeviceIDs  (plat,CL_DEVICE_TYPE_DEFAULT,1,&dev,nullptr);

    cl_context ctx  = clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err);
    cl_command_queue q =
        clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&err);

    cl_program pr = clCreateProgramWithSource(ctx,1,&kSrc,nullptr,&err);
    clBuildProgram(pr,1,&dev,"",nullptr,nullptr);
    cl_kernel  kn = clCreateKernel (pr,"accumulate",&err);

    // device buffers --------------------------------------------------------
    const size_t bytesSpec = Kc * sizeof(cpx);
    cl_mem dFDL  = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                  P*bytesSpec, nullptr, &err);
    cl_mem dH    = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                  P*bytesSpec, nullptr, &err);
    cl_mem dY    = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                  bytesSpec,  nullptr, &err);
    // upload constant filter spectra
    clEnqueueWriteBuffer(q,dH,CL_TRUE,0,P*bytesSpec,Hspec.data(),0,nullptr,nullptr);

    // set static kernel args
    clSetKernelArg(kn,0,sizeof(cl_mem), &dFDL);
    clSetKernelArg(kn,1,sizeof(cl_mem), &dH);
    clSetKernelArg(kn,2,sizeof(cl_mem), &dY);
    clSetKernelArg(kn,3,sizeof(cl_uint),&P);
    clSetKernelArg(kn,4,sizeof(cl_uint),&Kc);

    // -----------------------------------------------------------------------
    // 4.  streaming convolution
    // -----------------------------------------------------------------------
    std::vector<float> y(OUT_LEN, 0.0f);
    std::vector<cpx>   Xspec(K);      // host: current block spectrum
    std::vector<cpx>   Yspec(K);      // host: accumulated spectrum

    unsigned head = 0;                // FDL newest slot
    double kernel_ns = 0.0;

    const size_t gsz = Kc;            // one WI per spectral bin
    for (unsigned blk = 0; blk < BLKS_IN; ++blk)
    {
        // ---- 4.1  FFT of current block -----------------------------------
        float inBlock[B] = {0};
        const unsigned blkSamples = std::min<unsigned>(B,
                                    N_SAMPLES - blk*B);
        std::memcpy(inBlock, x.data()+blk*B, blkSamples*sizeof(float));
        fftRealToSpec(inBlock, Xspec);          // → Xspec[0 … K-1]

        // ---- 4.2  update FDL (newest spectra at 'head') -------------------
        clEnqueueWriteBuffer(q, dFDL, CL_TRUE,
                             head*bytesSpec, bytesSpec,
                             Xspec.data(), 0,nullptr,nullptr);

        // ---- 4.3  launch accumulate kernel -------------------------------
        clSetKernelArg(kn,5,sizeof(cl_uint), &head);

        cl_event ev;
        clEnqueueNDRangeKernel(q, kn, 1, nullptr, &gsz, nullptr,
                               0,nullptr,&ev);
        clWaitForEvents(1,&ev);
        cl_ulong ts,te;
        clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,sizeof(ts),&ts,nullptr);
        clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END  ,sizeof(te),&te,nullptr);
        kernel_ns += double(te-ts);
        clReleaseEvent(ev);

        // ---- 4.4  read spectrum → IFFT → save right half ------------------
        clEnqueueReadBuffer(q,dY,CL_TRUE,0,bytesSpec,
                            Yspec.data(),0,nullptr,nullptr);
        // rebuild the conjugate half (Hermitian)
        for(unsigned k=1;k<Kc-1;++k)
            Yspec[K-k] = std::conj(Yspec[k]);
        ifftSpecToReal(Yspec, inBlock);         // reuse inBlock as temp
        std::memcpy(y.data()+blk*B, inBlock, B*sizeof(float));

        // ---- 4.5 advance head  (circular) --------------------------------
        head = (head == 0) ? (P-1) : (head-1);
    }

    dump("gpu.bin", y);
    std::cout << "GPU complex‑MAC time: "
              << kernel_ns*1e-6 << " ms  ("
              << kernel_ns*1e-3/BLKS_IN << " µs/block)\n";

    // -----------------------------------------------------------------------
    // 5.  tidy up
    // -----------------------------------------------------------------------
    clReleaseMemObject(dFDL);  clReleaseMemObject(dH);  clReleaseMemObject(dY);
    clReleaseKernel(kn);       clReleaseProgram(pr);
    clReleaseCommandQueue(q);  clReleaseContext(ctx);

    std::cout << "done – binaries written.  Run python3 check_conv.py\n";
    return 0;
}
