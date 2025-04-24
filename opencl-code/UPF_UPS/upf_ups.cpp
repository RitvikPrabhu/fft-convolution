   #define  CL_TARGET_OPENCL_VERSION 120
   #define  _POSIX_C_SOURCE 199309L
   #include <CL/cl.h>
   
   #include <cmath>
   #ifndef M_PI
   #  define M_PI 3.14159265358979323846
   #endif
   #include <chrono>
   #include <cstring>
   #include <fstream>
   #include <iostream>
   #include <vector>
   
   #define CHECK(E) if((E)!=CL_SUCCESS){                                       \
       std::cerr<<"OpenCL error "<<(E)<<" @ line "<<__LINE__<<std::endl;       \
       std::exit(1);                                                           \
   }
   
   constexpr unsigned N_SAMPLES = 8192;   
   constexpr unsigned FIR_LEN   = 256;    
   constexpr unsigned B         = 1024;   
   constexpr unsigned L         = 256;    

   static_assert(N_SAMPLES % B   == 0);
   static_assert(FIR_LEN   % L   == 0);
   
   constexpr unsigned BLKS_IN  = N_SAMPLES / B;   // 8
   constexpr unsigned OUT_LEN  = N_SAMPLES + FIR_LEN - 1;
   
   static inline double now_ns()
   {
       return std::chrono::duration<double,std::nano>(
                  std::chrono::steady_clock::now().time_since_epoch()).count();
   }
   
   static void dump(const char* fn,const std::vector<float>& v)
   {
       std::ofstream(fn,std::ios::binary)
           .write(reinterpret_cast<const char*>(v.data()),
                  v.size()*sizeof(float));
   }
   
   static void hannLPF(std::vector<float>& h,float fc_norm)
   {
       const float M = float(h.size()-1);
       for(unsigned n=0;n<h.size();++n)
       {
           const float w  = 0.5f*(1.0f-std::cos(2.0f*M_PI*n/M));
           const float t  = float(n) - M/2.0f;
           const float si = (t==0.0f)
                          ? 1.0f
                          : sin(2.0f*M_PI*fc_norm*t)/(M_PI*t*fc_norm*2.0f);
           h[n] = w*si;
       }
   }
   
   static void refConv(const std::vector<double>& x,
                       const std::vector<double>& h,
                       std::vector<double>&       y)
   {
       const std::size_t Nx=x.size(), Nh=h.size(), Ny=Nx+Nh-1;
       y.assign(Ny,0.0);
       for(std::size_t n=0;n<Ny;++n)
       {
           const std::size_t kmin = (n>=Nh-1)? n-(Nh-1) : 0;
           const std::size_t kmax = (n< Nx-1)? n         : Nx-1;
           double acc=0.0;
           for(std::size_t k=kmin;k<=kmax;++k) acc += x[k]*h[n-k];
           y[n]=acc;
       }
   }
   
   static const char* kSrc =
   "__kernel void upfUps(__global const float* ring,"  /* B+L-1             */"\n"
   "                     __global const float* Hrev," /* L (reversed taps) */"\n"
   "                     __global       float* out, " /* B                 */"\n"
   "                     const uint   Ltap)                                            \n"
   "{                                                                                 \n"
   "    const uint gid = get_global_id(0);                                            \n"
   "    float acc = 0.0f;                                                             \n"
   "    #pragma unroll                                                                \n"
   "    for(uint k=0;k<Ltap;++k)                                                      \n"
   "        acc += ring[gid + k] * Hrev[k];                                           \n"
   "    out[gid] = acc;                                                               \n"
   "}";
   
   int main()
   {
       std::vector<float> x(N_SAMPLES);
       for(unsigned n=0;n<N_SAMPLES;++n)
           x[n]=sinf(2.0f*M_PI*100.0f*n/N_SAMPLES);   // 100 Hz tone
       dump("input.bin",x);
   
       std::vector<float> h(FIR_LEN);  hannLPF(h,500.0f/8192.0f);
       dump("fir.bin",h);
   
       std::vector<float> hrev(L);
       for(unsigned k=0;k<L;++k) hrev[k]=h[L-1-k];
   
       std::vector<double> xd(x.begin(),x.end()), hd(h.begin(),h.end()), yd;
       double t0=now_ns();  refConv(xd,hd,yd);  double t1=now_ns();
       std::cout<<"CPU reference  : "<<(t1-t0)*1e-6<<" ms\n";
       std::vector<float> yCPU(yd.begin(),yd.end()); dump("cpu.bin",yCPU);
   
       cl_int err; cl_platform_id plat; cl_device_id dev;
       CHECK(clGetPlatformIDs(1,&plat,nullptr));
       CHECK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_DEFAULT,1,&dev,nullptr));
   
       cl_context ctx=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err); CHECK(err);
       cl_command_queue q=clCreateCommandQueue(ctx,dev,
                       CL_QUEUE_PROFILING_ENABLE,&err); CHECK(err);
       cl_program pr=clCreateProgramWithSource(ctx,1,&kSrc,nullptr,&err);   CHECK(err);
       CHECK(clBuildProgram(pr,1,&dev,"",nullptr,nullptr));
       cl_kernel  kn=clCreateKernel(pr,"upfUps",&err);                       CHECK(err);
   
       const size_t ringBytes = (B + L - 1)*sizeof(float);
       cl_mem dRing=clCreateBuffer(ctx,CL_MEM_READ_ONLY ,ringBytes,nullptr,&err);CHECK(err);
       cl_mem dH   =clCreateBuffer(ctx,CL_MEM_READ_ONLY ,L*sizeof(float),nullptr,&err);CHECK(err);
       cl_mem dY   =clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,B*sizeof(float),nullptr,&err);CHECK(err);
   
       CHECK(clEnqueueWriteBuffer(q,dH,CL_TRUE,0,L*sizeof(float),
                                  hrev.data(),0,nullptr,nullptr));
       CHECK(clSetKernelArg(kn,1,sizeof(cl_mem),&dH));
       cl_uint Ltap=L;   CHECK(clSetKernelArg(kn,3,sizeof(Ltap),&Ltap));
   
       std::vector<float> ring(B+L-1,0.0f);
       std::vector<float> yGPU(OUT_LEN,0.0f);
   
       const size_t gsz = B;          // one work-item per output sample
       double kernel_ns = 0.0;
   
       for(unsigned blk=0; blk<=BLKS_IN; ++blk)           
       {
           std::memmove(ring.data(),
                        ring.data()+B,
                        (L-1)*sizeof(float));
   
           if(blk < BLKS_IN)
               std::memcpy(ring.data()+L-1,
                           x.data()+blk*B,
                           B*sizeof(float));
           else
               std::memset(ring.data()+L-1,0,B*sizeof(float));
   
           CHECK(clEnqueueWriteBuffer(q,dRing,CL_TRUE,0,ringBytes,
                                      ring.data(),0,nullptr,nullptr));
           CHECK(clSetKernelArg(kn,0,sizeof(cl_mem),&dRing));
           CHECK(clSetKernelArg(kn,2,sizeof(cl_mem),&dY));
   
           cl_event ev;
           CHECK(clEnqueueNDRangeKernel(q,kn,1,nullptr,&gsz,nullptr,0,nullptr,&ev));
           CHECK(clWaitForEvents(1,&ev));
   
           cl_ulong ts,te;
           clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,sizeof(ts),&ts,nullptr);
           clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END  ,sizeof(te),&te,nullptr);
           kernel_ns += double(te - ts);
           clReleaseEvent(ev);
   
           const unsigned copyLen = (blk<BLKS_IN) ? B : (L-1);
           CHECK(clEnqueueReadBuffer(q,dY,CL_TRUE,0,copyLen*sizeof(float),
                  yGPU.data()+blk*B,0,nullptr,nullptr));
       }
   
       std::cout<<"GPU kernel time: "<<kernel_ns*1e-6
                <<" ms  ("<<kernel_ns*1e-3/BLKS_IN<<" µs/block)\n";
   
       dump("gpu.bin",yGPU);
   
       clReleaseMemObject(dRing); clReleaseMemObject(dH); clReleaseMemObject(dY);
       clReleaseKernel(kn); clReleaseProgram(pr);
       clReleaseCommandQueue(q); clReleaseContext(ctx);
   
       std::cout<<"done – binaries written.  Run python3 check_conv.py\n";
       return 0;
   }
   