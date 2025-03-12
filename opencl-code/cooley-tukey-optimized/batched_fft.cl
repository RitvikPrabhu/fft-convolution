__kernel void bit_reverse_kernel(__global float2* input,
                                 __global float2* output,
                                 const int N)
{
    int gid = get_global_id(0);
    if (gid >= N) return;

    int logN = 0;
    {
        int tmp = N;
        while(tmp>1){
            tmp >>= 1;
            logN++;
        }
    }

    int x = gid;
    int reversed = 0;
    for(int i=0; i<logN; i++){
        reversed = (reversed<<1) | (x & 1);
        x >>= 1;
    }
    if(reversed < N){
        output[reversed] = input[gid];
    }
}

__kernel void two_stages_fft_kernel(__global float2* input,
                                    __global float2* output,
                                    const int N,
                                    const int startStage,   
                                    const int numStages,    
                                    const int inverse,
                                    __global float2* twiddles) 
{

    int gid = get_global_id(0);
    if(gid >= N/2) return; 

    for(int st = 0; st < numStages; st++){
        int s = startStage + st;
        if(s < 0) return; 

        int subFFTSize = 1 << (s+1);
        int halfSize   = subFFTSize >> 1;

        int blockId     = gid / halfSize;
        int blockOffset = gid % halfSize;

        int lowerIndex = blockId*subFFTSize + blockOffset;
        int upperIndex = lowerIndex + halfSize;

        float2 lowerVal = input[lowerIndex];
        float2 upperVal = input[upperIndex];

        int ratio = (N / subFFTSize); 
        int idx   = blockOffset * ratio;
        if(idx >= N/2) idx = idx % (N/2); 
        float2 w = twiddles[idx]; 
        float sign = (inverse != 0) ? 1.f : -1.f;
        float c = w.x;
        float ssin = sign * w.y; 

        float2 t;
        t.x = c*upperVal.x - ssin*upperVal.y;
        t.y = ssin*upperVal.x + c*upperVal.y;

        float2 outLower, outUpper;
        outLower.x = lowerVal.x + t.x;
        outLower.y = lowerVal.y + t.y;
        outUpper.x = lowerVal.x - t.x;
        outUpper.y = lowerVal.y - t.y;

        input[lowerIndex] = outLower;
        input[upperIndex] = outUpper;
    }

    {
        int finalS = startStage + numStages - 1;
        int finalSize = 1 << (finalS+1);
        int finalHalf = finalSize >> 1;

        int blockId2 = gid / finalHalf;
        int off2     = gid % finalHalf;
        int lowerIdx2= blockId2*finalSize + off2;
        int upperIdx2= lowerIdx2 + finalHalf;

        float2 L = input[lowerIdx2];
        float2 U = input[upperIdx2];

        output[lowerIdx2] = L;
        output[upperIdx2] = U;
    }
}


__kernel void final_scale_kernel(__global float2* data,
                                 const int N)
{
    int i = get_global_id(0);
    if(i < N){
        data[i].x /= (float)N;
        data[i].y /= (float)N;
    }
}
