__kernel void bit_reverse_kernel(__global float2* input,
                                 __global float2* output,
                                 const int N)
{
    int gid = get_global_id(0);
    if (gid >= N) return;

    int logN = 0;
    int tmpN = N;
    while (tmpN > 1) {
        tmpN >>= 1;
        logN++;
    }

    int x = gid;
    int reversed = 0;
    for(int i = 0; i < logN; i++) {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
    }

    output[reversed] = input[gid];
}


__kernel void fft_stage_kernel(__global float2* input,
                               __global float2* output,
                               const int N,
                               const int stage,
                               const int inverse)
{
    int gid = get_global_id(0);
    if (gid >= N/2) {
        return;
    }

    int subFFTSize = (1 << (stage + 1));
    int halfSize   = (subFFTSize >> 1);

    int blockId = gid / halfSize;
    int blockOffset = gid % halfSize;

    int lowerIndex = blockId*subFFTSize + blockOffset;
    int upperIndex = lowerIndex + halfSize;

    float2 lowerVal = input[lowerIndex];
    float2 upperVal = input[upperIndex];

    float sign = (inverse != 0) ? 1.0f : -1.0f;
    float angle = sign * 6.283185307f * blockOffset / (float)subFFTSize;
    float c = cos(angle);
    float s = sin(angle);

    // Multiply upperVal by twiddle
    float2 t;
    t.x = c*upperVal.x - s*upperVal.y;
    t.y = s*upperVal.x + c*upperVal.y;

    // Butterfly
    float2 outLower, outUpper;
    outLower.x = lowerVal.x + t.x;
    outLower.y = lowerVal.y + t.y;
    outUpper.x = lowerVal.x - t.x;
    outUpper.y = lowerVal.y - t.y;

    // Write to output
    output[lowerIndex] = outLower;
    output[upperIndex] = outUpper;
}


__kernel void final_scale_kernel(__global float2* data,
                                 const int N)
{
    int gid = get_global_id(0);
    if (gid >= N) return;

    data[gid].x /= (float)N;
    data[gid].y /= (float)N;
}

