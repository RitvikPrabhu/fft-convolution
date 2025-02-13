
__kernel void batched_radix2_fft(
    __global float2* input,   
    __global float2* output,  
    const int N,             
    const int batchCount,     
    const int inverse         
)
{
    int groupId = get_group_id(0);
    int localId = get_local_id(0);

    int base = groupId * N;

    if (groupId >= batchCount) {
        return;
    }

    __local float2 buffer[8];

    float2 val = input[base + localId];
    buffer[localId] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    int logN = 0;
    {
        int temp = N;
        while(temp > 1) {
            temp >>= 1;
            logN++;
        }
    }

    int reversed = 0;
    int x = localId;
    for(int i = 0; i < logN; i++) {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
    }

    if(localId < reversed) {
        float2 tmp = buffer[localId];
        buffer[localId] = buffer[reversed];
        buffer[reversed] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int size = 2; size <= N; size <<= 1) {
        int halfSize = size >> 1;

        float angleSign = (inverse != 0) ? 1.0f : -1.0f;
        float theta = angleSign * (6.283185307f / (float)size);

        int blockId = localId / size;
        int blockOffset = localId % size;

        if(blockOffset < halfSize) {
            int lowerIndex = blockId*size + blockOffset;
            int upperIndex = lowerIndex + halfSize;

            float2 lowerVal = buffer[lowerIndex];
            float2 upperVal = buffer[upperIndex];

            float k = (float)blockOffset;
            float c = cos(theta * k);
            float s = sin(theta * k);

            float2 t;
            t.x = c * upperVal.x - s * upperVal.y;
            t.y = s * upperVal.x + c * upperVal.y;

            float2 outLower, outUpper;
            outLower.x = lowerVal.x + t.x;
            outLower.y = lowerVal.y + t.y;

            outUpper.x = lowerVal.x - t.x;
            outUpper.y = lowerVal.y - t.y;

            buffer[lowerIndex] = outLower;
            buffer[upperIndex] = outUpper;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(inverse != 0) {
        buffer[localId].x /= (float)N;
        buffer[localId].y /= (float)N;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    output[base + localId] = buffer[localId];
}

