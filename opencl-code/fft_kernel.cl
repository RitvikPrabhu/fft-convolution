#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void bit_reverse(__global double2 *data, __global int *bit_rev, int n) {
    int i = get_global_id(0);             
    if(i < n) {
        int j = bit_rev[i];             
        if(i < j) {                       
            double2 temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
}

__kernel void fft_stage(__global double2 *data, int m, int inverse) {
    int gid = get_global_id(0);           
    int half_m = m / 2;                 
    
    int group = gid / half_m;
    int k = gid % half_m;
    
    int index1 = group * m + k;
    int index2 = index1 + half_m;
    
    double angle = 2.0 * 3.141592653589793 / m * (inverse ? 1.0 : -1.0);
    double cos_val = cos(angle * k);
    double sin_val = sin(angle * k);
    double2 twiddle = (double2)(cos_val, sin_val);
    
    double2 a = data[index1];
    double2 b = data[index2];
    
    double2 t;
    t.x = b.x * twiddle.x - b.y * twiddle.y;
    t.y = b.x * twiddle.y + b.y * twiddle.x;
    
    data[index1] = (double2)(a.x + t.x, a.y + t.y);
    data[index2] = (double2)(a.x - t.x, a.y - t.y);
}

