#define ACCUM_N 1024
__global__ void get_accelerations(float *a, float *pos, float * mass, int n)
{
    __shared__ float3 accumResult[ACCUM_N];

    //Loop over all the vectors
    for(int vec = blockIdx.x; vec < n; vec += gridDim.x){
        float3 c_pos = make_float3(pos[3*vec],pos[3*vec+1],pos[3*vec+2]);

        ////////////////////////////////////////////////////////////////////////
        // Each accumulator cycles through vectors with
        // stride equal to number of total number of accumulators ACCUM_N
        // At this stage ACCUM_N is only preferred be a multiple of warp size
        // to meet memory coalescing alignment constraints.
        ////////////////////////////////////////////////////////////////////////
        for(int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x){
            float3 r = make_float3(0,0,0);
            float r_3 = 0;
            float3 accel = make_float3(0,0,0);

            for(int pos_index = iAccum; pos_index  < n; pos_index += ACCUM_N){
                r.x = pos[ 3*pos_index ] - c_pos.x;
                r.y = pos[3*pos_index+1] - c_pos.y;
                r.z = pos[3*pos_index+2] - c_pos.z;
                r_3 = pow(pow(r.x,2) + pow(r.y,2) + pow(r.z,2),1.5f);
                if(r_3 > 0){
                    accel.x += r.x / r_3 * mass[pos_index];
                    accel.y += r.y / r_3 * mass[pos_index];
                    accel.z += r.z / r_3 * mass[pos_index];
                }

                accumResult[iAccum] = accel;
            }
        }

        ////////////////////////////////////////////////////////////////////////
        // Perform tree-like reduction of accumulators' results.
        // ACCUM_N has to be power of two at this stage
        ////////////////////////////////////////////////////////////////////////
        for(int stride = ACCUM_N / 2; stride > 0; stride >>= 1){
            __syncthreads();
            for(int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x){
                accumResult[iAccum].x += accumResult[stride + iAccum].x;
                accumResult[iAccum].y += accumResult[stride + iAccum].y;
                accumResult[iAccum].x += accumResult[stride + iAccum].z;
            }
        }

        if(threadIdx.x == 0){
            a[ 3*vec ] = accumResult[0].x;
            a[3*vec+1] = accumResult[0].y;
            a[3*vec+2] = accumResult[0].z;
        }
    }
}
