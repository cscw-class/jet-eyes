#define NDISP 48

typedef unsigned char uchar;
using namespace std;

__global__ void path_aggregate(int width, int height, int D, int P1, int P2,
                              int dir, int* Espace, cudaTextureObject_t Cost,
                              short* texturePerp, short* textureParallel)
{
    __shared__ short L_p0[NDISP+2];
    __shared__ short L_c0[NDISP+2];
    __shared__ short delta0;

    short Tp = 0, T = 0, Tp2 = 0;
    float Kt = 0;
    int P1a, P2a; int Td;
    int dy, dx;

    int tid = threadIdx.x;
    int d = tid+1;

    //just one thread to prevent bank conflicts
    if(tid == 0)
        L_p0[0] = L_p0[NDISP+1] = L_c0[0] = L_c0[NDISP+1] = SHRT_MAX;

    if(tid < D)
        L_p0[d] = SHRT_MAX;

    if(tid == 0)
        delta0 = SHRT_MAX;

    int x = 0;
    int y = 0;

    //set initial coordinates of a block
    switch(dir)
    {
        case 0:
            x = 0; y = blockIdx.x; dy = 0; dx = 1;
            break;
        case 1:
            if(blockIdx.x < height)
            {    y = height-1-(blockIdx.x); x = 0; dy=1; dx=1;}
            else
            {    y = 0; x = (blockIdx.x)-height;}
            break;
        case 2:
            y = 0; x = blockIdx.x; dy = 1; dx = 0;
            break;
        case 3:
            if(blockIdx.x < width)
            {    x = blockIdx.x; y = 0; dy = 1; dx = -1;}
            else
            {    y = blockIdx.x-width; x = width-1; dy = 1; dx = -1;}
            break;
        case 4:
            x = width-1; y = blockIdx.x; dy = 0; dx= -1;
            break;
        case 5:
            if(blockIdx.x < height)
            {    x = width-1; y = blockIdx.x; dy=-1; dx = -1;}
            else
            {    y = height-1; x = blockIdx.x-height; dy=-1; dx = -1;}
            break;
        case 6:
            y = height-1; x = blockIdx.x; dy = -1; dx = 0;
            break;
        case 7:
            if(blockIdx.x < height)
            {    x = 0; y = blockIdx.x; dy = -1; dx = 1;}
            else
            {    x = blockIdx.x-height; y = height-1; dy = -1; dx = 1;}
            break;
    }
    __syncthreads();
    //main loop of a thread
    while(x >= 0 && y >= 0 && x < width && y < height)
    {
        //read texture and adjust penalties
        Kt = 1;//textureParallel[y*width+x]/500;
        T = texturePerp[y*width+x];
        Td = abs(T-Tp2);
        short C;
        short bestsad; short bestdisp;
        P1a = P1;
        P2a = P2;

        Tp2 = Tp;
        Tp = T;

        if(tid < D)
        {
            C = tex2D<short>(Cost, (d-1)*width+x, y);
            if(C < SHRT_MAX-5)
            {
                L_c0[d] = C + min((int)L_p0[d], min((int)L_p0[d-1]+P1a, min((int)L_p0[d+1]+P1a, (int)delta0))) - delta0;
                Espace[y*width*D + x*D + d-1] += (int)L_c0[d]*Kt;
                L_p0[d] = L_c0[d];
            }
            else
            {
                L_c0[d] = SHRT_MAX;
                Espace[y*width*D + x*D + d-1] += SHRT_MAX;
                L_p0[d] = L_c0[d];
            }

            __syncthreads();

            for(int i = blockDim.x, n = blockDim.x/2; n>0; n /= 2)
            {
                if((d-1) < n)
                {
                    L_c0[d] = min((int)L_c0[d], (int)L_c0[d+n]);
                }

                if(n*2 != i)
                {
                    L_c0[n+1] = L_c0[2*n+1];
                    n = n+1;
                }
                i = n;

                __syncthreads();
            }

            delta0 = L_c0[1] + P2a;
        }

        __syncthreads();
        x += dx; y += dy;
    }
}

/* BM_cost
 *
 * Calculates block matching cost for D disparity levels
 * between image1 and image2
 */
__global__ void BM_cost(cudaTextureObject_t image1, cudaTextureObject_t image2,
                        short* cost, int width, int height, int D)
{
    extern __shared__ short I[];

    int x = threadIdx.x;
    int y = blockIdx.x;

    I[x] = tex2D<uchar>(image2, x, y);

    __syncthreads();

    short i = tex2D<uchar>(image1, x, y);
    int r = y*width*D + x;

    for(int d=0; d<D; d++)
    {
        if(x >= d)
        {
            cost[r + d*width] = abs(i - I[x-d]);
        }
        else
            cost[r + d*width] = SHRT_MAX;
    }
}

__global__ void energy_minimalize(int *Energy, short* disparity, short* disparity2,
                                  int D, int width, int height)
{
    __shared__ int L[NDISP];    //storage for disparity values
    __shared__ int cL[NDISP];

    int d = threadIdx.x;
    int x = blockIdx.x;
    int y = blockIdx.y;
    float d_refined;

    //read energy for each disparity (one operation)
    L[d] = Energy[y*width*D + x*D + d];
    cL[d] = L[d];

    __syncthreads();
    //use reduction algorithm to minimalize energy

    for(int i = blockDim.x, n = blockDim.x/2; n>0; n /= 2)
    {
        if(d < n)
        {
            L[d] = min(L[d], L[d+n]);
        }

        if(n*2 != i)
        {
            L[n] = L[2*n];
            n = n+1;
        }
        i = n;

        __syncthreads();
    }

    if(cL[d] == L[0])
    {
        if(0 < d && d < D-1)
        {
            int denom2 = max((Energy[(y * width * D) + (x * D) + d - 1]
                              + Energy[(y * width * D) + (x * D) + d + 1]
                              - 2 * Energy[(y * width * D) + (x * D) + d]), 1);
            d_refined = 10 * ((float) d) + (float)((Energy[(y * width * D)
                        + (x * D) + d - 1] - Energy[(y * width * D)
                        + (x * D) + d + 1]) + denom2) / (denom2 * 2);
        }
        else
        {
            d_refined = 10*(float)d;
        }

        disparity[y*width+x] = d_refined;
        if(x-1-d >= 0)
            disparity2[y*width+x-1-d] = d_refined;

    }
    __syncthreads();
}

__global__ void validate(short* disparity, short* disparity2, int width, int height, int disp12maxdiff)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    int d0 = disparity[y*width + x];
    if(d0 != -1)
    {
        int x2 = x - d0;

        if(0 <= x2 && abs(disparity2[y*width + x2]-d0) > 10*disp12maxdiff)
        {
            disparity[y*width+x] = -1;
        }
    }
}
