#include <stdio.h>
#include <cuda_runtime.h>
#include <core/core.hpp>
#include <gpu/gpu.hpp>

using namespace std;
using namespace cv;
typedef unsigned char uchar;

//laws masks column by column - stored in local memory
__constant__ int L3E3[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
__constant__ int E3L3[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int L3S3[9] = {-1, 2, -1, -2, 4, -2, -1, 2, -1};
__constant__ int E3E3[9] = {1, 0, -1, 0, 0, 0, -1, 0, 1};
__constant__ int E3S3[9] = {1, -2, 1, 0, 0, 0, -1, 2, -1};
__constant__ int S3S3[9] = {1, -2, 1, -2, 4, -2, 1, -2, 1};
__constant__ int S3L3[9] = {-1, -2, -1, 2, 4, 2, -1, -2, -1};
__constant__ int S3E3[9] = {1, 0, -1, -2, 0, 2, 1, 0, -1};

__constant__ short LAWS[6][9] = {
        {-1, -2, -1, 0, 0, 0, 1, 2, 1},    //L3E3
        {-1, 0, 1, -2, 0, 2, -1, 0, 1}, //E3L3
        {1, 0, -1, 0, 0, 0, -1, 0, 1},    //E3E3
        {-1, -2, -1, 2, 4, 2, -1, -2, -1}, //S3L3
        {-1, 2, -1, -2, 4, -2, -1, 2, -1}, //L3S3
        {1, -2, 1, -2, 4, -2, 1, -2, 1}, //S3S3
};

__global__ void laws_convolution(cudaTextureObject_t img, int width, int height, short* convoluted, int Tn)
{
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int x = blockIdx.x*blockDim.x+threadIdx.x;

    //read intensity tile into shared mem
    __shared__ uchar tile[34*4];
    tile[(threadIdx.y+1)*34 + threadIdx.x+1] = tex2D<uchar>(img, x, y);

    switch(threadIdx.y)
    {
        case 0:
            tile[threadIdx.y*34 + threadIdx.x+1] = tex2D<uchar>(img, x, y-1); break;
        case 3:
            tile[(threadIdx.y+2)*34 + threadIdx.x+1] = tex2D<uchar>(img, x, y+1); break;
    }
    switch(threadIdx.x)
    {
        case 0:
            tile[(threadIdx.y+1)*34 + threadIdx.x] = tex2D<uchar>(img, x-1, y); break;
        case 31:
            tile[(threadIdx.y+1)*34 + threadIdx.x+2] = tex2D<uchar>(img, x+1, y); break;
    }
    if(threadIdx.y == 0 && threadIdx.x == 0)
    {
        tile[(threadIdx.y)*34 + threadIdx.x] = tex2D<uchar>(img, x-1, y-1);
    }
    else if(threadIdx.y == 0 && threadIdx.x+1 == 32)
    {
        tile[(threadIdx.y)*34 + threadIdx.x+2] = tex2D<uchar>(img, x+1, y-1);
    }
    else if(threadIdx.y + 1 == 4 && threadIdx.x == 0)
    {
        tile[(threadIdx.y+2)*34 + threadIdx.x] = tex2D<uchar>(img, x-1, y+1);
    }
    else if(threadIdx.y + 1 == 4 && threadIdx.x+1 == 32)
    {
        tile[(threadIdx.y+2)*34 + threadIdx.x+2] = tex2D<uchar>(img, x+1, y+1);
    }

    __syncthreads();

    short tenergy = 0;
    //check if we are still in image
    if(x < width && y < height)
    {
        if(Tn == -1)
        {
            tenergy += tile[(threadIdx.y)*34 + threadIdx.x]*(LAWS[0][0]+LAWS[1][0]+(LAWS[3][0]+LAWS[4][0])+LAWS[5][0]);// + tile[(threadIdx.y)*34 + threadIdx.x]*E3L3[0];
            tenergy += tile[(threadIdx.y+1)*34 + threadIdx.x]*(LAWS[0][1]+LAWS[1][1]+(LAWS[3][1]+LAWS[4][1])+LAWS[5][1]);// + tile[(threadIdx.y+1)*34 + threadIdx.x]*E3L3[1];
            tenergy += tile[(threadIdx.y+2)*34 + threadIdx.x]*(LAWS[0][2]+LAWS[1][2]+(LAWS[3][2]+LAWS[4][2])+LAWS[5][2]);// + tile[(threadIdx.y+2)*34 + threadIdx.x]*E3L3[2];
            tenergy += tile[(threadIdx.y)*34 + threadIdx.x+1]*(LAWS[0][3]+LAWS[1][3]+(LAWS[3][3]+LAWS[4][3])+LAWS[5][3]);// + tile[(threadIdx.y)*34 + threadIdx.x+1]*E3L3[3];
            tenergy += tile[(threadIdx.y+1)*34 + threadIdx.x+1]*(LAWS[0][4]+LAWS[1][4]+(LAWS[3][4]+LAWS[4][4])+LAWS[5][4]);// + tile[(threadIdx.y+1)*34 + threadIdx.x+1]*E3L3[4];
            tenergy += tile[(threadIdx.y+2)*34 + threadIdx.x+1]*(LAWS[0][5]+LAWS[1][5]+(LAWS[3][5]+LAWS[4][5])+LAWS[5][5]);// + tile[(threadIdx.y+2)*34 + threadIdx.x+1]*E3L3[5];
            tenergy += tile[(threadIdx.y)*34 + threadIdx.x+2]*(LAWS[0][6]+LAWS[1][6]+(LAWS[3][6]+LAWS[4][6])+LAWS[5][6]);// + tile[(threadIdx.y)*34 + threadIdx.x+2]*E3L3[6];
            tenergy += tile[(threadIdx.y+1)*34 + threadIdx.x+2]*(LAWS[0][7]+LAWS[1][7]+(LAWS[3][7]+LAWS[4][7])+LAWS[5][7]);// + tile[(threadIdx.y+1)*34 + threadIdx.x+2]*E3L3[7];
            tenergy += tile[(threadIdx.y+2)*34 + threadIdx.x+2]*(LAWS[0][8]+LAWS[1][8]+(LAWS[3][8]+LAWS[4][8])+LAWS[5][8]);// + tile[(threadIdx.y+2)*34 + threadIdx.x+2]*E3L3[8];
            convoluted[y*width + x] = abs((int)tenergy);
        }
        else
        {
            tenergy += tile[(threadIdx.y)*34 + threadIdx.x]*LAWS[Tn][0];// + tile[(threadIdx.y)*34 + threadIdx.x]*E3L3[0];
            tenergy += tile[(threadIdx.y+1)*34 + threadIdx.x]*LAWS[Tn][1];// + tile[(threadIdx.y+1)*34 + threadIdx.x]*E3L3[1];
            tenergy += tile[(threadIdx.y+2)*34 + threadIdx.x]*LAWS[Tn][2];// + tile[(threadIdx.y+2)*34 + threadIdx.x]*E3L3[2];
            tenergy += tile[(threadIdx.y)*34 + threadIdx.x+1]*LAWS[Tn][3];// + tile[(threadIdx.y)*34 + threadIdx.x+1]*E3L3[3];
            tenergy += tile[(threadIdx.y+1)*34 + threadIdx.x+1]*LAWS[Tn][4];// + tile[(threadIdx.y+1)*34 + threadIdx.x+1]*E3L3[4];
            tenergy += tile[(threadIdx.y+2)*34 + threadIdx.x+1]*LAWS[Tn][5];// + tile[(threadIdx.y+2)*34 + threadIdx.x+1]*E3L3[5];
            tenergy += tile[(threadIdx.y)*34 + threadIdx.x+2]*LAWS[Tn][6];// + tile[(threadIdx.y)*34 + threadIdx.x+2]*E3L3[6];
            tenergy += tile[(threadIdx.y+1)*34 + threadIdx.x+2]*LAWS[Tn][7];// + tile[(threadIdx.y+1)*34 + threadIdx.x+2]*E3L3[7];
            tenergy += tile[(threadIdx.y+2)*34 + threadIdx.x+2]*LAWS[Tn][8];// + tile[(threadIdx.y+2)*34 + threadIdx.x+2]*E3L3[8];
            convoluted[y*width + x] = abs((int)tenergy);
        }
    }
}

__global__ void average_filter(short* convoluted, int width, int height, short* tex)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    if(x < width && y < height)
    {
        extern __shared__ short halo[];
        if(y > 0 && y < width-1)
        {    halo[(x+1)*3+0] = convoluted[(y-1)*width+x];
            halo[(x+1)*3+1] = convoluted[y*width+x];
            halo[(x+1)*3+2] = convoluted[(y+1)*width+x];
        }

        __syncthreads();

        int sum = 0;

        sum += halo[x*3];
        sum += halo[x*3+1];
        sum += halo[x*3+2];
        sum += halo[x*3+3];
        sum += halo[x*3+4];
        sum += halo[x*3+5];
        sum += halo[x*3+6];
        sum += halo[x*3+7];
        sum += halo[x*3+8];

        sum = sum/9;
        tex[y*width + x] = sum;

    }
}

__global__ void binary_treshold(cudaTextureObject_t T, int t, short* texture, gpu::GpuMat binMat, int width, int height)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    if(texture[y*width + x] > t)
    {
        *(binMat.data + y*binMat.step + x) = 0;
    }
    else
        *(binMat.data + y*binMat.step + x) = 1;
}
