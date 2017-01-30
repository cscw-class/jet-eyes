#include <cuda_runtime.h>
#include <core/core.hpp>

using namespace cv;
typedef unsigned char uchar;

//Global externs
extern int width;
extern int height;
extern int ndisp;

short* texture1;
short* texture2;
short* textureE3L3;
short* textureL3E3;
short* textureE3E3;

short* convoluted;
short* validpLUT;
short* h_validpLUT;
short* cost;
short* disp; short* disp2;
int* Espace; int* Espace2;
uchar* binary;
uchar *h_binary;
short *h_cost;
Mat mapx1, mapx2, mapy1, mapy2, roi, D2Dmap, ground;
cudaArray *image1;
cudaArray *image2;
cudaArray *bin_tex_arr;
cudaArray *cost_arr;


extern "C" void Init_cuda()
{
    cudaDeviceReset();
    cudaFree(0);

    //mallocs
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&image1, &channelDesc, width, height);
    cudaMallocArray(&image2, &channelDesc, width, height);
    cudaChannelFormatDesc texDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&cost_arr, &texDesc, width*ndisp, height);
    cudaMallocArray(&bin_tex_arr, &channelDesc, width, height);
    cudaMalloc(&texture1, width*height*sizeof(short));
    cudaMalloc(&texture2, width*height*sizeof(short));
    cudaMalloc(&textureE3L3, width*height*sizeof(short));
    cudaMalloc(&textureL3E3, width*height*sizeof(short));
    cudaMalloc(&textureE3E3, width*height*sizeof(short));

    cudaMalloc(&convoluted, width*height*sizeof(short));
    cudaMalloc(&binary, width*height*sizeof(uchar));

    h_binary = (uchar*)malloc(width*height*sizeof(uchar));
    h_validpLUT = (short*)malloc(width*height*2*sizeof(short));
    h_cost = (short*)malloc(width*ndisp*height*sizeof(short));

    cudaMalloc(&validpLUT, width*height*2*sizeof(short));
    cudaMalloc(&cost, width*height*ndisp*sizeof(short));
    cudaMalloc(&Espace, width*height*ndisp*sizeof(int));
    cudaMalloc(&Espace2, width*height*ndisp*sizeof(int));
    cudaMalloc(&disp, width*height*sizeof(short));
    cudaMalloc(&disp2, width*height*sizeof(short));
}
