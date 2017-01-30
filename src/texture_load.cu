#include <cuda_runtime.h>
#include <core/core.hpp>

using namespace std;
using namespace cv;

typedef unsigned char uchar;

extern cudaArray *image1; extern cudaArray *image2;
extern cudaArray *cost_arr;
extern int width, height; extern int ndisp;

extern cudaTextureObject_t texImg1;
extern cudaTextureObject_t texImg2;
extern cudaTextureObject_t texBin;
extern cudaTextureObject_t Tcost;

extern "C" void init_texture(uchar *img1, uchar *img2)
{
    cudaMemcpyToArray(image1, 0, 0, img1, width*height*sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(image2, 0, 0, img2, width*height*sizeof(uchar), cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc1, resDesc2;
    memset(&resDesc1, 0, sizeof(resDesc1));
    memset(&resDesc2, 0, sizeof(resDesc2));
    resDesc1.resType = cudaResourceTypeArray;
    resDesc1.res.array.array = image1;
    resDesc2.resType = cudaResourceTypeArray;
    resDesc2.res.array.array = image2;

    struct cudaTextureDesc texDesc1, texDesc2;
    memset(&texDesc1, 0, sizeof(texDesc1));
    memset(&texDesc2, 0, sizeof(texDesc2));
    texDesc1.addressMode[0] = cudaAddressModeClamp;
    texDesc1.addressMode[1] = cudaAddressModeClamp;
    texDesc1.filterMode = cudaFilterModePoint;
    texDesc1.readMode = cudaReadModeElementType;
    texDesc2.addressMode[0] = cudaAddressModeClamp;
    texDesc2.addressMode[1] = cudaAddressModeClamp;
    texDesc2.filterMode = cudaFilterModePoint;
    texDesc2.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&texImg1, &resDesc1, &texDesc1, NULL);
    cudaCreateTextureObject(&texImg2, &resDesc2, &texDesc2, NULL);
}

extern "C" void upload_cost(short *h_cost)
{
    cudaMemcpyToArray(cost_arr, 0, 0, h_cost, width*ndisp*height*sizeof(short), cudaMemcpyHostToDevice);

    struct cudaResourceDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.resType = cudaResourceTypeArray;
    texDesc.res.array.array = cost_arr;

    struct cudaTextureDesc Desc;
    memset(&Desc, 0, sizeof(Desc));
    Desc.addressMode[0] = cudaAddressModeClamp;
    Desc.addressMode[1] = cudaAddressModeClamp;
    Desc.filterMode = cudaFilterModePoint;
    Desc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&Tcost, &texDesc, &Desc, NULL);
}
