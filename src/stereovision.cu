#include <stdio.h>
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <calib3d/calib3d.hpp>
#include <imgproc/imgproc.hpp>
#include <gpu/gpu.hpp>
#include <cuda_runtime.h>
#include <features2d/features2d.hpp>

#define BASE 0.112
#define PIXEL_SIDE 0.000006
#define FOCAL 0.003
#define X0 160
#define Y0 120

using namespace std;
using namespace cv;

//function defines
void sendDisp(Mat disp);
extern "C" void init_texture(uchar *img1, uchar *img2);
extern "C" void upload_cost(short *h_cost);
__global__ void laws_convolution(cudaTextureObject_t img, int width, int height, short* convoluted, int Tn);
__global__ void average_filter(short* convoluted, int width, int height, short* tex);
__global__ void path_aggregate(int width, int height, int D, int P1, int P2, int dir, int* Espace, cudaTextureObject_t Cost, short* texturePerp, short* textureParallel);
__global__ void BM_cost(cudaTextureObject_t image1, cudaTextureObject_t image2, short* cost, int width, int height, int D);
__global__ void energy_minimalize(int *Energy, short* disparity, short* disparity2, int D, int width, int height);
__global__ void validate(short* disparity, short* disparity2, int width, int height, int disp12maxdiff);
__global__ void binary_treshold(cudaTextureObject_t T, int t, short* texture, gpu::GpuMat binMat, int width, int height);
void SGBM(Mat im_left, Mat im_right, int D, Mat& disp);

//Global externs
extern int width, height;
extern short* texture1; extern short* texture2;
extern short* textureE3L3; extern short* textureL3E3; extern short* textureE3E3;
extern short* convoluted;
extern Mat disparity;
extern short* cost;
extern int ndisp;
extern Mat h_texture;
extern short* h_cost;
extern short* disp; extern short* disp2; extern int* Espace;
extern bool loaded; extern bool stereo_end; extern bool disparity_READY;
extern int HORIZON;

//Global variables for stereovision
cudaTextureObject_t texImg1;
cudaTextureObject_t texImg2;
cudaTextureObject_t texBin;
cudaTextureObject_t Tcost;
Mat dI1, dI2;    //local image storage
Mat real3D; //3channel matrix containing real 3D coordinates of all points
float real_x, real_y, real_z;
Mat disparity_norm, disparity_normC;
int P1 = 20; int P2 = 60; int maxDispDiff = 4;
int min_area = 1500;

__device__ int valid_count;
int h_valid_count;
gpu::GpuMat binMatraw;
gpu::GpuMat binMateroded;
gpu::GpuMat binMat;
gpu::GpuMat texRaw;
Mat segmented;
Mat htexMain;
vector< vector <Point> > blobs;
vector< vector <int> > CoW;
vector<KeyPoint> keypoint_left, keypoints_right;
KeyPoint border;
vector<DMatch> match;
vector<DMatch> matches;
vector< vector <int> > borders;
Mat descriptor_left, descriptors_right;
bool use_textures = true;
bool refine = true;


extern "C" void lawTex(short* texture, cudaTextureObject_t tIm, int Tn)
{

    //use 128 threads per block -> best occupancy
    dim3 dimGrid((width+31)/32, (height+3)/4, 1);
    dim3 dimBlock(32, 4, 1);
    //use dynamic shared array in exe command
    laws_convolution<<<dimGrid, dimBlock, 0>>>(tIm, width, height, texture, Tn);
    //average_filter<<<height, width, (width+2)*3*sizeof(short)>>>(convoluted, width, height, texture);
}

extern "C" void binarySegmentation(short* texture)
{
    int treshold = 8;
    //segment image into binary map
    binMatraw = gpu::GpuMat(height, width, CV_8U);
    binary_treshold<<<height, width>>>(texImg1, treshold, texture, binMatraw, width, height);

    gpu::dilate(binMatraw, binMateroded, Mat(), Point(-1, -1), 1);
    gpu::erode(binMateroded, binMat, Mat(), Point(-1, -1), 2);
}

extern "C" void stereoBM()
{
    //clear previous data
    cudaMemset(Espace, 0, width*height*ndisp*sizeof(int));
    BM_cost<<<height, width, width*sizeof(short)>>>(texImg1, texImg2, cost, width, height, ndisp);
    cudaMemcpy(h_cost, cost, width*ndisp*height*sizeof(short), cudaMemcpyDeviceToHost);
    upload_cost(h_cost);

    dim3 dimBlock(ndisp, 1, 1);
    dim3 dimGrid(width, height, 1);

    path_aggregate<<<height, ndisp>>>(width, height, ndisp, P1, P2, 0, Espace, Tcost, textureE3L3, textureL3E3);
    path_aggregate<<<width, ndisp>>>(width, height, ndisp, P1, P2, 2, Espace, Tcost, textureL3E3, textureE3L3);
    path_aggregate<<<(width+height), ndisp>>>(width, height, ndisp, P1, P2, 1, Espace, Tcost, texture1, texture1);
    path_aggregate<<<(width+height), ndisp>>>(width, height, ndisp, P1, P2, 3, Espace, Tcost, texture1, texture1);
    path_aggregate<<<(height), ndisp>>>(width, height, ndisp, P1, P2, 4, Espace, Tcost, textureE3L3, textureL3E3);
    path_aggregate<<<(width+height), ndisp>>>(width, height, ndisp, P1, P2, 5, Espace, Tcost, texture1, texture1);
    path_aggregate<<<(width), ndisp>>>(width, height, ndisp, P1, P2, 6, Espace, Tcost, textureL3E3, textureE3L3);
    path_aggregate<<<(width+height), ndisp>>>(width, height, ndisp, P1, P2, 7, Espace, Tcost, texture1, texture1);

    energy_minimalize<<<dimGrid, dimBlock, 0>>>(Espace, disp, disp2, ndisp, width, height);
    validate<<<height, width>>>(disp, disp2, width, height, maxDispDiff);
}

extern "C" void detect_borders(vector<Point> blob, vector<Point>& borders, Mat binary_texture)
{
    Point border;
    float distance_threshold = 10;
    float min_dist;
    float dist;
    //detect border points
    for(int i = 0; i<blob.size(); i++)
    {
         for(int c = 0; c < 9; c++)
         {
              ///corner detected
              if(binary_texture.at<uchar>(blob[i].y-1+c/3, blob[i].x-1+c%3) == 0 && c != 4)
              {
                   border.x = blob[i].x;
                   border.y = blob[i].y;

                   /*check distance to other border points, if distance to a closest border point is
                    * lower than distance threshold reject this point
                    */
                   min_dist = INT_MAX;
                   for(int i = 0; i < borders.size(); i++)
                   {
                           dist = sqrt(pow(borders[i].x - border.x, 2) + pow(borders[i].y - border.y, 2));
                           if(dist < min_dist)
                                   min_dist = dist;
                   }

                   if(min_dist > distance_threshold)
                           borders.push_back(border);
              }
         }
    }
}

void disparity_interpolate(Mat& source)
{
    int T, d0, dy, dx, j0, i0, n = 0, m;
    int meanT[8]; int disp_candidate[8];
    int bestT, bestD;
    for(int j = 0; j < source.rows; j++)
    {
        for(int i = 0; i < source.cols; i++)
        {
            //interpolate missing pixels
            if(source.at<short>(j, i) == -1)
            {
                //eight directions, track its texture
                T = htexMain.at<short>(j, i);
                n = 0;
                for(int dir = 0; dir < 9; dir++)
                {
                    if(dir == 4)
                            continue;

                    j0 = j; i0 = i;
                    dx = -1 + dir/3;
                    dy = -1 + dir%3;
                    m = 0; meanT[n] = 0;

                    while(1)
                    {
                        j0 += dy; i0 += dx; m++;
                        meanT[n] += htexMain.at<short>(j0, i0);
                        d0 = source.at<short>(j0, i0);
                        if(d0 > 50)
                        {
                            meanT[n] = meanT[n]/m;
                            disp_candidate[n] = d0;
                            break;
                        }
                    }

                    n++;
                }

                //select best candidate with most similar path texture
                bestT = INT_MAX; bestD = 0;
                for(n = 0; n < 8; n++)
                {
                    if(abs(T-meanT[n]) < bestT)
                    {
                        bestT = abs(T-meanT[n]);
                        bestD = disp_candidate[n];
                    }
                }

                if(bestD > 0 && bestD < 500) {
                    source.at<short>(j, i) = bestD;
                }
            }
        }
    }
}

/* stereo_main
 *
 * Performs stereo matching and creates a disparity map.
 * Uses Laws texture energies to increase efficiency.
 */

extern "C" void * stereo_main(void * arguments)
{
    ostringstream strstream;
    string txt;

    //Time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    //Load frames into texture memory
    cudaEventRecord(start, 0);

    //load captured images into texture memory
    while(!loaded);
    init_texture(dI1.data, dI2.data);
    //calculate laws textures for future use in stereo matching
    lawTex(textureE3L3, texImg1, 0);
    lawTex(texture1, texImg1, -1);
    lawTex(textureL3E3, texImg1, 1);

    texRaw = gpu::GpuMat(height, width, CV_16S, texture1);
    texRaw.download(htexMain);

    //copy tresholded texture to Host
    binarySegmentation(texture1);
    binMat.download(h_texture);

    //Calculate initial disparity map
    stereoBM();
    //During calculations do segmentation
    medianBlur(h_texture, h_texture, 3);
    blobs.clear();
    h_texture.convertTo(segmented, CV_32S);
    int seg_count = 1; //255 and 0 are already "used"

    for( int y = 0; y < height; y++)
    {
        for( int x = 0; x < width; x++)
        {
            if(segmented.at<int>(y, x) != 1)
                continue;

            Rect rect;
            floodFill(segmented, cvPoint(x, y), Scalar(seg_count), &rect, Scalar(0), Scalar(0), 4);
            vector<Point> blob;
            for( int j = rect.y; j < (rect.y+rect.height); j++)
            {
                for( int i = rect.x; i < (rect.x+rect.width); i++)
                {
                   if(segmented.at<int>(j, i) != seg_count) {
                      continue;
                   }

                   blob.push_back(cvPoint(i, j));
                }
            }

            if(blob.size() > min_area)
            {
                blobs.push_back(blob);
                seg_count += 1;
            }
        }
    }

    //find convex hulls of blobs (border points)
    vector<vector<Point> > convex_border(blobs.size());
    for(int i = 0; i < blobs.size(); i++)
    {
        vector<Point2f> hull;
        detect_borders(blobs[i], convex_border[i], h_texture);
    }
    //calculate depth of selected segments
    //first, calculate depth of selected border points

    BriefDescriptorExtractor descriptor;
    BFMatcher matcher(NORM_HAMMING, false);
    vector<vector<short> > border_depth;
    vector<KeyPoint> base;
    vector<KeyPoint> match_points;
    Mat Bdescriptor, Mdescriptor;
    vector<DMatch> match;
    KeyPoint point;
    short z;
    int By, Bx;
    int sad; int bestsad; int bestdisp;
    int dispLR, dispRL, Bx2;
    float Sa; float ed; float edSUM = 0;

    for(int i = 0; i<convex_border.size(); i++)
    {
        vector<short> b0_depth;
        for(int j=0; j<convex_border[i].size(); j++)
        {
            //go through each border point
            By = convex_border[i][j].y;
            Bx = convex_border[i][j].x;

            KeyPoint KP;
            match_points.clear();
            base.clear();
            KP.pt.x = Bx; KP.pt.y = By; KP.size = 25.0f;
            base.push_back(KP);

            for(int d0 = 0; d0 < ndisp; d0++)
            {
                if(Bx - d0 > 0)
                {
                        KP.pt.x = Bx-d0; KP.pt.y = By; KP.size = 25.0f;
                        match_points.push_back(KP);
                }
            }
            descriptor.compute(dI1, base, Bdescriptor);
            descriptor.compute(dI2, match_points, Mdescriptor);
            matcher.match(Bdescriptor, Mdescriptor, match, Mat());

            if(match.size() == 1)
                dispLR = abs(base[0].pt.x - match_points[match[0].trainIdx].pt.x);
            else
                dispLR = -1;
            //ght Left check
            if(dispLR != -1)
            {
                Bx2 = Bx-dispLR;
                match_points.clear();
                base.clear();
                match.clear();
                KP.pt.x = Bx2; KP.pt.y = By; KP.size = 25.0f;
                base.push_back(KP);

                for(int d0 = 0; d0 < ndisp; d0++)
                {
                    if(Bx2 + d0 < width && Bx2+d0 > 0)
                    {
                        KP.pt.x = Bx2+d0; KP.pt.y = By; KP.size = 25.0f;
                        match_points.push_back(KP);
                    }
                }

                descriptor.compute(dI2, base, Bdescriptor);
                descriptor.compute(dI1, match_points, Mdescriptor);
                matcher.match(Bdescriptor, Mdescriptor, match, Mat());
                if(match.size() == 1)
                    dispRL = abs(base[0].pt.x - match_points[match[0].trainIdx].pt.x);
                else
                    dispRL = -1;
            }

            if(dispLR != -1 && dispRL != -1 && abs(dispLR-dispRL) < 8)// && abs(dispLR-dispRL) < 8)
            {
                z = (short)dispLR; //bestdisp; //(BASE*FOCAL)/(bestdisp*PIXEL_SIDE);
                b0_depth.push_back(z);
            }
            else
            {
                z = -1;
                b0_depth.push_back(z);
            }
        }
        border_depth.push_back(b0_depth);
    }

    cudaMemcpy(disparity.ptr(), disp, width*height*sizeof(short), cudaMemcpyDeviceToHost);

    for(int i=0; i<blobs.size(); i++)
    {
        for(int j = 0; j<blobs[i].size(); j++)
        {
            //Weighted average of border pixels
            Sa = 0;
            ed=0;
            edSUM = 0;
            for(int m = 0; m < convex_border[i].size(); m++)
            {
                if(border_depth[i][m] != -1)
                {
                    ed = sqrt( pow(blobs[i][j].x-convex_border[i][m].x, 2) + pow(blobs[i][j].y-convex_border[i][m].y, 2) );
                    if(ed != 0)
                    {
                        Sa += (short)border_depth[i][m]/ed;//sqrt((pow(blobs[i][j].x - convex_border[i][m].x, 2) + pow(blobs[i][j].y - convex_border[i][m].y, 2)));
                        edSUM += 1/ed;
                    }
                    else
                    {
                        Sa += border_depth[i][m];
                        edSUM+=1;
                    }
                }
            }

            if(border_depth[i].size() > 0)
            {
                Sa = Sa/edSUM;
            }
            else
            {
                Sa = 255;
            }

            if(use_textures)
            {
                disparity.at<short>(blobs[i][j].y, blobs[i][j].x) = 10*(short)Sa;
            }
        }
    }
    //Nail polish
    if(refine)
    {
        disparity_interpolate(disparity);
        medianBlur(disparity, disparity, 3);
        //filterSpeckles(disparity, -1, 200, 20);
        //disparity_interpolate(disparity);
    }
    else
    {
        medianBlur(disparity, disparity, 3);
        //filterSpeckles(disparity, -1, 200, 20);
    }

    //calculate real 3D coordinates of viewed scene
    real3D = Mat(disparity.rows, disparity.cols, CV_32FC3);
    for(int j = 0; j < disparity.rows; j++)
    {
        for(int i = 0; i < disparity.cols; i++)
        {
            if(disparity.at<uchar>(j, i) != 0)
            {
                real_z = 10*BASE*FOCAL/((float)disparity.at<short>(j, i)*PIXEL_SIDE);
                real_x = ((X0 - i)*PIXEL_SIDE*real_z)/FOCAL;
                real_y = 0.25+((Y0 - j)*PIXEL_SIDE*real_z)/FOCAL;

                real3D.at<Vec3f>(j, i)[0] = real_x;
                real3D.at<Vec3f>(j, i)[1] = real_y;
                real3D.at<Vec3f>(j, i)[2] = real_z;
            }
            else
            {
                real3D.at<Vec3f>(j, i)[0] = 0;
                real3D.at<Vec3f>(j, i)[1] = 100;
                real3D.at<Vec3f>(j, i)[2] = 100;
            }
        }
    }

    //look for horizon
    float horizon_thres = 0.1;
    int n;
    for(int j = 0; j < real3D.rows; j++)
    {
        n = 0;
        for(int i = 0; i < real3D.cols; i++)
        {
            if(real3D.at<Vec3f>(j, i)[1] <= horizon_thres)
                n++;
        }
        if(n > 70)
        {
            if( j > 100 && j < 215)
                HORIZON = 2*j;
            else
                HORIZON = 300;
            break;
        }
    }

    disparity_normC = disparity.clone();
    normalize(disparity, disparity_norm, 0, 255, CV_MINMAX, CV_8U);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaDestroyTextureObject(Tcost);
    cudaDestroyTextureObject(texImg1);
    cudaDestroyTextureObject(texImg2);

    stereo_end = true;
    return 0;
}
