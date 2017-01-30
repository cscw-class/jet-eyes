#include <iostream>
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include <calib3d/calib3d.hpp>
#include <pthread.h>

#include <cuda_runtime.h>

#include "../headers/rectification.hpp"

using namespace std;
using namespace cv;

//Global scope
Mat i1, i2, i10, i20;
Mat disparity;
Mat iMatch;
Mat h_texture;
cv::Rect ROI;
int width = 320, height = 240;
int ndisp, roi_x, roi_y, roi_width, roi_height;
bool loaded, stereo_end = true, motion_analysis = true;
extern bool START;

extern Mat mapx1, mapx2, mapy1, mapy2, roi, D2Dmap, ground;
extern Mat dI1;
extern Mat dI2;

extern "C" void * stereo_main(void * arguments);
extern "C" void Init_cuda();

void motionCapture();
void * grab1(void * arguments);
void * grab2(void * arguments);
void * com_interface(void * arguments);
void initKalman();

//Will execute as first, perform all initialization, create threads
int main()
{
    setenv("DISPLAY", ":0", 0); //for remote development

    //Init rectification - load distortion params and region of interest
    get_correction_params(mapx1, mapy1, mapx2, mapy2, roi, D2Dmap, ground);
    roi_x = roi.at<short>(0, 0), roi_y = roi.at<short>(1, 0), roi_width = roi.at<short>(2, 0), roi_height = roi.at<short>(3, 0);
    ROI = cv::Rect(roi_x, roi_y, roi_width, roi_height);
    //Init misc
    ndisp = 48;
    //Init camera
    VideoCapture capture1(0);
    VideoCapture capture2(1);
    //Init Kalman filter
    initKalman();
    //Init threads
    pthread_t stereovision, motionProc, com;
    pthread_t get1, get2;
    //Init GPU
    Init_cuda();

    //start communication
    pthread_create(&com, NULL, com_interface, NULL);
    //wait for command from chief computer
    while(START == false)
    {
            waitKey(1);
            capture1.grab();
            capture2.grab();
    }

    cout << "Main routine is GO" << endl;

    //main loop
    while (1)
    {
        capture1.grab();    //Assuming this is left
        capture2.grab();    //And this is right
        capture1.retrieve(i10, 0);
        capture2.retrieve(i20, 0);

        loaded = false;
        cvtColor(i10, i10, CV_RGB2GRAY);
        remap(i10, i10, mapx1, mapy1, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
        i10(ROI).copyTo(i1);
        resize(i1, dI1, Size(320, 240), 0, 0, INTER_LINEAR);

        cvtColor(i20, i20, CV_RGB2GRAY);
        remap(i20, i20, mapx2, mapy2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
        i20(ROI).copyTo(i2);
        resize(i2, dI2, Size(320, 240), 0, 0, INTER_LINEAR);
        loaded = true;

        if(stereo_end)
        {
            h_texture = Mat::zeros(height, width, CV_8U);
            disparity = Mat::zeros(height, width, CV_16S);
            pthread_create(&stereovision, NULL, stereo_main, NULL);
            pthread_detach(stereovision);
            stereo_end = false;
        }

        motionCapture();
        waitKey(1);
    }

    return 0;
}
