#include <core/core.hpp>
#include <calib3d/calib3d.hpp>
#include <imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

extern Mat mapx1, mapx2, mapy1, mapy2;
extern Mat i10;
extern Mat i20;
extern Mat i2;
extern Mat i1;
extern Mat dI1, dI2;
extern cv::Rect ROI;

void * grab1(void * arguments)
{
    cvtColor(i10, i10, CV_RGB2GRAY);
    remap(i10, i10, mapx1, mapy1, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    i10(ROI).copyTo(i1);
    resize(i1, dI1, Size(320, 240), 0, 0, INTER_LINEAR);

       return 0;
}

void * grab2(void * arguments)
{
    cvtColor(i20, i20, CV_RGB2GRAY);
    remap(i20, i20, mapx2, mapy2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    i20(ROI).copyTo(i2);
    resize(i2, dI2, Size(320, 240), 0, 0, INTER_LINEAR);
    return 0;
}
