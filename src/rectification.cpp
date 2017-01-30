#include <iostream>
#include <fstream>
#include <core/core.hpp>

using namespace cv;
using namespace std;

void get_correction_params(Mat& mapx1, Mat& mapy1, Mat& mapx2, Mat& mapy2, Mat& roi, Mat& D2Dmap, Mat& ground)
{
    FileStorage fs;
    fs.open("../Rectification/mapx1.xml", FileStorage::READ);
    fs["mapx1"] >> mapx1;
    fs.release();
    fs.open("../Rectification/mapx2.xml", FileStorage::READ);
    fs["mapx2"] >> mapx2;
    fs.release();
    fs.open("../Rectification/mapy1.xml", FileStorage::READ);
    fs["mapy1"] >> mapy1;
    fs.release();
    fs.open("../Rectification/mapy2.xml", FileStorage::READ);
    fs["mapy2"] >> mapy2;
    fs.release();
    fs.open("../Rectification/roi.xml", FileStorage::READ);
    fs["roi"] >> roi;
    fs.release();
    fs.open("../Rectification/D2Dmap.xml", FileStorage::READ);
    fs["D2Dmap"] >> D2Dmap;
    fs.open("../Rectification/temp.xml", FileStorage::READ);
    fs["perspective"] >> ground;
    fs.release();
}




