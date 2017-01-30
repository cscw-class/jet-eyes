#define BASE 0.112
#define PIXEL_SIDE 0.000006
#define FOCAL 0.003
#define X0 293
#define Y0 223
#define PPANGLE 0.1125
#define PPCM 0.6
#define PI 3.14159

#include <iostream>
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include <features2d/features2d.hpp>
#include <calib3d/calib3d.hpp>
#include <video/video.hpp>
#include <sstream>
#include <cmath>
#include <string>
#include <pthread.h>
#include <time.h>

using namespace cv;
using namespace std;

//Global externs
extern int width;
extern int height;
extern Mat i1;
extern Mat i2;
extern Mat iMatch;
extern int ndisp;
extern bool motion_analysis;
extern bool loaded;
extern Mat ground;

bool base = true;
vector<Point3f> base3D, match3D;
vector<Point2f> previousR, currentR;
vector<Point2f> previousL, currentL;
int nPoints, nGpts, nSpts;
vector<int> Bscene_depth;
vector<KeyPoint> BsceneKP;
vector<int> Mscene_depth;
vector<KeyPoint> MsceneKP;
Mat prevLeft, prevRight;
ORB orb_detector(500, 1.2f, 8, 10, 0, 2, ORB::HARRIS_SCORE, 50);
BriefDescriptorExtractor descriptor;
BFMatcher matcher(NORM_HAMMING, false);
Mat mI1, mI2;
vector<Point2f> featuresL;
vector<Point2f> featuresR;
vector<KeyPoint> leftKP;
vector<KeyPoint> rightKP;
Mat Mdescriptors_left, Mdescriptors_right;
Mat drawing, groundArea, groundArea2;
int repopThres = 30, groundThres = 15, skyThres = 10;
int HORIZON = 300;

//loop data
vector< vector <Point2f> > statSequence;
int seq_count = 0;
int seqSize = 7;

//movement data
float speed;
float yaw=0;
float X=0, Y=0;    //real spatial coordinates;
float deltaY = 0, deltaX = 0;
timespec time1, time2;
KalmanFilter KF = KalmanFilter(2, 1, 0, CV_32F);    //set up Kalman filter, dynamic - x position and x speed, measurement z movement
Mat measurement = Mat::zeros(1, 1, CV_32F);
Mat estimated;

void initKalman()
{
    KF.transitionMatrix = *(Mat_<float>(2,2) << 1, 1, 0, 1);
    measurement.setTo(Scalar(0));

    X = 0;
    Y = 0;

    KF.statePre.at<float>(0) = 0;    //position
    KF.statePre.at<float>(1) = 0;    //speed

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));
}

void * detectL(void * arguments)
{
    int max_corn = 500;    //maximum number of features
    double qualityLvl = 0.008;
    double minDist = 20;
    int blockSz = 21;
    double k = 0.005;

    //Harris looks better (more stable keypoints)
    goodFeaturesToTrack(i1, featuresL, max_corn, qualityLvl, minDist, noArray(), blockSz, false, k);

    KeyPoint K;
    for(int i = 0; i < featuresL.size(); i++)
    {
        K.pt.x = featuresL[i].x;
        K.pt.y = featuresL[i].y;
        K.size = 15.0f;
        leftKP.push_back(K);
    }
    featuresL.clear();
    descriptor.compute(i1, leftKP, Mdescriptors_left);
}

void detect_kp(std::vector<Point2f>& scene, Mat src)
{
    vector<DMatch> matches;
    vector<DMatch> filtered_matches;
    vector<DMatch> Hfiltered_matches;
    vector<KeyPoint> outputKP;
    Point2f rightPoint;
    Point3f outPoint3D;
    int x, y, x0, y0, t;
    ostringstream strstream;
    string txt;
    pthread_t feature_detectL;
    pthread_t feature_detectR;
    float z;
    int sad, bestsad, bestdisp, sad_err = 500;

    int max_corn = 2000;    //maximum number of features
    double qualityLvl = 0.01;
    double minDist = 15;
    int blockSz = 5;
    double k = 0.005;

    //Detect keypoints in left image
    //Start with Shi-Tomasi feature detector
    //detect features in both left and right image
    goodFeaturesToTrack(src, scene, max_corn, qualityLvl, minDist, noArray(), blockSz, false, k);

}

timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

void coordinates3D(int x, int y, int disp, Point3f& out)
{
    //Calculates real world coordinates of an image point
    float realx;
    float realy;
    float realz;

    realz = (BASE*FOCAL)/(disp*PIXEL_SIDE);
    realx = ((X0-x)*PIXEL_SIDE*realz)/FOCAL;
    realy = ((Y0-y)*PIXEL_SIDE*realz)/FOCAL;

    out.x = realx;
    out.y = realy;
    out.z = realz;
}

void motionCapture()
{
    Mat base_descriptors, match_descriptors;
    vector<vector<int> > cons_matchesPC;
    vector<vector<float> > matches_valid;
    vector<Point3f> validBase3D, validMatch3D, validFBase3D, validFMatch3D;
    vector<Point2f> previousL0, currentL0;
    vector<Point2f> currentR0, previousR0;
    vector<float> correlations;
    vector<float> match;
    float x, y, z, x0, y0, z0;
    int x_c, y_c, z_c; //number of changes for each direction
    int n, disp;
    float av_ang; int n_ang;
    float K, k0;
    bool valid; int EdX, EdY;
    ostringstream strstream;
    string txt;
    Point3f validbase, validmatch;
    vector<Point2f> dummy;

    //statistical variables
    float mean, variance, sd;

    //first image
    if(base == true)
    {
        previousL.clear();
        detect_kp(previousL, i1);

        nPoints = previousL.size();
        nGpts = nPoints;
        nSpts = nPoints;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
        prevLeft = i1.clone();
        base = false;

    }
    else
    {
        float search_radX = 60; //search window around landmark from frame0, where Im looking for match in frame1
        float search_radY = 20;
        int x, y, x0, y0;
        float z;
        int SW = 25; int SW_s = sqrt(SW);
        float sad, bestsad, bestmatch, ncc;
        x_c = 0; y_c = 0; z_c = 0;
        float mean_corr = 0;
        float rotation_correct;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

        //locate landmarks in second image
        currentL0.clear(); previousL0.clear();

        //re-populate if necessary
        if(nPoints < repopThres || nGpts < groundThres || nSpts < skyThres)
        {
            previousL.clear();
            detect_kp(previousL, prevLeft);

            nPoints = previousL.size();
            nGpts = nPoints;
            nSpts = nPoints;
            statSequence.clear();
            seq_count = 0;
        }

        previousL0 = previousL;

        //Match previous frame with current one
        vector<uchar> statusL; vector<float> errorsL;
        calcOpticalFlowPyrLK(prevLeft, i1, previousL0, currentL0, statusL, errorsL, Size(21, 21), 3, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.2), 0);

        for(int i=0; i<previousL0.size(); i++)
        {
            EdX = previousL0[i].x - currentL0[i].x;
            EdY = previousL0[i].y - currentL0[i].y;
        }

        //Add points into statistical folder
        vector<Point2f> curSeq;
        for(int i = 0; i < currentL0.size(); i++)
        {
            if(statusL[i] != 0)
            {
                curSeq.push_back(currentL0[i]);
            }
            else
            {
                Point2f Pt;
                Pt.x = -1; Pt.y = -1;
                curSeq.push_back(Pt);
            }
        }
        statSequence.push_back(curSeq);


        //analyze stat folder
        if(statSequence.size() == seqSize)
        {
            //do stuff
            nPoints = 0; nGpts = 0;
            vector<int> rotation, translation;
            vector<Point2f> groundPt, groundPtT;
            vector<vector<Point2f> > groundPtsT;
            drawing = i1.clone();
            av_ang = 0; n_ang = 0;

            groundArea = Mat(i1.size(), i1.type());
            warpPerspective(i1, groundArea, ground, i1.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar());

            for(int i = 0; i < statSequence[0].size(); i++)
            {
                valid = true;
                for(int j = 0; j < seqSize; j++)
                {
                    if(statSequence[j][i].x == -1)
                    {
                        valid = false;
                    }
                }
                if(valid == false)
                    continue;
                nPoints++;
                if(statSequence[seqSize-1][i].y < HORIZON)
                {
                    //analyze direction change
                    //calculate first - last direction change
                    if( (statSequence[seqSize-1][i].x-statSequence[0][i].x) != 0)
                        K = (statSequence[seqSize-1][i].y-statSequence[0][i].y)/(statSequence[seqSize-1][i].x-statSequence[0][i].x);
                    else
                        K = statSequence[seqSize-1][i].y-statSequence[0][i].y;
                    for(int j = 0; j < seqSize-1; j++)
                    {
                        //direction change between two neighbourhood frames
                        if( (statSequence[j+1][i].x-statSequence[j][i].x) != 0)
                            k0 = (statSequence[j+1][i].y-statSequence[j][i].y)/(statSequence[j+1][i].x-statSequence[j][i].x);
                        else
                            k0 = statSequence[j+1][i].y-statSequence[j][i].y;
                         if(abs(K-k0) > 2)
                         {    valid = false;
                         }
                     }



                    if(valid == false)
                        continue;

                    rotation.push_back(statSequence[seqSize-1][i].x - statSequence[0][i].x);
                }
                else if(statSequence[seqSize-1][i].y > HORIZON)
                {
                    //just store ground points
                    groundPt.clear(); groundPtT.clear();
                    for(int j = 0; j < seqSize; j++)
                    {
                        groundPt.push_back(statSequence[j][i]);
                    }
                    perspectiveTransform(groundPt, groundPtT, ground);
                    groundPtsT.push_back(groundPtT);
                }
            }
            nSpts = rotation.size();
            sort(rotation.begin(), rotation.end());
            if(rotation.size() > 5)
            {
                //increment rotation
                deltaY = rotation[rotation.size()/2] * PPANGLE;
                Y -= deltaY;

                if(Y > 180)
                    Y = -179;
                else if(Y < -180)
                    Y = 179;

                //cout << "Rot: " << Y << endl;
            }

            for(int i = 0; i < groundPtsT.size(); i++)
            {
                //checking ground points

                //check smoothness
                //perform heuristic analysis
                mean = 0; variance = 0;
                for(int j = 0; j < seqSize-1; j++)
                {
                    mean += abs( groundPtsT[i][j].x - groundPtsT[i][j+1].x);
                }
                mean = mean/(seqSize-1);
                for(int j = 0; j < seqSize-1; j++)
                {
                    variance += pow( abs(groundPtsT[i][j].x - groundPtsT[i][j+1].x) - mean, 2 );
                }
                variance = variance/(seqSize-1);
                sd = sqrt(variance);

                valid = true;
                for(int j = 0; j < seqSize-1; j++)
                {
                    if( (abs(groundPtsT[i][j].x - groundPtsT[i][j+1].x) - mean) > 2*sd )
                        valid = false;
                }
                if(valid == false)
                    continue;

                nGpts++;
                rotation_correct = (cos( (PI/180) * deltaY)*groundPtsT[i][0].y) - groundPtsT[i][0].y;
                translation.push_back( (groundPtsT[i][seqSize-1].y - groundPtsT[i][0].y) - rotation_correct);
            }

            sort(translation.begin(), translation.end());

            if(translation.size() > 10)
            {
                deltaX = translation[translation.size()/2]*PPCM;
                X += deltaX;
                //cout << X << endl;
            }

            line(drawing, Point(0, HORIZON), Point(drawing.cols, HORIZON), Scalar(255, 0, 0, 0), 1, 8, 0);
            groundArea2 = drawing.clone();

            statSequence.clear();
            //re-populate?
            //nPoints = 0;
        }

        previousL = currentL0;
        prevLeft = i1.clone();

    }
}
