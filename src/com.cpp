#include <iostream>
#include <vector>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <core/core.hpp>
#include <highgui/highgui.hpp>

using namespace std;
using namespace cv;

typedef unsigned char uchar;

struct termios config;
int fd;

int listenfd = 0, connfd = 0;
struct sockaddr_in serv_addr;
char control_buffer[500];
vector<char> message;
pthread_mutex_t mut;

bool START = false;
bool disparity_READY = false;
bool sendDisparity = false;

extern Mat disparity_normC;
extern Mat groundArea2;
extern Mat dI1;
extern Mat dI2;
extern float deltaX, Y, X;
extern int roi_x, roi_y, roi_width, roi_height;
extern bool use_textures;
extern bool refine;
extern int P1, P2, maxDispDiff, min_area;

void changeRspeed(uchar speed)
{
    uchar command[1];
    size_t ret;
    command[0] = 'T';
    ret = write(fd, command, 1);

    command[0] = speed;
    ret = write(fd, command, 1);
}

void changeMspeed(uchar speed)
{
    uchar command[1];
    size_t ret;
    command[0] = 'M';
    ret = write(fd, command, 1);
    command[0] = speed;
    ret = write(fd, command, 1);
}

void moveForward()
{
    uchar command[1];
    size_t ret;
    command[0] = 'F';

    ret = write(fd, command, 1);
}

void moveBackward()
{
    uchar command[1];
    size_t ret;
    command[0] = 'B';

    ret = write(fd, command, 1);
}

void moveLeft()
{
    uchar command[1];
    size_t ret;
    command[0] = 'L';

    ret = write(fd, command, 1);
}

void moveRight()
{
    uchar command[1];
    size_t ret;
    command[0] = 'R';

    ret = write(fd, command, 1);
}

void StopEng()
{
    uchar command[1];
    size_t ret;
    command[0] = 'S';

    ret = write(fd, command, 1);
}

void get_message(char buffer[], vector<char>& msg, int len)
{
    msg.clear();
    for(int i=0; i<len; i++)
    {
        msg.push_back(buffer[i]);
    }
}

int compare(vector<char> & msg, const char * data)
{
    int val;
    if(msg.size() == strlen(data))
    {
        for(int i=0; i<strlen(data); i++)
        {
            if(msg[i] != data[i])
            {
                return 0;
            }
        }
        return(1);
    }
    return 0;
}

void startSLAM()
{
    START = true;
}

void serverInit()
{
    listenfd = socket(AF_INET, SOCK_STREAM, 0);
    memset(&serv_addr, '0', sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(5000);

    bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    listen(listenfd, 10);
}

void * com_interface(void * arguments)
{
    //Communication interface for talking to vehicle and to a remote
    //computer through wifi.
    //Maintains an infinite loop and constant flow of state data and commands

    //Init variables
    groundArea2 = Mat::ones(roi_height, roi_width, CV_8U);
    disparity_normC = Mat::ones(240, 320, CV_16S);
    dI1 = Mat::ones(240, 320, CV_8U);
    dI2 = Mat::ones(240, 320, CV_8U);

    int buffSize, n;
    pthread_mutex_init(&mut, 0);
    int speed;

    //Setup WiFi server
    serverInit();

    //set up serial port
    const char *device = "/dev/ttyTHS0";
    fd = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
    if(fd == -1) {
        cout <<    "failed to open port\n" << endl;
    }
    config.c_iflag &= ~(IGNBRK | BRKINT | ICRNL |
                         INLCR | PARMRK | INPCK | ISTRIP | IXON);
    config.c_oflag = 0;
    config.c_lflag &= ~(ECHO | ECHONL | ICANON | IEXTEN | ISIG);
    config.c_cflag &= ~(CSIZE | PARENB);
    config.c_cflag |= CS8;
    config.c_cc[VMIN]  = 1;
    config.c_cc[VTIME] = 0;
    if(cfsetispeed(&config, B9600) < 0 || cfsetospeed(&config, B9600) < 0) {
        cout << "error setting speed" << endl;
    }
    if(tcsetattr(fd, TCSAFLUSH, &config) < 0) {
        cout << "error writing tty params" << endl;
    }

    //Wait for support computer to connect
    while(1)
    {
        cout << "waiting for connection" << endl;
        connfd = accept(listenfd, (struct sockaddr*)NULL, NULL);
        if(connfd != -1)
        {
            cout << "Connected! \n" << endl;
            break;
        }
    }

    while(1)
    {
        //Main communication loop
        //Handle TCP communication
        if(recv(connfd, control_buffer, 8, 0) > 0)
        {
            get_message(control_buffer, message, 8);
            if(compare(message, "GOGOGOGO"))
            {
                //start main loop
                startSLAM();
            }
            else if(compare(message, "MFORWARD"))
            {
                moveForward();
            }
            else if(compare(message, "MVBKWARD"))
            {
                moveBackward();
            }
            else if(compare(message, "MOVELEFT"))
            {
                moveLeft();
            }
            else if(compare(message, "MOVRIGHT"))
            {
                moveRight();
            }
            else if(compare(message, "STOPENG0"))
            {
                StopEng();
            }
            else if(compare(message, "SETMSPED"))
            {
                n = recv(connfd, &speed, sizeof(int), 0);
                changeMspeed((uchar)speed);
            }
            else if(compare(message, "SETRSPED"))
            {
                n = recv(connfd, &speed, sizeof(int), 0);
                changeRspeed((uchar)speed);
            }
            else if(compare(message, "SETMDIFF"))
            {
                n = recv(connfd, &maxDispDiff, sizeof(int), 0);
            }
            else if(compare(message, "SETP1000"))
            {
                n = recv(connfd, &P1, sizeof(int), 0);
            }
            else if(compare(message, "SETP2000"))
            {
                n = recv(connfd, &P2, sizeof(int), 0);
            }
            else if(compare(message, "SMINAREA"))
            {
                n = recv(connfd, &min_area, sizeof(int), 0);
            }
            else if(compare(message, "STOPSTOP"))
            {
                //stop the program if control computer says so
                StopEng();
                exit(0);
            }
            else if(compare(message, "USETEXT0"))
            {
                //stop the program if control computer says so
                if(use_textures == true)
                    use_textures = false;
                else
                    use_textures = true;
            }
            else if(compare(message, "REFINE00"))
            {
                //stop the program if control computer says so
                if(refine == true)
                    refine = false;
                else
                    refine = true;
            }
            else if(compare(message, "SENDDATA"))
            {
                pthread_mutex_lock(&mut);
                //send left image
                n = send(connfd, groundArea2.data, groundArea2.total()*groundArea2.elemSize(), 0);
                //send disparity map
                n = send(connfd, disparity_normC.data, disparity_normC.total()*disparity_normC.elemSize(), 0);
                //send left frame
                n = send(connfd, dI1.data, dI1.total()*dI1.elemSize(), 0);
                //send current position
                n = send(connfd, &deltaX, sizeof(deltaX), 0);
                n = send(connfd, &Y, sizeof(Y), 0);
                pthread_mutex_unlock(&mut);
            }
            memset(control_buffer, 0, sizeof(control_buffer));
        }
    }

    pthread_mutex_destroy(&mut);
}


