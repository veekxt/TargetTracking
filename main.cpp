#include <iostream>
#include <cstdio>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//读取路径和格式有规律的素材
class MyPicStream{
    /*连续显示一段图像,迭代模板
    MyPicStream my = MyPicStream("Hallway",1,0,450);
    Mat tmp;
    tmp=my.getPic();
    while (!tmp.empty())
    {
        waitKey(50);
        puts("one");
        imshow("test_mat",tmp);

        tmp=my.getPic();
    }
    */
private:
public:
    int start_num;//初始和结束图片号
    int end_num;
    int pic_type;//图片类型,1普通rgb ，2深度
    const char *head;//文件名前缀
    int i;//当前图片号
    MyPicStream(const char *a_head,int a_type,int a_start_num,int a_end_num)
    {
        i=a_start_num;
        start_num=a_start_num;
        end_num=a_end_num;
        pic_type=a_type;
        head=a_head;
    };
    Mat getPic()
    {
        i++;
        if(i>end_num)
        {
            Mat tmp;
            tmp.data=NULL;
            return tmp;
        }
        char name[50];
        sprintf(name,"pic/01-%s/%s%s_%s_%d.bmp",
                      head,pic_type==2?"/Depth/":"",head,pic_type==1?"L":"disp_kinect",i);
        Mat target = imread(name);
        if(target.empty())
        {
            cout<<"cant open file:"<<name<<" may to end !"<<endl;
			waitKey(0);
            exit(1);
        }

        return target;
    };
};


//删除小的噪声点
//简单腐蚀
Mat& del_small(Mat &a,int k,int repeat)
{
    Mat tmp;
    erode(a,tmp,Mat(k,k,CV_8U),Point(-1,-1),repeat);
    //morphologyEx(a,tmp,MORPH_CLOSE,Mat(3,3,CV_8U));
    a=tmp;
    return a;
}


//参考深度处理结果和正常结果（a、b） 产生新的结果（c）
Mat& do_depth_and_normal(Mat &a,Mat &b,Mat &c)
{
    CV_Assert(a.depth() != sizeof(uchar));
    const int channels = a.channels();
    switch(channels)
    {
    case 1:
        {
            //三个迭代器分别迭代a、b、c
            MatIterator_<uchar> itA, Aend;
            MatIterator_<uchar> itB, Bend;
            MatIterator_<uchar> itC, Cend;

            for( itA = a.begin<uchar>(),Aend = a.end<uchar>(),
                 itB = b.begin<uchar>(),Bend = b.end<uchar>(),
                 itC = c.begin<uchar>(),Cend = c.end<uchar>();
                 itA != Aend;
                 ++itA,++itB,++itC )
            {
                //处理
                //note  ：
                //简单并：填补空洞但是增加噪声
                //简单交：减少噪声但增加空洞
                if(*itA>127 || *itB>127)
                *itC = 255;
                else *itC = 0;
            }
            break;
        }
        /*
    case 3:
        {
            MatIterator_<Vec3b> it, end;
            for( it = b.begin<Vec3b>(), end = b.end<Vec3b>(); it != end; ++it)
            {
                (*it)[0] = table_a[(*it)[0]];
                (*it)[1] = table_a[(*it)[1]];
                (*it)[2] = table_a[(*it)[2]];
            }
        }
        */
    }
    return c;
}

//空洞填充
//[参考](http://bbs.csdn.net/topics/340140568)
//[另](http://blog.sina.com.cn/s/blog_79bb01d00101btsq.html)
Mat imFillHoles(Mat imInput)
{
    Mat imShow = Mat::zeros(imInput.size(),CV_8UC3);    // for show result

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(imInput, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    if( !contours.empty() && !hierarchy.empty() )
    {
        for (unsigned int idx=0;idx < contours.size();idx++)
        {
            drawContours(imShow,contours,idx,Scalar::all(255),CV_FILLED,8);
        }
    }

    Mat imFilledHoles;
    cvtColor(imShow,imFilledHoles,CV_BGR2GRAY);
    imFilledHoles = imFilledHoles > 0;

    imShow.release();

    return imFilledHoles;
}

//去除小的噪声点(寻找最大的目标)
//[参考](http://www.cnblogs.com/tornadomeet/archive/2012/06/02/2531705.html)
//参数：原始32 img（只需要它的大小） 8位img 目标img
void del_small2(const Mat& img, Mat& mask, Mat& dst)
{
    int niters = 1;// default :3

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat temp;

    dilate(mask, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);//腐蚀
    dilate(temp, temp, Mat(), Point(-1,-1), niters);

    findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//找轮廓

    dst = Mat::zeros(img.size(), CV_8UC1);

    if( contours.size() == 0 )
        return;

    int idx = 0, largestComp = 0;
    double maxArea = 0;

    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(Mat(c)));
        if( area > maxArea )
        {
            maxArea = area;
            largestComp = idx;//找出包含面积最大的轮廓
        }
    }
    Scalar color( 250, 250, 250 );//使用白色来输出bgr
    drawContours( dst, contours, largestComp, color, CV_FILLED, 8, hierarchy );
}


#define SC1 "ChairBox"
#define SC2 "Hallway"
#define SC3 "Shelves"
#define SC4 "Wall"

const char *SC(int i)
{
    return i==1?SC1:(i==2?SC2:(i==3?SC3:(i==4?SC4:NULL)));
}

int main(int argc,char **argv)
{
    int delay_t,scn,start_pic;
    if(argc>1) delay_t = atoi(argv[1]);
    else delay_t=70;
    if(argc>2) scn = atoi(argv[2]);
    else scn = 1;
	if(argc>3) start_pic = atoi(argv[3]);
    else start_pic = 200;

    MyPicStream myb = MyPicStream(SC(scn),1,start_pic,600);
    MyPicStream myf = MyPicStream(SC(scn),2,start_pic,600);

    //ToDo ：使用自己实现的的GMM算法
    BackgroundSubtractorMOG2 bgSubtractor(30,16,true);

    namedWindow("picgmm",CV_WINDOW_NORMAL);
    namedWindow("picgmm_depth",CV_WINDOW_NORMAL);
    namedWindow("src",CV_WINDOW_NORMAL);
    namedWindow("tar",CV_WINDOW_NORMAL);

    Mat tmp,tmpf,picgmm,picgmmf;
    tmp=myb.getPic();
    tmpf=myf.getPic();

    while (!tmp.empty())
    {
        char key = waitKey(delay_t);
        switch(key)
        {
            //空格键暂停
            case ' ':waitKey(0);break;
            default:;
        }
        cout<<"# to "<<myb.i<<" th pic #"<<endl;
        bgSubtractor(tmp,picgmm,0.005);
        bgSubtractor(tmpf,picgmmf,0.005);

        // picgmm = del_small(picgmm,3,3);
        // picgmmf = del_small(picgmmf,3,3);

        imshow("picgmm",picgmm);            //rgb图结果
        imshow("picgmm_depth",picgmmf);     //深度图结果
        imshow("src",tmp);              //rgb原图
        /*
        Mat bk;
        bgSubtractor.getBackgroundImage(bk);
        imshow("background",bk);              //显示背景
        */
        del_small2(tmp,picgmm,picgmm);
        del_small2(tmp,picgmmf,picgmmf);

        /*写入文件，加入暂停后基本不需要了
        char name[20];
        sprintf(name,"target/L_%d.png",myb.i-1);
        imwrite(name,picgmm);
        */
        Mat tar=picgmm.clone();
        Mat tmp2;
        tar = do_depth_and_normal(picgmm,picgmmf,tar);

        //refineSegments(tmp,tar,tar);

        //del_small2(tar,tar,tar);
		tar = imFillHoles(tar);         //空洞填充
        imshow("tar",tar);

        tmp=myb.getPic();
        tmpf=myf.getPic();
    }
    puts("end");
    waitKey();
    return 0;
}
//ToDo :使用滑块来控制速度与其他参数
