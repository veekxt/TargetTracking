#include <iostream>
#include <cstdio>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//一些常量定义，但是可能要动态修改，仍声明为变量

//最小有效区域，低于此面积将被直接认定为噪声点
double MINAREA = 85.0;
//使用前多少张图像训练？
int TRAIN = 30;
//延时
int delay_t = 60;

#define SC1 "ChairBox"
#define SC2 "Hallway"
#define SC3 "Shelves"
#define SC4 "Wall"

const char *SC(int i)
{
    return i==1?SC1:(i==2?SC2:(i==3?SC3:(i==4?SC4:NULL)));
}

//读取路径和格式有规律的素材
class MyPicStream
{
private:
public:
    int start_num;      //初始和结束图片号
    int end_num;
    int pic_type;       //图片类型,1普通rgb ，2深度
    const char *head;   //文件名前缀
    int i;              //当前图片号
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
        char nameBuf[50];
        sprintf(nameBuf,"pic/01-%s/%s%s_%s_%d.bmp",
                head,pic_type==2?"/Depth/":"",head,pic_type==1?"L":"disp_kinect",i);
        Mat target = imread(nameBuf);
        if(target.empty())
        {
            cout<<"cant open file:"<<nameBuf<<" may to end !"<<endl;
            waitKey(0);
            exit(1);
        }

        return target;
    };
};


//参考深度处理结果和正常结果（a、b） 产生新的结果（c）
//note :已经弃用
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

//删除小的噪声点
void del_small(const Mat mask,Mat &dst)
{
    int niters = 1;// default :3

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat temp=mask.clone();

    dilate(mask, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);//腐蚀
    dilate(temp, temp, Mat(), Point(-1,-1), niters);

    findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE );//找轮廓

    dst = Mat::zeros(mask.size(), CV_8UC1);

    if( contours.size() == 0 )
        return;

    int idx = 0 ;

    //使用容器保存符合条件的区域
    vector<int> all_big_area;

    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(c));
        if( area > MINAREA )
        {
            all_big_area.push_back(idx);//添加到容器中
        }
    }
    Scalar color(255);//使用白色来输出
    vector<int>::iterator it;
    for(it=all_big_area.begin(); it!=all_big_area.end(); it++)
        drawContours( dst, contours, *it, color,CV_FILLED, 8, hierarchy );
}
//结构，轮廓和记录其中一些轮廓的vector
struct ValidContours{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;            //分层信息
    vector<int> valids;
};

//检查两个轮廓的重合率
//一个轮廓在一个轮廓内部的点所占自身的大小，粗略地表达重合面积
//供getValidContours函数使用
double coincidenceRateContours(vector<Point> a,vector<Point> b)
{
    double s=0.0;
    for(int i=a.size()-1;i>=0;i--)
    {
        if(pointPolygonTest(b,Point2f((double)a[i].y,(double)a[i].x),false)>=0)
        {
            s=s+1.0;
        }
    }
    return s / (double)a.size();
}

//根据前一帧和当前帧，判断一个轮廓是否是前景，并返回所有的轮廓和轮廓编号
struct ValidContours getValidContours(Mat dep,Mat dep_pre)
{
    vector<vector<Point> > contours;
    vector<vector<Point> > contours_pre;
    vector<Vec4i> hierarchy;
    vector<Vec4i> hierarchy_pre;

    vector<int> valid;

    //当前帧轮廓
    findContours( dep.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE );
    //前一帧轮廓
    findContours( dep_pre.clone(), contours_pre, hierarchy_pre, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE );
    //遍历单层轮廓，判断是否重合
    for(int i=0;i>=0;i=hierarchy[i][0])
    {
        if(hierarchy[i][3]<0)   //判断是顶层轮廓，即该轮廓没有父轮廓
        {
            for(int j=0;j>=0;j=hierarchy_pre[j][0])     //遍历上一帧的轮廓
            {
                if(hierarchy_pre[j][3]<0)
                {
                    if(coincidenceRateContours(contours[i],contours_pre[j])>0.3)
                    {
                        valid.push_back(i);
                        break;
                    }
                }
            }
        }
    }
    if(valid.empty())cout<<"None Valid Contours"<<endl;
    struct ValidContours v={contours,hierarchy,valid};
    return v;
}

//求一个二值图的前景面积,用于判断大面积光照
double getArea(Mat src)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat src2=src.clone();
    findContours( src2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
    double s=0;
    if(!contours.empty())
    for(int idx =0 ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(c));
        s+=area;
    }
    return s;
}

//判断是否有突然的光照
bool is_suddenly_light(Mat rgb,Mat rgb_pre,Mat dep,Mat dep_pre)
{
    double s_rgb,s_rgb_pre,s_dep,s_dep_pre,pro_rgb,pro_dep,s_pic;

    s_rgb = getArea(rgb);
    s_dep = getArea(dep);
    s_rgb_pre = getArea(rgb_pre);
    s_dep_pre = getArea(dep_pre);
    s_pic = rgb.rows*rgb.cols;

    if(s_rgb<=2)
    {
        return false;
    }
    pro_rgb = s_rgb / s_pic - s_rgb_pre / s_pic;//s_rgb / (s_rgb + s_rgb_pre);
    pro_dep = s_dep / s_pic - s_dep_pre / s_pic;//s_dep / (s_dep + s_dep_pre);

    if(pro_rgb >0.5 && pro_rgb - pro_dep > 0.3)//实验得出?
    {
        cout<<pro_rgb<<"*"<<pro_rgb - pro_dep<<endl;
        return true;
    }
    return false;
}
//判断fg或bg的可靠性
//参数 ：fg还是bg（0或1），坐标，src

/*             w2
            +--------+---------+
            |        |         |
            |     w1 |         |
            |    +---+---+     |
            |    |       |     |
            +----+       +-----+
            |    +---+---+     |
            |        |         |
            |        |         |
            +--------+---------+
*/
bool is_fbg_com_pre_2(int fg_or_bg,int x0,int y0,Mat src)
{
    double w1 = 13.0;
    double w2 = w1 * sqrt(5);
    double s = w1 * w1 * 4;
    int sum[5]={0,0,0,0,0,};

    int start_x = (x0 - w2)<0?0:(x0 - w2);
    int start_y = (y0 - w2)<0?0:(y0 - w2);
    int end_x   = (x0 + w2)>src.rows?src.rows:(x0 + w2);
    int end_y   = (y0 + w2)>src.cols?src.cols:(y0 + w2);

    //cout<<end_x-start_x<<"**"<<end_y-start_y<<"**"<<end_x<<"**"<<end_y<<endl;

    for(int x=start_x; x<end_x; x++)
    {
        for(int y=start_y; y<end_y; y++)
        {
            //todo:优化分类应该可以减少判断次数
            bool is_fbg;
            if(fg_or_bg==0)is_fbg = src.at<uchar>(x,y) > 0?true:false;
            else is_fbg = src.at<uchar>(x,y) < 200?true:false;

            if     ((x<x0 && y<y0-w1 )||(x<x0-w1 && y<y0)){if(is_fbg)sum[1]++;}
            else if((x>x0 && y<y0-w1 )||(x>x0+w1 && y<y0)){if(is_fbg)sum[2]++;}
            else if((x>x0 && y>y0+w1 )||(x>x0+w1 && y>y0)){if(is_fbg)sum[3]++;}
            else if((x<x0 && y>y0+w1 )||(x<x0-w1 && y>y0)){if(is_fbg)sum[4]++;}
            else                                          {if(is_fbg)sum[0]++;}
        }
    }
    int tmp=0;
    for(int i=0;i<5;i++)
    {
        if(sum[i]>tmp)
        {
            tmp=sum[i];
        }
    }
    if((double)tmp / s >(fg_or_bg==0?0.8 :0.8))
    {
        //cout<<sum[0]+sum[1]+sum[2]+sum[3]+sum[4]<<"**"<<tmp<<"**"<<tmp_i<<endl;
        return true;
    }
    return false;
}

//通过和src对比判断这个点是否为前景，是则写入,rgb=bk,dep=fg时使用
//参数：坐标、src、dst
//update:已弃用
void is_fg_com_pre(int i,int j,Mat src,Mat &dst)
{
    int w=60;               //设置宽度,决定周围区域的大小
    int s=0;
    int s_min = 9;          //重合部分最小比例
    int start_x = (i - w/2)<0?0:(i - w/2);
    int start_y = (j - w/2)<0?0:(j - w/2);
    int end_x   = (i + w/2)>src.rows?src.rows:(i + w/2);
    int end_y   = (j + w/2)>src.cols?src.cols:(j + w/2);


    for(; start_x<end_x; start_x++)
    {
        for(; start_y<end_y; start_y++)
        {
            if(src.at<uchar>(start_x,start_y)>0)s++;
        }
    }
    if(1000*s>(w*w)*s_min)
    {
        /*
        if(s>0)
        {
            printf("%d,%d\n%d#%d,%d#%d\n",i,j,start_x,start_y,end_x,end_y);
            printf("$ %d $\n",s);
        }
        */
        dst.at<uchar>(i,j)=127;   //有某些重合部分，判定为前景
    }
    //if(dep_pre.at<uchar>(i,j)>100)dst.at<uchar>(i,j)=255;
}

//分析4个输入图，产生新图
void analysis(bool have_suddenly_light,Mat rgb,Mat rgb_pre,Mat dep,Mat dep_pre,Mat &dst)
{
    if(!have_suddenly_light)
    {
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours( dep.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

        for( int i = 0; i < dep.rows; ++i)
        {
            for( int j = 0; j < dep.cols; ++j )
            {
                if(rgb.at<uchar>(i,j)>100)
                {
                    if(dep.at<uchar>(i,j)>100)dst.at<uchar>(i,j)=255; //确定为前景
                    else
                    {
                        //可能是局部的光照和阴影
                        if(!is_fbg_com_pre_2(1,i,j,dep_pre))
                        {
                            //如果深度图判定为”背景不可靠“
                            dst.at<uchar>(i,j) = 255;
                        }else
                        {
                            //如果深度图判定为”背景可靠“，应该是局部光照
                            //dst.at<uchar>(i,j) = 0;//默认是0，不需要显式赋值
                        }
                    }
                }
                if(dep.at<uchar>(i,j)>100)      //dep为前景
                {
                    if(rgb.at<uchar>(i,j)>100)  //rgb也为前景
                    {
                        dst.at<uchar>(i,j)=255; //确定为前景
                    }
                    else                        //rgb为背景，进一步检查
                    {
                        if(is_fbg_com_pre_2(0,i,j,dep_pre))
                        {
                            //dst.at<uchar>(i,j)=190;
                            for(int i_c=contours.size()-1;i_c>=0;i_c--)
                             {
                                double distance = pointPolygonTest(contours[i_c],Point(j,i),true);
                                //cout<<abs(distance)<<"<=distance"<<endl;
                                if((distance >= 6.0))
                                {
                                    //cout<<"a point in edge !########"<<endl;
                                    //cout<<distance<<"-"<<i<<"-"<<j<<endl;
                                    dst.at<uchar>(i,j)=255;
                                    break;
                                }
                            }

                        }
                    }
                }
            }
        }
    }
    else
    {
        //使用is_fg_com_pre判断噪声

        for( int i = 0; i < dep.rows; ++i)
        {
            for( int j = 0; j < dep.cols; ++j)
            {
                if(dep.at<uchar>(i,j)>100)
                {
                    if(is_fbg_com_pre_2(0,i,j,dep_pre))
                    {
                        dst.at<uchar>(i,j)=50;
                    }
                }
            }
        }

        /*
        //update：似乎也可以直接is_fg_com_pre判断，但is_fg_com_pre似乎有bug！
        //突然光照使得rgb结果没有参考价值，这里纯粹使用深度结果
        //深度图很多噪声只有一瞬间，因此对于一个区域来说，
        //前一帧和当前帧的轮廓应该有交集，没有交集可以判定为背景

        //找到dep可用的轮廓
        struct ValidContours v = getValidContours(dep,dep_pre);
        //填充可用轮廓
        for(int i=v.valids.size()-1;i>=0;i--)
        {
            drawContours(dst,v.contours,v.valids[i],Scalar(255),CV_FILLED,8,v.hierarchy);
        }
        */
        //但深度结果噪声太大，所以这里提高del_small的目标大小上限减少一些偏小噪声（note:甚至可以只保留最大值？）
        //也可以不使用del_small
        double tmp = MINAREA;   //暂存阈值
        MINAREA = 450.0;
        del_small(dst,dst);
        MINAREA = tmp;          //恢复阈值
    }
}
//
void imSmallHoles(Mat src,Mat &dst)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //cout<<"call imHoles"<<endl;
    //两层轮廓
    findContours(src.clone(), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    //cout<<"call imHoles2"<<endl;

    if( !contours.empty() && !hierarchy.empty() )
    {
        for(int i=contours.size()-1;i>=0;i--)
        {
            //drawContours( dst, contours, i, Scalar(255),1, 8, hierarchy );
            //cout<<hierarchy[i][3]<<endl;
            double area = fabs(contourArea(contours[i]));
            if(hierarchy[i][3]>=0 && area<270.0)
            {
                //cout<<"find !"<<endl;
                drawContours( dst, contours, i, Scalar(255),CV_FILLED, 8, hierarchy );
            }
        }
    }
}

//空洞填充
//note:已弃用，空洞可能是本来就有的，只应去除很小的内部轮廓，见imSmallHoles
//[参考](http://bbs.csdn.net/topics/340140568)
//[另](http://blog.sina.com.cn/s/blog_79bb01d00101btsq.html)
Mat imFillHoles(Mat imInput)
{
    Mat imShow = Mat::zeros(imInput.size(),CV_8UC3);    // for show result

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(imInput, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    if( !contours.empty() && !hierarchy.empty() )
    {
        for (unsigned int idx=0; idx < contours.size(); idx++)
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
//note:弃用，见 del_small()
//[参考](http://www.cnblogs.com/tornadomeet/archive/2012/06/02/2531705.html)
//参数：原始32 img（只需要它的大小） 8位img 目标img
void del_small2(Mat& mask, Mat& dst)
{
    int niters = 1;// default :3

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat temp;

    dilate(mask, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);//腐蚀
    dilate(temp, temp, Mat(), Point(-1,-1), niters);

    findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE );//找轮廓

    dst = Mat::zeros(mask.size(), CV_8UC1);

    if( contours.size() == 0 )
        return;

    int idx = 0, largestComp = 0;
    double maxArea = 0;

    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(c));
        if( area > maxArea )
        {
            maxArea = area;
            largestComp = idx;//找出包含面积最大的轮廓
        }
    }
    Scalar color( 250, 250, 250 );//使用白色来输出
    drawContours( dst, contours, largestComp, color, CV_FILLED, 8, hierarchy );
}


int main(int argc,char **argv)
{
    const bool IS_WRITE_TOFILE = 1;//是否写到文件
    const bool IS_SHOW = 0;//是否显示
    int scn,start_pic;
    if(argc>1) delay_t = atoi(argv[1]);
    else ;
    if(argc>2) scn = atoi(argv[2]);
    else scn = 1;
    if(argc>3) start_pic = atoi(argv[3]);
    else start_pic = 150;

    MyPicStream myb = MyPicStream(SC(scn),1,start_pic,600);
    MyPicStream myf = MyPicStream(SC(scn),2,start_pic,600);

    //ToDo ：使用自己实现的的GMM算法
    BackgroundSubtractorMOG2 bgSubtractor(30,16,true);

    //namedWindow("picgmm",CV_WINDOW_NORMAL);
    //namedWindow("picgmm_depth",CV_WINDOW_NORMAL);
    //namedWindow("src",CV_WINDOW_NORMAL);
    //namedWindow("tar",CV_WINDOW_NORMAL);

    Mat src,src_dep,rgb,dep,rgb_pre,dep_pre,dst;
    src=myb.getPic();
    src_dep=myf.getPic();
    int i=0;                            //图片计数
    const int FIT=12;                    //使用多少张图来适应光照
    int fit = FIT;
    bool is_light=false;
    double v_rgb=0.003,v_rgb_train=0.003,v_dep=0.0025,v_dep_train=0.009;    //学习速度
    while (!src.empty())
    {
        char key = waitKey(i<51?1:delay_t);
        switch(key)
        {
        //空格键暂停
        case ' ':
            waitKey(0);
            break;
        default:
            ;
        }
        cout<<"# to "<<myb.i<<" th pic #"<<endl;
        if(i<TRAIN)             //训练期间
        {
            i++;
            bgSubtractor(src,rgb,v_rgb_train);
            bgSubtractor(src_dep,dep,v_dep_train);

            src=myb.getPic();
            src_dep=myf.getPic();
            continue;
        }
        else if(i>TRAIN)
        {
            bgSubtractor(src,rgb,v_rgb);
            bgSubtractor(src_dep,dep,v_dep);
            if(IS_SHOW)
            {
                imshow("rgb",rgb);              //rgb图结果
                imshow("dep",dep);              //深度图结果
                imshow("src",src);              //rgb原图
            }
            del_small(rgb,rgb);
            //del_small(rgb_pre,rgb_pre);
            del_small(dep,dep);
            //del_small(dep_pre,dep_pre);

            if(!is_light)         //如果不在光照影响期,检查是否光照
            {
                is_light = is_suddenly_light(rgb,rgb_pre,dep,dep_pre);
                if(is_light)
                {
                    v_rgb=0.01;    //加快学习速度
                    cout<<"sudenly light !"<<endl;
                }
            }
            else                  //正在光照影响期
            {
                fit--;
                if(fit==0)        //fit减到零说明适应结束
                {
                    is_light=false;
                    fit=FIT;
                    v_rgb=0.003;   //复原学习速度
                    cout<<"light fit end !"<<endl;
                }
            }

            dst=Mat(src.rows,src.cols,CV_8U,Scalar(0));
            analysis(is_light,rgb,rgb_pre,dep,dep_pre,dst);

            //del_small(dst,dst);//删除小的点
            imSmallHoles(dst,dst);//填补内部小空洞
            if(IS_SHOW)imshow("dst",dst);

            if(IS_WRITE_TOFILE)
            {
                //写入文件，需要先手动建立target文件夹和target/rgb以及target/dep两个子文件夹
                char name[20];
                sprintf(name,"target/T_%d.png",myb.i-1);
                imwrite(name,dst);
                sprintf(name,"target/dep/dep_%d.png",myb.i-1);
                imwrite(name,dep);
                sprintf(name,"target/rgb/rgb_%d.png",myb.i-1);
                imwrite(name,rgb);
            }
        }
        rgb_pre=rgb.clone();
        dep_pre=dep.clone();
        src=myb.getPic();
        src_dep=myf.getPic();
        i++;
    }
    puts("end");
    waitKey();
    return 0;
}

//ToDo :使用滑块来控制速度与其他参数
//ToDo :小的空洞填充(OK)
//ToDo :轮廓重合检测(OK)
//ToDo :深度图噪声过大(?)
//ToDo :生成文件以后再遍历文件，因为不同条件下处理速度不一致，影响观察体验
//Issue:边缘问题，局部光照/阴影
/*

//流程图

                        +----------------------+
                        |                      |
+--------------+        |     调整学习参数     |
|              |        |     只使用深度图     |
| 大面积光照?  +---+--->+                      |
|              |   |  Y |                      |
+--------------+   |    +----------------------+
                   |
                   |
                   |
                   |
                   |    +-----------------+         +------------------+
                   |  N |    寻找DEP_FG   |       Y |                  |
                   +--->+    判断RGB_FG   +----+--->+      判定FG      +
                        |                 |    |    |                  |
                        +-------+---------+    |    +------------------+
                                |              |
                                |              |
                                |              |     +--------------------+       +----------------+
                                |              |   N |                    |     Y |                |
                                |              +---->+ 判断DEP(t-1)聚集点 +------>+   判定FG       |
                                |                    |                    |       |                |
                                |                    +--------------------+       +----------------+
                                |
                                |
                                |
                                |      +-----------------+         +----------------+
                                |      |                 |         |                |
                                |      |   寻找RGB_FG    |      Y  |                |
                                +----->+   判断DEP_FG    +--+----->+      判定FG    +
                                       |                 |  |      |                |
                                       +-----------------+  |      +----------------+
                                                            |
                                                            |
                                                            |     +-----------------+      +-----------------+
                                                            |   N |1、局部光照/阴影 |    1 |                 |
                                                            +-----+                 +---+-->     判定BK      +
                                                                  |2、靠近墙壁      |   |  |                 |
                                                                  +-----------------+   |  +-----------------+
                                                                                        |
                                                                                        |
                                                                                        |  +-----------------+
                                                                                        |  |   判定FG        |
                                                                                        +-->                 |
                                                                                         2 +-----------------+


*/
