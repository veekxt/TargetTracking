#include <iostream>
#include <cstdio>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//----------------- 结合相关-----------------//

//一些常量定义，但是可能要动态修改，仍声明为变量

//最小有效区域，低于此面积将被直接认定为噪声点
double MINAREA = 85.0;
//使用前多少张图像训练？
int TRAIN = 10;
//延时
int delay_t = 1;

enum scene{
    SCE_NORMAL,
    SCE_NOT_LIGHT,
};

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
            Mat tmp;
            tmp.data=NULL;
            return tmp;
        }
        return target;
    };
};

//删除小的噪声点
void del_small(const Mat mask,Mat &dst)
{
    int niters = 1;// default :3

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat temp=mask.clone();

    dilate(temp, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
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
    Scalar color(255);//输出的颜色
    vector<int>::iterator it;
    for(it=all_big_area.begin(); it!=all_big_area.end(); it++)
        drawContours( dst, contours, *it, color,CV_FILLED, 8, hierarchy );
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
bool is_fbg_com_pre_2(int fg_or_bg,int x0,int y0,Mat src)
{
    /*         w2
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
    int tmp=sum[0];
    for(int i=1;i<5;i++)
    {
        if(sum[i]>tmp)
        {
            tmp=sum[i];
        }
    }
    if((double)tmp / s >(fg_or_bg==0?0.7 :0.8))
    {
        //cout<<sum[0]+sum[1]+sum[2]+sum[3]+sum[4]<<"**"<<tmp<<"**"<<tmp_i<<endl;
        return true;
    }
    return false;
}

//分析4个输入图，产生新图
void analysis(bool have_suddenly_light,Mat rgb,Mat rgb_pre,Mat dep,Mat dep_pre,Mat &dst,enum scene S)
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
                if(rgb.at<uchar>(i,j)==255)
                {
                    //如果场景被设置为无光照,将直接可信
                    if(S==SCE_NOT_LIGHT)
                    {
                        dst.at<uchar>(i,j) = 255;
                        continue;
                    }
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
                    if(rgb.at<uchar>(i,j)==255)  //rgb也为前景
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
                                if((distance >= 9.0))
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
                        dst.at<uchar>(i,j)=255;
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

int add_depth_gmm_test(int delay_t,int scn,int start_pic,bool IS_WRITE_TOFILE,bool IS_SHOW,enum scene S,double v_rgb)
{
    MyPicStream myb = MyPicStream(SC(scn),1,start_pic,500);
    MyPicStream myf = MyPicStream(SC(scn),2,start_pic,500);

    //ToDo ：使用自己实现的的GMM算法
    BackgroundSubtractorMOG2 bgSubtractor(10,16,true);
    BackgroundSubtractorMOG2 bgSubtractor_dep(10,16,true);
    //先建立窗口以实现窗口的拖放（CV_WINDOW_NORMAL）
    if(0 && IS_SHOW)
    {
        namedWindow("src",CV_WINDOW_NORMAL);
        namedWindow("rgb",CV_WINDOW_NORMAL);
        namedWindow("dep",CV_WINDOW_NORMAL);
        namedWindow("target",CV_WINDOW_NORMAL);
    }
    Mat src,src_dep,rgb,dep,rgb_pre,dep_pre,dst;
    src=myb.getPic();
    src_dep=myf.getPic();
    int i=0;                            //图片计数
    const int FIT=12;                    //使用多少张图来适应光照
    int fit = FIT;
    bool is_light=false;
    double v_rgb_s=v_rgb;
    double v_rgb_n=v_rgb_s,v_rgb_train=0.003,v_dep=0.001,v_dep_train=0.009;    //学习速度
    while (!src.empty())
    {
        char key = waitKey(i<TRAIN?1:delay_t);
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
            bgSubtractor_dep(src_dep,dep,v_dep_train);

            src=myb.getPic();
            src_dep=myf.getPic();
            continue;
        }
        else if(i>TRAIN)
        {
            bgSubtractor(src,rgb,v_rgb_n);
            bgSubtractor_dep(src_dep,dep,v_dep);

            //阴影直接删除
            threshold(rgb,rgb,200,255,THRESH_BINARY);
            threshold(dep,dep,0,255,THRESH_BINARY);

            if(IS_SHOW)
            {
                imshow("rgb",rgb);              //rgb图结果
                imshow("dep",dep);              //深度图结果
                imshow("src",src);              //rgb原图
            }
            //del_small(rgb,rgb);
            //del_small(rgb_pre,rgb_pre);
            //del_small(dep,dep);
            //del_small(dep_pre,dep_pre);
            if(!is_light)         //如果不在光照影响期,检查是否光照
            {
                is_light = is_suddenly_light(rgb,rgb_pre,dep,dep_pre);
                if(is_light)
                {
                    v_rgb_n=0.01;    //加快学习速度
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
                    v_rgb=v_rgb;   //复原学习速度
                    cout<<"light fit end !"<<endl;
                }
            }

            dst=Mat(src.rows,src.cols,CV_8U,Scalar(0));
            //del_isolated(rgb,dep);
            analysis(is_light,rgb,rgb_pre,dep,dep_pre,dst,S);

            del_small(dst,dst);//删除小的点
            imSmallHoles(dst,dst);//填补内部小空洞
            if(IS_SHOW)imshow("target",dst);

            if(IS_WRITE_TOFILE)
            {
                char name[20];
                sprintf(name,"target%d/T_%d.png",scn,myb.i-1);
                imwrite(name,dst);
                sprintf(name,"target%d/dep/dep_%d.png",scn,myb.i-1);
                imwrite(name,dep);
                sprintf(name,"target%d/rgb/rgb_%d.png",scn,myb.i-1);
                imwrite(name,rgb);
            }
        }
        rgb_pre=rgb.clone();
        dep_pre=dep.clone();
        src=myb.getPic();
        src_dep=myf.getPic();
        i++;
    }
    puts("end one");
    return 0;
}

//----------------编码本相关-----------------//
#define CHANNELS 3
typedef struct ce{
    uchar learnHigh[CHANNELS];
    uchar learnLow[CHANNELS];
    uchar max[CHANNELS];
    uchar min[CHANNELS];
    int t_last_update;
    int stale;
}code_element;

typedef struct code_book{
    code_element **cb;
    int numEntries;
    int t;
}codeBook;

#define CVCONTOUR_APPROX_LEVEL 2   // Approx.threshold - the bigger it is, the simpler is the boundary
#define CVCLOSE_ITR 1                // How many iterations of erosion and/or dialation there should be

#define CV_CVX_WHITE    CV_RGB(0xff,0xff,0xff)
#define CV_CVX_BLACK    CV_RGB(0x00,0x00,0x00)


Mat ipl2mat(IplImage *src)
{
	return Mat(src);
}

IplImage *mat2ipl(Mat src)
{
	IplImage *rr = new IplImage;
	*rr = IplImage(src);
	return rr;
}

///////////////////////////////////////////////////////////////////////////////////
// int updateCodeBook(uchar *p, codeBook &c, unsigned cbBounds)
// Updates the codebook entry with a new data point
//
// p            Pointer to a YUV pixel
// c            Codebook for this pixel
// cbBounds     Learning bounds for codebook (Rule of thumb: 10)
// numChannels  Number of color channels we're learning
//
// NOTES:
//      cvBounds must be of size cvBounds[numChannels]
//
// RETURN
//  codebook index    更新码本

int update_codebook(uchar* p,codeBook &c, unsigned* cbBounds, int numChannels)
{
    if(c.numEntries==0)c.t=0;    //码本中码元数为0初始化时间为0
    c.t+=1;                        //每调用一次时间数加1

    int n;
    unsigned int high[3],low[3];
    for(n=0;n<numChannels;n++)
    {
        //加减cbBonds作为此像素阈值上下界
        high[n]=*(p+n) + *(cbBounds+n);        //直接使用指针操作更快
        if(high[n]>255)    high[n]=255;
        low[n]=*(p+n)-*(cbBounds+n);
        if(low[n]<0)  low[n]=0;
    }

    int matchChannel;
    int i;
    for(i=0;i<c.numEntries;i++)
    {
        matchChannel=0;
        for(n=0;n<numChannels;n++)
        {
            if((c.cb[i]->learnLow[n]<=*(p+n)) && (*(p+n)<=c.cb[i]->learnHigh[n]))
            {
                matchChannel++;
            }
        }
        if(matchChannel==numChannels)
        {
            c.cb[i]->t_last_update=c.t;      //更新码元时间
            for(n=0;n<numChannels;n++)         //调整码元各通道最大最小值
            {
                if(c.cb[i]->max[n]<*(p+n))
                    c.cb[i]->max[n]=*(p+n);
                else if(c.cb[i]->min[n]>*(p+n))
                    c.cb[i]->min[n]=*(p+n);
            }
            break;
        }
    }

    //像素p不满足码本中任何一个码元,创建一个新码元
    if(i == c.numEntries)
    {
        code_element **foo=new code_element*[c.numEntries+1];
        for(int ii=0;ii<c.numEntries;ii++)
            foo[ii]=c.cb[ii];
        foo[c.numEntries]=new code_element;
        if(c.numEntries)delete[]c.cb;
        c.cb=foo;
        for(n=0;n<numChannels;n++)
        {
            c.cb[c.numEntries]->learnHigh[n]=high[n];
            c.cb[c.numEntries]->learnLow[n]=low[n];
            c.cb[c.numEntries]->max[n]=*(p+n);
            c.cb[c.numEntries]->min[n]=*(p+n);
        }
        c.cb[c.numEntries]->t_last_update = c.t;
        c.cb[c.numEntries]->stale = 0;
        c.numEntries += 1;
    }

    //计算码元上次更新到现在的时间
    for(int s=0; s<c.numEntries; s++)
    {
        int negRun=c.t-c.cb[s]->t_last_update;
        if(c.cb[s]->stale < negRun)
            c.cb[s]->stale = negRun;
    }

    //如果像素通道值在高低阈值内,但在码元阈值之外,则缓慢调整此码元学习界限(max,min相当于外墙,粗调;learnHigh,learnLow相当于内墙,细调)
    for(n=0; n<numChannels; n++)
    {
        if(c.cb[i]->learnHigh[n]<high[n])
            c.cb[i]->learnHigh[n]+=1;
        if(c.cb[i]->learnLow[n]>low[n])
            c.cb[i]->learnLow[n]-=1;
    }

    return i;
}

// 删除一定时间内未访问的码元,避免学习噪声的codebook
int cvclearStaleEntries(codeBook &c)
{
    int staleThresh=c.t>>1;                //设定刷新时间
    int *keep=new int[c.numEntries];
    int keepCnt=0;                        //记录不删除码元码元数目
    for(int i=0; i<c.numEntries; i++)
    {
        if(c.cb[i]->stale > staleThresh)
            keep[i]=0;        //保留标志符
        else
        {
            keep[i]=1;        //删除标志符
            keepCnt+=1;
        }
    }

    c.t=0;
    code_element **foo=new code_element*[keepCnt];
    int k=0;
    for(int ii=0; ii<c.numEntries; ii++)
    {
        if(keep[ii])
        {
            foo[k]=c.cb[ii];
            foo[k]->stale=0;    //We have to refresh these entries for next clearStale
            foo[k]->t_last_update=0;
            k++;
        }
    }

    delete[] keep;
    delete[] c.cb;
    c.cb=foo;
    int numCleared=c.numEntries-keepCnt;
    c.numEntries=keepCnt;
    return numCleared;      //返回删除的码元
}

///////////////////////////////////////////////////////////////////////////////////
// uchar cvbackgroundDiff(uchar *p, codeBook &c, int minMod, int maxMod)
// Given a pixel and a code book, determine if the pixel is covered by the codebook
//
// p        pixel pointer (YUV interleaved)
// c        codebook reference
// numChannels  Number of channels we are testing
// maxMod   Add this (possibly negative) number onto max level when code_element determining if new pixel is foreground
// minMod   Subract this (possible negative) number from min level code_element when determining if pixel is foreground
//
// NOTES:
// minMod and maxMod must have length numChannels, e.g. 3 channels => minMod[3], maxMod[3].
//
// Return
// 0 => background, 255 => foreground  背景差分,寻找前景目标
uchar cvbackgroundDiff(uchar *p, codeBook &c, int numChannels, int *minMod, int *maxMod)
{
    int matchChannel;
    int i;
    for(i=0; i<c.numEntries; i++)
    {
        matchChannel=0;
        for(int n=0; n<numChannels; n++)
        {
            if((c.cb[i]->min[n]-minMod[n]<=*(p+n)) && (*(p+n)<=c.cb[i]->max[n]+maxMod[n]))
                matchChannel++;
            else
                break;
        }
        if(matchChannel==numChannels)
            break;
    }
    if(i==c.numEntries)        //像素p各通道值不满足所有的码元,则为前景,返回白色
        return 255;
    return 0;                //匹配到一个码元时,则为背景,返回黑色
}

///////////////////////////////////////////////////////////////////////////////////////////
//void cvconnectedComponents(IplImage *mask, int poly1_hull0, float perimScale, int *num, CvRect *bbs, CvPoint *centers)
// This cleans up the foreground segmentation mask derived from calls to cvbackgroundDiff
//
// mask         Is a grayscale (8 bit depth) "raw" mask image which will be cleaned up
//
// OPTIONAL PARAMETERS:
// poly1_hull0  If set, approximate connected component by (DEFAULT) polygon, or else convex hull (0)
// perimScale   Len = image (width+height)/perimScale.  If contour len < this, delete that contour (DEFAULT: 4)
void cvconnectedComponents(IplImage *mask, int poly1_hull0, float perimScale)
{
    static CvMemStorage* mem_storage=NULL;
    static CvSeq* contours=NULL;
    cvMorphologyEx( mask, mask, NULL, NULL, CV_MOP_OPEN, CVCLOSE_ITR);
    cvMorphologyEx( mask, mask, NULL, NULL, CV_MOP_CLOSE, CVCLOSE_ITR);

    if(mem_storage==NULL)
        mem_storage=cvCreateMemStorage(0);
    else
        cvClearMemStorage(mem_storage);

    CvContourScanner scanner=cvStartFindContours(mask,mem_storage,sizeof(CvContour),CV_RETR_EXTERNAL);
    CvSeq* c;
    int numCont=0;                //轮廓数
    while((c=cvFindNextContour(scanner))!=NULL)
    {
        double len=cvContourPerimeter(c);
        double q=(mask->height+mask->width)/perimScale;   //轮廓长度阀值设定
        if(len<q)
            cvSubstituteContour(scanner,NULL);           //删除太短轮廓
        else
        {
            CvSeq* c_new;
            if(poly1_hull0)                                  //用多边形拟合轮廓
                c_new = cvApproxPoly(c, sizeof(CvContour), mem_storage,
                    CV_POLY_APPROX_DP, CVCONTOUR_APPROX_LEVEL);
            else                                        //计算轮廓Hu矩
                c_new = cvConvexHull2(c,mem_storage, CV_CLOCKWISE, 1);

            cvSubstituteContour(scanner,c_new);            //替换拟合后的多边形轮廓
            numCont++;
        }
    }
    contours = cvEndFindContours(&scanner);  //结束扫描,并返回最高层的第一个轮廓指针

    cvZero(mask);
    for(c=contours; c!=NULL; c=c->h_next)
        cvDrawContours(mask,c,CV_CVX_WHITE, CV_CVX_BLACK,-1,CV_FILLED,8);
}

int main()
{
    ///////////////////////////////////////
    // 需要使用的变量
    //CvCapture* capture=NULL;
    IplImage*  rawImage=NULL;            //视频的每一帧原图像
    IplImage*  rawImageDep=NULL;
    IplImage*  yuvImage=NULL;            //比经验角度看绝大部分背景中的变化倾向于沿亮度轴,而不是颜色轴,故YUV颜色空间效果更好
    IplImage*  yuvImageDep=NULL;
    IplImage* ImaskCodeBook=NULL;        //掩模图像
    IplImage* ImaskCodeBookDep=NULL;
    IplImage* ImaskCodeBook_pre=NULL;        //掩模图像
    IplImage* ImaskCodeBookDep_pre=NULL;
    IplImage* ImaskCodeBookCC=NULL;        //清除噪声后并采用多边形法拟合轮廓连通域的掩模图像
    //IplImage* ImaskCodeBookCCDep=NULL;

    codeBook* cB=NULL;
    codeBook* cB_dep=NULL;
    unsigned cbBounds[CHANNELS];
    unsigned cbBoundsDep[CHANNELS];
    uchar* pColor=NULL;                    //yuvImage像素指针
    uchar* pColorDep=NULL;
    int imageLen=0;
    int nChannels=CHANNELS;
    int minMod[CHANNELS];
    int maxMod[CHANNELS];

    //////////////////////////////////////////////////////////////////////////
    // 初始化各变量
    //cvNamedWindow("原图");
    //cvNamedWindow("掩模图像");
    //cvNamedWindow("连通域掩模图像");

    //capture = cvCreateFileCapture("C:/Users/shark/Desktop/eagle.flv");
    //capture=cvCreateCameraCapture(0);
    //CvCapture* capture = cvCaptureFromFile("test002.3gp");
    //if(!capture)
    //{
    //    printf("Couldn't open the capture!");
    //    return -1;
    //}

    //rawImage=cvQueryFrame(capture);
    //int width=(int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH);
    //int height=(int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT);
    int scn=1;
	MyPicStream mypic = MyPicStream(SC(scn),1,200,450);
	MyPicStream mypic_dep = MyPicStream(SC(scn),2,200,450);
	//MyPicStream mypic = MyPicStream("Hallway",1,200,450);
	//MyPicStream mypic = MyPicStream("Shelves",1,200,450);
	//MyPicStream mypic = MyPicStream("Wall",1,200,450);

    Mat rgb = mypic.getPic();
    Mat dep = mypic_dep.getPic();
    rawImage = mat2ipl(rgb);
    rawImageDep = mat2ipl(dep);
    int width=rawImage->width;
    int height=rawImage->height;


    CvSize size=cvSize(width,height);
    yuvImage=cvCreateImage(size,8,3);
    yuvImageDep=cvCreateImage(size,8,3);
    ImaskCodeBook = cvCreateImage(size, IPL_DEPTH_8U, 1);
    ImaskCodeBookDep = cvCreateImage(size, IPL_DEPTH_8U, 1);
    ImaskCodeBookCC = cvCreateImage(size, IPL_DEPTH_8U, 1);
    cvSet(ImaskCodeBook,cvScalar(255));
    cvSet(ImaskCodeBookDep,cvScalar(255));

    imageLen=width*height;
    cB=new codeBook[imageLen];    //得到与图像像素数目长度一样的一组码本,以便对每个像素进行处理
    cB_dep=new codeBook[imageLen];
    for(int i=0;i<imageLen;i++)
    {
        cB[i].numEntries=0;
        cB_dep[i].numEntries=0;
    }

    for(int i=0;i<nChannels;i++)
    {
        cbBounds[i]=10;
        cbBoundsDep[i]=10;
        minMod[i]=20;        //用于背景差分函数中
        maxMod[i]=20;        //调整其值以达到最好的分割
    }

    //////////////////////////////////////////////////////////////////////////
    // 开始处理视频每一帧图像

    for(int i=0;;i++)
    {
        //if(!(rawImage=cvQueryFrame(capture)))
        Mat rgb = mypic.getPic();
        Mat dep = mypic_dep.getPic();
        if(rgb.empty())break;
        rawImage = mat2ipl(rgb);
        if(dep.empty())break;
        rawImageDep = mat2ipl(dep);

        cvCvtColor(rawImage, yuvImage, CV_BGR2YCrCb);
        cvCvtColor(rawImageDep, yuvImageDep, CV_BGR2YCrCb);
        // 色彩空间转换,将rawImage 转换到YUV色彩空间,输出到yuvImage
        // 即使不转换效果依然很好
        // yuvImage = cvCloneImage(rawImage);

        if(i<=30)            //前30帧进行背景学习
        {
            pColor=(uchar*)yuvImage->imageData;
            pColorDep=(uchar*)yuvImageDep->imageData;
            for(int c=0; c<imageLen; c++)
            {
                update_codebook(pColor, cB[c], cbBounds, nChannels);   //对每个像素调用此函数
                update_codebook(pColorDep, cB_dep[c], cbBoundsDep, nChannels);
                pColor+=3;
                pColorDep+=3;
            }
            if(i==30)
            {
                for(int c=0;c<imageLen;c++)
                {
                    cvclearStaleEntries(cB_dep[c]);
                    cvclearStaleEntries(cB[c]);            //第30时帧时,删除每个像素码本中陈旧的码元
                }
            }
        }
        else
        {
            uchar maskPixel;
            uchar maskPixelDep;
            pColor=(uchar*)yuvImage->imageData;
            pColorDep=(uchar*)yuvImageDep->imageData;
            uchar* pMask=(uchar*)ImaskCodeBook->imageData;
            uchar* pMaskDep=(uchar*)ImaskCodeBookDep->imageData;
            for(int c=0;c<imageLen;c++)
            {
                maskPixel=cvbackgroundDiff(pColor,cB[c],nChannels,minMod,maxMod);
                maskPixelDep=cvbackgroundDiff(pColorDep,cB_dep[c],nChannels,minMod,maxMod);
                *pMask++ = maskPixel;
                *pMaskDep++ = maskPixelDep;
                pColor+=3;
                pColorDep+=3;
            }
            if(i>31)
            {
                cvShowImage("rgb_cb",ImaskCodeBook);
                cvShowImage("dep_cb",ImaskCodeBookDep);

                Mat dst = Mat(rawImage->height,rawImage->width,CV_8U,Scalar(0));
                Mat rgb = ipl2mat(ImaskCodeBook);
                Mat dep = ipl2mat(ImaskCodeBookDep);
                Mat rgb_pre = ipl2mat(ImaskCodeBook_pre);
                Mat dep_pre = ipl2mat(ImaskCodeBookDep_pre);
                analysis(0,rgb,rgb_pre,dep,dep_pre,dst,SCE_NORMAL);
                del_small(dst,dst);
                imSmallHoles(dst,dst);
                //imshow("tar",dst);

                char name[20];
                sprintf(name,"target%d/T_%d.png",scn,mypic.i-1);
                imwrite(name,dst);
                sprintf(name,"target%d/dep/dep_%d.png",scn,mypic.i-1);
                imwrite(name,dep);
                sprintf(name,"target%d/rgb/rgb_%d.png",scn,mypic.i-1);
                imwrite(name,rgb);

            }
            ImaskCodeBook_pre=ImaskCodeBook;
            ImaskCodeBookDep_pre=ImaskCodeBookDep;

        }
        cvShowImage("src",rawImage);
        int key = cvWaitKey(80);
        switch(key)
        {
        //空格键暂停
        case ' ':
            waitKey(0);
            break;
        default:
            ;
        }
    }

    //cvReleaseCapture(&capture);
    if (yuvImage)
        cvReleaseImage(&yuvImage);
    if(ImaskCodeBook)
        cvReleaseImage(&ImaskCodeBook);
    cvDestroyAllWindows();
    delete [] cB;

    return 0;

}
