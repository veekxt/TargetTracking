#include <iostream>
#include <cstdio>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//----------------- ������-----------------//

//һЩ�������壬���ǿ���Ҫ��̬�޸ģ�������Ϊ����

//��С��Ч���򣬵��ڴ��������ֱ���϶�Ϊ������
double MINAREA = 85.0;
//ʹ��ǰ������ͼ��ѵ����
int TRAIN = 10;
//��ʱ
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

//��ȡ·���͸�ʽ�й��ɵ��ز�
class MyPicStream
{
private:
public:
    int start_num;      //��ʼ�ͽ���ͼƬ��
    int end_num;
    int pic_type;       //ͼƬ����,1��ͨrgb ��2���
    const char *head;   //�ļ���ǰ׺
    int i;              //��ǰͼƬ��
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

//ɾ��С��������
void del_small(const Mat mask,Mat &dst)
{
    int niters = 1;// default :3

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    Mat temp=mask.clone();

    dilate(temp, temp, Mat(), Point(-1,-1), niters);//���ͣ�3*3��element����������Ϊniters
    erode(temp, temp, Mat(), Point(-1,-1), niters*2);//��ʴ
    dilate(temp, temp, Mat(), Point(-1,-1), niters);

    findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE );//������

    dst = Mat::zeros(mask.size(), CV_8UC1);

    if( contours.size() == 0 )
        return;

    int idx = 0 ;

    //ʹ�����������������������
    vector<int> all_big_area;

    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        const vector<Point>& c = contours[idx];
        double area = fabs(contourArea(c));
        if( area > MINAREA )
        {
            all_big_area.push_back(idx);//��ӵ�������
        }
    }
    Scalar color(255);//�������ɫ
    vector<int>::iterator it;
    for(it=all_big_area.begin(); it!=all_big_area.end(); it++)
        drawContours( dst, contours, *it, color,CV_FILLED, 8, hierarchy );
}

//��һ����ֵͼ��ǰ�����,�����жϴ��������
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

//�ж��Ƿ���ͻȻ�Ĺ���
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

    if(pro_rgb >0.5 && pro_rgb - pro_dep > 0.3)//ʵ��ó�?
    {
        cout<<pro_rgb<<"*"<<pro_rgb - pro_dep<<endl;
        return true;
    }
    return false;
}
//�ж�fg��bg�Ŀɿ���
//���� ��fg����bg��0��1�������꣬src
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
            //todo:�Ż�����Ӧ�ÿ��Լ����жϴ���
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

//����4������ͼ��������ͼ
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
                    //�������������Ϊ�޹���,��ֱ�ӿ���
                    if(S==SCE_NOT_LIGHT)
                    {
                        dst.at<uchar>(i,j) = 255;
                        continue;
                    }
                    if(dep.at<uchar>(i,j)>100)dst.at<uchar>(i,j)=255; //ȷ��Ϊǰ��
                    else
                    {
                        //�����Ǿֲ��Ĺ��պ���Ӱ
                        if(!is_fbg_com_pre_2(1,i,j,dep_pre))
                        {
                            //������ͼ�ж�Ϊ���������ɿ���
                            dst.at<uchar>(i,j) = 255;
                        }else
                        {
                            //������ͼ�ж�Ϊ�������ɿ�����Ӧ���Ǿֲ�����
                            //dst.at<uchar>(i,j) = 0;//Ĭ����0������Ҫ��ʽ��ֵ
                        }
                    }
                }
                if(dep.at<uchar>(i,j)>100)      //depΪǰ��
                {
                    if(rgb.at<uchar>(i,j)==255)  //rgbҲΪǰ��
                    {
                        dst.at<uchar>(i,j)=255; //ȷ��Ϊǰ��
                    }
                    else                        //rgbΪ��������һ�����
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
        //ʹ��is_fg_com_pre�ж�����

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
        //update���ƺ�Ҳ����ֱ��is_fg_com_pre�жϣ���is_fg_com_pre�ƺ���bug��
        //ͻȻ����ʹ��rgb���û�вο���ֵ�����﴿��ʹ����Ƚ��
        //���ͼ�ܶ�����ֻ��һ˲�䣬��˶���һ��������˵��
        //ǰһ֡�͵�ǰ֡������Ӧ���н�����û�н��������ж�Ϊ����

        //�ҵ�dep���õ�����
        struct ValidContours v = getValidContours(dep,dep_pre);
        //����������
        for(int i=v.valids.size()-1;i>=0;i--)
        {
            drawContours(dst,v.contours,v.valids[i],Scalar(255),CV_FILLED,8,v.hierarchy);
        }
        */
        //����Ƚ������̫�������������del_small��Ŀ���С���޼���һЩƫС������note:��������ֻ�������ֵ����
        //Ҳ���Բ�ʹ��del_small
        double tmp = MINAREA;   //�ݴ���ֵ
        MINAREA = 450.0;
        del_small(dst,dst);
        MINAREA = tmp;          //�ָ���ֵ
    }
}

void imSmallHoles(Mat src,Mat &dst)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //cout<<"call imHoles"<<endl;
    //��������
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

    //ToDo ��ʹ���Լ�ʵ�ֵĵ�GMM�㷨
    BackgroundSubtractorMOG2 bgSubtractor(10,16,true);
    BackgroundSubtractorMOG2 bgSubtractor_dep(10,16,true);
    //�Ƚ���������ʵ�ִ��ڵ��Ϸţ�CV_WINDOW_NORMAL��
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
    int i=0;                            //ͼƬ����
    const int FIT=12;                    //ʹ�ö�����ͼ����Ӧ����
    int fit = FIT;
    bool is_light=false;
    double v_rgb_s=v_rgb;
    double v_rgb_n=v_rgb_s,v_rgb_train=0.003,v_dep=0.001,v_dep_train=0.009;    //ѧϰ�ٶ�
    while (!src.empty())
    {
        char key = waitKey(i<TRAIN?1:delay_t);
        switch(key)
        {
        //�ո����ͣ
        case ' ':
            waitKey(0);
            break;
        default:
            ;
        }
        cout<<"# to "<<myb.i<<" th pic #"<<endl;
        if(i<TRAIN)             //ѵ���ڼ�
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

            //��Ӱֱ��ɾ��
            threshold(rgb,rgb,200,255,THRESH_BINARY);
            threshold(dep,dep,0,255,THRESH_BINARY);

            if(IS_SHOW)
            {
                imshow("rgb",rgb);              //rgbͼ���
                imshow("dep",dep);              //���ͼ���
                imshow("src",src);              //rgbԭͼ
            }
            //del_small(rgb,rgb);
            //del_small(rgb_pre,rgb_pre);
            //del_small(dep,dep);
            //del_small(dep_pre,dep_pre);
            if(!is_light)         //������ڹ���Ӱ����,����Ƿ����
            {
                is_light = is_suddenly_light(rgb,rgb_pre,dep,dep_pre);
                if(is_light)
                {
                    v_rgb_n=0.01;    //�ӿ�ѧϰ�ٶ�
                    cout<<"sudenly light !"<<endl;
                }
            }
            else                  //���ڹ���Ӱ����
            {
                fit--;
                if(fit==0)        //fit������˵����Ӧ����
                {
                    is_light=false;
                    fit=FIT;
                    v_rgb=v_rgb;   //��ԭѧϰ�ٶ�
                    cout<<"light fit end !"<<endl;
                }
            }

            dst=Mat(src.rows,src.cols,CV_8U,Scalar(0));
            //del_isolated(rgb,dep);
            analysis(is_light,rgb,rgb_pre,dep,dep_pre,dst,S);

            del_small(dst,dst);//ɾ��С�ĵ�
            imSmallHoles(dst,dst);//��ڲ�С�ն�
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

//----------------���뱾���-----------------//
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
//  codebook index    �����뱾

int update_codebook(uchar* p,codeBook &c, unsigned* cbBounds, int numChannels)
{
    if(c.numEntries==0)c.t=0;    //�뱾����Ԫ��Ϊ0��ʼ��ʱ��Ϊ0
    c.t+=1;                        //ÿ����һ��ʱ������1

    int n;
    unsigned int high[3],low[3];
    for(n=0;n<numChannels;n++)
    {
        //�Ӽ�cbBonds��Ϊ��������ֵ���½�
        high[n]=*(p+n) + *(cbBounds+n);        //ֱ��ʹ��ָ���������
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
            c.cb[i]->t_last_update=c.t;      //������Ԫʱ��
            for(n=0;n<numChannels;n++)         //������Ԫ��ͨ�������Сֵ
            {
                if(c.cb[i]->max[n]<*(p+n))
                    c.cb[i]->max[n]=*(p+n);
                else if(c.cb[i]->min[n]>*(p+n))
                    c.cb[i]->min[n]=*(p+n);
            }
            break;
        }
    }

    //����p�������뱾���κ�һ����Ԫ,����һ������Ԫ
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

    //������Ԫ�ϴθ��µ����ڵ�ʱ��
    for(int s=0; s<c.numEntries; s++)
    {
        int negRun=c.t-c.cb[s]->t_last_update;
        if(c.cb[s]->stale < negRun)
            c.cb[s]->stale = negRun;
    }

    //�������ͨ��ֵ�ڸߵ���ֵ��,������Ԫ��ֵ֮��,������������Ԫѧϰ����(max,min�൱����ǽ,�ֵ�;learnHigh,learnLow�൱����ǽ,ϸ��)
    for(n=0; n<numChannels; n++)
    {
        if(c.cb[i]->learnHigh[n]<high[n])
            c.cb[i]->learnHigh[n]+=1;
        if(c.cb[i]->learnLow[n]>low[n])
            c.cb[i]->learnLow[n]-=1;
    }

    return i;
}

// ɾ��һ��ʱ����δ���ʵ���Ԫ,����ѧϰ������codebook
int cvclearStaleEntries(codeBook &c)
{
    int staleThresh=c.t>>1;                //�趨ˢ��ʱ��
    int *keep=new int[c.numEntries];
    int keepCnt=0;                        //��¼��ɾ����Ԫ��Ԫ��Ŀ
    for(int i=0; i<c.numEntries; i++)
    {
        if(c.cb[i]->stale > staleThresh)
            keep[i]=0;        //������־��
        else
        {
            keep[i]=1;        //ɾ����־��
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
    return numCleared;      //����ɾ������Ԫ
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
// 0 => background, 255 => foreground  �������,Ѱ��ǰ��Ŀ��
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
    if(i==c.numEntries)        //����p��ͨ��ֵ���������е���Ԫ,��Ϊǰ��,���ذ�ɫ
        return 255;
    return 0;                //ƥ�䵽һ����Ԫʱ,��Ϊ����,���غ�ɫ
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
    int numCont=0;                //������
    while((c=cvFindNextContour(scanner))!=NULL)
    {
        double len=cvContourPerimeter(c);
        double q=(mask->height+mask->width)/perimScale;   //�������ȷ�ֵ�趨
        if(len<q)
            cvSubstituteContour(scanner,NULL);           //ɾ��̫������
        else
        {
            CvSeq* c_new;
            if(poly1_hull0)                                  //�ö�����������
                c_new = cvApproxPoly(c, sizeof(CvContour), mem_storage,
                    CV_POLY_APPROX_DP, CVCONTOUR_APPROX_LEVEL);
            else                                        //��������Hu��
                c_new = cvConvexHull2(c,mem_storage, CV_CLOCKWISE, 1);

            cvSubstituteContour(scanner,c_new);            //�滻��Ϻ�Ķ��������
            numCont++;
        }
    }
    contours = cvEndFindContours(&scanner);  //����ɨ��,��������߲�ĵ�һ������ָ��

    cvZero(mask);
    for(c=contours; c!=NULL; c=c->h_next)
        cvDrawContours(mask,c,CV_CVX_WHITE, CV_CVX_BLACK,-1,CV_FILLED,8);
}

int main()
{
    ///////////////////////////////////////
    // ��Ҫʹ�õı���
    //CvCapture* capture=NULL;
    IplImage*  rawImage=NULL;            //��Ƶ��ÿһ֡ԭͼ��
    IplImage*  rawImageDep=NULL;
    IplImage*  yuvImage=NULL;            //�Ⱦ���Ƕȿ����󲿷ֱ����еı仯��������������,��������ɫ��,��YUV��ɫ�ռ�Ч������
    IplImage*  yuvImageDep=NULL;
    IplImage* ImaskCodeBook=NULL;        //��ģͼ��
    IplImage* ImaskCodeBookDep=NULL;
    IplImage* ImaskCodeBook_pre=NULL;        //��ģͼ��
    IplImage* ImaskCodeBookDep_pre=NULL;
    IplImage* ImaskCodeBookCC=NULL;        //��������󲢲��ö���η����������ͨ�����ģͼ��
    //IplImage* ImaskCodeBookCCDep=NULL;

    codeBook* cB=NULL;
    codeBook* cB_dep=NULL;
    unsigned cbBounds[CHANNELS];
    unsigned cbBoundsDep[CHANNELS];
    uchar* pColor=NULL;                    //yuvImage����ָ��
    uchar* pColorDep=NULL;
    int imageLen=0;
    int nChannels=CHANNELS;
    int minMod[CHANNELS];
    int maxMod[CHANNELS];

    //////////////////////////////////////////////////////////////////////////
    // ��ʼ��������
    //cvNamedWindow("ԭͼ");
    //cvNamedWindow("��ģͼ��");
    //cvNamedWindow("��ͨ����ģͼ��");

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
    cB=new codeBook[imageLen];    //�õ���ͼ��������Ŀ����һ����һ���뱾,�Ա��ÿ�����ؽ��д���
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
        minMod[i]=20;        //���ڱ�����ֺ�����
        maxMod[i]=20;        //������ֵ�Դﵽ��õķָ�
    }

    //////////////////////////////////////////////////////////////////////////
    // ��ʼ������Ƶÿһ֡ͼ��

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
        // ɫ�ʿռ�ת��,��rawImage ת����YUVɫ�ʿռ�,�����yuvImage
        // ��ʹ��ת��Ч����Ȼ�ܺ�
        // yuvImage = cvCloneImage(rawImage);

        if(i<=30)            //ǰ30֡���б���ѧϰ
        {
            pColor=(uchar*)yuvImage->imageData;
            pColorDep=(uchar*)yuvImageDep->imageData;
            for(int c=0; c<imageLen; c++)
            {
                update_codebook(pColor, cB[c], cbBounds, nChannels);   //��ÿ�����ص��ô˺���
                update_codebook(pColorDep, cB_dep[c], cbBoundsDep, nChannels);
                pColor+=3;
                pColorDep+=3;
            }
            if(i==30)
            {
                for(int c=0;c<imageLen;c++)
                {
                    cvclearStaleEntries(cB_dep[c]);
                    cvclearStaleEntries(cB[c]);            //��30ʱ֡ʱ,ɾ��ÿ�������뱾�г¾ɵ���Ԫ
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
        //�ո����ͣ
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
