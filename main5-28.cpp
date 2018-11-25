#include <iostream>
#include <cstdio>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

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

void my_mkdir(int n){
    //��windows����
    char name[50];
    sprintf(name,"mkdir target%d",n);
    system(name);
    sprintf(name,"mkdir target%d\\rgb",n);
    system(name);
    sprintf(name,"mkdir target%d\\dep",n);
    system(name);
}

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

//��ȡ·���͸�ʽ�й��ɵ��ز�
class MyDemo
 {
 private:
 public:
     int start_num;      //��ʼ�ͽ���ͼƬ��
     int end_num;
     int pic_type;       //ͼƬ����,1��ͨrgb ��2���
     int number;
     int i;              //��ǰͼƬ��
     MyDemo(int a_number,int a_start_num,int a_end_num)
     {
         number=a_number;
         i=a_start_num;
         start_num=a_start_num;
         end_num=a_end_num;
     };

     Mat getPic(int pic_type)
     {
         //i++;
         if(i>end_num)
         {
             Mat tmp;
             tmp.data=NULL;
             return tmp;
         }
         char nameBuf[50];
         sprintf(nameBuf,"target%d%s/%s_%d.png",
                 number,pic_type==2?"/rgb":(pic_type==3?"/dep":""),pic_type==2?"rgb":(pic_type==3?"dep":"T"),i);
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
    /**        w2
            +--------+---------+
            |        |         |
            |     w1 |         |
            |    +---+---+     |
            |    |       |     |
            +----+   s   +-----+
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
                                if((distance >= 9.8))
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
//��������ʱ���ڼ����زģ���ʼ֡��д���ļ�������ʾ��������rgbѧϰ�ٶ�
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
                    v_rgb_n=v_rgb;   //��ԭѧϰ�ٶ�
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

void demo(int n,int s,int e,int delay)
 {
     MyDemo m = MyDemo(n,s,e);
     MyPicStream mypic = MyPicStream(SC(n),1,s,e);
     for(int i=m.start_num;i<=m.end_num;i++){
         imshow("target",m.getPic(1));
         cout<<"to "<<i<<" pic"<<endl;
         imshow("rgb",m.getPic(2));
         imshow("dep",m.getPic(3));
         Mat src = mypic.getPic();
         if(!src.empty())imshow("src",src);
         m.i++;
         switch(waitKey(delay))
         {
         //�ո����ͣ
         case ' ':
             waitKey(0);
         default:
             ;
         }
     }
 }

int main(int argc,char **argv)
{
    //�Ƿ�д���ļ�
    //��Ҫ���ֶ�����target<n>�ļ��к�target<n>/rgb�Լ�target<n>/dep�������ļ���
    //�����<n>�滻���زı�ű���target1
    const bool IS_WRITE_TOFILE = 1;
    //�Ƿ���ʾ
    const bool IS_SHOW = 0;
    bool is_demo = 1;
    int scn,start_pic;
    /**
    * 3�������в���
    * ��ʱ���ڼ����زģ��ӵڼ�֡��ʼ
    */
    if(argc>1) delay_t = atoi(argv[1]);
    else ;
    if(argc>2) scn = atoi(argv[2]);
    else scn = 1;
    if(argc>3) start_pic = atoi(argv[3]);
    else start_pic = 150;

    //add_depth_gmm_test(delay_t,scn,start_pic,IS_WRITE_TOFILE,IS_SHOW,SCE_NOT_LIGHT);
    //add_depth_gmm_test(1,1,1,1,1,SCE_NORMAL,0.003);
    /*
    * ��4���زĵ�ͼƬ�������浽�ļ���
    */
    //��������ʱ���ڼ����زģ���ʼ֡��д���ļ�������ʾ��������rgbѧϰ�ٶ�
    if(is_demo)
    {
        demo(1,200,500,70);
        return 0;
    }
    my_mkdir(1);
    my_mkdir(2);
    my_mkdir(3);
    my_mkdir(4);
    add_depth_gmm_test(1,1,1,1,0,SCE_NORMAL,0.002);
    add_depth_gmm_test(1,2,1,1,0,SCE_NORMAL,0.003);
    add_depth_gmm_test(1,3,1,1,0,SCE_NORMAL,0.003);
    add_depth_gmm_test(1,4,1,1,0,SCE_NOT_LIGHT,0.001);



    waitKey();
    return 0;
}

//ToDo :ʹ�û����������ٶ�����������
//ToDo :С�Ŀն����(OK)
//ToDo :�����غϼ��(OK)
//ToDo :���ͼ��������(?)
//ToDo :�����ļ��Ժ��ٱ����ļ�����Ϊ��ͬ�����´����ٶȲ�һ�£�Ӱ��۲�����
//Issue:��Ե���⣬�ֲ�����/��Ӱ
/*

//����ͼ

                        +----------------------+
                        |                      |
+--------------+        |     ����ѧϰ����     |
|              |        |     ֻʹ�����ͼ     |
| ���������?  +---+--->+                      |
|              |   |  Y |                      |
+--------------+   |    +----------------------+
                   |
                   |
                   |
                   |
                   |    +-----------------+         +------------------+
                   |  N |    Ѱ��DEP_FG   |       Y |                  |
                   +--->+    �ж�RGB_FG   +----+--->+      �ж�FG      +
                        |                 |    |    |                  |
                        +-------+---------+    |    +------------------+
                                |              |
                                |              |
                                |              |     +--------------------+       +----------------+
                                |              |   N |                    |     Y |                |
                                |              +---->+ �ж�DEP(t-1)�ۼ��� +------>+   �ж�FG       |
                                |                    |                    |       |                |
                                |                    +--------------------+       +----------------+
                                |
                                |
                                |
                                |      +-----------------+         +----------------+
                                |      |                 |         |                |
                                |      |   Ѱ��RGB_FG    |      Y  |                |
                                +----->+   �ж�DEP_FG    +--+----->+      �ж�FG    +
                                       |                 |  |      |                |
                                       +-----------------+  |      +----------------+
                                                            |
                                                            |
                                                            |     +-----------------+      +-----------------+
                                                            |   N |1���ֲ�����/��Ӱ |    1 |                 |
                                                            +-----+                 +---+-->     �ж�BK      +
                                                                  |2������ǽ��      |   |  |                 |
                                                                  +-----------------+   |  +-----------------+
                                                                                        |
                                                                                        |
                                                                                        |  +-----------------+
                                                                                        |  |   �ж�FG        |
                                                                                        +-->                 |
                                                                                         2 +-----------------+


*/
