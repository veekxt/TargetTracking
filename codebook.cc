#include "cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

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

//��ȡ·���͸�ʽ�й��ɵ��ز�
class MyPicStream{
    /*������ʾһ��ͼ��,����ģ��
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
    int start_num;//��ʼ�ͽ���ͼƬ��
    int end_num;
    int pic_type;//ͼƬ����,1��ͨrgb ��2���
    const char *head;//�ļ���ǰ׺
    int i;//��ǰͼƬ��
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

	MyPicStream mypic = MyPicStream("ChairBox",1,200,450);
	MyPicStream mypic_dep = MyPicStream("ChairBox",2,200,450);
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
            //cvCopy(ImaskCodeBook,ImaskCodeBookCC);
            //cvconnectedComponents(ImaskCodeBookCC,1,4.0);
            cvShowImage("rgb_cb",ImaskCodeBook);
            cvShowImage("dep_cb",ImaskCodeBookDep);
            //cvShowImage("codebook2",ImaskCodeBookCC);
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
