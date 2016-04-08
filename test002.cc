 #include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
 #include <opencv2/core/core.hpp>
 #include <stdio.h>
#include <iostream>
#include "cv.hpp"
#include "cxcore.hpp"
using namespace cv;
using namespace std;



//行人检测函数
 HOGDescriptor people_detect_hog; //HOG特征检测器 
void  Found_people(IplImage *get_imge,CvRect rect)
{
	
	  vector<Rect> found, found_filtered; //矩形框数组  
       //对输入的图片进行多尺度行人检测，检测窗口移动步长为(8,8)  
	  people_detect_hog.detectMultiScale(get_imge, found, 0, Size(8, 8), Size(64, 128), 1.05, 2);  
        //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中  
	  cout<<found.size()<<endl;
        for(int i=0; i < found.size(); i++)  
        { 
            Rect r = found[i];  
			cout<<r.x<<" "<<r.y<<endl;
            int j=0;  
            for(; j < found.size(); j++)  
                if(j != i && (r & found[j]) == r)  
                    break;  
            if( j == found.size())  
                found_filtered.push_back(r);  
        }  
        //画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整  
       for(int i=0; i<found_filtered.size(); i++)  
        {  
			//cout<<"进入"<<endl;
            Rect r = found_filtered[i];  
			
		    //X方向位置修订，y方向无需修订
			r.x=rect.x-20;
            
            r.width = cvRound(r.width*0.8);  
       
            r.height = cvRound(r.height*0.8);  
           // cvRectangle(, r.tl(), r.br(), Scalar(0,255,0), 1);  
        }    
}
  
int main(int argc, char* argv[])  
{  
	
	cvNamedWindow("back",CV_WINDOW_KEEPRATIO);  
    cvNamedWindow("fore",CV_WINDOW_KEEPRATIO); 
	cvNamedWindow("原图",CV_WINDOW_KEEPRATIO);
	
    CvCapture *capture=NULL;//
	capture=cvCreateFileCapture("test002.3gp"); 
	//capture=cvCreateCameraCapture(0);
    IplImage *mframe,*current,*frg,*test;  
    int *fg,*bg_bw,*rank_ind;  
    double *w,*mean,*sd,*u_diff,*rank;  
    int C,M,sd_init,i,j,k,m,rand_temp=0,rank_ind_temp=0,min_index=0,x=0,y=0,counter_frame=0;  
    double D,alph,thresh,p,temp;  
    CvRNG state;  
    int match,height,width;  
    mframe=cvQueryFrame(capture);  
  
    frg = cvCreateImage(cvSize(mframe->width,mframe->height),IPL_DEPTH_8U,1);  
    current = cvCreateImage(cvSize(mframe->width,mframe->height),IPL_DEPTH_8U,1);  
    test = cvCreateImage(cvSize(mframe->width,mframe->height),IPL_DEPTH_8U,1);  
      
    C = 4;                      //number of gaussian components (typically 3-5)  
    M = 4;                      //number of background components  
    sd_init = 6;                //initial standard deviation (for new components) var = 36 in paper  
    alph = 0.01;                //learning rate (between 0 and 1) (from paper 0.01)  
    D = 2.5;                    //positive deviation threshold  
    thresh = 0.25;              //foreground threshold (0.25 or 0.75 in paper)  
    p = alph/(1/C);         //initial p variable (used to update mean and sd)  
  
    height=current->height;width=current->widthStep;  
      
    fg = (int *)malloc(sizeof(int)*width*height);                   //foreground array  
    bg_bw = (int *)malloc(sizeof(int)*width*height);                //background array  
    rank = (double *)malloc(sizeof(double)*1*C);                    //rank of components (w/sd)  
    w = (double *)malloc(sizeof(double)*width*height*C);            //weights array  
    mean = (double *)malloc(sizeof(double)*width*height*C);         //pixel means  
    sd = (double *)malloc(sizeof(double)*width*height*C);           //pixel standard deviations  
    u_diff = (double *)malloc(sizeof(double)*width*height*C);       //difference of each pixel from mean  
      
    for (i=0;i<height;i++)  
    {  
        for (j=0;j<width;j++)  
        {  
            for(k=0;k<C;k++)  
            {  
                mean[i*width*C+j*C+k] = cvRandReal(&state)*255;  
                w[i*width*C+j*C+k] = (double)1/C;  
                sd[i*width*C+j*C+k] = sd_init;  
            }  
        }  
    }  
  
    while(1){  
        rank_ind = (int *)malloc(sizeof(int)*C);  
        cvCvtColor(mframe,current,CV_BGR2GRAY);  
        // calculate difference of pixel values from mean  
        for (i=0;i<height;i++)  
        {  
            for (j=0;j<width;j++)  
            {  
                for (m=0;m<C;m++)  
                {  
                    u_diff[i*width*C+j*C+m] = abs((uchar)current->imageData[i*width+j]-mean[i*width*C+j*C+m]);  
                }  
            }  
        }  
        //update gaussian components for each pixel  
        for (i=0;i<height;i++)  
        {  
            for (j=0;j<width;j++)  
            {  
                match = 0;  
                temp = 0;  
                for(k=0;k<C;k++)  
                {  
                    if (abs(u_diff[i*width*C+j*C+k]) <= D*sd[i*width*C+j*C+k])      //pixel matches component  
                    {  
                         match = 1;                                                 // variable to signal component match  
                           
                         //update weights, mean, sd, p  
                         w[i*width*C+j*C+k] = (1-alph)*w[i*width*C+j*C+k] + alph;  
                         p = alph/w[i*width*C+j*C+k];                    
                         mean[i*width*C+j*C+k] = (1-p)*mean[i*width*C+j*C+k] + p*(uchar)current->imageData[i*width+j];  
                         sd[i*width*C+j*C+k] =sqrt((1-p)*(sd[i*width*C+j*C+k]*sd[i*width*C+j*C+k]) + p*(pow((uchar)current->imageData[i*width+j] - mean[i*width*C+j*C+k],2)));  
                    }else{  
                        w[i*width*C+j*C+k] = (1-alph)*w[i*width*C+j*C+k];           // weight slighly decreases  
                    }  
                    temp += w[i*width*C+j*C+k];  
                }  
                  
                for(k=0;k<C;k++)  
                {  
                    w[i*width*C+j*C+k] = w[i*width*C+j*C+k]/temp;  
                }  
              
                temp = w[i*width*C+j*C];  
                bg_bw[i*width+j] = 0;  
                for (k=0;k<C;k++)  
                {  
                    bg_bw[i*width+j] = bg_bw[i*width+j] + mean[i*width*C+j*C+k]*w[i*width*C+j*C+k];  
                    if (w[i*width*C+j*C+k]<=temp)  
                    {  
                        min_index = k;  
                        temp = w[i*width*C+j*C+k];  
                    }  
                    rank_ind[k] = k;  
                }  
  
                test->imageData[i*width+j] = (uchar)bg_bw[i*width+j];  
  
                //if no components match, create new component  
                if (match == 0)  
                {  
                    mean[i*width*C+j*C+min_index] = (uchar)current->imageData[i*width+j];  
                    //printf("%d ",(uchar)bg->imageData[i*width+j]);  
                    sd[i*width*C+j*C+min_index] = sd_init;  
                }  
                for (k=0;k<C;k++)  
                {  
                    rank[k] = w[i*width*C+j*C+k]/sd[i*width*C+j*C+k];  
                    //printf("%f ",w[i*width*C+j*C+k]);  
                }  
  
                //sort rank values  
                for (k=1;k<C;k++)  
                {  
                    for (m=0;m<k;m++)  
                    {  
                        if (rank[k] > rank[m])  
                        {  
                            //swap max values  
                            rand_temp = rank[m];  
                            rank[m] = rank[k];  
                            rank[k] = rand_temp;  
  
                            //swap max index values  
                            rank_ind_temp = rank_ind[m];  
                            rank_ind[m] = rank_ind[k];  
                            rank_ind[k] = rank_ind_temp;  
                        }  
                    }  
                }  
  
                //calculate foreground  
                match = 0;k = 0;  
                //frg->imageData[i*width+j]=0;  
                while ((match == 0)&&(k<M)){  
                    if (w[i*width*C+j*C+rank_ind[k]] >= thresh)  
                        if (abs(u_diff[i*width*C+j*C+rank_ind[k]]) <= D*sd[i*width*C+j*C+rank_ind[k]]){  
                            frg->imageData[i*width+j] = 0;  
                            match = 1;  
                        }  
                        else  
                            frg->imageData[i*width+j] = (uchar)current->imageData[i*width+j];       
                    k = k+1;  
                }  
            }  
        }  



		cvThreshold(frg,frg,100,255,0);
        mframe = cvQueryFrame(capture);  
        cvShowImage("fore",frg);  
        cvShowImage("back",test); 
		cvShowImage("原图",mframe);
        char s=cvWaitKey(33);  
        if(s==27) break;  
        free(rank_ind);  
    }  
      
    free(fg);free(w);free(mean);free(sd);free(u_diff);free(rank);  
    //cvNamedWindow("back",CV_WINDOW_KEEPRATIO);  
    //cvNamedWindow("fore",CV_WINDOW_KEEPRATIO); 
	//cvNamedWindow("原图",CV_WINDOW_KEEPRATIO);
	
    cvReleaseCapture(&capture);  
    cvDestroyWindow("fore");  
    cvDestroyWindow("back");  
    return 0;  
}  
