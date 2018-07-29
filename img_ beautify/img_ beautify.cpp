#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat image, image_gray;      //定义两个Mat变量，用于存储每一帧的图像

    image = imread("img1.jpg");
    imshow("原图", image);

    cvtColor(image, image_gray, CV_BGR2GRAY);//转为灰度图
    equalizeHist(image_gray, image_gray);//直方图均衡化，增加对比度方便处理

	//------------------------拉普拉斯算子增强--------------------------
	//-------------------------实现图片锐化-----------------------------
	/*Mat imageEnhance;
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
	filter2D(image, imageEnhance, CV_8UC3, kernel);
	imageEnhance=imageEnhance+image;
	imshow("拉普拉斯算子图像增强效果", imageEnhance);*/

	//------------------------磨皮部分子程序-----------------------------
	Mat dst;
	int value1 = 4, value2 = 1;     //磨皮程度与细节程度的确定

	int dx = value1 * 5;    //双边滤波参数之一  
	double fc = value1*12.5; //双边滤波参数之一  12.5
	int p = 50; //透明度  
	Mat temp1, temp2, temp3, temp4;

	//双边滤波  
	bilateralFilter(image, temp1, dx, fc, fc);
	//imshow("双边滤波后", temp1);
	temp2 = (temp1 - image + 128);
	//imshow("temp2", temp2);
	//高斯模糊  恢复细节 使图片更有质感 
	//-------------------------------------------------------------------
	//---这里相当于对双边滤波的逆效果进行了高斯滤波，其效果相当于是使----
	//---脸上的褶皱和不光滑的地方凹陷进去，在加上原图使其有质感----------
	//-------------------------------------------------------------------
	GaussianBlur(temp2, temp3, Size(2 * value2+1 , 2 * value2+1 ), 3, 3); 
	//imshow("temp3", temp3);
	temp4 = image + 2 * temp3 - 256;
	//imshow("temp4", temp4);
	dst = (image*(100 - p) + temp4*p) / 100;
	dst.copyTo(image);
	//-------------------------------------------------------------------

    CascadeClassifier eye_Classifier;  //载入分类器
    CascadeClassifier face_cascade;    //载入分类器
	CascadeClassifier mouth_cascade;    //载入分类器
    //加载分类训练器，OpenCv官方文档提供的xml文档，可以直接调用
    //xml文档路径  opencv\sources\data\haarcascades 
    if (!eye_Classifier.load("haarcascade_eye.xml"))  //需要将xml文档放在自己指定的路径下
    {  
        cout << "Load haarcascade_eye.xml failed!" << endl;
        return 0;
    }

    if (!face_cascade.load("haarcascade_frontalface_alt.xml"))
    {
        cout << "Load haarcascade_frontalface_alt failed!" << endl;
        return 0;
    }

	if (!mouth_cascade.load("haarcascade_mcs_mouth.xml"))
    {
        cout << "Load haarcascade_mcs_mouth failed!" << endl;
        return 0;
    }

    //vector 是个类模板 需要提供明确的模板实参 vector<Rect>则是个确定的类 模板的实例化
    vector<Rect> eyeRect;
    vector<Rect> faceRect;
	vector<Rect> mouthRect;
    //检测关于眼睛部位位置
    //eye_Classifier.detectMultiScale(image_gray, eyeRect, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    //for (size_t eyeIdx = 0; eyeIdx < eyeRect.size(); eyeIdx++)
    //{   
    //    rectangle(image, eyeRect[eyeIdx], Scalar(0, 0, 255));   //用矩形画出检测到的位置
    //}

    //检测关于脸部位置
    face_cascade.detectMultiScale(image_gray, faceRect, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (size_t i = 0; i < faceRect.size(); i++)
    {   
        rectangle(image, faceRect[i], Scalar(0, 0, 255));      //用矩形画出检测到的位置
    }

	//mouth_cascade.detectMultiScale(image_gray, mouthRect, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
 //   for (size_t mouthIdx = 0; mouthIdx < mouthRect.size(); mouthIdx++)
 //   {   
 //       rectangle(image, mouthRect[mouthIdx], Scalar(0, 0, 255));   //用矩形画出检测到的位置
 //   }

    imshow("人脸识别图", image);         //显示当前帧
    waitKey(0);
	return 0;

}