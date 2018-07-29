#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat image, image_gray;      //��������Mat���������ڴ洢ÿһ֡��ͼ��

    image = imread("img1.jpg");
    imshow("ԭͼ", image);

    cvtColor(image, image_gray, CV_BGR2GRAY);//תΪ�Ҷ�ͼ
    equalizeHist(image_gray, image_gray);//ֱ��ͼ���⻯�����ӶԱȶȷ��㴦��

	//------------------------������˹������ǿ--------------------------
	//-------------------------ʵ��ͼƬ��-----------------------------
	/*Mat imageEnhance;
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
	filter2D(image, imageEnhance, CV_8UC3, kernel);
	imageEnhance=imageEnhance+image;
	imshow("������˹����ͼ����ǿЧ��", imageEnhance);*/

	//------------------------ĥƤ�����ӳ���-----------------------------
	Mat dst;
	int value1 = 4, value2 = 1;     //ĥƤ�̶���ϸ�ڳ̶ȵ�ȷ��

	int dx = value1 * 5;    //˫���˲�����֮һ  
	double fc = value1*12.5; //˫���˲�����֮һ  12.5
	int p = 50; //͸����  
	Mat temp1, temp2, temp3, temp4;

	//˫���˲�  
	bilateralFilter(image, temp1, dx, fc, fc);
	//imshow("˫���˲���", temp1);
	temp2 = (temp1 - image + 128);
	//imshow("temp2", temp2);
	//��˹ģ��  �ָ�ϸ�� ʹͼƬ�����ʸ� 
	//-------------------------------------------------------------------
	//---�����൱�ڶ�˫���˲�����Ч�������˸�˹�˲�����Ч���൱����ʹ----
	//---���ϵ�����Ͳ��⻬�ĵط����ݽ�ȥ���ڼ���ԭͼʹ�����ʸ�----------
	//-------------------------------------------------------------------
	GaussianBlur(temp2, temp3, Size(2 * value2+1 , 2 * value2+1 ), 3, 3); 
	//imshow("temp3", temp3);
	temp4 = image + 2 * temp3 - 256;
	//imshow("temp4", temp4);
	dst = (image*(100 - p) + temp4*p) / 100;
	dst.copyTo(image);
	//-------------------------------------------------------------------

    CascadeClassifier eye_Classifier;  //���������
    CascadeClassifier face_cascade;    //���������
	CascadeClassifier mouth_cascade;    //���������
    //���ط���ѵ������OpenCv�ٷ��ĵ��ṩ��xml�ĵ�������ֱ�ӵ���
    //xml�ĵ�·��  opencv\sources\data\haarcascades 
    if (!eye_Classifier.load("haarcascade_eye.xml"))  //��Ҫ��xml�ĵ������Լ�ָ����·����
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

    //vector �Ǹ���ģ�� ��Ҫ�ṩ��ȷ��ģ��ʵ�� vector<Rect>���Ǹ�ȷ������ ģ���ʵ����
    vector<Rect> eyeRect;
    vector<Rect> faceRect;
	vector<Rect> mouthRect;
    //�������۾���λλ��
    //eye_Classifier.detectMultiScale(image_gray, eyeRect, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    //for (size_t eyeIdx = 0; eyeIdx < eyeRect.size(); eyeIdx++)
    //{   
    //    rectangle(image, eyeRect[eyeIdx], Scalar(0, 0, 255));   //�þ��λ�����⵽��λ��
    //}

    //����������λ��
    face_cascade.detectMultiScale(image_gray, faceRect, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    for (size_t i = 0; i < faceRect.size(); i++)
    {   
        rectangle(image, faceRect[i], Scalar(0, 0, 255));      //�þ��λ�����⵽��λ��
    }

	//mouth_cascade.detectMultiScale(image_gray, mouthRect, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
 //   for (size_t mouthIdx = 0; mouthIdx < mouthRect.size(); mouthIdx++)
 //   {   
 //       rectangle(image, mouthRect[mouthIdx], Scalar(0, 0, 255));   //�þ��λ�����⵽��λ��
 //   }

    imshow("����ʶ��ͼ", image);         //��ʾ��ǰ֡
    waitKey(0);
	return 0;

}