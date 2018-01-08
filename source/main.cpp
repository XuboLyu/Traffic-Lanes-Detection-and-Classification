#include <math.h> 
#include <cmath>
#include <typeinfo>
#include "contours.h"
#include "Camera.h"
#include <ctime>
#include <random>
#include <fstream>
#include "opencv/cv.h"
#include <sstream>
#include <iostream>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
extern "C" {
#include <vl/generic.h>
}
#include <vl/lbp.h>
#include <vl/sift.h>
#include <vl/hog.h>
#include <opencv2/hal.hpp>


using namespace cv;
using namespace cv::ml;

/**************methods  declaration***************************************/
void split(std::string & txt, std::vector<std::string> & strs, char ch);
int  CountLines(char *filename);
void show(std::vector<std::string> &strs);
void show(std::vector<float> &floats);
void multiple_delete(std::vector<std::string> & strs, std::vector<int> delete_pos);
void multiple_delete(std::vector<float> & floats, std::vector<int> delete_pos);
void vector_ele_attr(std::vector<std::string> & strs, char * attr, std::vector<float>& floats);
void vector_ele_attr(std::vector<std::string> & strs, char * attr, std::vector<float>& ints);
bool write_mat_to_file(cv::Mat &m, std::string filename,std::string mode,std::string type);
bool read_mat_from_file(cv::Mat &m, std::string filename,std::string mode);
bool contains(std::vector<ContourPtr> collection, ContourPtr element);
bool contains(std::vector<int> collection, int element);
bool contains(Rect rect, Contour ct1, Contour ct2);
void DrawContourss(const cv::Mat &ipm, std::vector<ContourPtr> &ipm_contours);
Mat deleteCols(Mat src, std::vector<int> numCols, int n);

static float calcOrientationHist(const Mat& img, Point pt, int radius, float sigma, float* hist, int n);
float angle(Point2f A, Point2f B);
std::vector<int> compute_chaincode_hist(Contour ct1, Contour ct2);
void standardization(cv::Mat inputM, cv::Mat& outputM);
float twoPointsDistance(cv::Point2f pt1, cv::Point2f pt2);
float computeDistance(Contour ct1, Contour ct2, Contour realContour);
void compute_roi(Camera& camera, int img_h, int img_w, float& a_left, float& b_left, float& a_right, float& b_right);
std::vector<float> compute_descriptor(cv::Mat ipm_gray, cv::Mat ipm_grad, cv::Mat mergeForSift,
	float* LBP_features, float * HOG_features, VlSiftFilt * sift, Contour ct1, Contour ct2, vl_size cellSize);
void compute_gradient(cv::Mat ipm_gray, cv::Mat& ipm_grad, cv::Mat& mergeForSift);
float* lbpFeatPerFrame(cv::Mat ipm_gray);
float * hogFeatPerFrame(cv::Mat ipm_gray);
void cal_evaluation(cv::Mat real, cv::Mat test, float& precision, float& recall);


int getSingleFrame(VideoCapture vc, int frameindex, cv::Mat& frame);
void compute_roi(Camera& camera, int img_h, int img_w, float& a_left, float& b_left, float& a_right, float& b_right);
Contour obtainRealContourBase(Camera camera, float a, float b, float c, float y_bottom, float y_top);
std::vector<ContourPtr> computeNegtive(std::vector<ContourPtr> ipm_contours, std::vector<ContourPtr> qualifiedContours);
std::vector<ContourPtr> resampleNegtive(std::vector<ContourPtr> negtiveContours, std::vector<ContourPtr> qualifiedContours, float ratio);
cv::Mat constructLabelMat(std::vector<ContourPtr> negtiveContours, std::vector<ContourPtr> qualifiedContours);
cv::Mat constructDescMat(cv::Mat ipm_gray, cv::Mat ipm_grad, cv::Mat mergeForSift, float * LBP_features, float * HOG_features, VlSiftFilt * sift, vl_size cellSize, std::vector<ContourPtr> negtiveContours, std::vector<ContourPtr> qualifiedContours, int dim);
void predict_test(Camera camera, cv::Mat frame, Ptr<RTrees> rtrees, std::vector<float> setting);
void onMouse(int event, int x, int y, int flags, void *ustc);
void normalization_train(cv::Mat &inputM, cv::Mat &outputM, char *filename);
void normalization_predict(cv::Mat &inputM, cv::Mat &outputM, char *filename);

/*******external definition**************/
cv::Mat image;
cv::Mat ipm, ipm_gray;

bool pauseFlag = false; //左键单击后视频暂停播放的标志位 
bool rectDone = false;
bool dupCalFlag = false;
cv::Point2f originalPoint; //矩形框起点  
cv::Point2f endPoint;

int count;
cv::VideoWriter demo("C:\\Users\\lvxubo\\Desktop\\SWT\\demo\\output.avi", CV_FOURCC('D', 'I', 'V', 'X'), 18, cv::Size(640,480), true);




/**********************functional methods********************/
void split(std::string & txt, std::vector<std::string> & strs, char ch)
{
	unsigned int pos = txt.find(ch);
	unsigned int initialPos = 0;
	strs.clear();

	// Decompose statement
	while (pos != std::string::npos) {
		strs.push_back(txt.substr(initialPos, pos - initialPos + 1));
		initialPos = pos + 1;

		pos = txt.find(ch, initialPos);
	}

	// Add the last one
	strs.push_back(txt.substr(initialPos, min(pos, txt.size()) - initialPos + 1));

	//return strs.size();
}
int  CountLines(char *filename)
{
	std::ifstream ReadFile;
	int n = 0;
	std::string tmp;
	ReadFile.open(filename, std::ios::in);//ios::in 表示以只读的方式读取文件
	if (ReadFile.fail())//文件打开失败:返回0
	{
		return 0;
	}
	else//文件存在
	{
		while (getline(ReadFile, tmp, '\n'))
		{
			n++;
		}
		ReadFile.close();
		return n;
	}
}
void show(std::vector<std::string> &strs)
{
	std::cout << "[";
	for (int i = 0; i < strs.size(); i++)
	{
		std::cout << strs[i] << " ";
	}
	std::cout << "]";
}
void show(std::vector<float> &floats)
{
	std::cout << "[";
	for (int i = 0; i < floats.size(); i++)
	{
		std::cout << floats[i] << " ";
	}
	std::cout << "]";
}
void multiple_delete(std::vector<std::string> & strs,std::vector<int> delete_pos)
{
	for (int i = 0; i < delete_pos.size(); i++)
	{
		strs.erase(strs.begin()+ delete_pos[i]-i);
	}
}
void multiple_delete(std::vector<float> & floats, std::vector<int> delete_pos)
{
	for (int i = 0; i < delete_pos.size(); i++)
	{
		floats.erase(floats.begin() + delete_pos[i] - i);
	}
}
void vector_ele_attr(std::vector<std::string> & strs, char * attr, std::vector<float>& floats)
{
	
	floats.clear();
	if (attr == "float")
	{
		for (int i = 0; i < strs.size(); i++)
		{
			floats.push_back(stof(strs[i]));
 		}
	}
}
void vector_ele_attr(std::vector<std::string> & strs, char * attr, std::vector<int>& ints)
{
	ints.clear();
	if (attr == "int")
	{
		for (int i = 0; i < strs.size(); i++)
		{
			ints.push_back(stoi(strs[i]));
		}
	}
}
bool write_mat_to_file(cv::Mat &m, std::string filename, std::string mode,std::string type)
{
	std::ofstream WriteFile;
	if (mode == "app")
	{
		WriteFile.open(filename, std::ios::app);
	}
	else if (mode == "out")
	{
		WriteFile.open(filename, std::ios::out);//ios::in 表示以只读的方式读取文件
	}
	if (WriteFile.fail())//文件打开失败:返回0
	{
		return false;
	}
	else
	{
		for (int i = 0; i < m.rows; i++)
		{
			if (type == "float")
			{
				const float* inData = m.ptr<float>(i);

				for (int j = 0; j < m.cols - 1; j++)
				{
					WriteFile << inData[j] << " ";
				}
				WriteFile << inData[m.cols - 1];
				WriteFile << "\n";
			}
			else if (type == "int")
			{
				const int* inData = m.ptr<int>(i);

				for (int j = 0; j < m.cols - 1; j++)
				{
					WriteFile << inData[j] << " ";
				}
				WriteFile << inData[m.cols - 1];
				WriteFile << "\n";
			}
		}
	}
	WriteFile.close();
}
bool read_mat_from_file(cv::Mat &m, std::string filename, std::string mode)
{
	std::ifstream ReadFile;
	ReadFile.open(filename, std::ios::in);//ios::in 表示以只读的方式读取文件
	std::string tmp;
	std::vector<std::string> tmp_split;
	
	
	
	if (ReadFile.fail())//文件打开失败:返回0
	{
		return false;
	}
	else//文件存在
	{
		if (mode == "float")
		{
			std::vector<std::vector<float>> sum_float;
			while (getline(ReadFile, tmp, '\n'))
			{
				std::vector<float> to_float;
				split(tmp, tmp_split, ' ');
				vector_ele_attr(tmp_split, "float", to_float);
				sum_float.push_back(to_float);
			}
			vconcat(sum_float, m);
		}
		else if (mode == "int")
		{
			std::vector<std::vector<int>> sum_int;
			while (getline(ReadFile, tmp, '\n'))
			{
				std::vector<int> to_int;
				split(tmp, tmp_split, ' ');
				vector_ele_attr(tmp_split, "int", to_int);
				sum_int.push_back(to_int);
			}
			vconcat(sum_int, m);
		}
		
	}
	
	ReadFile.close();
}
bool contains(std::vector<ContourPtr> collection, ContourPtr element)
{
	if (collection.empty())
		return false;
	if (std::find(collection.begin(), collection.end(), element) != collection.end())
	{
		return true;
	}
	else
	{
		return false;
	}

}
bool contains(std::vector<int> collection, int element)
{
	if (collection.empty())
		return false;
	if (std::find(collection.begin(), collection.end(), element) != collection.end())
	{
		return true;
	}
	else
	{
		return false;
	}
}
bool contains(Rect rect, Contour ct1, Contour ct2)
{
	Point2f p1, p2, p3, p4;
	p1 = ct1[0];
	p2 = ct1[ct1.size() - 1];
	p3 = ct2[0];
	p4 = ct2[ct2.size() - 1];

	if (rect.x < p1.x && p1.x < (rect.x + rect.width) && rect.x < p2.x && p2.x < (rect.x + rect.width) && rect.x < p3.x && p3.x < (rect.x + rect.width) && rect.x < p4.x && (p4.x < rect.x + rect.width) && rect.y < p1.y && (p1.y < rect.y + rect.height) && rect.y < p2.y && (p2.y < rect.y + rect.height) && rect.y < p3.y && (p3.y < rect.y + rect.height) && rect.y < p4.y && (p4.y < rect.y + rect.height))
	{
		return true;
	}
	else
	{
		return false;
	}
}
void DrawContourss(const cv::Mat &ipm, std::vector<ContourPtr> &ipm_contours)
{
	for (std::vector<ContourPtr>::iterator it = ipm_contours.begin(); it != ipm_contours.end(); it += 2)
	{
		std::vector<Point2f> c1 = *(*it);
		std::vector<Point2f> c2 = *(*(it + 1));
		for (int i = 0; i < c1.size() - 1; i++)
		{
			cv::line(ipm, c1[i], c1[i + 1], Scalar(0, 0, 255), 2);
		}
		for (int j = 0; j < c2.size() - 1; j++)
		{
			cv::line(ipm, c2[j], c2[j + 1], Scalar(0, 255, 0), 2);
		}

	}
}
Mat deleteCols(Mat src, std::vector<int> numCols, int n)
{
	Mat dst;
	bool singal = true;
	for (int i = 0; i < src.cols; i++)
	{
		//判断第i列是否属于被删除的列。
		for (int j = 0; j < n; j++)
		{
			int value = numCols[j];
			if (i == value)
			{
				singal = false;  //如果是需要删除的列，那么给singal幅值为false；                    
			}
		}
		//如果是需要删除的列，singal为假，跳过下面的if语句，从而将该列删除。
		if (singal)
		{
			Mat temp = src.col(i).t();
			dst.push_back(temp);
		}
		singal = true;
	}
	return dst.t();
}

/********************************algorithmic methods***************************/
static float calcOrientationHist(const Mat& img, Point pt, int radius, float sigma, float* hist, int n)
{
	//len：2r+1也就是以r为半径的圆（正方形）像素个数  
	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1);

	float expf_scale = -1.f / (2.f * sigma * sigma);
	AutoBuffer<float> buf(len * 4 + n + 4);
	float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
	float* temphist = W + len + 2;

	for (i = 0; i < n; i++)
		temphist[i] = 0.f;

	// 图像梯度直方图统计的像素范围  
	for (i = -radius, k = 0; i <= radius; i++)
	{
		int y = pt.y + i;
		if (y <= 0 || y >= img.rows - 1)
			continue;
		for (j = -radius; j <= radius; j++)
		{
			int x = pt.x + j;
			if (x <= 0 || x >= img.cols - 1)
				continue;

			float dx = (float)(img.at<uchar>(y, x + 1) - img.at<uchar>(y, x - 1));
			float dy = (float)(img.at<uchar>(y - 1, x) - img.at<uchar>(y + 1, x));

			X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
			k++;
		}
	}

	len = k;

	// compute gradient values, orientations and the weights over the pixel neighborhood  
	cv::hal::exp(W, W, len);
	cv::hal::fastAtan2(Y, X, Ori, len, true);
	cv::hal::magnitude(X, Y, Mag, len);

	// 计算直方图的每个bin  
	for (k = 0; k < len; k++)
	{
		int bin = cvRound((n / 360.f)*Ori[k]);
		if (bin >= n)
			bin -= n;
		if (bin < 0)
			bin += n;
		temphist[bin] += W[k] * Mag[k];
	}

	// smooth the histogram  
	// 高斯平滑  
	temphist[-1] = temphist[n - 1];
	temphist[-2] = temphist[n - 2];
	temphist[n] = temphist[0];
	temphist[n + 1] = temphist[1];
	for (i = 0; i < n; i++)
	{
		hist[i] = (temphist[i - 2] + temphist[i + 2])*(1.f / 16.f) +
			(temphist[i - 1] + temphist[i + 1])*(4.f / 16.f) +
			temphist[i] * (6.f / 16.f);
	}

	// 得到主方向  
	float maxval = hist[0];
	for (i = 1; i < n; i++)
		maxval = max(maxval, hist[i]);
	for (i = 0; i < n; i++)
	{
		if (hist[i] == maxval)
		{
			float mainAngle = i;
			break;
		}
	}
	//return maxval;
	return 45 * i * 2 * PI / 360.f;
}
float angle(Point2f A, Point2f B)
{
	if (B.x == A.x)
	{
		if (B.x <= A.x)
		{
			return 270;
		}
		else
		{
			return 90;
		}
		
	}
	float val = (B.y - A.y) / (B.x - A.x); // calculate slope between the two points
	val = val - pow(val, 3) * 1.0 / 3.0 + pow(val, 5) * 1.0 / 5.0;
	val = ((int)(val * 180 / 3.14)); // Convert the angle in radians to degrees
	
	if (B.x < A.x)
	{
		val += 180;
	}
	if (val < 0)
	{
		val += 360;
	}
	
	return val;
}
float twoPointsDistance(cv::Point2f pt1, cv::Point2f pt2)
{
	return (pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y);
}
float computeDistance(Contour ct1, Contour ct2, Contour realContour)
{
	//std::vector<int> label;
	cv::Point2f pt11 = ct1[0];
	cv::Point2f pt12 = ct1[ct1.size() / 2];
	cv::Point2f pt13 = ct1[ct1.size() - 1];

	cv::Point2f pt21 = ct2[0];
	cv::Point2f pt22 = ct2[ct2.size() / 2];
	cv::Point2f pt23 = ct2[ct2.size() - 1];

	cv::Point2f basicPt1, basicPt2, basicPt3;
	bool flag1, flag2, flag3;
	flag1 = false;
	flag2 = false;
	flag3 = false;

	for (int i = 0; i < realContour.size(); i++)
	{
		if (int(realContour[i].y) == pt11.y)
		{
			basicPt1.x = int(realContour[i].x);
			basicPt1.y = int(realContour[i].y);
			flag1 = true;
		}
		else if (int(realContour[i].y) == pt12.y)
		{
			basicPt2.x = int(realContour[i].x);
			basicPt2.y = int(realContour[i].y);
			flag2 = true;
		}
		else if (int(realContour[i].y) == pt13.y)
		{
			basicPt3.x = int(realContour[i].x);
			basicPt3.y = int(realContour[i].y);
			flag3 = true;
		}

		if (flag1 & flag2 & flag3)
		{
			break;
		}
	}
	//std::cout << "[" << pt12.x << pt12.y << "]" << "[" << pt22.x << pt22.y << "]" << "[" << basicPt.x << basicPt.y << "]" << std::endl;
	// 	int distance = (cv::norm(pt11 - basicPt) + cv::norm(pt12 - basicPt) + cv::norm(pt13 - basicPt)
	// 		+ cv::norm(pt21 - basicPt) + cv::norm(pt22 - basicPt) + cv::norm(pt23 - basicPt)) / 6;
	float distance = twoPointsDistance(pt11, basicPt1) + twoPointsDistance(pt12, basicPt2) + twoPointsDistance(pt13, basicPt3) + twoPointsDistance(pt21, basicPt1) + twoPointsDistance(pt22, basicPt2) + twoPointsDistance(pt23, basicPt3);
	distance = distance / 6;
	return distance;

}

std::vector<int> compute_chaincode_hist(Contour ct1, Contour ct2)
{
	
	int size = ct1.size();
	std::vector<int> hist(8);
	for (int i = 0; i < size - 1; i++)
	{
		float ale1 = angle(ct1[i], ct1[i + 1]);
		float ale2 = angle(ct2[i], ct2[i + 1]);
		hist[int(ale1 / 45)]++;
		hist[int(ale2 / 45)]++;
	}

	//cv::Mat histMat(hist, true);
	return hist;
}
void standardization(cv::Mat inputM, cv::Mat& outputM)
{
	std::ofstream standard_setting("standard_setting.ini");
	double max, min;
	std::vector<cv::Mat> vm;

	for (int i = 0; i < inputM.cols; i++)
	{
		cv::Mat col = inputM.col(i);
		minMaxLoc(col, &min, &max);
		cv::Mat tmpMin(inputM.rows, 1, CV_32FC1, min);
		vm.push_back((col - tmpMin)*1.0 / (max - min));
		standard_setting << min << " " << max << std::endl;
	}
	cv::hconcat(vm, outputM);
	standard_setting.close();

}
//void standardization(cv::Mat inputM, cv::Mat& outputM)
//{
//	std::ofstream standard_setting("standard_setting.ini");
//	
//	Scalar mean, var;
//	std::vector<cv::Mat> vm;
//	for (int i = 0; i < inputM.cols; i++)
//	{
//		cv::Mat col = inputM.col(i);
//		cv::meanStdDev(col, mean, var);
//		cv::Mat tmpMean(inputM.rows, 1, CV_32FC1, mean[0]);
//		vm.push_back((col - tmpMean) * 1.0 / var[0]);
//		standard_setting << mean[0] << " " << var[0];
//		standard_setting << std::endl;
//		//std::cout << col.size() << " " << col ;
//	}
//	cv::hconcat(vm,outputM);
//	standard_setting.close();
//	//std::cout << vm.size() << std::endl;
//	//std::cout << outputM;
//	//std::cout << std::endl;
//}
void compute_roi(Camera& camera, int img_h, int img_w, float& a_left, float& b_left, float& a_right, float& b_right)
{
	// compute out of view range 
	cv::Point2f im_bl(0, img_h - 1), im_br(img_w - 1, img_h - 1);
	cv::Point2f im_bl2(0, img_h - 40), im_br2(img_w - 1, img_h - 40);
	cv::Point2f ipm_bl = camera.cvtImageToIPM(im_bl), ipm_br = camera.cvtImageToIPM(im_br);
	cv::Point2f ipm_bl2 = camera.cvtImageToIPM(im_bl2), ipm_br2 = camera.cvtImageToIPM(im_br2);
	a_left = (ipm_bl2.y - ipm_bl.y) / (ipm_bl2.x - ipm_bl.x);
	a_right = (ipm_br2.y - ipm_br.y) / (ipm_br2.x - ipm_br.x);
	b_left = -a_left * ipm_bl.x + ipm_bl.y;
	b_right = -a_right * ipm_br.x + ipm_br.y;
}
//compute features of every contour pair.
std::vector<float> compute_descriptor(cv::Mat ipm_gray, cv::Mat ipm_grad, cv::Mat mergeForSift,
	float* LBP_features, float * HOG_features, VlSiftFilt * sift, Contour ct1, Contour ct2, vl_size cellSize)
{


	std::vector<float> descriptor;

	//float posMid;

	int size1 = ct1.size();
	int size2 = ct2.size();
	int size = min(size1, size2);

	vl_size width = ipm_gray.cols;
	vl_size height = ipm_gray.rows;
	vl_size hogWidth = width / cellSize;
	vl_size hogHeight = height / cellSize;

	vl_size lbpWidth = hogWidth;
	vl_size lbpHeight = hogHeight;


	// 长度
	descriptor.push_back(size);




	// 梯度横扫方向方差、均值 以及 亮度的方差、均值
	Scalar light_mean, light_var;
	std::vector<uchar> gray_vec;

	float grad_mean = 0.0, grad_var = 0.0;

	for (int i = 0; i < size; i++)
	{
		std::vector<uchar> gradient_vec;
		Scalar tmp_grad_mean, tmp_grad_var;

		for (int j = ct1[i].x; j < ct2[i].x; j++)
		{
			Scalar intensity = ipm_grad.at<uchar>(ct1[i].y, j);
			Scalar lightsity = ipm_gray.at<uchar>(ct1[i].y, j);

			gradient_vec.push_back(intensity.val[0]);
			gray_vec.push_back(lightsity.val[0]);
		}
		cv::meanStdDev(gradient_vec, tmp_grad_mean, tmp_grad_var);
		grad_var = grad_var + tmp_grad_var[0];
		grad_mean = grad_mean + tmp_grad_mean[0];


	}
	grad_var = grad_var / size;
	grad_mean = grad_mean / size;
	cv::meanStdDev(ipm_gray, light_mean, light_var);

	descriptor.push_back(grad_var);
	descriptor.push_back(grad_mean);

	descriptor.push_back(light_mean[0]);
	descriptor.push_back(light_var[0]);


	//中心相对位置  该特征不具备共性，有时候车道线会在边上
	/*posMid = (ct1[size / 2].x + ct2[size / 2].x + ct1[0].x + ct2[0].x) / 2;
	float posMidToCenter = abs(posMid - ipm_gray_gradient.size[1]);*/


	//descriptor.push_back(posMidToCenter);


	// 拟合直线斜率
	/*std::vector<float> param0, param1;
	cv::fitLine(ct1, param0, CV_DIST_L2, 0, 0.01, 0.01);
	cv::fitLine(ct2, param1, CV_DIST_L2, 0, 0.01, 0.01);*/

	//std::cout << param0[0] << " " << param0[1] << " " << param1[0] << " " << param1[1] << std::endl;
	//std::cout << (param0[0] + param1[0]) / 2 << " " << (param0[1] + param1[1]) / 2 << std::endl;
	//descriptor.push_back((param0[0] + param1[0]) / 2);
	//descriptor.push_back((param0[1] + param1[1]) / 2);

	// 宽高比（比较过，慎用）
	float line_length = (ct1[size - 1].y - ct1[0].y + ct2[size - 1].y - ct2[0].y) / 2;
	float line_width = (abs(ct1[size - 1].x - ct2[size - 1].x) + abs(ct1[0].x - ct2[0].x)) / 2;
	float ratioOfLengthWidth = line_length / line_width;
	descriptor.push_back(ratioOfLengthWidth);


	std::vector<int> chaincode(8);
	chaincode = compute_chaincode_hist(ct1, ct2);
	for (std::vector<int>::iterator r = chaincode.begin(); r != chaincode.end(); r++)
	{
		descriptor.push_back(*r);
	}


	std::vector<std::vector<float>> ContourHOG, ContourLBP, ContourSIFT;
	cv::Mat mat_HOG, mat_LBP, mat_SIFT;
	// HOG and LBP and SIFT FEATURES
	for (int i = 0; i < size; i++)
	{
		Point2f p1(ct1[i].x, ct1[i].y);
		Point2f p2(ct2[i].x, ct2[i].y);
		std::vector<float> hogWanted;
		std::vector<float> lbpWanted;
		std::vector<float> siftWanted;

		float *desc1, *desc2, *Hist1, *Hist2;
		try {
			desc1 = new float[128];
			desc2 = new float[128];
			Hist1 = new float[8];
			Hist2 = new float[8];
		}
		catch (std::bad_alloc)
		{
			std::cerr << "out of memory" << std::endl;
			exit(1);
		}
		float maxOrientation1 = 0.0;
		float maxOrientation2 = 0.0;
		maxOrientation1 = calcOrientationHist(ipm_gray, p1, 2.5*0.5, 0.5, Hist1, 8);
		maxOrientation2 = calcOrientationHist(ipm_gray, p2, 2.5*0.5, 0.5, Hist2, 8);
		vl_sift_calc_raw_descriptor(sift, (float*)mergeForSift.data, desc1, width, height, p1.x, p1.y, 0.5, maxOrientation1);
		vl_sift_calc_raw_descriptor(sift, (float*)mergeForSift.data, desc2, width, height, p2.x, p2.y, 0.5, maxOrientation2);

		vl_size inx1 = p1.x / cellSize;
		vl_size iny1 = p1.y / cellSize;
		vl_size in1 = hogWidth* iny1 + hogHeight;

		vl_size inx2 = p2.x / cellSize;
		vl_size iny2 = p2.y / cellSize;
		vl_size in2 = lbpWidth* iny2 + lbpHeight;
		//std::cout << inx << " " << iny << " " << in;
		for (int i = 0; i < 32; i++)
		{
			hogWanted.push_back(*(HOG_features + in1 + i * hogWidth*hogHeight));
			//std::cout << hogWanted[i] << std::endl;
		}
		for (int j = 0; j < 32; j++)
		{
			hogWanted.push_back(*(HOG_features + in2 + j * hogWidth*hogHeight));
		}

		for (int i = 0; i < 58; i++)
		{
			lbpWanted.push_back(*(LBP_features + in1 + i * lbpWidth*lbpHeight));
			//std::cout << hogWanted[i] << std::endl;
		}
		for (int j = 0; j < 58; j++)
		{
			lbpWanted.push_back(*(LBP_features + in2 + j * lbpWidth*lbpHeight));
		}


		for (int i = 0; i < 128; i++)
		{
			siftWanted.push_back(*(desc1 + i));
		}
		for (int j = 0; j < 128; j++)
		{
			siftWanted.push_back(*(desc2 + j));
		}
		ContourHOG.push_back(hogWanted);
		ContourLBP.push_back(lbpWanted);
		ContourSIFT.push_back(siftWanted);
		delete[] desc1;
		delete[] desc2;
		delete[] Hist1;
		delete[] Hist2;
		desc1 = NULL;
		desc2 = NULL;
		Hist1 = NULL;
		Hist2 = NULL;
	}
	vconcat(ContourHOG, mat_HOG);
	mat_HOG = mat_HOG.t();
	double min1, max1;
	cv::Mat m(mat_HOG.rows, 1, CV_32FC1);
	for (int c = 0; c < mat_HOG.rows; c++)
	{
		minMaxIdx(mat_HOG.row(c), &min1, &max1);
		m.at<float>(c, 0) = max1;
		descriptor.push_back(max1);
	}

	vconcat(ContourLBP, mat_LBP);
	mat_LBP = mat_LBP.t();
	double min2, max2;
	cv::Mat mm(mat_LBP.rows, 1, CV_32FC1);
	for (int c = 0; c < mat_LBP.rows; c++)
	{
		minMaxIdx(mat_LBP.row(c), &min2, &max2);
		mm.at<float>(c, 0) = max2;
		descriptor.push_back(max2);
	}

	vconcat(ContourSIFT, mat_SIFT);
	mat_SIFT = mat_SIFT.t();
	double min3, max3;
	cv::Mat mmm(mat_SIFT.rows, 1, CV_32FC1);
	for (int c = 0; c < mat_SIFT.rows; c++)
	{
		minMaxIdx(mat_SIFT.row(c), &min3, &max3);
		mmm.at<float>(c, 0) = max3;
		descriptor.push_back(max3);
	}

	//std::cout << "HOG SIZE: " << m.rows << " " << "HOG FEAT: " <<m.t() << std::endl; 

	//std::cout << "LBP SIZE: " << mm.rows << " " << "LBP FEAT: " <<mm.t() << std::endl;

	// SIFT features just TRY!


	//std::cout << "mat_SIFT.rows: " << mat_SIFT.rows << std::endl;
	return descriptor;

}
void compute_gradient(cv::Mat ipm_gray, cv::Mat& ipm_grad, cv::Mat& mergeForSift)
{
	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	cv::Sobel(ipm_gray, grad_x, CV_32F, 1, 0, 3, 0.5, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);


	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(ipm_gray, grad_y, CV_32F, 0, 1, 3, 0.5, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);


	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, ipm_grad);

	/******************************************************/
	std::vector<float> array_x(ipm_gray.rows*ipm_gray.cols);
	if (grad_x.isContinuous())
		array_x.assign((float*)grad_x.datastart, (float*)grad_x.dataend);


	std::vector<float> array_y(ipm_gray.rows*ipm_gray.cols);
	if (grad_y.isContinuous())
		array_y.assign((float*)grad_y.datastart, (float*)grad_y.dataend);


	cv::Mat Mag, Ori;

	cartToPolar(array_x, array_y, Mag, Ori);

	std::vector<cv::Mat> to;
	to.push_back(Mag);
	to.push_back(Ori);
	hconcat(to, mergeForSift);
	//imshow("gradient", ipm_grad);
	//cv::waitKey(0);


}
float* lbpFeatPerFrame(cv::Mat ipm_gray)
{
	cv::Mat to_Float;
	ipm_gray.convertTo(to_Float, CV_32F);
	float *vlimage = (float*)to_Float.data;
	VlLbp * lbp = vl_lbp_new(VlLbpUniform, false);
	vl_size dimension = vl_lbp_get_dimension(lbp);
	vl_size cellSize = 8;
	vl_size lbpWidth = ipm_gray.cols / cellSize;
	vl_size lbpHeight = ipm_gray.rows / cellSize;
	static float * LBP_features = (float*)vl_malloc(floor(lbpWidth)*floor(lbpHeight)*dimension * sizeof(float));
	vl_lbp_process(lbp, LBP_features, vlimage, ipm_gray.cols, ipm_gray.rows, cellSize);
	vl_lbp_delete(lbp);
	return LBP_features;
}
float * hogFeatPerFrame(cv::Mat ipm_gray)
{
	cv::Mat to_Float;
	ipm_gray.convertTo(to_Float, CV_32F);
	float *vlimage = (float*)to_Float.data;

	vl_size cellSize = 8;
	vl_size num_channels = 1;
	VlHog * hog = vl_hog_new(VlHogVariantDalalTriggs, 8, VL_FALSE);
	vl_hog_put_image(hog, vlimage, ipm_gray.cols, ipm_gray.rows, num_channels, cellSize);
	vl_size hogWidth = vl_hog_get_width(hog);
	vl_size hogHeight = vl_hog_get_height(hog);
	vl_size hogDimension = vl_hog_get_dimension(hog);
	//std::cout << hogWidth << " " << hogHeight << " " << hogDimension;
	static float * HOG_features = (float *)vl_malloc(hogWidth*hogHeight*hogDimension * sizeof(float));
	vl_hog_extract(hog, HOG_features);
	vl_hog_delete(hog);
	return HOG_features;
}
void cal_evaluation(cv::Mat real, cv::Mat test, float& precision, float& recall)
{
	//std::cout << real.type()<< "\n";  //32S
	//std::cout << test.type() << "\n"; //32F
	int intersecPosNum = 0;
	int realPosNum = 0;
	int testPosNum = 0;
	int rtmp;
	float ttmp;
	for (int i = 0; i < real.rows; i++)
	{

		rtmp = real.at<int>(i, 0);
		ttmp = test.at<float>(i, 0);
		//std::cout << ttmp << "\n";
		if (rtmp == 1 && ttmp == 1)
		{
			intersecPosNum++;
		}
		if (rtmp == 1)
		{
			realPosNum++;
		}
		if (ttmp == 1)
		{
			testPosNum++;
		}
	}
	//std::cout << intersecPosNum << " " << testPosNum << " " << realPosNum;
	precision = intersecPosNum * 1.0 / testPosNum;
	recall = intersecPosNum * 1.0 / realPosNum;
}



/****************************logical methods****************************/
int getSingleFrame(VideoCapture vc, int frameindex,cv::Mat& frame)
{

	if (!vc.isOpened())
	{

		std::cout << "can not open the video" << std::endl;
		return -1;
	}
	
	int frameNum = vc.get((CV_CAP_PROP_FRAME_COUNT));
	
	if (frameindex < 0 || frameindex >= frameNum)
	{
		
		std::cout << "invalid frame index";
		return -1;
	}
	vc.set(CV_CAP_PROP_POS_FRAMES, frameindex);
	bool success = vc.read(frame);
	if (!success)
	{
		std::cout << " can not read the frame";
		return -1;
	}
	
	return 1;
}
Contour obtainRealContourBase(Camera camera, float a, float b, float c, float y_bottom, float y_top)
{
	float x = 0.0;
	cv::Point2f pt;
	Contour realContour;
	for (float y = y_bottom; y <= y_top; y+=0.1)
	{
		x = a * y * y * 1.0 + b * y * 1.0 + c;
		pt.x = x;
		pt.y = y;
		realContour.push_back(camera.cvtGroundToIPM(pt));
		//std::cout << camera.cvtGroundToIPM(pt).x << " " << camera.cvtGroundToIPM(pt).y << std::endl;
		
	}
	
	return realContour;
}
std::vector<ContourPtr> computeNegtive(std::vector<ContourPtr> ipm_contours, std::vector<ContourPtr> qualifiedContours)
{
	std::vector<ContourPtr> negtiveContours;
	for (int i = 0; i < ipm_contours.size(); i+=2)
	{
		if (!contains(qualifiedContours, ipm_contours[i]))
		{
			negtiveContours.push_back(ipm_contours[i]);
			negtiveContours.push_back(ipm_contours[i+1]);
		}
	}
	return negtiveContours;
}
std::vector<ContourPtr> resampleNegtive(std::vector<ContourPtr> negtiveContours, std::vector<ContourPtr> qualifiedContours, float ratio)
{
	int deleteNum = negtiveContours.size()/2 - ceil(qualifiedContours.size()  * ratio / 2);
	
	std::vector<int> negIndex;
	std::vector<int> delIndex;

	std::vector<ContourPtr> resampleNegtiveContours;

	for (int i = 0; i < negtiveContours.size()/2; i++)
	{
		negIndex.push_back(i);
	}
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(negIndex.begin(), negIndex.end(), g);

	for (int j = 0; j < deleteNum; j++)
	{
		delIndex.push_back(negIndex[j]);
	}

	// delIndex contains index of contour pair. attention!! it's the index of pair , not single

	for (int k = 0; k < negtiveContours.size() / 2; k++)
	{
		if (!contains(delIndex, k))
		{
			resampleNegtiveContours.push_back(negtiveContours[2 * k]);
			resampleNegtiveContours.push_back(negtiveContours[2 * k + 1]);
		}
	}



	return resampleNegtiveContours;

	
	

}
cv::Mat constructLabelMat(std::vector<ContourPtr> negtiveContours, std::vector<ContourPtr> qualifiedContours)
{
	int negRows = negtiveContours.size();
	int posRows = qualifiedContours.size();
	int rows = negRows + posRows;

	cv::Mat M(rows/2, 1, CV_32SC1);
	

	for (int i = 0; i < posRows/2; i++)
	{
		M.at<int>(i, 0) = 1;
		
	}
	for (int j = posRows / 2; j < rows/2; j++)
	{
		M.at<int>(j, 0) = -1;
		
	}
	
// 	std::cout << posRows << " ";
// 	std::cout << negRows << " ";
	std::cout << M.t() << std::endl;
	return M;

}
cv::Mat constructDescMat(cv::Mat ipm_gray,cv::Mat ipm_grad,cv::Mat mergeForSift, float * LBP_features, float * HOG_features, VlSiftFilt * sift,vl_size cellSize,std::vector<ContourPtr> negtiveContours, std::vector<ContourPtr> qualifiedContours,int dim)
{
	int negRows = negtiveContours.size();
	int posRows = qualifiedContours.size();
	//std::cout << "neg num: " << negRows << " " <<"pos num: "<< posRows << std::endl;
	int rows = negRows + posRows;
	// 8U --> uchar
	cv::Mat M(rows/2, dim, CV_32FC1);

	std::vector<float> desc;
	

	for (int i = 0; i < posRows / 2; i++)
	{
		desc = compute_descriptor(ipm_gray,ipm_grad,mergeForSift,LBP_features,HOG_features, sift,*(qualifiedContours[2 * i]), *(qualifiedContours[2 * i + 1]),cellSize);
		for (int k = 0; k < dim; k++)
		{
			//float q = desc[k];
			M.at<float>(i, k) = desc[k];
		}
	}
	for (int j = posRows / 2; j < rows/2; j++)
	{
		desc = compute_descriptor(ipm_gray, ipm_grad, mergeForSift, LBP_features, HOG_features, sift,*(negtiveContours[2 * (j-posRows/2)]), *(negtiveContours[2 * (j - posRows / 2) + 1]), cellSize);
		for (int k = 0; k < dim; k++)
		{
			//float q = desc[k];
			M.at<float>(j, k) = desc[k];
		}
	}


	return M;
}
void predict_test(Camera camera, cv::Mat frame, Ptr<RTrees> rtrees, std::vector<float> setting)
{
	//Ptr<ml::SVM> svm = ml::SVM::load<ml::SVM>(svm_path);
	
	float a_left, b_left, a_right, b_right;
	compute_roi(camera, frame.rows, frame.cols, a_left, b_left, a_right, b_right);
	cv::Mat ipm, ipm_gray, ipm_grad, mergeForSift;
	float *lbpFeat, *hogFeat;
	std::vector<ContourPtr> ipm_contours;

	camera.GetIPMImage(frame, ipm);
	cv::cvtColor(ipm, ipm_gray, cv::COLOR_BGR2GRAY);
	ScanlineContours(ipm_gray, ipm_contours, 20.f, 2, 12, 5, a_left, b_left, a_right, b_right);

	compute_gradient(ipm_gray, ipm_grad, mergeForSift);

	lbpFeat = lbpFeatPerFrame(ipm_gray);
	hogFeat = hogFeatPerFrame(ipm_gray);
	VlSiftFilt  * sift = vl_sift_new(ipm_gray.cols, ipm_gray.rows, 4, 5, 0);
	std::vector<float> desc;

	std::vector<ContourPtr> PosContours;
	float min, max;
	//std::cout << setting.size() << std::endl;
	
	std::vector<int> pos_to_del = { 11,13 };
	for (std::vector<ContourPtr>::iterator it = ipm_contours.begin(); it != ipm_contours.end(); it += 2)
	{
		desc = compute_descriptor(ipm_gray, ipm_grad, mergeForSift, lbpFeat, hogFeat, sift, *(*it), *(*(it + 1)), 8);
		
		for (int z = 0; z < desc.size(); z++)
		{

			min = setting[z*2];
			max = setting[z * 2 + 1];
			if (min != max)
			{
				desc[z] = (desc[z] - min) / (max - min);
			}

		}
		multiple_delete(desc, pos_to_del);
		//std::cout << desc.size() << "\n";
		//show(desc);
		//std::cout << std::endl;
		float response = rtrees->predict(desc);

		//std::cout << response << std::endl;
		//现在可以确定 训练数据中特征矩阵和标签矩阵的对应关系没有错 2016/8/22
		if (response == 1.0)
		{
			PosContours.push_back(*(it));
			PosContours.push_back(*(it + 1));
		}
	}
	vl_sift_delete(sift);

	namedWindow("result");
	DrawContourss(ipm, PosContours);
	cv::imshow("result", ipm);
	//demo.write(ipm);
	
}
//void onMouse(int event, int x, int y, int flags, void *ustc)
//{
//	if (event == EVENT_RBUTTONDOWN)
//	{
//		rectDone = false;
//		
//		originalPoint = Point(0, 0);
//		endPoint = Point(0, 0);
//		count++;
//		if (count % 2 == 1)
//		{
//			pauseFlag = true; //标志位
//		}
//		else
//		{
//			pauseFlag = false; //标志位
//		}
//	}
//	
//
//	if (event == CV_EVENT_LBUTTONDOWN)
//	{
//		rectDone = false;
//		
//		originalPoint = Point(x, y);  //设置左键按下点的矩形起点  
//		//processPoint = originalPoint;
//		
//	}
//	//if (event == CV_EVENT_MOUSEMOVE&&flags == EVENT_FLAG_LBUTTON&&pauseFlag)
//	//{
//	//	//imageCopy = ipm_gray.clone();
//	//	//processPoint = Point(x, y);
//	//	//if (originalPoint != processPoint)
//	//	//{
//	//	//	//在复制的图像上绘制矩形  
//	//	//	rectangle(imageCopy, originalPoint, processPoint, Scalar(0, 255, 0), 2);
//	//	//}
//	//	//imshow("ipm_gray", imageCopy);
//	//}
//	if (event == CV_EVENT_LBUTTONUP && pauseFlag)
//	{
//		
//		endPoint = Point(x, y);
//		if (originalPoint != endPoint)
//		{ 
//			Mat rectImage = ipm_gray(Rect(originalPoint, endPoint)); //子图像显示  
//			rectDone = true;
//			imshow("sub video", rectImage);
//		}
//		
//		
//		
//	}
//	
//}

bool GetFrameBaseLineParam(std::map<int, std::vector<float>>& baseLineParam)
{
	std::ifstream fin("C:\\Users\\lvxubo\\Desktop\\SWT\\Yaml\\OV10635_20160518_1547_lane.txt");
	if (!fin.is_open())
	{
		std::cout << "load file error" << std::endl;
		return false;
	}
	std::string s;
	int frameNum, realContourNum;
	float a = 0.0, b = 0.0, c = 0.0;
	while (getline(fin, s))
	{
		std::stringstream ss(s);
		std::vector<float> tmpParamPerFrame;
		ss >> frameNum >> realContourNum;
		for (int i = 0; i < realContourNum; i++)
		{
			ss >> c >> b >> a;
			tmpParamPerFrame.push_back(a);
			tmpParamPerFrame.push_back(b);
			tmpParamPerFrame.push_back(c);
		}
		baseLineParam[frameNum] = tmpParamPerFrame;	
	}
	fin.close();
	return true;
}
void computeContours(Camera camera, std::vector<ContourPtr> &ipm_contours ,int frameNum,std::vector<ContourPtr>& positive_contours,std::vector<ContourPtr>& negative_contours,std::map<int, std::vector<float>> baseLineParamMap)
{
	/*cv::Mat m, ipm, ipm_gray;
	std::vector<ContourPtr> ipm_contours;
	getSingleFrame(vc, frameNum, m);
	camera.GetIPMImage(m, ipm);
	cv::cvtColor(ipm, ipm_gray, cv::COLOR_BGR2GRAY);
	ScanlineContours(ipm_gray, ipm_contours, 20.f, 2, 12, 5, a_left, b_left, a_right, b_right);*/
		
	std::vector<Contour> baseLineVec;
	for (int i = 0; i < baseLineParamMap[frameNum].size(); i += 3)
	{
		Contour baseLine = obtainRealContourBase(camera, baseLineParamMap[frameNum][i + 0], baseLineParamMap[frameNum][i + 1], baseLineParamMap[frameNum][i + 2], 0.0, 50.0);
		cv::Mat curve(baseLine, true);
		curve.convertTo(curve, CV_32S);
		//cv::polylines(ipm, curve, false, CV_RGB(0, 255, 0));
		baseLineVec.push_back(baseLine);
	}
	for (int i = 0; i < baseLineVec.size(); i++)
	{
		std::vector<std::vector<int>> allParamVec;
		int markIndex = 0;
		for (std::vector<ContourPtr>::iterator pos = ipm_contours.begin(); pos != ipm_contours.end(); pos += 2)
		{
	
			int tempDis = computeDistance(*(*pos), *(*(pos + 1)), baseLineVec[i]);
			std::vector<int> sigleParamVec;
			sigleParamVec.push_back(tempDis);
			sigleParamVec.push_back(markIndex);
			sigleParamVec.push_back(markIndex + 1);
	
			allParamVec.push_back(sigleParamVec);
	
			markIndex += 2;
	
		}
	
		std::sort(allParamVec.begin(), allParamVec.end(), [](const std::vector<int>& a, std::vector<int>& b) {return a[0] < b[0]; });
	
		// store and display positive results
		for (int i = 0; i < min(allParamVec.size(), 3); i++)
		{
			ContourPtr tmp_ipm_contours_1 = ipm_contours[allParamVec[i][1]];
			ContourPtr tmp_ipm_contours_2 = ipm_contours[allParamVec[i][2]];
			if ((*tmp_ipm_contours_1).size() > 40)
			{
				positive_contours.push_back(ipm_contours[allParamVec[i][1]]);
				positive_contours.push_back(ipm_contours[allParamVec[i][2]]);
			}
			else
			{
				std::cout << "invalid qualified contours" << std::endl;
			}
		}
	}
	negative_contours = computeNegtive(ipm_contours, positive_contours);
	//std::vector<ContourPtr> resampleNegtiveContours = resampleNegtive(negtiveContours, positive_contours, 1.5);
	
}
void sampleSplit(cv::Mat &m, int ratio, cv::Mat &trainMat, cv::Mat &testMat)
{
	int end_rowNum = m.rows - 1;
	int middle_rowNum = m.rows * ratio / (ratio + 1) - 1;
	trainMat = m.rowRange(0, middle_rowNum + 1).clone();
	testMat = m.rowRange(middle_rowNum + 1, end_rowNum + 1).clone();

}
void normalization_train(cv::Mat &inputM, cv::Mat &outputM,char *filename)
{
	std::ofstream standard_setting("standard_setting.ini");
	double max, min;
	std::vector<cv::Mat> vm;
	std::vector<int> exceptIndex;
	for (int i = 0; i < inputM.cols; i++)
	{
		cv::Mat col = inputM.col(i);
		minMaxLoc(col, &min, &max);
		if (max == min)
		{
			exceptIndex.push_back(i);
		}
		cv::Mat tmpMin(inputM.rows, 1, CV_32FC1, min);
		vm.push_back((col - tmpMin)*1.0 / (max - min));
		standard_setting << min << " " << max << std::endl;
	}
	cv::hconcat(vm, outputM);
	standard_setting.close();

	outputM = deleteCols(outputM, exceptIndex, exceptIndex.size());

	
}
void normalization_predict(cv::Mat &inputM, cv::Mat &outputM, char *filename)
{
	std::ifstream standard_setting("standard_setting.ini");
	double max, min;
	std::vector<cv::Mat> vm;
	std::vector<int> exceptIndex;
	for (int i = 0; i < inputM.cols; i++)
	{
		standard_setting >> min >> max;
		cv::Mat col = inputM.col(i);
		if (max == min)
		{
			exceptIndex.push_back(i);
		}
		cv::Mat tmpMin(inputM.rows, 1, CV_32FC1, min);
		vm.push_back((col - tmpMin)*1.0 / (max - min));
		
	}
	cv::hconcat(vm, outputM);
	standard_setting.close();

	outputM = deleteCols(outputM, exceptIndex, exceptIndex.size());
}


int main(int argc, char** argv)
{
	CSimpleIniExt ini;
	if (argc < 1 || ini.LoadFile(argv[1]) != SI_OK) {
		std::cerr << "Load ini Failed.";
		return 0;
	}
	std::string videofile;
	ini.GetValueExt("video_loader", "video_fname", videofile);
	cv::VideoCapture vc(videofile);
	
	Camera camera;
	camera.init(ini, "camera_settings");
	InitLUT();
	 /******************features check part********************/
	//cv::Mat frame, ipm, ipm_gray,grad;
	//
	//float a_left, b_left, a_right, b_right;
	//std::vector<float> desc;
	//bool ret = vc.read(frame);
	//compute_roi(camera, frame.rows, frame.cols, a_left, b_left, a_right, b_right);
	//int frame_count = 0;
	//while (ret)
	//{
	//	std::cout << "Frame: " << frame_count <<std::endl;
	//	float * lbpFeat, *hogFeat;
	//	std::vector<ContourPtr> ipm_contours;
	//	camera.GetIPMImage(frame, ipm);
	//	cv::cvtColor(ipm, ipm_gray, cv::COLOR_BGR2GRAY);
	//	ScanlineContours(ipm_gray, ipm_contours, 20.f, 2, 12, 5, a_left, b_left, a_right, b_right);
	//	compute_gradient(ipm_gray, grad);
	//	lbpFeat = lbpFeatPerFrame(ipm_gray);
	//	hogFeat = hogFeatPerFrame(ipm_gray);
	//	
	//	//std::vector<int> hist;
	//	int contour_count = 0;
	//	for (std::vector<ContourPtr>::iterator it = ipm_contours.begin(); it != ipm_contours.end(); it += 2)
	//	{
	//		std::cout << "Contour pair: " << contour_count << std::endl;
	//		desc = compute_descriptor(grad,lbpFeat,hogFeat,*(*it), *(*(it + 1)),8);
	//		
	//		//hist = compute_chaincode_hist(*(*it), *(*(it + 1)));
	//		/*for (std::vector<int>::iterator q = hist.begin(); q != hist.end(); q++)
	//		{
	//			std::cout << *q << " ";
	//		}
	//		std::cout << std::endl;*/
	//		contour_count++;
	//	}
	//	std::cout << "frame " << frame_count << " is done " << std::endl;
	//	//system("pause");
	//	//cv::waitKey(0);
	//	ret = vc.read(frame);
	//	frame_count++;
	//}
	//
	//
	//return 0;
	//
	//
	 /********************prediction for per frame**************************/
	/*cv::Mat m;
	Ptr<RTrees> rtrees = ml::RTrees::load<RTrees>("C:\\Users\\lvxubo\\Desktop\\SWT\\rfc\\rtrees_2.xml");
	std::vector<float> setting;
	std::ifstream standard_setting("standard_setting.ini");
	float min, max;
	for (int l = 0; l < CountLines("standard_setting.ini"); l++)
	{
		standard_setting >> min >> max;
		setting.push_back(min);
		setting.push_back(max);
	}
	standard_setting.close();
	
	for (int z = 25000; z < 30000; z++)
	{
		getSingleFrame(vc, z, m);
		
		predict_test(camera, m, rtrees,setting);
		cv::waitKey(1);
	}
	demo.release();
	vc.release();
	return 0;*/
	

	 /*******from video frames obtain original samples divided to train part and test part********/
	//cv::Mat frame;
	//bool ret = vc.read(frame);
	//float a_left, b_left, a_right, b_right;
	//compute_roi(camera, frame.rows, frame.cols, a_left, b_left, a_right, b_right);	
	//
	//
	//
	//std::map<int, std::vector<float>> baseLineParamMap;
	//GetFrameBaseLineParam(baseLineParamMap);
	//
	//cv::Mat m, ipm, ipm_gray, ipm_grad, mergeForSift;
	//std::vector<cv::Mat> Features_v, Labels_v;
	////vc.get(CV_CAP_PROP_FRAME_COUNT)
	//for (int i = 0; i < 20000; i++)
	//{
	//	if ((baseLineParamMap.find(i)!= baseLineParamMap.end()))
	//	{
	//		std::vector<ContourPtr> ipm_contours,positive_contours, negative_contours, resampleNegtiveContours;
	//		getSingleFrame(vc, i, m);
	//		camera.GetIPMImage(m, ipm);
	//		cv::cvtColor(ipm, ipm_gray, cv::COLOR_BGR2GRAY);
	//		ScanlineContours(ipm_gray, ipm_contours, 20.f, 2, 12, 5, a_left, b_left, a_right, b_right);
	//		computeContours(camera, ipm_contours, i, positive_contours, negative_contours, baseLineParamMap);
	//		resampleNegtiveContours = resampleNegtive(negative_contours, positive_contours, 1.5);
	//		
	//		
	//
	//		compute_gradient(ipm_gray, ipm_grad, mergeForSift);
	//		float * lbpFeat, *hogFeat;
	//		lbpFeat = lbpFeatPerFrame(ipm_gray);
	//		hogFeat = hogFeatPerFrame(ipm_gray);
	//		VlSiftFilt  * sift = vl_sift_new(ipm_gray.cols, ipm_gray.rows, 4, 5, 0);
	//
	//		cv::Mat DM = constructDescMat(ipm_gray,ipm_grad,mergeForSift, lbpFeat, hogFeat,sift, 8, resampleNegtiveContours,positive_contours, 450);
	//		cv::Mat LM = constructLabelMat(resampleNegtiveContours, positive_contours);
	//
	//		Features_v.push_back(DM);
	//		Labels_v.push_back(LM);
	//
	//		DrawContourss(ipm, positive_contours);
	//		DrawContourss(ipm, resampleNegtiveContours);
	//		cv::imshow("ipm", ipm);
	//		cv:waitKey(1);
	//		vl_sift_delete(sift);
	//
	//	}
	//
	//}
	//cv::Mat Features_m, Labels_m;
	//
	//vconcat(Features_v, Features_m);
	//vconcat(Labels_v, Labels_m);
	//
	//// split total samples to train part and test part
	//cv::Mat trainFeatures, trainLabels, testFeatures, testLabels;
	//sampleSplit(Features_m, 3, trainFeatures, testFeatures);
	//sampleSplit(Labels_m, 3, trainLabels, testLabels);
	//
	//
	//
	//write_mat_to_file(trainFeatures, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\trainFeaturesMat.txt","out","float");
	//write_mat_to_file(trainLabels, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\trainLablesMat.txt", "out", "int");
	//write_mat_to_file(testFeatures, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\testFeaturesMat.txt", "out", "float");
	//write_mat_to_file(testLabels, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\testLabelsMat.txt", "out", "int");

	/*********************normalization with considering abnormal issues***********/
	//train preparation for instant model training 
	/*cv::Mat Features, wideFeatures, Labels, wideLabels;
	read_mat_from_file(Features, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\trainFeaturesMat.txt", "float");
	read_mat_from_file(wideFeatures, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\wideLaneFeaturesMat.txt", "float");
	read_mat_from_file(Labels, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\trainLablesMat.txt", "int");
	read_mat_from_file(wideLabels, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\wideLaneLabelsMat.txt", "int");

	vconcat(Features, wideFeatures, Features);
	vconcat(Labels, wideLabels, Labels);
	cv::Mat norm_f;
	normalization_train(Features, norm_f, "standard_setting.ini");
	std::cout << "norm done!!" << std::endl;*/
	
	// predict preparation for model predicting
	/*cv::Mat Features,Labels;
	read_mat_from_file(Features, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\testFeaturesMat.txt", "float");
	read_mat_from_file(Labels, "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\testLabelsMat.txt", "int");
	std::cout << "read done!!" << std::endl;
	cv::Mat norm_f;
	normalization_predict(Features, norm_f, "standard_setting.ini");
	std::cout << "norm done!!" << std::endl;*/



	/*********************SVM PART : TRAIN***************/
	/*Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

	std::cout << "SVM Training start..." << std::endl;

	Ptr<ml::TrainData> td = ml::TrainData::create(standard_SF, ml::ROW_SAMPLE, standard_SL);

	svm->train(td);
	svm->save("C:\\Users\\lvxubo\\Desktop\\SWT\\SVM_Model.xml");
	std::cout << "SVM Training Complete" << std::endl;*/

	/*********************SVM PART : PREDICT***************/
	//cv::Mat test_SL;
	//float precision, recall;
	//Ptr<SVM> svm = ml::SVM::load<SVM>("C:\\Users\\lvxubo\\Desktop\\SWT\\SVM_Model.xml");
	//svm ->predict(standard_SF, test_SL);
	//
	////std::cout << standard_SL.type() << std::endl;
	////std::cout << test_SL.type();
	//
	//
	//cal_evaluation(standard_SL, test_SL, precision, recall);
	//std::cout << "precision: " << precision << "\n";
	//std::cout << "recall: " << recall << "\n";


	/**********************RANDOM FOREST PART : TRAIN*******************/
	/*auto rtrees = cv::ml::RTrees::create();
	rtrees->setMaxDepth(10);
	rtrees->setMinSampleCount(2);
	rtrees->setRegressionAccuracy(0);
	rtrees->setUseSurrogates(false);
	rtrees->setMaxCategories(16);
	rtrees->setPriors(cv::Mat());
	rtrees->setCalculateVarImportance(false);
	rtrees->setActiveVarCount(0);
	rtrees->setTermCriteria({ cv::TermCriteria::MAX_ITER, 100, 0 });

	std::cout << "rtrees train starts:..." << std::endl;
	rtrees->train(norm_f ,cv::ml::ROW_SAMPLE, Labels);
	rtrees->save("C:\\Users\\lvxubo\\Desktop\\SWT\\rfc\\rtrees_2.xml");
	std::cout << "rtrees train ends:..." << std::endl;*/


	/**********************RANDOM FOREST PART : PREDICT*******************/
	//cv::Mat test_SL;
	//float precision, recall;
	//Ptr<RTrees> rtrees2 = ml::RTrees::load<RTrees>("C:\\Users\\lvxubo\\Desktop\\SWT\\rfc\\rtrees_1.xml");
	//rtrees2->predict(norm_f, test_SL);
	//
	////std::cout << standard_SL.type() << std::endl;
	////std::cout << test_SL.type();
	//
	//
	//cal_evaluation(Labels, test_SL, precision, recall);
	//std::cout << "precision: " << precision << "\n";
	//std::cout << "recall: " << recall << "\n";

	

	system("pause");
	return 0;
	
}

//int main(int argc, char** argv)
//{
//	CSimpleIniExt ini;
//	if (argc < 1 || ini.LoadFile(argv[1]) != SI_OK) {
//		std::cerr << "Load ini Failed.";
//		return 0;
//	}
//	std::string videofile;
//	ini.GetValueExt("video_loader", "video_fname", videofile);
//	cv::VideoCapture vc(videofile);
//
//	if (!vc.isOpened())
//	{
//		std::cout << "fail to open!" << std::endl;
//	}
//		
//	Camera camera;
//	camera.init(ini, "camera_settings");
//	InitLUT();
//
//	/*****************************part for extraction validation test****************/
//	/*vc >> image;
//	float  a_left = 0.0, b_left = 0.0, a_right = 0.0, b_right = 0.0;
//	bool ret = vc.read(image);
//	compute_roi(camera, image.rows, image.cols, a_left, b_left, a_right, b_right);
//
//	cv::Mat m, ipm, ipm_gray;
//	std::vector<ContourPtr> ipm_contours, draw_contours;
//
//	getSingleFrame(vc, 20, m);
//	camera.GetIPMImage(m, ipm);
//	cv::cvtColor(ipm, ipm_gray, cv::COLOR_BGR2GRAY);
//	ScanlineContours(ipm_gray, ipm_contours, 20.f, 2, 12, 5, a_left, b_left, a_right, b_right);
//	
//	draw_contours.push_back(ipm_contours[528]);
//	draw_contours.push_back(ipm_contours[529]);
//	draw_contours.push_back(ipm_contours[680]);
//	draw_contours.push_back(ipm_contours[681]);
//	
//	namedWindow("test", 1);
//	DrawContourss(ipm, draw_contours);
//	imshow("test", ipm);
//	cv::waitKey(0);*/
//
//	/****************************part for reproduce extraction result******************/
//	/* 1.get wide lane location information (including frameNum and lane index)*/
//
	//std::ifstream f_wideLine("C:\\Users\\lvxubo\\Desktop\\SWT\\manual\\wideLineInfo.ini");
//	//std::string s;
//	//int currentFrame = 0, currentIndex1 = 0, currentIndex2 = 0, oldFrame = 0;
//
//	//std::map<int, std::vector<int>> wideParamMap;
//	//while (getline(f_wideLine, s))
//	//{
//	//	std::stringstream ss(s);
//	//	ss >> currentFrame >> currentIndex1 >> currentIndex2;
//	//	if (currentFrame != oldFrame)
//	//	{
//	//		std::vector<int> indexesPeriod;
//	//		wideParamMap[currentFrame] = indexesPeriod;
//	//		oldFrame = currentFrame;
//	//		wideParamMap[currentFrame].push_back(currentIndex1);
//	//		wideParamMap[currentFrame].push_back(currentIndex2);
//	//	}
//	//	else
//	//	{
//	//		wideParamMap[currentFrame].push_back(currentIndex1);
//	//		wideParamMap[currentFrame].push_back(currentIndex2);
//	//	}
//	//}
//	//f_wideLine.close();
//
//
//	/* 2. convert wide lane samples into positive samples feature mat*/
//
//	//float  a_left = 0.0, b_left = 0.0, a_right = 0.0, b_right = 0.0;
//	//bool ret = vc.read(image);
//	//if (ret)
//	//{
//	//	compute_roi(camera, image.rows, image.cols, a_left, b_left, a_right, b_right);
//	//}
//
//
//	//cv::Mat m, ipm, ipm_gray, ipm_grad, mergeForSift;
//
//
//	//std::vector<cv::Mat> allDescForWide, allLabelForWide;
//	//cv::Mat allDescForWideMat, allLabelForWideMat;
//	//for (std::map<int, std::vector<int>>::iterator it_map = wideParamMap.begin(); it_map != wideParamMap.end(); it_map++)
//	//{
//	//	std::vector<ContourPtr> ipm_contours,  wide_contours, empty;
//	//	getSingleFrame(vc, (*it_map).first, m);
//	//	camera.GetIPMImage(m, ipm);
//	//	cv::cvtColor(ipm, ipm_gray, cv::COLOR_BGR2GRAY);
//	//	ScanlineContours(ipm_gray, ipm_contours, 20.f, 2, 12, 5, a_left, b_left, a_right, b_right);
//	//	for (int i = 0; i < (*it_map).second.size(); i++)
//	//	{
//
//	//		wide_contours.push_back(ipm_contours[(*it_map).second[i]]);
//
//	//	}
//	//	DrawContourss(ipm, wide_contours);
//
//	//	compute_gradient(ipm_gray, ipm_grad, mergeForSift);
//	//	float * lbpFeat, *hogFeat;
//	//	lbpFeat = lbpFeatPerFrame(ipm_gray);
//	//	hogFeat = hogFeatPerFrame(ipm_gray);
//	//	VlSiftFilt  * sift = vl_sift_new(ipm_gray.cols, ipm_gray.rows, 4, 5, 0);
//
//	//	cv::Mat DM = constructDescMat(ipm_gray, ipm_grad, mergeForSift, lbpFeat, hogFeat, sift, 8, empty, wide_contours, 450);
//	//	cv::Mat LM = constructLabelMat(empty, wide_contours);
//
//	//	allDescForWide.push_back(DM);
//	//	allLabelForWide.push_back(LM);
//
//	//	vl_sift_delete(sift);
//	//	imshow("test", ipm);
//	//	cv::waitKey(1);
//	//}
//	//vconcat(allDescForWide, allDescForWideMat);
//	//vconcat(allLabelForWide, allLabelForWideMat);
//
//	////std::cout << CountLines("C:\\Users\\lvxubo\\Desktop\\SWT\\manual\\wideLineInfo.ini") << "\n";
//	////std::cout << allDescForWideMat.rows << " " << allLabelForWideMat.rows;
//
//	/*3. gathering negative samples in order to match with positive samples,keep ratio balanced*/
//
//	//std::map<int, std::vector<float>> baseLineParamMap;
//	//int negativeNum = 0;
//	//std::vector<cv::Mat> allDescNegative, allLabelNegative;
//	//cv::Mat allDescNegMat, allLabelNegMat;
//
//	//GetFrameBaseLineParam(baseLineParamMap);
//	//for (int i = 0; i < vc.get(CV_CAP_PROP_FRAME_COUNT); i++)
//	//{
//	//	if ((baseLineParamMap.find(i)!= baseLineParamMap.end()))
//	//	{
//	//		std::vector<ContourPtr> ipm_contours,positive_contours, negative_contours, resampleNegtiveContours, empty;
//	//		getSingleFrame(vc, i, m);
//	//		camera.GetIPMImage(m, ipm);
//	//		cv::cvtColor(ipm, ipm_gray, cv::COLOR_BGR2GRAY);
//	//		ScanlineContours(ipm_gray, ipm_contours, 20.f, 2, 12, 5, a_left, b_left, a_right, b_right);
//	//		
//	//		computeContours(camera, ipm_contours,i, positive_contours, negative_contours,baseLineParamMap);
//	//		resampleNegtiveContours = resampleNegtive(negative_contours, positive_contours,1.5);
//	//		
//	//		negativeNum += resampleNegtiveContours.size() / 2;
//	//		if (negativeNum > 1299 * 1.5)
//	//		{
//	//			break;
//	//		}
//	//		DrawContourss(ipm, resampleNegtiveContours);
//
//
//	//		compute_gradient(ipm_gray, ipm_grad, mergeForSift);
//	//		float * lbpFeat, *hogFeat;
//	//		lbpFeat = lbpFeatPerFrame(ipm_gray);
//	//		hogFeat = hogFeatPerFrame(ipm_gray);
//	//		VlSiftFilt  * sift = vl_sift_new(ipm_gray.cols, ipm_gray.rows, 4, 5, 0);
//	//		cv::Mat DM = constructDescMat(ipm_gray, ipm_grad, mergeForSift, lbpFeat, hogFeat, sift, 8, resampleNegtiveContours, empty, 450);
//	//		cv::Mat LM = constructLabelMat(resampleNegtiveContours, empty);
//	//		allDescNegative.push_back(DM);
//	//		allLabelNegative.push_back(LM);
//
//	//		vl_sift_delete(sift);
//	//		imshow("test", ipm);
//	//		cv::waitKey(1);
//	//		
//	//	}
//
//	//}
//	//vconcat(allDescNegative, allDescNegMat);
//	//vconcat(allLabelNegative, allLabelNegMat);
//
//	//char *filename1 = "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\wideLaneFeaturesMat.txt";
//	//char *filename2 = "C:\\Users\\lvxubo\\Desktop\\SWT\\Features\\wideLaneLabelsMat.txt";
//	//for (int i = 0; i < 5; i++)
//	//{
//	//	write_mat_to_file(allDescForWideMat, filename1, "app","float");
//	//	write_mat_to_file(allDescNegMat, filename1,"app","float");
//
//	//	write_mat_to_file(allLabelForWideMat, filename2, "app","int");
//	//	write_mat_to_file(allLabelNegMat, filename2, "app","int");
//
//	//}
//
//	//std::cout << CountLines(filename1) << std::endl;
//	//std::cout << CountLines(filename2) << std::endl;
//	
//	
//	
//	
//	/*************************************tool for extract specific samples*******************/
//	//std::ifstream fin("C:\\Users\\lvxubo\\Desktop\\SWT\\Yaml\\OV10635_20160518_1547_lane.txt");
//	//if (!fin.is_open())
//	//{
//	//	std::cout << "load file error" << std::endl;
//	//	return 0;
//	//}
//
//	//std::string s;
//	//int frameNum, realContourNum;
//	//float a=0.0, b=0.0, c=0.0;
//	//
//	//std::map<int, std::vector<float>> frameIdMap;
//	//int lineNum = 0;
//	//
//	//
//	//
//	//while (getline(fin, s))
//	//{
//	//	
//	//	std::stringstream ss(s);
//	//	
//	//	std::vector<float> tmpParamPerFrame;
//	//	ss >> frameNum >> realContourNum;
//	//	for (int i = 0; i < realContourNum; i++)
//	//	{
//	//		
//	//		ss >> c >> b >> a;
//	//		tmpParamPerFrame.push_back(a);
//	//		tmpParamPerFrame.push_back(b);
//	//		tmpParamPerFrame.push_back(c);
//	//	}
//	//	frameIdMap[frameNum] = tmpParamPerFrame;
//	//	
//	//	
//	//	
//	//}
//
//	//fin.close();
//
//	//
//	//namedWindow("video", 1);//显示视频原图像的窗口	
//	//namedWindow("sub video", 2);
//	//namedWindow("ipm_gray", 3);
//
//	//setMouseCallback("ipm_gray", onMouse,0);//捕捉鼠标
//
//	//float a_left, b_left, a_right, b_right;
//	//bool ret = vc.read(image);
//	//compute_roi(camera, image.rows, image.cols, a_left, b_left, a_right, b_right);
//	//
//	//int frameIndex;
//	//
//	//std::ofstream wideLineInfo("C:\\Users\\lvxubo\\Desktop\\SWT\\manual\\wideLineInfo.ini");
//	//while (true)
//	//{
//	//	std::vector<ContourPtr> ipm_contours;
//	//	if (!pauseFlag) //判定鼠标右键没有按下，采取播放视频，否则暂停  
//	//	{ 
//	//		vc >> image;
//	//		frameIndex = vc.get(cv::CAP_PROP_POS_FRAMES);
//	//
//	//		std::cout << frameIndex << "\n";
//	//		
//	//		//std::map<int, std::vector<float>>::iterator it_map = frameIdMap.find(frameIndex);;
//	//		
//	//		camera.GetIPMImage(image, ipm);
//	//		cv::cvtColor(ipm, ipm_gray, cv::COLOR_BGR2GRAY); 
//	//		/*if (it_map != frameIdMap.end())
//	//		{
//	//			std::cout << "VID:" << frameIndex << std::endl;
//	//			
//	//			
//	//		}*/
//
//	//	}
//	//	else
//	//	{
//	//		
//	//		ScanlineContours(ipm_gray, ipm_contours, 20.f, 2, 12, 5, a_left, b_left, a_right, b_right);
//	//		
//	//		if (rectDone)
//	//		{
//	//			frameIndex = vc.get(cv::CAP_PROP_POS_FRAMES);
//	//			//std::cout << frameIndex << "\n";
//	//			int contour_index = 0;
//	//			Rect rect(originalPoint, endPoint);
//	//			//std::cout << rect.width << std::endl;
//	//			//std::cout << "ori pt: " << originalPoint << "end pt: " << endPoint << std::endl;
//	//			for (std::vector<ContourPtr>::iterator pi = ipm_contours.begin(); pi != ipm_contours.end(); pi += 2)
//	//			{
//	//				if (contains(rect, *(*pi), *(*(pi + 1))))
//	//				{
//	//					std::cout << frameIndex-1 << " " << contour_index * 2 << " " << contour_index * 2 + 1 << "\n"; //the pair index !!!
//	//					wideLineInfo << frameIndex - 1 << " " << contour_index * 2 << " " << contour_index * 2 + 1 << "\n";
//	//				}
//	//				contour_index++;
//	//			}
//	//			
//	//			rectDone = false; //in case of duplicate calculation
//	//		}
//
//	//	}
//
//	//	if (!image.data || waitKey(33) == 27)  //图像为空或Esc键按下退出播放  
//	//	{
//	//		break;
//	//		
//	//	}
//
//	//	imshow("video", image);
//	//	if (!ipm_gray.empty())
//	//	{
//	//		imshow("ipm_gray", ipm_gray);
//	//		cv::waitKey(100);
//	//	}
//	//	
//	//	
//	//}
//	//wideLineInfo.close();
//	/*vc.release();
//	system("pause");
//	return 0;*/	
//}


//int main(int argc, char** argv)
//{
//	Mat a = (Mat_<int>(5, 5) << 1, 2, 3, 4, 5,
//		7, 8, 9, 10, 11,
//		12, 13, 14, 15, 16,
//		17, 18, 19, 20, 21,
//		22, 23, 24, 25, 26);
//	//std::cout <<a.rowRange(2, 3);
//	int cols[] = { 0,3,4 };
//	std::cout << deleteCols(a,cols, 3);
//	system("pause");
//	return 0;
//}