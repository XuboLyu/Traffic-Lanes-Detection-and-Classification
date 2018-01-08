#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
typedef std::vector<cv::Point2f> Contour;
typedef std::shared_ptr<Contour> ContourPtr;
#define PI 3.1415926f

int ScanlineContoursSmoothed(const cv::Mat& gray_ipm, std::vector<ContourPtr>& contours, float thresh, int min_width, int max_width, float max_lr_mag_ratio, float a_left, float b_left, float a_right, float b_right);
int ScanlineContoursImageSpace(const cv::Mat& gray_img, std::vector<ContourPtr>& contours, cv::Point2f& vp, float thresh, int min_width, int max_width, float max_lr_mag_ratio);
int ScanlineContours(const cv::Mat& gray_ipm, std::vector<ContourPtr>& contours, float thresh, int min_width, int max_width, float max_lr_mag_ratio, float a_left, float b_left, float a_right, float b_right);
void DrawContours(const cv::Mat& img, const std::vector<ContourPtr>& contours);
void InitLUT();
static short ang_lut[511][511];
static short mag_lut[256][256];