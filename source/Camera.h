#ifndef __CAMERA__
#define __CAMERA__

#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/opencv.hpp>
#include "SimpleIniExt.h"

namespace cv {
  typedef cv::Rect_<float> Rect2f;
}

class Camera {
public:
	Camera();
	Camera(const Camera& cam);
	Camera(float fu, float fv, float cu, float cv, float rx, float ry, float rz, float tx = 0.0f, float ty = 0.0f, float tz = 0.0f, float theta_rad = M_PI_2);
	
  Camera & operator =(const Camera &cam);
  
  void initCamera(float fu, float fv, float cu, float cv, float rx, float ry, float rz, float tx = 0.0f, float ty = 0.0f, float tz = 0.0f, float theta_rad = M_PI_2);
	void init(const CSimpleIniExt & ini, const char * section_name);

	// Setters
	void setK(float fu, float fv, float cu, float cv, float theta_rad = M_PI_2);
	void setR(float rx, float ry, float rz);
	void setDeltaR(float rx, float ry, float rz);
	void setT(float tx, float ty, float tz);
	void setDeltaT(float dx, float dy, float dz);
	void setIPMInfo(cv::Rect2f roi, int ipm_width, int ipm_height);

	// Transformations
	// Image <=> World
	cv::Point3f cvtImageToWorld(const cv::Point2f& pt, float z = 0);
	cv::Mat cvtImageToWorld(const cv::Mat& pts, float z = 0);
	cv::Point2f cvtWorldToImage(const cv::Point3f& pt);
	cv::Mat cvtWorldToImage(const cv::Mat& pts);

	// Image <=> Ground
  cv::Point2f cvtImageToGround(const cv::Point2f& pt);
  cv::Mat cvtImageToGround(const cv::Mat& pts);
  cv::Point2f cvtGroundToImage(const cv::Point2f& pt);
  cv::Mat cvtGroundToImage(const cv::Mat& pts);

	// Image <=> IPM
	cv::Point2f cvtImageToIPM(const cv::Point2f& pt);
	cv::Mat cvtImageToIPM(const cv::Mat& pts);
	cv::Point2f cvtIPMToImage(const cv::Point2f& pt);
	cv::Mat cvtIPMToImage(const cv::Mat& pts);

	// Ground <=> IPM
  cv::Point2f cvtGroundToIPM(const cv::Point2f& pt);
  cv::Mat cvtGroundToIPM(const cv::Mat& pts);
  cv::Point2f cvtIPMToGround(const cv::Point2f& pt);
  cv::Mat cvtIPMToGround(const cv::Mat& pts);


	// useful utils
	cv::Point2f getVanishingPoint();
	// Get birds' view image
	void GetIPMImage(const cv::Mat& img, cv::Mat& ipm);

	// Inverse Perspective Mapping related members
	cv::Rect2f roi_;
	cv::Mat M_; // ground to image
	cv::Mat Minv_; // image to ground
	cv::Mat T_; // IPM to ground
	cv::Size ipm_size_;

	// Camera related members
	cv::Mat P_; // 3x4, world to image
	cv::Mat Pinv_; // 4x3, image to world
	cv::Mat R_; // 3x3, rotation
	cv::Mat K_; // 3x3
  cv::Mat Kinv_;
	cv::Mat t_; // 3x1
  // Camera position, according to the center of world's coordinate (the center of front axle)
	cv::Point3f camara_center_; 
	float pitch_, roll_, yaw_; // extrinsic rotations
	float focal_u_, focal_v_; // focal length
	cv::Point2f optical_center_; // principal point
protected:
	void updateCamera();
	void updateIPM();
};

#endif // __CAMERA__