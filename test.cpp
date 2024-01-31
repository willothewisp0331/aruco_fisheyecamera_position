// DpvrServiceCmdTest.cpp
#include <thread>
#include "DpvrSharedFisheyeImage.h"
using namespace std;
#include <Windows.h>
#include <conio.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
using namespace cv;
#pragma comment(lib, "Winmm.lib")
#pragma comment(lib, "C:\\ProgramData\\DPVR Assistant 4\\SDK\\lib\\DpvrSharedFisheyeImage_x64.lib")

HRESULT InitWindow(HINSTANCE hInstance, int nCmdShow) { return (HRESULT)0; }
HRESULT InitDevice() { return (HRESULT)0; }
void CleanupDevice() {}
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM) { return (LRESULT)0; }
void Render() {}

void UpdateTexture(void* pixels0, void* pixels1, void* pixels2, void* pixels3) {}

std::mutex imageLocker;

dpvr_fisheye_pixel* g_pixels0 = new dpvr_fisheye_pixel[640 * 480];
dpvr_fisheye_pixel* g_pixels1 = new dpvr_fisheye_pixel[640 * 480];
dpvr_fisheye_pixel* g_pixels2 = new dpvr_fisheye_pixel[640 * 480];
dpvr_fisheye_pixel* g_pixels3 = new dpvr_fisheye_pixel[640 * 480];

class RateCounter
{
	int rate = 0;
	DWORD _currTime = 0;
	unsigned short color_index = 0;
public:
	RateCounter(unsigned short color_index)
		:color_index(color_index)
	{
		_currTime = timeGetTime();
	}

	void RunFrame(const char* mark)
	{
		auto currTime = timeGetTime();
		auto deltaTime = currTime - _currTime;
		if (deltaTime > 1000)
		{
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color_index);
			printf("%s : rate = %.3f HZ\n", mark, rate * 1000 / (float)deltaTime);
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY);

			rate = 0;
			_currTime = currTime;
		}
		rate++;
	}
};

bool isRotationMatrix(const Mat& R)
{
	Mat Rt;
	transpose(R, Rt);
	Mat shouldBeIdentity = Rt * R;
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());
	return norm(I, shouldBeIdentity) < 1e-6;
}

Vec3f rotationMatrixToEulerAngles(const Mat& R)
{
	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6;

	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else
	{
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}

	return Vec3f(x, y, z);
}

Mat ConvertPixelsToMat(dpvr_fisheye_pixel* pixels, int width = 640, int height = 480)
{
	Mat imageMat(height, width, CV_8UC1, pixels);
	return imageMat.clone();
}

Mat Detector(Mat image)
{
	Mat K = (Mat_<double>(3, 3) << 158.92757205, 0., 152.07775471, 0., 164.87005228, 203.70307396, 0., 0., 1.);
	Mat D = (Mat_<double>(1, 4) << -0.09769812, 0.36214841, -0.41720733, 0.15493034);

	Mat imageCopy, imageCopy_r, flipMatrix;
	resize(image, imageCopy, Size(400, 300), 0, 0, INTER_LINEAR);
	rotate(imageCopy, imageCopy_r, ROTATE_90_CLOCKWISE);

	vector<int> ids;
	vector<vector<Point2f>> corners, rejectedCandidates;
	Ptr<aruco::DetectorParameters> parameters;
	Ptr<aruco::Dictionary> aruco_dict = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

	vector<Vec3d> rvecs, tvecs;

	flipMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, -1);

	aruco::detectMarkers(imageCopy_r, aruco_dict, corners, ids);

	if (ids.size() > 0)
	{
		aruco::estimatePoseSingleMarkers(corners, 0.14, K, D, rvecs, tvecs);
		for (int i = 0; i < ids.size(); i++)
		{
			double fontScale = 0.5;
			int thickness = 1;

			aruco::drawAxis(imageCopy_r, K, D, rvecs[i], tvecs[i], 0.07);
			Mat matTmp, matTmpTransposed, test;
			Rodrigues(rvecs[i], matTmp);
			transpose(matTmp, matTmpTransposed);
			Vec3f eulerAngles = rotationMatrixToEulerAngles(matTmpTransposed);

			stringstream yawss, pitchss, rollss;
			yawss << "Yaw: " << fixed << setprecision(1) << eulerAngles[0] * 180.0 / CV_PI;
			pitchss << "Pitch: " << fixed << setprecision(1) << eulerAngles[1] * 180.0 / CV_PI;
			rollss << "Roll: " << fixed << setprecision(1) << eulerAngles[2] * 180.0 / CV_PI;
			string pitchText = pitchss.str(); 
			string yawText = yawss.str(); 
			string rollText = rollss.str();
			
			Size yawTextSize = getTextSize(yawText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);
			Size pitchTextSize = getTextSize(pitchText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);
			Size rollTextSize = getTextSize(rollText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);

			Point yawTextOrg(10, 250);
			Point pitchTextOrg(10, 270);
			Point rollTextOrg(10, 290);

			int rectMargin = 5;
			Rect yawRect(yawTextOrg.x - rectMargin, yawTextOrg.y - yawTextSize.height - rectMargin,
				yawTextSize.width + 2 * rectMargin, yawTextSize.height + 2 * rectMargin);
			Rect pitchRect(pitchTextOrg.x - rectMargin, pitchTextOrg.y - pitchTextSize.height - rectMargin,
				pitchTextSize.width + 2 * rectMargin, pitchTextSize.height + 2 * rectMargin);
			Rect rollRect(rollTextOrg.x - rectMargin, rollTextOrg.y - rollTextSize.height - rectMargin,
				rollTextSize.width + 2 * rectMargin, rollTextSize.height + 2 * rectMargin);

			rectangle(imageCopy_r, yawRect, Scalar(255, 255, 255), -1);
			rectangle(imageCopy_r, pitchRect, Scalar(255, 255, 255), -1);
			rectangle(imageCopy_r, rollRect, Scalar(255, 255, 255), -1);

			putText(imageCopy_r, yawText, yawTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
			putText(imageCopy_r, pitchText, pitchTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
			putText(imageCopy_r, rollText, rollTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);

			Vec3d tvecs_flipped = tvecs[0] * -1;

			float tx = tvecs_flipped[0] * 100;
			float ty = tvecs_flipped[1] * 100;
			float tz = tvecs_flipped[2] * 100;
			float distance = sqrt(tx * tx + ty * ty + tz * tz);

			stringstream xss, yss, zss, dss;
			xss << "x: " << fixed << setprecision(2) << tx << " cm";
			yss << "y: " << fixed << setprecision(2) << ty << " cm";
			zss << "z: " << fixed << setprecision(2) << tz << " cm";
			dss << "dist: " << fixed << setprecision(2) << distance << " cm";
			string xText = xss.str();
			string yText = yss.str();
			string zText = zss.str();
			string distanceText = dss.str();

			Size xTextSize = getTextSize(xText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);
			Size yTextSize = getTextSize(yText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);
			Size zTextSize = getTextSize(zText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);
			Size distanceTextSize = getTextSize(distanceText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);

			Point xTextOrg(10, 330);
			Point yTextOrg(10, 350);
			Point zTextOrg(10, 370);
			Point distanceTextOrg(10, 390);

			Rect xRect(xTextOrg.x - rectMargin, xTextOrg.y - xTextSize.height - rectMargin,
				xTextSize.width + 2 * rectMargin, xTextSize.height + 2 * rectMargin);
			Rect yRect(yTextOrg.x - rectMargin, yTextOrg.y - yTextSize.height - rectMargin,
				yTextSize.width + 2 * rectMargin, yTextSize.height + 2 * rectMargin);
			Rect zRect(zTextOrg.x - rectMargin, zTextOrg.y - zTextSize.height - rectMargin,
				zTextSize.width + 2 * rectMargin, zTextSize.height + 2 * rectMargin);
			Rect distanceRect(distanceTextOrg.x - rectMargin, distanceTextOrg.y - distanceTextSize.height - rectMargin,
				distanceTextSize.width + 2 * rectMargin, distanceTextSize.height + 2 * rectMargin);

			rectangle(imageCopy_r, xRect, Scalar(255, 255, 255), -1);
			rectangle(imageCopy_r, yRect, Scalar(255, 255, 255), -1);
			rectangle(imageCopy_r, zRect, Scalar(255, 255, 255), -1);
			rectangle(imageCopy_r, distanceRect, Scalar(255, 255, 255), -1);

			putText(imageCopy_r, xText, xTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
			putText(imageCopy_r, yText, yTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
			putText(imageCopy_r, zText, zTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
			putText(imageCopy_r, distanceText, distanceTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
		}
	}
	else
	{
		putText(imageCopy_r, "No detection", cv::Point(30, 64), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, LINE_AA);
	}

	return imageCopy_r;
}

void OnFisheyeImageUpdate(dpvr_Pose* pose, dpvr_fisheye_pixel* imagePixels0, dpvr_fisheye_pixel* imagePixels1, dpvr_fisheye_pixel* imagePixels2, dpvr_fisheye_pixel* imagePixels3)
{
	//const int pixel_size = 640 * 480;
	//dpvr_fisheye_pixel* pixels = new dpvr_fisheye_pixel[pixel_size];
	//memcpy(pixels, imagePixels0, pixel_size);

	//static int index = 0;
	//thread thread0([&](dpvr_fisheye_pixel* pixels)
	//	{
	//		char name[128] = {};
	//		sprintf(name, "images/fisheye_image_0_%d.png", index++);
	//		stbi_write_png(name, 640, 480, 1, pixels, 640);

	//		delete pixels;

	//	}, pixels);

	//thread0.detach();
	printf("fisheye : time = %lf, position(%lf, %lf, %lf) rotation(%lf, %lf, %lf, %lf) velocity(%lf, %lf, %lf) angularVelocity(%lf, %lf, %lf) acceleration(%lf, %lf, %lf) angularAcceleration(%lf, %lf, %lf)\n",
		pose->hostTimestamp, pose->position.x, pose->position.y, pose->position.z, pose->rotation.w, pose->rotation.x, pose->rotation.y, pose->rotation.z,
		pose->velocity.x, pose->velocity.y, pose->velocity.z, pose->angularVelocity.x, pose->angularVelocity.y, pose->angularVelocity.z,
		pose->acceleration.x, pose->acceleration.y, pose->acceleration.z, pose->angularAcceleration.x, pose->angularAcceleration.y, pose->angularAcceleration.z);

	static RateCounter rate(FOREGROUND_RED);
	rate.RunFrame(__FUNCTION__);

	imageLocker.lock();

	memcpy(g_pixels0, imagePixels0, 640 * 480);
	memcpy(g_pixels1, imagePixels1, 640 * 480);
	memcpy(g_pixels2, imagePixels2, 640 * 480);
	memcpy(g_pixels3, imagePixels3, 640 * 480);

	imageLocker.unlock();
}

void OnHmdImuUpdated(dpvr_HmdImu* hmdImu)
{
	if (hmdImu == nullptr)
		return;

	static RateCounter rate(FOREGROUND_GREEN);
	rate.RunFrame(__FUNCTION__);

	printf("hmdImu : time = %lf, gyro(%lf, %lf, %lf) accel(%lf, %lf, %lf) accelSaturation(%s, %s, %s) magneto(%lf, %lf, %lf) temperature = %lf\n",
		hmdImu->hostTimestamp, hmdImu->gyro.x, hmdImu->gyro.y, hmdImu->gyro.z, hmdImu->accel.x, hmdImu->accel.y, hmdImu->accel.z,
		hmdImu->accelSaturation.x ? "true" : "false", hmdImu->accelSaturation.y ? "true" : "false", hmdImu->accelSaturation.z ? "true" : "false",
		hmdImu->magneto.x, hmdImu->magneto.y, hmdImu->magneto.z, hmdImu->temperature);
}

int main1()
{


	// Main message loop
	MSG msg = { 0 };
	while (WM_QUIT != msg.message)
	{
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			Render();
		}
	}

	CleanupDevice();

	return (int)msg.wParam;
}

int main(int argc, char* argv[])
{
	// new
	void* fisheyeImage = dpvr_SharedFisheyeImage_new();

	// open
	auto result = dpvr_SharedFisheyeImage_Open(fisheyeImage);
	if (!result)
	{
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED);
		printf("device open failed! please startup DPVRAssistant4 first\n");
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY);
		system("pause");
		return -1;
	}

	BOOL nCmdShow = TRUE;

	HINSTANCE hInstance = GetModuleHandle(nullptr);

	if (FAILED(InitWindow(hInstance, nCmdShow)))
		return 0;

	if (FAILED(InitDevice()))
	{
		CleanupDevice();
		return 0;
	}

	// wait for device opened
	dpvr_SharedFisheyeImage_WaitForDeviceOpened(fisheyeImage, -1);

	// while (dpvr_SharedFisheyeImage_WaitForDeviceOpened(image, 100) == false);

	// read calibration
	auto calib0 = dpvr_SharedFisheyeImage_ReadCalibration(fisheyeImage, 0);
	auto calib1 = dpvr_SharedFisheyeImage_ReadCalibration(fisheyeImage, 1);
	auto calib2 = dpvr_SharedFisheyeImage_ReadCalibration(fisheyeImage, 2);
	auto calib3 = dpvr_SharedFisheyeImage_ReadCalibration(fisheyeImage, 3);

	// handle image updated
	dpvr_SharedFisheyeImage_OnFisheyImageUpdate(fisheyeImage, OnFisheyeImageUpdate);

	dpvr_SharedFisheyeImage_OnHmdImuUpdated(fisheyeImage, OnHmdImuUpdated);




	RateCounter rateCounter(FOREGROUND_GREEN | FOREGROUND_RED);
	MSG msg = { 0 };
	while (WM_QUIT != msg.message)
	{
		if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			imageLocker.lock();
			UpdateTexture(g_pixels0, g_pixels1, g_pixels2, g_pixels3);
			imageLocker.unlock();
			Mat image3 = ConvertPixelsToMat(g_pixels3, 640, 480);

			Mat image3_d = Detector(image3); 
			imshow("right", image3_d);
			//Mat image3_d = remapper(image3, mapx, mapy);
			//imshow("right_d", image3_d);
			Render();

			rateCounter.RunFrame("render fish eye image");
		}
	}

	// close
	dpvr_SharedFisheyeImage_Close(fisheyeImage);

	// delete
	dpvr_SharedFisheyeImage_delete(fisheyeImage);

	fisheyeImage = nullptr;


	CleanupDevice();

    return 0;
}

