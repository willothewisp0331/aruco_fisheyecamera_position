// DpvrServiceCmdTest.cpp
#include "DpvrSharedFisheyeImage.h"
#include "output.h"
using namespace std;
#include <Windows.h>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <deque>
#include <algorithm>
#include <numeric>
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

deque<Vec3f> angleHistory3, distHistory3;
deque<Vec3f> angleHistory2, distHistory2;
int windowSize = 50;

struct DetectionResults
{
	Vec3f angles;
	Vec3f dists;
	bool detected = false;
}lastResults3, lastResults2, output;

struct PoseData
{
	Vec3f angles;
	Vec3f dists;
	//0PoseData(Vec3f orient, Vec3f pos) : angles(orient), dists(pos) {}
}offsets3, offsets2;

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

Vec3f mean(const deque<Vec3f>& data) // Mean
{
	if (data.empty()) {
		return Vec3f(0, 0, 0);
	}

	Vec3f sum(0, 0, 0);
	for (const auto& vec : data) {
		sum += vec;
	}

	return sum / static_cast<float>(data.size());
}

Vec3f stddev(const deque<Vec3f>& data, const Vec3f& mean) // Standard Deviation
{
	if (data.size() <= 1) {
		return Vec3f(0, 0, 0);
	}

	Vec3f sumSquares(0, 0, 0);
	for (const auto& vec : data) {
		Vec3f diff = vec - mean;
		sumSquares += diff.mul(diff);
	}
	return Vec3f(
		sqrt(sumSquares[0] / (data.size() - 1)), 
		sqrt(sumSquares[1] / (data.size() - 1)), 
		sqrt(sumSquares[2] / (data.size() - 1))
	);
}

bool updateHistory(deque<Vec3f>& history, const Vec3f& newData)
{
	history.push_back(newData);
	if (history.size() > windowSize)
	{
		history.pop_front();
	}
	
	if (history.size() < windowSize) {
		return false;
	}

	Vec3f currentMean = mean(history);
	Vec3f currentStdDev = stddev(history, currentMean);

	bool isOutlier = false;

	for (int i = 0; i < 3; i++)
	{
		if (abs(newData[i] - currentMean[i]) > 2 * currentStdDev[i])
		{
			isOutlier = true;
			break;
		}
	}

	if (isOutlier)
	{
		history.pop_back();
	}

	return isOutlier;
}

bool isRotationMatrix(const Mat& R) // Check if Matrix is a Rotation Matrix
{
	Mat Rt;
	transpose(R, Rt);
	Mat shouldBeIdentity = Rt * R;
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());
	return norm(I, shouldBeIdentity) < 1e-6;
}

Vec3f rotationMatrixToEulerAngles(const Mat& R) // Convert rotation Matrix to Angles
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

Mat ConvertPixelsToMat(dpvr_fisheye_pixel* pixels, int width = 640, int height = 480) // Convert Pixel to Matrix
{
	Mat imageMat(height, width, CV_8UC1, pixels);
	return imageMat.clone();
}

Mat remapImage(const Mat& image, const Mat& mapx, const Mat& mapy) // Remap Fisheye camera to undistorted image
{
	Mat image_r;
	rotate(image, image_r, ROTATE_90_CLOCKWISE);

	if (mapx.empty() || mapy.empty())
	{
		cerr << "Error: mapx or mapy is empty." << endl;
		return Mat();
	}

	Mat remappedImage;

	remap(image_r, remappedImage, mapx, mapy, INTER_LINEAR);

	return remappedImage;
}

void DrawDetectionResult(Mat& image, DetectionResults results, PoseData offsets) // Draw Detection Result
{

	double fontScale = 0.5;
	int thickness = 1;

	Vec3f eulerAngles = results.angles - offsets.angles;
	Vec3f dists = results.dists - offsets.dists;

	stringstream yawss, pitchss, rollss;
	yawss << "vertical: " << setw(5) << fixed << setprecision(1) << eulerAngles[0];
	pitchss << "horizontal: " << setw(5) << fixed << setprecision(1) << eulerAngles[1];
	rollss << "clockwise: " << setw(5) << fixed << setprecision(1) << eulerAngles[2];
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

	rectangle(image, yawRect, Scalar(255, 255, 255), -1);
	rectangle(image, pitchRect, Scalar(255, 255, 255), -1);
	rectangle(image, rollRect, Scalar(255, 255, 255), -1);

	putText(image, yawText, yawTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
	putText(image, pitchText, pitchTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
	putText(image, rollText, rollTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);

	float tx = dists[0];
	float ty = dists[1];
	float tz = dists[2];
	float distance = sqrt(tx * tx + ty * ty + tz * tz);

	stringstream xss, yss, zss, dss;
	xss << "x: " << setw(5) << fixed << setprecision(2) << tx << " cm";
	yss << "y: " << setw(5) << fixed << setprecision(2) << ty << " cm";
	zss << "z: " << setw(5) << fixed << setprecision(2) << tz << " cm";
	//dss << "dist: " << fixed << setprecision(2) << distance << " cm";
	string xText = xss.str();
	string yText = yss.str();
	string zText = zss.str();
	//string distanceText = dss.str();

	Size xTextSize = getTextSize(xText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);
	Size yTextSize = getTextSize(yText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);
	Size zTextSize = getTextSize(zText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);
	//Size distanceTextSize = getTextSize(distanceText, FONT_HERSHEY_SIMPLEX, fontScale, thickness, 0);

	Point xTextOrg(10, 330);
	Point yTextOrg(10, 350);
	Point zTextOrg(10, 370);
	//Point distanceTextOrg(10, 390);

	Rect xRect(xTextOrg.x - rectMargin, xTextOrg.y - xTextSize.height - rectMargin,
		xTextSize.width + 2 * rectMargin, xTextSize.height + 2 * rectMargin);
	Rect yRect(yTextOrg.x - rectMargin, yTextOrg.y - yTextSize.height - rectMargin,
		yTextSize.width + 2 * rectMargin, yTextSize.height + 2 * rectMargin);
	Rect zRect(zTextOrg.x - rectMargin, zTextOrg.y - zTextSize.height - rectMargin,
		zTextSize.width + 2 * rectMargin, zTextSize.height + 2 * rectMargin);
	//Rect distanceRect(distanceTextOrg.x - rectMargin, distanceTextOrg.y - distanceTextSize.height - rectMargin,
		//distanceTextSize.width + 2 * rectMargin, distanceTextSize.height + 2 * rectMargin);

	rectangle(image, xRect, Scalar(255, 255, 255), -1);
	rectangle(image, yRect, Scalar(255, 255, 255), -1);
	rectangle(image, zRect, Scalar(255, 255, 255), -1);
	//rectangle(image, distanceRect, Scalar(255, 255, 255), -1);

	putText(image, xText, xTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
	putText(image, yText, yTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
	putText(image, zText, zTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
	//putText(image, distanceText, distanceTextOrg, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 0, 0), thickness);
}

DetectionResults  Detector(Mat& image, const Mat& K, const Mat& D)
{
	DetectionResults results;

	vector<int> ids;
	vector<vector<Point2f>> corners, rejectedCandidates;
	Ptr<aruco::DetectorParameters> parameters;
	Ptr<aruco::Dictionary> aruco_dict = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

	vector<Vec3d> rvecs, tvecs;

	aruco::detectMarkers(image, aruco_dict, corners, ids);

	if (ids.size() > 0)
	{
		results.detected = true;
		aruco::estimatePoseSingleMarkers(corners, 0.07, K, D, rvecs, tvecs);
		for (size_t i = 0; i < ids.size(); i++)
		{
			aruco::drawAxis(image, K, D, rvecs[i], tvecs[i], 0.07);
			Mat matTmp, matTmpTransposed, test;
			Rodrigues(rvecs[i], matTmp);
			transpose(matTmp, matTmpTransposed);
			Vec3f eulerAngles = rotationMatrixToEulerAngles(matTmpTransposed);
			Mat p = -matTmpTransposed * Mat(tvecs[0]);
			results.angles = eulerAngles * 180.0 / CV_PI;
			results.dists = Vec3f(p)*100;
		}
	}
	else
	{
		putText(image, "No detection", cv::Point(30, 64), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2, LINE_AA);
	}

	return results;
}

void LoadRemapFiles(const string& yamlPath, Mat& mapx, Mat& mapy)
{
	FileStorage fs(yamlPath, FileStorage::READ);
	if (!fs.isOpened())
	{
		cerr << "Error: Could not open." << yamlPath << endl;
		return;
	}

	fs["mapx"] >> mapx;
	fs["mapy"] >> mapy;
	fs.release();
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
	/*printf("fisheye : time = %lf, position(%lf, %lf, %lf) rotation(%lf, %lf, %lf, %lf) velocity(%lf, %lf, %lf) angularVelocity(%lf, %lf, %lf) acceleration(%lf, %lf, %lf) angularAcceleration(%lf, %lf, %lf)\n",
		pose->hostTimestamp, pose->position.x, pose->position.y, pose->position.z, pose->rotation.w, pose->rotation.x, pose->rotation.y, pose->rotation.z,
		pose->velocity.x, pose->velocity.y, pose->velocity.z, pose->angularVelocity.x, pose->angularVelocity.y, pose->angularVelocity.z,
		pose->acceleration.x, pose->acceleration.y, pose->acceleration.z, pose->angularAcceleration.x, pose->angularAcceleration.y, pose->angularAcceleration.z);*/

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

	/*printf("hmdImu : time = %lf, gyro(%lf, %lf, %lf) accel(%lf, %lf, %lf) accelSaturation(%s, %s, %s) magneto(%lf, %lf, %lf) temperature = %lf\n",
		hmdImu->hostTimestamp, hmdImu->gyro.x, hmdImu->gyro.y, hmdImu->gyro.z, hmdImu->accel.x, hmdImu->accel.y, hmdImu->accel.z,
		hmdImu->accelSaturation.x ? "true" : "false", hmdImu->accelSaturation.y ? "true" : "false", hmdImu->accelSaturation.z ? "true" : "false",
		hmdImu->magneto.x, hmdImu->magneto.y, hmdImu->magneto.z, hmdImu->temperature); */
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

	int frameCounter = 0;

	// Camera Intrinsic and Extrinsic Matrix
	// I used chessboard calibration method to get these values
	Mat K3 = (Mat_<double>(3, 3) << 232.15773862, 0., 239.29255483, 0., 224.65161846, 320.90631547, 0., 0., 1.);
	Mat D3 = (Mat_<double>(1, 4) << 0.07505029, -0.05127031, 0.04389267, -0.01600546);
	Mat K2 = (Mat_<double>(3, 3) << 242.10123445, 0., 238.55286211, 0., 243.5493872, 325.3093086, 0., 0., 1.);
	Mat D2 = (Mat_<double>(1, 4) << 0.05602751, -0.04683471, 0.01998477, -0.00428105);
	
	// Remapping fisheye camera to undistorted image
	Mat mapxR, mapyR, mapxL, mapyL;
	// file for remapping
	LoadRemapFiles("./mapsR.yaml", mapxR, mapyR);
	LoadRemapFiles("./mapsL.yaml", mapxL, mapyL);
	
	int reset = 0;
	
	PoseData nooffset = { Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f)};

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
			Mat imageCopy3;
			
			Mat image2 = ConvertPixelsToMat(g_pixels2, 640, 480);
			Mat imageCopy2;
		
			Mat image3_r;
			rotate(imageCopy3, image3_r, ROTATE_90_CLOCKWISE);
			Mat image2_r;
			rotate(imageCopy2, image2_r, ROTATE_90_CLOCKWISE);

			image3_r = remapImage(image3, mapxR, mapyR);
			image2_r = remapImage(image2, mapxL, mapyL);
			lastResults3 = Detector(image3_r, K3, D3);
			lastResults2 = Detector(image2_r, K2, D2);

			if (frameCounter == 10)
			{
				lastResults3 = Detector(image3_r, K3, D3);
				lastResults2 = Detector(image2_r, K2, D2);
				frameCounter = 0;
			}
			else
			{
				Detector(image3_r, K3, D3);
				Detector(image2_r, K2, D2);
			}
			frameCounter++;

			if (lastResults2.detected && lastResults3.detected) {
				// update offset
				offsets3.angles = lastResults3.angles - (lastResults3.angles + lastResults2.angles) / 2;
				offsets3.dists = lastResults3.dists - (lastResults3.dists + lastResults2.dists) / 2;
				offsets2.angles = lastResults2.angles - (lastResults3.angles + lastResults2.angles) / 2;
				offsets2.dists = lastResults2.dists - (lastResults3.dists + lastResults2.dists) / 2;
				// output is average of 2 camera
				output.angles = (lastResults3.angles + lastResults2.angles) / 2;
				output.dists = (lastResults3.dists + lastResults2.dists) / 2;
				output.detected = true;
			}
			else if (lastResults2.detected) {
				// using lastest offsets
				output.angles = lastResults2.angles - offsets2.angles;
				output.dists = lastResults2.dists - offsets2.dists;
				output.detected = true;
			}
			else if (lastResults3.detected) {
				// using lastest offsets
				output.angles = lastResults3.angles - offsets3.angles;
				output.dists = lastResults3.dists - offsets3.dists;
				output.detected = true;
			}
			else {
				output.detected = false;
			}
			if (output.detected) {
				DrawDetectionResult(image3_r, output, nooffset);
				DrawDetectionResult(image2_r, output, nooffset);
			}
						
			imshow("right", image3_r);
			imshow("left", image2_r);
						
			Render();

			rateCounter.RunFrame("render fish eye image");
		}
	}

	// close
	dpvr_SharedFisheyeImage_Close(fisheyeImage);

	// delete
	dpvr_SharedFisheyeImage_delete(fisheyeImage);

	fisheyeImage = nullptr;

	cv::destroyAllWindows();
	CleanupDevice();

    return 0;
}

float GetAngleX() {
	return output.angles[0];
}

float GetAngleY() {
	return output.angles[1];
}

float GetAngleZ() {
	return output.angles[2];
}

float GetDistX() {
	return output.dists[0];
}

float GetDistY() {
	return output.dists[1];
}

float GetDistZ() {
	return output.dists[2];
}

bool GetDetected() {
	return output.detected;
}
