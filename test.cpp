#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cmath>

bool isRotationMatrix(const cv::Mat &R) {
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, R.type());
    return norm(I - shouldBeIdentity) < 1e-6;
}

cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat &R) {
    assert(isRotationMatrix(R));

    float sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6;

    float x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }

    return cv::Vec3f(x, y, z);
}

int main() {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 158.92757205, 0., 152.07775471, 0., 164.87005228, 203.70307396, 0., 0., 1.);
    cv::Mat D = (cv::Mat_<double>(1, 4) << -0.09769812, 0.36214841, -0.41720733, 0.15493034);

    cv::VideoCapture cap("../aruco3right.mp4");
    cv::VideoWriter out("aruco2right_detect.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(300, 400));
    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    cv::Mat frame;
    cv::namedWindow("frame", cv::WINDOW_NORMAL);

    while (true) {
        cap.read(frame);
        if (frame.empty()) break;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> rejectedImgPoints;

        cv::aruco::detectMarkers(gray, aruco_dict, corners, ids, parameters, rejectedImgPoints);

        if (!ids.empty()) {
            cv::Vec3d rvec, tvec;
            cv::aruco::estimatePoseSingleMarkers(corners, 0.05, K, D, rvec, tvec);

            for (size_t i = 0; i < rvec.rows; i++) {
                cv::aruco::drawAxis(frame, K, D, rvec.row(i), tvec.row(i), 0.03);
                cv::aruco::drawDetectedMarkers(frame, corners, ids);
            }

            cv::Vec3f angles = rotationMatrixToEulerAngles(cv::Mat(rvec));
            cv::putText(frame, "pitch(right): " + std::to_string(round(angles[0] * 180.0 / CV_PI, 1)),
                        cv::Point(10, 250), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::putText(frame, "roll(up): " + std::to_string(round(angles[1] * 180.0 / CV_PI, 1)),
                        cv::Point(10, 270), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::putText(frame, "yaw(clockwise): " + std::to_string(round(angles[2] * 180.0 / CV_PI, 1)),
                        cv::Point(10, 290), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

            cv::Vec3f realworld_tvec = cv::Mat(cv::Mat(-rvec) * tvec);

            float tx = realworld_tvec[0] * 2.54 * 100;
            float ty = realworld_tvec[1] * 2.54 * 100;
            float tz = realworld_tvec[2] * 2.54 * 100;
            float distance = std::sqrt(tx * tx + ty * ty + tz * tz);

            cv::putText(frame, "x: " + std::to_string(round(tx, 2)) + " cm", cv::Point(10, 330),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::putText(frame, "y: " + std::to_string(round(ty, 2)) + " cm", cv::Point(10, 350),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::putText(frame, "z: " + std::to_string(round(tz, 2)) + " cm", cv::Point(10, 370),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::putText(frame, "distance: " + std::to_string(round(distance, 2)) + " cm", cv::Point(10, 390),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        } else {
            cv::putText(frame, "No detection", cv::Point(0, 64), cv::FONT_HERSHEY
