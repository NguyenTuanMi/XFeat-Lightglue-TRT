#ifndef UTILS_H
#define UTILS_H

// NOTE: torch/torch.h is intentionally absent.
// Helper functions now work directly with std::vector<float> which is the
// output format of the refactored XFeat class.

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

// ---------------------------------------------------------------------------
// Convert a flat [N*2] float vector (x0,y0, x1,y1, …) to cv::Point2f vector.
// ---------------------------------------------------------------------------
inline std::vector<cv::Point2f> floatVecToPoints(const std::vector<float>& vec)
{
    std::vector<cv::Point2f> pts;
    if (vec.size() % 2 != 0) {
        std::cerr << "[utils] floatVecToPoints: vector size must be even\n";
        return pts;
    }
    pts.reserve(vec.size() / 2);
    for (size_t i = 0; i < vec.size(); i += 2)
        pts.emplace_back(vec[i], vec[i + 1]);
    return pts;
}

// ---------------------------------------------------------------------------
// Reject outlier matches using the fundamental matrix + RANSAC.
// ---------------------------------------------------------------------------
static void reject_outliers(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>&   _matches,
    std::vector<cv::DMatch>&         _inliers)
{
    std::vector<cv::Point2f> points1, points2;
    for (const auto& m : _matches) {
        points1.push_back(keypoints1[m.queryIdx].pt);
        points2.push_back(keypoints2[m.trainIdx].pt);
    }
    std::vector<uchar> mask(_matches.size());
    cv::findFundamentalMat(points1, points2, mask, 3, 0.99, cv::FM_RANSAC);
    for (size_t i = 0; i < _matches.size(); ++i)
        if (mask[i]) _inliers.push_back(_matches[i]);
}

// ---------------------------------------------------------------------------
// Visualise keypoints and matches (for debug / non-ROS test binary).
// ---------------------------------------------------------------------------
static void VisualizeMatching(
    const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& image2, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& _matches,
    cv::Mat& output_image,
    double cost_time = -1)
{
    cv::Mat img1 = image1, img2 = image2;
    if (img1.rows != 480 || img1.cols != 640) cv::resize(img1, img1, cv::Size(640, 480));
    if (img2.rows != 480 || img2.cols != 640) cv::resize(img2, img2, cv::Size(640, 480));

    cv::drawMatches(img1, keypoints1, img2, keypoints2, _matches, output_image,
                    cv::Scalar(0,255,0), cv::Scalar(0,0,255));
    const double sc = std::min(img1.rows / 640.0, 2.0);
    const int ht = static_cast<int>(30 * sc);
    const std::string title = "XFeat TensorRT";
    cv::putText(output_image, title,
                cv::Point(static_cast<int>(8*sc), ht),
                cv::FONT_HERSHEY_DUPLEX, 1.0*sc, cv::Scalar(0,0,0), 2, cv::LINE_AA);
    cv::putText(output_image, title,
                cv::Point(static_cast<int>(8*sc), ht),
                cv::FONT_HERSHEY_DUPLEX, 1.0*sc, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    const std::string kp_str = "Keypoints: " + std::to_string(keypoints1.size())
                              + ":" + std::to_string(keypoints2.size());
    cv::putText(output_image, kp_str,
                cv::Point(static_cast<int>(8*sc), ht*2),
                cv::FONT_HERSHEY_DUPLEX, 1.0*sc, cv::Scalar(0,0,0), 2, cv::LINE_AA);
    cv::putText(output_image, kp_str,
                cv::Point(static_cast<int>(8*sc), ht*2),
                cv::FONT_HERSHEY_DUPLEX, 1.0*sc, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    const std::string m_str = "Matches: " + std::to_string(_matches.size());
    cv::putText(output_image, m_str,
                cv::Point(static_cast<int>(8*sc), ht*3),
                cv::FONT_HERSHEY_DUPLEX, 1.0*sc, cv::Scalar(0,0,0), 2, cv::LINE_AA);
    cv::putText(output_image, m_str,
                cv::Point(static_cast<int>(8*sc), ht*3),
                cv::FONT_HERSHEY_DUPLEX, 1.0*sc, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    if (cost_time > 0) {
        const std::string fps_str = "FPS: " + std::to_string(1000.0 / cost_time);
        cv::putText(output_image, fps_str,
                    cv::Point(static_cast<int>(8*sc), ht*4),
                    cv::FONT_HERSHEY_DUPLEX, 1.0*sc, cv::Scalar(0,0,0), 2, cv::LINE_AA);
        cv::putText(output_image, fps_str,
                    cv::Point(static_cast<int>(8*sc), ht*4),
                    cv::FONT_HERSHEY_DUPLEX, 1.0*sc, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    }
}

#endif  // UTILS_H