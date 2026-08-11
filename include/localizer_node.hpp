#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "xfeat.h"
#include "lightglue.h"
#include <torch/torch.h>

#include <string>
#include <vector>
#include <memory>

namespace xfeat_localizer {

class LocalizerNode : public rclcpp::Node {
public:
    explicit LocalizerNode(const rclcpp::NodeOptions& options);
    ~LocalizerNode() = default;

private:
    // ── Initialisation ────────────────────────────────────────────────────
    void declareParameters();
    void initInferenceEngines();
    void loadReferenceImage();

    // ── ROS2 callbacks ────────────────────────────────────────────────────
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // ── Pipeline stages ───────────────────────────────────────────────────
    // Returns false if not enough matches found
    bool runMatching(
        const cv::Mat& query_gray,
        std::vector<cv::Point2f>& pts_query,
        std::vector<cv::Point2f>& pts_ref
    );

    // Lifts 2D ref-image matches → 3D board-frame coordinates
    void lift2Dto3D(
        const std::vector<cv::Point2f>& pts_ref_img,
        const std::vector<cv::Point2f>& pts_query_img,
        std::vector<cv::Point3f>& pts_3d,
        std::vector<cv::Point2f>& pts_2d
    );

    // solvePnP → geometry_msgs::PoseStamped
    bool estimatePose(
        const std::vector<cv::Point3f>& pts_3d,
        const std::vector<cv::Point2f>& pts_2d,
        const std_msgs::msg::Header& header,
        geometry_msgs::msg::PoseStamped& pose_out
    );

    // ── Derkai inference objects ──────────────────────────────────────────
    std::unique_ptr<XFeat> xfeat_;
    std::unique_ptr<Lightglue> lightglue_;

    // Reference image cached features
    // (exact types depend on Derkai's API — see xfeat.h)
    // We store them as whatever xfeat_->detectAndCompute() returns
    torch::Tensor ref_keypoints_;
    torch::Tensor ref_descriptors_;
    cv::Size ref_image_size_;
    torch::Tensor ref_heatmap_;
    std::string config_path_;
    bool ref_loaded_ = false;

    // ── Camera intrinsics ─────────────────────────────────────────────────
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    bool camera_ready_ = false;

    // ── Board 3D model ────────────────────────────────────────────────────
    std::vector<cv::Point3f> board_corners_3d_;
    cv::Mat H_ref_to_board_;   // homography: ref px → board XY metres

    // ── Config ────────────────────────────────────────────────────────────
    std::string xfeat_engine_path_;
    std::string lightglue_engine_path_;
    std::string reference_image_path_;
    int         image_width_;
    int         image_height_;
    int         top_k_;
    double      board_width_m_;
    double      board_height_m_;
    double      pnp_reproj_thresh_;
    int         min_inliers_;

    // ── ROS2 interfaces ───────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr      image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
};

}  // namespace xfeat_localizer