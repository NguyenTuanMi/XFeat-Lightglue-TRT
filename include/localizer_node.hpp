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

// NOTE: torch/torch.h intentionally removed — XFeat now returns std::vector<float>
#include "xfeat.h"
#include "lightglue.h"

#include <string>
#include <vector>
#include <memory>

namespace xfeat_localizer {

class LocalizerNode : public rclcpp::Node {
public:
    explicit LocalizerNode(const rclcpp::NodeOptions& options);
    ~LocalizerNode() = default;

private:
    // ── Initialisation ────────────────────────────────────────────────────────
    void declareParameters();
    void initInferenceEngines();
    void loadReferenceImage();

    // ── ROS2 callbacks ────────────────────────────────────────────────────────
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // ── Pipeline stages ───────────────────────────────────────────────────────
    bool runMatching(
        const cv::Mat& query_gray,
        std::vector<cv::Point2f>& pts_query,
        std::vector<cv::Point2f>& pts_ref);

    void lift2Dto3D(
        const std::vector<cv::Point2f>& pts_ref_img,
        const std::vector<cv::Point2f>& pts_query_img,
        std::vector<cv::Point3f>& pts_3d,
        std::vector<cv::Point2f>& pts_2d);

    bool estimatePose(
        const std::vector<cv::Point3f>& pts_3d,
        const std::vector<cv::Point2f>& pts_2d,
        const std_msgs::msg::Header& header,
        geometry_msgs::msg::PoseStamped& pose_out);

    // ── Inference engines ─────────────────────────────────────────────────────
    std::unique_ptr<XFeat>    xfeat_;
    std::unique_ptr<Lightglue> lightglue_;

    // ── Reference image cached features ──────────────────────────────────────
    // Previously torch::Tensor — now plain vectors from the LibTorch-free API.
    //   ref_kp_vec_   : [ref_kp_count_ * 2]  (x0,y0, x1,y1, …)
    //   ref_desc_vec_ : [ref_kp_count_ * 64] (one 64-D descriptor per keypoint)
    //   ref_kp_count_ : number of valid reference keypoints
    std::vector<float> ref_kp_vec_;
    std::vector<float> ref_desc_vec_;
    std::vector<float> ref_scores_vec_;  // kept for diagnostics; not used in matching
    int                ref_kp_count_ = 0;
    cv::Size           ref_image_size_;
    bool               ref_loaded_ = false;

    // ── Camera intrinsics ─────────────────────────────────────────────────────
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    bool    camera_ready_ = false;

    // ── Board geometry ────────────────────────────────────────────────────────
    cv::Mat H_ref_to_board_;   // homography: ref px → board XY metres

    // ── Parameters ───────────────────────────────────────────────────────────
    std::string config_path_;
    std::string xfeat_engine_path_;
    std::string lightglue_engine_path_;
    std::string reference_image_path_;
    int         image_width_       = 640;
    int         image_height_      = 480;
    double      board_width_m_     = 0.60;
    double      board_height_m_    = 0.40;
    double      pnp_reproj_thresh_ = 3.0;
    int         min_inliers_       = 10;

    // ── ROS2 interfaces ───────────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr      image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
};

}  // namespace xfeat_localizer