#include "localizer_node.hpp"
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/LinearMath/Quaternion.h>

namespace xfeat_localizer {

// ─────────────────────────────────────────────────────────────────────────────
LocalizerNode::LocalizerNode(const rclcpp::NodeOptions& options)
: Node("localizer_node", options)
{
    declareParameters();
    initInferenceEngines();
    loadReferenceImage();

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        this->get_parameter("camera_topic").as_string(),
        rclcpp::SensorDataQoS(),
        std::bind(&LocalizerNode::imageCallback, this, std::placeholders::_1));

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        this->get_parameter("camera_info_topic").as_string(),
        rclcpp::SensorDataQoS(),
        std::bind(&LocalizerNode::cameraInfoCallback, this, std::placeholders::_1));

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        this->get_parameter("pose_topic").as_string(), 10);

    RCLCPP_INFO(this->get_logger(), "LocalizerNode ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
void LocalizerNode::declareParameters()
{
    this->declare_parameter("config_path",         "");
    this->declare_parameter("xfeat_engine",        "");
    this->declare_parameter("lightglue_engine",    "");
    this->declare_parameter("reference_image",     "");
    this->declare_parameter("image_width",         640);
    this->declare_parameter("image_height",        480);
    this->declare_parameter("board_width_m",       0.60);
    this->declare_parameter("board_height_m",      0.40);
    this->declare_parameter("pnp_reproj_thresh",   3.0);
    this->declare_parameter("min_inliers",         10);
    this->declare_parameter("camera_topic",        "/camera/image_raw");
    this->declare_parameter("camera_info_topic",   "/camera/camera_info");
    this->declare_parameter("pose_topic",          "/auv/board_pose");

    config_path_           = this->get_parameter("config_path").as_string();
    xfeat_engine_path_     = this->get_parameter("xfeat_engine").as_string();
    lightglue_engine_path_ = this->get_parameter("lightglue_engine").as_string();
    reference_image_path_  = this->get_parameter("reference_image").as_string();
    image_width_           = this->get_parameter("image_width").as_int();
    image_height_          = this->get_parameter("image_height").as_int();
    board_width_m_         = this->get_parameter("board_width_m").as_double();
    board_height_m_        = this->get_parameter("board_height_m").as_double();
    pnp_reproj_thresh_     = this->get_parameter("pnp_reproj_thresh").as_double();
    min_inliers_           = this->get_parameter("min_inliers").as_int();
}

// ─────────────────────────────────────────────────────────────────────────────
void LocalizerNode::initInferenceEngines()
{
    RCLCPP_INFO(this->get_logger(), "Loading XFeat engine: %s",
                xfeat_engine_path_.c_str());
    xfeat_ = std::make_unique<XFeat>(config_path_, xfeat_engine_path_);

    RCLCPP_INFO(this->get_logger(), "Loading LightGlue engine: %s",
                lightglue_engine_path_.c_str());
    lightglue_ = std::make_unique<Lightglue>(config_path_, lightglue_engine_path_);
}

// ─────────────────────────────────────────────────────────────────────────────
void LocalizerNode::loadReferenceImage()
{
    cv::Mat ref = cv::imread(reference_image_path_, cv::IMREAD_COLOR);
    if (ref.empty()) {
        RCLCPP_FATAL(this->get_logger(),
                     "Cannot load reference image: %s", reference_image_path_.c_str());
        throw std::runtime_error("Reference image not found.");
    }
    ref_image_size_ = ref.size();

    // Convert to grayscale for XFeat (1-channel input)
    cv::Mat ref_gray;
    cv::cvtColor(ref, ref_gray, cv::COLOR_BGR2GRAY);

    // --- Formerly: xfeat_->detectAndCompute(ref_gray, ref_keypoints_, ref_descriptors_, ref_heatmap_)
    //     where the outputs were torch::Tensor.
    //     Now: std::vector<float> + count integer — no LibTorch required.
    xfeat_->detectAndCompute(ref_gray,
                             ref_kp_vec_,
                             ref_desc_vec_,
                             ref_scores_vec_,
                             ref_kp_count_);

    if (ref_kp_count_ == 0) {
        RCLCPP_FATAL(this->get_logger(), "XFeat found no features in reference image.");
        throw std::runtime_error("Reference feature extraction failed.");
    }

    // Build homography: reference image pixels → board XY plane (metres)
    const float rw = static_cast<float>(ref_image_size_.width);
    const float rh = static_cast<float>(ref_image_size_.height);
    std::vector<cv::Point2f> ref_px_corners   = {{0,0},{rw,0},{rw,rh},{0,rh}};
    std::vector<cv::Point2f> board_xy_corners  = {
        {0.0f,                  0.0f},
        {(float)board_width_m_, 0.0f},
        {(float)board_width_m_, (float)board_height_m_},
        {0.0f,                  (float)board_height_m_}
    };
    H_ref_to_board_ = cv::findHomography(ref_px_corners, board_xy_corners);

    ref_loaded_ = true;
    RCLCPP_INFO(this->get_logger(),
                "Reference loaded: %d keypoints from %s",
                ref_kp_count_, reference_image_path_.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
void LocalizerNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (camera_ready_) return;
    camera_matrix_ = (cv::Mat_<double>(3,3) <<
        msg->k[0], msg->k[1], msg->k[2],
        msg->k[3], msg->k[4], msg->k[5],
        msg->k[6], msg->k[7], msg->k[8]);
    dist_coeffs_ = cv::Mat(msg->d).clone();
    camera_ready_ = true;
    RCLCPP_INFO(this->get_logger(),
                "Camera intrinsics received: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
}

// ─────────────────────────────────────────────────────────────────────────────
void LocalizerNode::imageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!camera_ready_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Waiting for camera_info…");
        return;
    }
    if (!ref_loaded_) return;

    cv::Mat frame;
    try {
        frame = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge: %s", e.what());
        return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> pts_query, pts_ref;
    if (!runMatching(gray, pts_query, pts_ref)) return;

    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    lift2Dto3D(pts_ref, pts_query, pts_3d, pts_2d);

    if ((int)pts_3d.size() < min_inliers_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Too few 3D correspondences: %zu", pts_3d.size());
        return;
    }

    geometry_msgs::msg::PoseStamped pose;
    if (estimatePose(pts_3d, pts_2d, msg->header, pose))
        pose_pub_->publish(pose);
}

// ─────────────────────────────────────────────────────────────────────────────
bool LocalizerNode::runMatching(
    const cv::Mat& query_gray,
    std::vector<cv::Point2f>& pts_query,
    std::vector<cv::Point2f>& pts_ref)
{
    // ── Extract query features ────────────────────────────────────────────────
    // Previously: torch::Tensor query_kp, query_desc, query_heatmap
    // Now:        std::vector<float> + count
    std::vector<float> query_kp_vec, query_desc_vec, query_scores_vec;
    int query_kp_count = 0;
    xfeat_->detectAndCompute(query_gray,
                             query_kp_vec,
                             query_desc_vec,
                             query_scores_vec,
                             query_kp_count);

    if (query_kp_count == 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "XFeat: no keypoints in query frame.");
        return false;
    }

    // ── Run LightGlue ─────────────────────────────────────────────────────────
    // ref_kp_vec_ / ref_desc_vec_ are already flat [N*2] / [N*64] vectors —
    // exactly the format Lightglue::matching() expects.  Previously we called
    // TensorToVectorKeypoints() to flatten torch::Tensors; that step is gone.
    std::vector<MatchPoint> matches;
    lightglue_->matching(
        ref_kp_vec_,   query_kp_vec,
        ref_desc_vec_, query_desc_vec,
        matches,
        std::vector<float>{static_cast<float>(image_width_),
                           static_cast<float>(image_height_)});

    if ((int)matches.size() < min_inliers_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "LightGlue: only %zu matches (need %d).",
                             matches.size(), min_inliers_);
        return false;
    }

    // ── Unpack matched pairs into cv::Point2f vectors ─────────────────────────
    pts_ref.reserve(matches.size());
    pts_query.reserve(matches.size());

    const size_t n_ref   = static_cast<size_t>(ref_kp_count_);
    const size_t n_query = static_cast<size_t>(query_kp_count);

    for (const auto& m : matches) {
        if (m.idx1 < 0 || m.idx2 < 0) continue;
        const size_t i1 = static_cast<size_t>(m.idx1);
        const size_t i2 = static_cast<size_t>(m.idx2);
        if (i1 >= n_ref || i2 >= n_query) continue;

        // Previously: ref_keypoints_[idx1][0].item<float>()
        // Now: flat index into ref_kp_vec_
        pts_ref.emplace_back(ref_kp_vec_[i1 * 2 + 0],
                             ref_kp_vec_[i1 * 2 + 1]);
        pts_query.emplace_back(query_kp_vec[i2 * 2 + 0],
                               query_kp_vec[i2 * 2 + 1]);
    }

    RCLCPP_DEBUG(this->get_logger(), "Matches: %zu", matches.size());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
void LocalizerNode::lift2Dto3D(
    const std::vector<cv::Point2f>& pts_ref_img,
    const std::vector<cv::Point2f>& pts_query_img,
    std::vector<cv::Point3f>& pts_3d,
    std::vector<cv::Point2f>& pts_2d)
{
    std::vector<cv::Point2f> board_xy;
    cv::perspectiveTransform(pts_ref_img, board_xy, H_ref_to_board_);

    pts_3d.reserve(board_xy.size());
    pts_2d.reserve(board_xy.size());
    for (size_t i = 0; i < board_xy.size(); ++i) {
        pts_3d.emplace_back(board_xy[i].x, board_xy[i].y, 0.0f);
        pts_2d.push_back(pts_query_img[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
bool LocalizerNode::estimatePose(
    const std::vector<cv::Point3f>& pts_3d,
    const std::vector<cv::Point2f>& pts_2d,
    const std_msgs::msg::Header& header,
    geometry_msgs::msg::PoseStamped& pose_out)
{
    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool ok = cv::solvePnPRansac(
        pts_3d, pts_2d,
        camera_matrix_, dist_coeffs_,
        rvec, tvec,
        false, 100, static_cast<float>(pnp_reproj_thresh_), 0.99,
        inliers, cv::SOLVEPNP_ITERATIVE);

    if (!ok || (int)inliers.size() < min_inliers_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "solvePnP failed or too few inliers (%zu).",
                             inliers.size());
        return false;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * tvec;

    Eigen::Matrix3d Re;
    Re << R_inv.at<double>(0,0), R_inv.at<double>(0,1), R_inv.at<double>(0,2),
          R_inv.at<double>(1,0), R_inv.at<double>(1,1), R_inv.at<double>(1,2),
          R_inv.at<double>(2,0), R_inv.at<double>(2,1), R_inv.at<double>(2,2);
    Eigen::Quaterniond q(Re);
    q.normalize();

    pose_out.header           = header;
    pose_out.header.frame_id  = "board";
    pose_out.pose.position.x  = t_inv.at<double>(0);
    pose_out.pose.position.y  = t_inv.at<double>(1);
    pose_out.pose.position.z  = t_inv.at<double>(2);
    pose_out.pose.orientation.x = q.x();
    pose_out.pose.orientation.y = q.y();
    pose_out.pose.orientation.z = q.z();
    pose_out.pose.orientation.w = q.w();

    RCLCPP_DEBUG(this->get_logger(),
                 "Pose: x=%.3f y=%.3f z=%.3f | inliers=%zu",
                 t_inv.at<double>(0), t_inv.at<double>(1), t_inv.at<double>(2),
                 inliers.size());
    return true;
}

}  // namespace xfeat_localizer

RCLCPP_COMPONENTS_REGISTER_NODE(xfeat_localizer::LocalizerNode)