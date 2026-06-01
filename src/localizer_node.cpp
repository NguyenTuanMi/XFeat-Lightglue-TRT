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
        std::bind(&LocalizerNode::imageCallback, this, std::placeholders::_1)
    );

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        this->get_parameter("camera_info_topic").as_string(),
        rclcpp::SensorDataQoS(),
        std::bind(&LocalizerNode::cameraInfoCallback, this, std::placeholders::_1)
    );

    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        this->get_parameter("pose_topic").as_string(), 10
    );

    RCLCPP_INFO(this->get_logger(), "LocalizerNode ready.");
}

// ─────────────────────────────────────────────────────────────────────────────
void LocalizerNode::declareParameters()
{
    this->declare_parameter("config_path", "");
    this->declare_parameter("xfeat_engine",     "");
    this->declare_parameter("lightglue_engine", "");
    this->declare_parameter("reference_image",  ""); // The image in direct orientation
    this->declare_parameter("image_width",  640); // Image size 
    this->declare_parameter("image_height", 480);
    this->declare_parameter("board_width_m",  0.60);
    this->declare_parameter("board_height_m", 0.40);
    this->declare_parameter("pnp_reproj_thresh", 3.0);
    this->declare_parameter("min_inliers",  10);
    this->declare_parameter("camera_topic",      "/camera/image_raw");
    this->declare_parameter("camera_info_topic", "/camera/camera_info");
    this->declare_parameter("pose_topic",        "/auv/board_pose");

    config_path_ = this->get_parameter("config_path").as_string();
    xfeat_engine_path_ = this->get_parameter("xfeat_engine").as_string();
    lightglue_engine_path_ = this->get_parameter("lightglue_engine").as_string();
    reference_image_path_ = this->get_parameter("reference_image").as_string();
    image_width_ = this->get_parameter("image_width").as_int();
    image_height_ = this->get_parameter("image_height").as_int();
    board_width_m_ = this->get_parameter("board_width_m").as_double();
    board_height_m_ = this->get_parameter("board_height_m").as_double();
    pnp_reproj_thresh_ = this->get_parameter("pnp_reproj_thresh").as_double(); // PNP Reprojection Error is between 2 to 8 pixels
    min_inliers_ = this->get_parameter("min_inliers").as_int(); // 
}

// ─────────────────────────────────────────────────────────────────────────────
void LocalizerNode::initInferenceEngines()
{
    // Construct Derkai's XFeat and LightGlue objects.
    // These load the .engine files internally.
    RCLCPP_INFO(this->get_logger(), "Loading XFeat engine: %s",
                xfeat_engine_path_.c_str());
    xfeat_ = std::make_unique<XFeat>(config_path_, xfeat_engine_path_);

    RCLCPP_INFO(this->get_logger(), "Loading LightGlue engine: %s",
                lightglue_engine_path_.c_str());
    lightglue_ = std::make_unique<Lightglue>(config_path_, lightglue_engine_path_);
}

void LocalizerNode::loadReferenceImage()
{
    cv::Mat ref = cv::imread(reference_image_path_, cv::IMREAD_COLOR);
    if (ref.empty()) {
        RCLCPP_FATAL(this->get_logger(),
                     "Cannot load reference image: %s", reference_image_path_.c_str());
        throw std::runtime_error("Reference image not found.");
    }
    ref_image_size_ = ref.size();

    // Convert to grayscale for XFeat
    cv::Mat ref_gray;
    cv::cvtColor(ref, ref_gray, cv::COLOR_BGR2GRAY);

    xfeat_->detectAndCompute(ref_gray, ref_keypoints_, ref_descriptors_, ref_heatmap_);

    if (ref_keypoints_.numel() == 0) {
        RCLCPP_FATAL(this->get_logger(), "XFeat found no features in reference image.");
        throw std::runtime_error("Reference feature extraction failed.");
    }

    // Build homography: ref image pixels → board plane XY (metres)
    // Homography explanation: https://www.geeksforgeeks.org/python/python-opencv-findhomography-inputs/
    // Reference image corners (TL, TR, BR, BL) in pixel space
    float rw = (float)ref_image_size_.width;
    float rh = (float)ref_image_size_.height;
    std::vector<cv::Point2f> ref_px_corners  = {{0,0},{rw,0},{rw,rh},{0,rh}};
    // Corresponding board XY coords (Z=0, Y axis flipped: image-down = board-up)
    std::vector<cv::Point2f> board_xy_corners = {
        {0.0f,                  0.0f},
        {(float)board_width_m_, 0.0f},
        {(float)board_width_m_, (float)board_height_m_},
        {0.0f,                  (float)board_height_m_}
    };
    H_ref_to_board_ = cv::findHomography(ref_px_corners, board_xy_corners);

    ref_loaded_ = true;
    RCLCPP_INFO(this->get_logger(),
                "Reference loaded: %zu keypoints from %s",
                ref_keypoints_.numel(), reference_image_path_.c_str());
}

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

void LocalizerNode::imageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!camera_ready_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Waiting for camera_info...");
        return;
    }
    if (!ref_loaded_) return;

    // ── Convert to OpenCV ─────────────────────────────────────────────────
    cv::Mat frame;
    try {
        frame = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge: %s", e.what());
        return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Firstly, do an XFeat + LightGlue matching
    std::vector<cv::Point2f> pts_query, pts_ref;
    if (!runMatching(gray, pts_query, pts_ref)) return;

    // Secondly, lift 2D ref matches to obtain 3D board coords 
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    lift2Dto3D(pts_ref, pts_query, pts_3d, pts_2d);

    if ((int)pts_3d.size() < min_inliers_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Too few 3D correspondences: %zu", pts_3d.size());
        return;
    }

    // Finally, use RANSAC solvePnP and then publish
    geometry_msgs::msg::PoseStamped pose;
    if (estimatePose(pts_3d, pts_2d, msg->header, pose)) {
        pose_pub_->publish(pose);
    }
}

bool LocalizerNode::runMatching(
    const cv::Mat& query_gray,
    std::vector<cv::Point2f>& pts_query,
    std::vector<cv::Point2f>& pts_ref)
{
    // Run XFeat on live query frame
    torch::Tensor query_kp;
    torch::Tensor query_heatmap_;
    torch::Tensor query_desc; 
    xfeat_->detectAndCompute(query_gray, query_kp, query_desc, query_heatmap_);

    if (query_kp.numel() == 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "XFeat: no keypoints in query frame.");
        return false;
    }

    // Run LightGlue: match ref vs query
    // Adjust the method name / signature to match Derkai's lightglue.h exactly
    std::vector<MatchPoint> matches;
    std::vector<float> k1_v, k2_v;
    std::vector<float> f1_v, f2_v;
    
    TensorToVectorKeypoints(ref_keypoints_, k1_v);
    TensorToVectorKeypoints(query_kp, k2_v);
    TensorToVectorKeypoints(ref_descriptors_, f1_v);
    TensorToVectorKeypoints(query_desc, f2_v);

    lightglue_->matching(k1_v, k2_v, 
                        f1_v, f2_v,
                        matches, std::vector<float>{image_width_, image_height_});
    
    // pts_ref = floatVecToPoints(k1_v);
    // pts_query = floatVecToPoints(k2_v);

    // This is an additional test, can drop 
    if ((int)matches.size() < min_inliers_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "LightGlue: only %zu matches (need %d).",
                             matches.size(), min_inliers_);
        return false;
    }

    // Unpack into point vectors
    pts_ref.reserve(matches.size());
    pts_query.reserve(matches.size());

    const size_t n_ref   = static_cast<size_t>(ref_keypoints_.size(0));
    const size_t n_query = static_cast<size_t>(query_kp.size(0));

    for (const auto& m : matches) { // loop throught matched keypoints
        if (m.idx1 < 0 || m.idx2 < 0) {
            continue;
        }
        const size_t idx1 = static_cast<size_t>(m.idx1);
        const size_t idx2 = static_cast<size_t>(m.idx2);
        
        if (idx1 >= n_ref || idx2 >= n_query) {
            continue;
        }
        auto kp1 = ref_keypoints_[idx1];
        pts_ref.push_back(cv::Point2f(kp1[0].item<float>(), kp1[1].item<float>()));
        
        auto kp2 = query_kp[idx2];
        pts_query.push_back(cv::Point2f(kp2[0].item<float>(), kp2[1].item<float>()));
    }

    RCLCPP_DEBUG(this->get_logger(), "Matches: %zu", matches.size());
    return true;
}

void LocalizerNode::lift2Dto3D(
    const std::vector<cv::Point2f>& pts_ref_img,
    const std::vector<cv::Point2f>& pts_query_img,
    std::vector<cv::Point3f>& pts_3d,
    std::vector<cv::Point2f>& pts_2d)
{
    // Map each reference-image pixel to board XY (metres) using pre-computed H
    std::vector<cv::Point2f> board_xy;
    cv::perspectiveTransform(pts_ref_img, board_xy, H_ref_to_board_);

    pts_3d.reserve(board_xy.size());
    pts_2d.reserve(board_xy.size());
    for (size_t i = 0; i < board_xy.size(); ++i) {
        pts_3d.push_back(cv::Point3f(board_xy[i].x, board_xy[i].y, 0.0f));
        pts_2d.push_back(pts_query_img[i]);
    }
}

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
        false,              // useExtrinsicGuess
        100,                // iterations
        (float)pnp_reproj_thresh_,
        0.99,               // confidence
        inliers,
        cv::SOLVEPNP_ITERATIVE
    );

    if (!ok || (int)inliers.size() < min_inliers_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "solvePnP failed or too few inliers (%zu).",
                             inliers.size());
        return false;
    }

    // This part are just transforming back to the board-frame from camera fram obtained from solvePnP
    
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * tvec;

    // R_inv → quaternion via Eigen
    Eigen::Matrix3d Re;
    Re << R_inv.at<double>(0,0), R_inv.at<double>(0,1), R_inv.at<double>(0,2),
          R_inv.at<double>(1,0), R_inv.at<double>(1,1), R_inv.at<double>(1,2),
          R_inv.at<double>(2,0), R_inv.at<double>(2,1), R_inv.at<double>(2,2);
    Eigen::Quaterniond q(Re);
    q.normalize();

    pose_out.header          = header;
    pose_out.header.frame_id = "board";
    pose_out.pose.position.x = t_inv.at<double>(0);
    pose_out.pose.position.y = t_inv.at<double>(1);
    pose_out.pose.position.z = t_inv.at<double>(2);
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