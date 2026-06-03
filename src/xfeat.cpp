#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "xfeat.h"

#define CUDA_CHECK(status)                                          \
    do {                                                            \
        auto _ret = (status);                                       \
        if (_ret != cudaSuccess) {                                  \
            std::cerr << "CUDA error " << _ret << " ("             \
                      << cudaGetErrorString(_ret) << ") at line "  \
                      << __LINE__ << std::endl;                    \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

using namespace nvinfer1;

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
XFeat::XFeat(const std::string& config_path, const std::string& engine_path)
    : interp_nearest_("nearest"), interp_bilinear_("bilinear")
{
    YAML::Node cfg = YAML::LoadFile(config_path);

    inputH_     = cfg["image_height"].as<int>();
    inputW_     = cfg["image_width"].as<int>();
    top_k_      = cfg["max_keypoints"].as<int>();
    threshold_  = cfg["feat_threshold"].as<float>();
    kernel_size_= cfg["kernel_size"].as<int>();
    softmaxTemp_= cfg["softmaxTemp"].as<float>();

    // Pad to nearest multiple of 32 (XFeat backbone requirement)
    _H_ = (inputH_ / 32) * 32;
    _W_ = (inputW_ / 32) * 32;
    outputH_ = _H_ / 8;
    outputW_ = _W_ / 8;
    rh_ = static_cast<float>(inputH_) / static_cast<float>(_H_);
    rw_ = static_cast<float>(inputW_) / static_cast<float>(_W_);

    loadEngine(engine_path);

    context_ = std::unique_ptr<IExecutionContext, DestroyObjects>(
        engine_->createExecutionContext());
    if (!context_)
        throw std::runtime_error("XFeat: failed to create TRT execution context");

    // NOTE: do NOT call setInputShape here — we don't yet know the channel count.
    //       allocateBuffers() is called lazily on the first detectAndCompute().
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor
// ─────────────────────────────────────────────────────────────────────────────
XFeat::~XFeat()
{
    if (d_input_) cudaFree(d_input_);
    if (d_feats_) cudaFree(d_feats_);
    if (d_kpts_)  cudaFree(d_kpts_);
    if (d_hmap_)  cudaFree(d_hmap_);
}

// ─────────────────────────────────────────────────────────────────────────────
// allocateBuffers
// ─────────────────────────────────────────────────────────────────────────────
void XFeat::allocateBuffers(int inputC)
{
    if (bufs_allocated_ && buf_inputC_ == inputC) return;

    if (d_input_) cudaFree(d_input_);
    if (d_feats_) cudaFree(d_feats_);
    if (d_kpts_)  cudaFree(d_kpts_);
    if (d_hmap_)  cudaFree(d_hmap_);

    CUDA_CHECK(cudaMalloc(&d_input_, static_cast<size_t>(1 * inputC * _H_ * _W_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_feats_, static_cast<size_t>(1 * 64 * outputH_ * outputW_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kpts_,  static_cast<size_t>(1 * 65 * outputH_ * outputW_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hmap_,  static_cast<size_t>(1 *  1 * outputH_ * outputW_) * sizeof(float)));

    bufs_allocated_ = true;
    buf_inputC_     = inputC;
}

// ─────────────────────────────────────────────────────────────────────────────
// detectAndCompute  (main entry point)
// ─────────────────────────────────────────────────────────────────────────────
void XFeat::detectAndCompute(
    const cv::Mat& img,
    std::vector<float>& keypoints,
    std::vector<float>& descriptors,
    std::vector<float>& scores,
    int& num_kpts)
{
    // ── 1. Pre-process: resize, HWC→CHW float, upload ─────────────────────────
    const int inputC = preprocessImage(img);

    // ── 2. TRT inference ───────────────────────────────────────────────────────
    context_->setTensorAddress("images",    d_input_);
    context_->setTensorAddress("feats",     d_feats_);
    context_->setTensorAddress("keypoints", d_kpts_);
    context_->setTensorAddress("heatmaps",  d_hmap_);
    if (!context_->enqueueV3(0))
        throw std::runtime_error("XFeat: TRT enqueueV3 failed");

    // ── 3. Download TRT outputs to host ────────────────────────────────────────
    const size_t feats_n = static_cast<size_t>(64 * outputH_ * outputW_);
    const size_t kpts_n  = static_cast<size_t>(65 * outputH_ * outputW_);
    const size_t hmap_n  = static_cast<size_t>( 1 * outputH_ * outputW_);

    std::vector<float> h_feats(feats_n), h_kpts(kpts_n), h_hmap(hmap_n);
    CUDA_CHECK(cudaMemcpy(h_feats.data(), d_feats_, feats_n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_kpts.data(),  d_kpts_,  kpts_n  * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hmap.data(),  d_hmap_,  hmap_n  * sizeof(float), cudaMemcpyDeviceToHost));

    // ── 4. L2-normalise feature map along channel axis ─────────────────────────
    // Mirrors: featsData = F.normalize(featsData, dim=1)
    l2NormalizeChannels(h_feats, 64, outputH_, outputW_);

    // ── 5. Softmax + pixel-shuffle → full-resolution keypoint heatmap ──────────
    // Mirrors: keypointsData = get_kpts_heatmap(keypointsData, softmaxTemp_)
    // h_kpts  [65, outputH_, outputW_] → heatmap_full [_H_, _W_]
    std::vector<float> heatmap_full;
    computeHeatmap(h_kpts, heatmap_full);

    // ── 6. NMS → candidate keypoint positions ─────────────────────────────────
    std::vector<std::array<float, 2>> mkpts;  // (x, y) in [0, _W_-1] x [0, _H_-1]
    NMS(heatmap_full, mkpts);

    if (mkpts.empty()) {
        keypoints.clear(); descriptors.clear(); scores.clear(); num_kpts = 0;
        return;
    }

    // ── 7. Score each candidate ────────────────────────────────────────────────
    // score = nearest_sample(heatmap_full, kpt) * bilinear_sample(h_hmap, kpt)
    // nearest on heatmap_full [1,_H_,_W_] — same-res, direct lookup
    std::vector<float> s_kpts, s_hmap;
    interp_nearest_.forward(heatmap_full, 1, _H_, _W_, mkpts, _H_, _W_, s_kpts);
    interp_bilinear_.forward(h_hmap,      1, outputH_, outputW_, mkpts, _H_, _W_, s_hmap);

    const int Nall = static_cast<int>(mkpts.size());
    std::vector<float> all_scores(Nall);
    for (int i = 0; i < Nall; i++)
        all_scores[i] = s_kpts[i] * s_hmap[i];

    // ── 8. Sort by descending score, take top_k ─────────────────────────────────
    std::vector<int> order(Nall);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return all_scores[a] > all_scores[b]; });

    const int K = std::min(top_k_, Nall);

    // ── 9. Gather descriptors for top-K keypoints ──────────────────────────────
    // bilinear_sample(featsData [64, outputH_, outputW_], top_k mkpts)
    std::vector<std::array<float, 2>> top_kpts(K);
    std::vector<float> top_scores(K);
    for (int k = 0; k < K; k++) {
        top_kpts[k]  = mkpts[order[k]];
        top_scores[k]= all_scores[order[k]];
    }

    std::vector<float> top_descs;
    interp_bilinear_.forward(h_feats, 64, outputH_, outputW_, top_kpts, _H_, _W_, top_descs);

    // ── 10. L2-normalise per-descriptor ────────────────────────────────────────
    // Mirrors: feats = F.normalize(feats, dim=-1)
    l2NormalizeRows(top_descs, K, 64);

    // ── 11. Scale keypoints back to original input resolution ─────────────────
    // Mirrors: mkpts = mkpts * scale   where scale = {rw_, rh_}
    for (auto& kp : top_kpts) {
        kp[0] *= rw_;
        kp[1] *= rh_;
    }

    // ── 12. Filter: keep only valid (score > 0) ────────────────────────────────
    keypoints.clear();
    descriptors.clear();
    scores.clear();

    for (int k = 0; k < K; k++) {
        if (top_scores[k] <= 0.0f) continue;
        keypoints.push_back(top_kpts[k][0]);
        keypoints.push_back(top_kpts[k][1]);
        for (int c = 0; c < 64; c++)
            descriptors.push_back(top_descs[k * 64 + c]);
        scores.push_back(top_scores[k]);
    }

    num_kpts = static_cast<int>(scores.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// preprocessImage
// Resize → float32 → HWC→CHW → cudaMemcpy to d_input_
// Returns the number of channels in the image.
// ─────────────────────────────────────────────────────────────────────────────
int XFeat::preprocessImage(const cv::Mat& img)
{
    const int C = img.channels();

    // Lazy first-time allocation (or re-allocation on channel count change)
    allocateBuffers(C);

    // Resize to (_H_, _W_) if needed
    cv::Mat resized;
    if (img.rows != _H_ || img.cols != _W_)
        cv::resize(img, resized, cv::Size(_W_, _H_), 0, 0, cv::INTER_LINEAR);
    else
        resized = img;

    // Convert to float32 (raw pixel values 0–255, no normalisation — matches
    // the original MatToTensor which also did not divide by 255)
    cv::Mat floatMat;
    resized.convertTo(floatMat, CV_32F);
    CV_Assert(floatMat.isContinuous());

    // HWC → CHW  (TensorRT expects NCHW)
    if (C == 1) {
        // Single-channel: HWC layout == CHW for 1 channel → direct copy
        CUDA_CHECK(cudaMemcpy(d_input_, floatMat.data,
                              static_cast<size_t>(_H_ * _W_) * sizeof(float),
                              cudaMemcpyHostToDevice));
    } else {
        // Multi-channel (e.g. BGR): split channels then pack plane-by-plane
        std::vector<cv::Mat> planes;
        cv::split(floatMat, planes);
        for (int c = 0; c < C; c++) {
            CV_Assert(planes[c].isContinuous());
            CUDA_CHECK(cudaMemcpy(
                d_input_ + static_cast<ptrdiff_t>(c) * _H_ * _W_,
                planes[c].data,
                static_cast<size_t>(_H_ * _W_) * sizeof(float),
                cudaMemcpyHostToDevice));
        }
    }

    // Set TRT input shape (called every frame; cheap for a fixed-shape engine)
    context_->setInputShape("images", nvinfer1::Dims4{1, C, _H_, _W_});

    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
// computeHeatmap
// Replicates get_kpts_heatmap():
//   softmax(kpts * softmaxTemp_, dim=channel).narrow(channel, 0, 64)
//   then pixel-shuffle [1,64,outputH_,outputW_] → [1,1,_H_,_W_]
//
// Layout of kpts_raw (flat, C-major): index = c * outputH_ * outputW_ + h * outputW_ + w
// Mapping: heatmap_full[(h*8+i) * _W_ + (w*8+j)] = softmax_output[i*8+j][h][w]
// ─────────────────────────────────────────────────────────────────────────────
void XFeat::computeHeatmap(
    const std::vector<float>& kpts_raw,
    std::vector<float>& heatmap_full) const
{
    heatmap_full.assign(static_cast<size_t>(_H_ * _W_), 0.0f);
    const int HW_out = outputH_ * outputW_;

    for (int h = 0; h < outputH_; h++) {
        for (int w = 0; w < outputW_; w++) {
            const int base = h * outputW_ + w;

            // --- softmax with temperature over 65 channels ---
            // Find max for numerical stability
            float max_v = -1e30f;
            for (int c = 0; c < 65; c++) {
                const float v = kpts_raw[c * HW_out + base] * softmaxTemp_;
                if (v > max_v) max_v = v;
            }
            // Compute softmax denominator
            float sum = 0.0f;
            float exps[65];
            for (int c = 0; c < 65; c++) {
                exps[c] = std::exp(kpts_raw[c * HW_out + base] * softmaxTemp_ - max_v);
                sum += exps[c];
            }
            // Normalise (only first 64 channels used; channel 64 = dustbin)
            const float inv_sum = 1.0f / sum;

            // --- pixel-shuffle ---
            // channel (i*8 + j) → spatial position (h*8+i, w*8+j)
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    const int channel = i * 8 + j;   // 0..63
                    const int hy = h * 8 + i;
                    const int hx = w * 8 + j;
                    heatmap_full[hy * _W_ + hx] = exps[channel] * inv_sum;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NMS
// Replicates the PyTorch NMS using max_pool2d(kernel_size, stride=1, pad).
// A position is a valid keypoint iff:
//   heatmap[y,x] > threshold_  AND
//   heatmap[y,x] == max over kernel_size×kernel_size neighbourhood
//
// Out-of-bounds positions contribute 0 (zero-padding) so they never override
// a real positive value (all softmax outputs > 0).
// ─────────────────────────────────────────────────────────────────────────────
void XFeat::NMS(
    const std::vector<float>& heatmap_full,
    std::vector<std::array<float, 2>>& mkpts) const
{
    mkpts.clear();
    const int pad = kernel_size_ / 2;

    for (int y = 0; y < _H_; y++) {
        for (int x = 0; x < _W_; x++) {
            const float val = heatmap_full[y * _W_ + x];
            if (val <= threshold_) continue;

            // Compute local max over the kernel neighbourhood (zero-padded)
            float local_max = 0.0f;
            for (int dy = -pad; dy <= pad; dy++) {
                const int ny = y + dy;
                if (ny < 0 || ny >= _H_) continue;
                for (int dx = -pad; dx <= pad; dx++) {
                    const int nx = x + dx;
                    if (nx < 0 || nx >= _W_) continue;
                    const float nb = heatmap_full[ny * _W_ + nx];
                    if (nb > local_max) local_max = nb;
                }
            }

            // val is the local maximum if it equals the neighbourhood max.
            // Using exact float comparison is correct here: local_max is one
            // of the stored values (not a computed result), so it is bit-exact.
            if (val == local_max) {
                mkpts.push_back({static_cast<float>(x), static_cast<float>(y)});
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// l2NormalizeChannels
// In-place L2 normalisation along the channel axis for [1, C, H, W].
// Mirrors: F.normalize(feat, dim=1)
// ─────────────────────────────────────────────────────────────────────────────
void XFeat::l2NormalizeChannels(std::vector<float>& feat, int C, int H, int W)
{
    const int HW = H * W;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            const int base = y * W + x;
            float norm_sq = 0.0f;
            for (int c = 0; c < C; c++)
                norm_sq += feat[c * HW + base] * feat[c * HW + base];
            const float inv = 1.0f / std::max(std::sqrt(norm_sq), 1e-8f);
            for (int c = 0; c < C; c++)
                feat[c * HW + base] *= inv;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// l2NormalizeRows
// In-place L2 normalisation for each row of [N, C].
// Mirrors: F.normalize(feats, dim=-1)
// ─────────────────────────────────────────────────────────────────────────────
void XFeat::l2NormalizeRows(std::vector<float>& mat, int N, int C)
{
    for (int i = 0; i < N; i++) {
        float norm_sq = 0.0f;
        for (int c = 0; c < C; c++)
            norm_sq += mat[i * C + c] * mat[i * C + c];
        const float inv = 1.0f / std::max(std::sqrt(norm_sq), 1e-8f);
        for (int c = 0; c < C; c++)
            mat[i * C + c] *= inv;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine loading helpers (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<char> XFeat::readEngineFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("XFeat: cannot open engine file: " + path);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    if (!file.read(buf.data(), size))
        throw std::runtime_error("XFeat: cannot read engine file: " + path);
    return buf;
}

void XFeat::loadEngine(const std::string& path)
{
    auto data = readEngineFile(path);
    runtime_ = std::unique_ptr<IRuntime, DestroyObjects>(createInferRuntime(gLogger_));
    if (!runtime_)
        throw std::runtime_error("XFeat: cannot create TRT runtime");
    initLibNvInferPlugins(nullptr, "");
    ICudaEngine* raw = runtime_->deserializeCudaEngine(data.data(), data.size());
    if (!raw)
        throw std::runtime_error("XFeat: cannot deserialize TRT engine: " + path);
    engine_ = std::unique_ptr<ICudaEngine, DestroyObjects>(raw);
}