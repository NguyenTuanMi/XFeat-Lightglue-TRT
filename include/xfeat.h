#ifndef XFEAT_H_
#define XFEAT_H_

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <iostream>

#include "InterpolateSparse2D.h"

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// TensorRT logger
// ---------------------------------------------------------------------------
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// ---------------------------------------------------------------------------
// Custom deleter for TensorRT objects (uses destroy() member) and for raw
// CUDA device memory (falls back to cudaFree).
// ---------------------------------------------------------------------------
struct DestroyObjects {
    template <typename T>
    void operator()(T* ptr) const { if (ptr) destroy(ptr); }
private:
    template <typename T>
    typename std::enable_if<
        std::is_member_function_pointer<decltype(&T::destroy)>::value>::type
    destroy(T* ptr) const { ptr->destroy(); }

    void destroy(void* ptr) const { cudaFree(ptr); }
};

// ---------------------------------------------------------------------------
// XFeat — keypoint detector / descriptor extractor backed by TensorRT.
//
// All post-processing (NMS, softmax, pixel-shuffle, sparse grid-sample,
// top-K selection, L2 normalisation) runs on the CPU so that the class
// compiles and runs correctly with a CPU-only (whl/cpu) PyTorch install,
// which supplies no CUDA device backend to libtorch.
//
// TensorRT inference itself still runs entirely on GPU via raw cudaMalloc
// device pointers, exactly as the LightGlue class does.
//
// Public API output format (flat row-major vectors):
//   keypoints   : [N*2]  — (x0,y0, x1,y1, …)  in input-image pixel coords
//   descriptors : [N*64] — one 64-D L2-normalised descriptor per keypoint
//   scores      : [N]    — confidence (> 0 guaranteed for all returned kpts)
//   num_kpts    : N
// ---------------------------------------------------------------------------
class XFeat
{
public:
    XFeat(const std::string& config_path, const std::string& engine_path);
    ~XFeat();

    // Main entry point — LibTorch-free.
    void detectAndCompute(
        const cv::Mat& img,
        std::vector<float>& keypoints,
        std::vector<float>& descriptors,
        std::vector<float>& scores,
        int& num_kpts);

private:
    // ── Pre-processing ────────────────────────────────────────────────────────
    // Resize img to (_H_, _W_), convert to CHW float32, upload to d_input_.
    // Returns the number of input channels detected from the image.
    int preprocessImage(const cv::Mat& img);

    // ── CPU post-processing helpers ───────────────────────────────────────────

    // Softmax(temp) along channel dim then pixel-shuffle:
    //   kpts_raw [1, 65, outputH_, outputW_] → heatmap_full [_H_, _W_]
    void computeHeatmap(
        const std::vector<float>& kpts_raw,
        std::vector<float>& heatmap_full) const;

    // Local-maximum suppression on heatmap_full [_H_, _W_].
    // Returns integer (x, y) positions above threshold_ that are local maxima.
    void NMS(
        const std::vector<float>& heatmap_full,
        std::vector<std::array<float, 2>>& mkpts) const;

    // L2-normalise feat map [1, C, H, W] along the channel (dim=1) axis.
    // Modifies in-place.
    static void l2NormalizeChannels(std::vector<float>& feat, int C, int H, int W);

    // L2-normalise each row of a [N, C] matrix (per-descriptor normalisation).
    // Modifies in-place.
    static void l2NormalizeRows(std::vector<float>& mat, int N, int C);

    // ── Engine I/O ────────────────────────────────────────────────────────────
    std::vector<char> readEngineFile(const std::string& path);
    void loadEngine(const std::string& path);

    // Allocate persistent GPU buffers once we know the channel count.
    void allocateBuffers(int inputC);

    // ── TensorRT objects ──────────────────────────────────────────────────────
    std::unique_ptr<IRuntime, DestroyObjects>         runtime_;
    std::unique_ptr<ICudaEngine, DestroyObjects>      engine_;
    std::unique_ptr<IExecutionContext, DestroyObjects> context_;
    Logger gLogger_;

    // ── Dimensions (computed from config + engine at construction time) ────────
    int inputH_ = 0, inputW_ = 0;   // image resolution from config
    int _H_ = 0, _W_ = 0;           // padded to nearest multiple of 32
    int outputH_ = 0, outputW_ = 0; // = _H_/8, _W_/8
    float rh_ = 1.0f, rw_ = 1.0f;  // scale factors: inputH/_H, inputW/_W

    // ── Config values ─────────────────────────────────────────────────────────
    int   top_k_       = 512;
    float threshold_   = 0.05f;
    int   kernel_size_ = 5;
    float softmaxTemp_ = 1.0f;

    // ── GPU device buffers (persistent, allocated once) ────────────────────────
    float* d_input_ = nullptr;   // [1, C, _H_, _W_]
    float* d_feats_ = nullptr;   // [1, 64, outputH_, outputW_]
    float* d_kpts_  = nullptr;   // [1, 65, outputH_, outputW_]
    float* d_hmap_  = nullptr;   // [1,  1, outputH_, outputW_]
    bool   bufs_allocated_ = false;
    int    buf_inputC_     = 0;

    // ── Interpolators ─────────────────────────────────────────────────────────
    InterpolateSparse2D interp_nearest_;   // nearest — for keypoints heatmap
    InterpolateSparse2D interp_bilinear_;  // bilinear — for heatmap & feats
};

#endif  // XFEAT_H_