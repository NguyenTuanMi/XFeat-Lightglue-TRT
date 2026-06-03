#ifndef LIGHTGLUE_H_
#define LIGHTGLUE_H_

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <type_traits>
#include <iostream>
#include <string>

// NOTE: torch/torch.h is intentionally absent.  The matching() implementation
// in lightglue.cpp already uses raw cudaMalloc / cudaMemcpy exclusively — it
// never touches LibTorch.  The dead torch:: private methods that were present
// in the original header (detectDense, match, preprocessImages, MatToTensor,
// get_kpts_heatmap) have been removed because they were never called by the
// localizer and only introduced a hard dependency on CUDA-enabled LibTorch.

using namespace nvinfer1;

// ---------------------------------------------------------------------------
struct MatchPoint {
    int   idx1;
    int   idx2;
    float score;
};

// ---------------------------------------------------------------------------
class GlueLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// ---------------------------------------------------------------------------
struct GlueDestroyObjects {
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
// Lightglue — feature matcher backed by TensorRT.
//
// All I/O is via std::vector<float> / std::vector<MatchPoint> on the host.
// Internally the implementation uses raw cudaMalloc / cudaMemcpy — no LibTorch.
// ---------------------------------------------------------------------------
class Lightglue
{
public:
    Lightglue(const std::string& config_path, const std::string& engine_path);

    // Match two keypoint sets.
    //   keypoints1/2 : flat [N*2] (x0,y0, x1,y1, …) — already padded to kpt_num=512
    //   feats1/2     : flat [N*64]
    //   matches      : output (idx_ref, idx_query, score) pairs above threshold
    //   image_size   : [width, height] used to normalise keypoint coords inside TRT
    void matching(
        std::vector<float> keypoints1,
        std::vector<float> keypoints2,
        std::vector<float> feats1,
        std::vector<float> feats2,
        std::vector<MatchPoint>& matches,
        std::vector<float> image_size);

private:
    std::vector<char> readEngineFile(const std::string& path);
    void              loadEngine(const std::string& path);

    std::unique_ptr<IRuntime, GlueDestroyObjects>          runtime_;
    std::unique_ptr<ICudaEngine, GlueDestroyObjects>       engine_;
    std::unique_ptr<IExecutionContext, GlueDestroyObjects> context_;
    GlueLogger gLogger_;

    int   maxMatches_ = 512;
    float threShold_  = 0.7f;
};

#endif  // LIGHTGLUE_H_