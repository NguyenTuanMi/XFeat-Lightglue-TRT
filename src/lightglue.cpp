#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "lightglue.h"

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
Lightglue::Lightglue(const std::string& config_path, const std::string& engine_path)
{
    YAML::Node cfg = YAML::LoadFile(config_path);
    maxMatches_ = cfg["max_matches"].as<int>();
    threShold_  = static_cast<float>(cfg["match_threshold"].as<double>());

    loadEngine(engine_path);

    context_ = std::unique_ptr<IExecutionContext, GlueDestroyObjects>(
        engine_->createExecutionContext());
    if (!context_)
        throw std::runtime_error("Lightglue: failed to create TRT execution context");

    // Set static input shapes (512 keypoints, fixed at export time)
    constexpr int kpt_num = 512;
    nvinfer1::Dims dim2;
    dim2.nbDims = 1;
    dim2.d[0]   = 2;
    context_->setInputShape("image0_size", dim2);
    context_->setInputShape("image1_size", dim2);
    context_->setInputShape("mkpts0",  nvinfer1::Dims3{1, kpt_num, 2});
    context_->setInputShape("mkpts1",  nvinfer1::Dims3{1, kpt_num, 2});
    context_->setInputShape("feats0",  nvinfer1::Dims3{1, kpt_num, 64});
    context_->setInputShape("feats1",  nvinfer1::Dims3{1, kpt_num, 64});
    if (!context_->allInputDimensionsSpecified())
        throw std::runtime_error("Lightglue: not all input dimensions specified");
}

// ─────────────────────────────────────────────────────────────────────────────
// matching
// ─────────────────────────────────────────────────────────────────────────────
void Lightglue::matching(
    std::vector<float> keypoints1,
    std::vector<float> keypoints2,
    std::vector<float> feats1,
    std::vector<float> feats2,
    std::vector<MatchPoint>& matches,
    std::vector<float> image_size)
{
    constexpr size_t kpt_num = 512;

    const size_t img_sz   = 2 * sizeof(float);
    const size_t kpts_sz  = 1 * kpt_num * 2 * sizeof(int);   // int on wire
    const size_t desc_sz  = 1 * kpt_num * 64 * sizeof(float);
    const size_t match_sz = static_cast<size_t>(maxMatches_) * 2 * sizeof(int);
    const size_t score_sz = static_cast<size_t>(maxMatches_) * sizeof(float);

    float *d_img0, *d_img1, *d_kpts0, *d_kpts1, *d_desc0, *d_desc1, *d_scores;
    int   *d_matches;
    CUDA_CHECK(cudaMalloc(&d_img0,    img_sz));
    CUDA_CHECK(cudaMalloc(&d_img1,    img_sz));
    CUDA_CHECK(cudaMalloc(&d_kpts0,   kpts_sz));
    CUDA_CHECK(cudaMalloc(&d_kpts1,   kpts_sz));
    CUDA_CHECK(cudaMalloc(&d_desc0,   desc_sz));
    CUDA_CHECK(cudaMalloc(&d_desc1,   desc_sz));
    CUDA_CHECK(cudaMalloc(&d_matches, match_sz));
    CUDA_CHECK(cudaMalloc(&d_scores,  score_sz));

    CUDA_CHECK(cudaMemcpy(d_img0,  image_size.data(),   img_sz,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img1,  image_size.data(),   img_sz,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kpts0, keypoints1.data(),   kpts_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kpts1, keypoints2.data(),   kpts_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_desc0, feats1.data(),       desc_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_desc1, feats2.data(),       desc_sz, cudaMemcpyHostToDevice));

    context_->setTensorAddress("image0_size",  d_img0);
    context_->setTensorAddress("image1_size",  d_img1);
    context_->setTensorAddress("mkpts0",       d_kpts0);
    context_->setTensorAddress("mkpts1",       d_kpts1);
    context_->setTensorAddress("descriptors0", d_desc0);
    context_->setTensorAddress("descriptors1", d_desc1);
    context_->setTensorAddress("matches",      d_matches);
    context_->setTensorAddress("scores",       d_scores);

    if (!context_->enqueueV3(0))
        throw std::runtime_error("Lightglue: TRT enqueueV3 failed");

    const nvinfer1::Dims match_dims = context_->getTensorShape("matches");
    const int num_matches = match_dims.d[0];

    std::vector<int>   h_matches(num_matches * 2);
    std::vector<float> h_scores(num_matches);
    CUDA_CHECK(cudaMemcpy(h_matches.data(), d_matches,
                          static_cast<size_t>(num_matches) * 2 * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_scores,
                          static_cast<size_t>(num_matches) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    matches.clear();
    for (int i = 0; i < num_matches; ++i) {
        if (h_scores[i] >= threShold_)
            matches.push_back({h_matches[i*2], h_matches[i*2+1], h_scores[i]});
    }

    cudaFree(d_img0);   cudaFree(d_img1);
    cudaFree(d_kpts0);  cudaFree(d_kpts1);
    cudaFree(d_desc0);  cudaFree(d_desc1);
    cudaFree(d_matches); cudaFree(d_scores);
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<char> Lightglue::readEngineFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Lightglue: cannot open engine file: " + path);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    if (!file.read(buf.data(), size))
        throw std::runtime_error("Lightglue: cannot read engine file: " + path);
    return buf;
}

void Lightglue::loadEngine(const std::string& path)
{
    auto data = readEngineFile(path);
    runtime_ = std::unique_ptr<IRuntime, GlueDestroyObjects>(createInferRuntime(gLogger_));
    if (!runtime_)
        throw std::runtime_error("Lightglue: cannot create TRT runtime");
    initLibNvInferPlugins(nullptr, "");
    ICudaEngine* raw = runtime_->deserializeCudaEngine(data.data(), data.size());
    if (!raw)
        throw std::runtime_error("Lightglue: cannot deserialize TRT engine: " + path);
    engine_ = std::unique_ptr<ICudaEngine, GlueDestroyObjects>(raw);
}