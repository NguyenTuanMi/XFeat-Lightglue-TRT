#ifndef INTER_SPARSE_2D_H
#define INTER_SPARSE_2D_H

#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// CPU replacement for PyTorch F.grid_sample with zeros padding.
//
// Replicates the normgrid + grid_sample pattern used by the original code:
//   normgrid:    norm = 2.0 * (pos / [W-1, H-1]) - 1.0   (align_corners=True)
//   grid_sample: sample feat[C, H_src, W_src] at normalised positions
//
// The net effect: kpt pixel coord (px, py) in [0, W_norm-1] x [0, H_norm-1]
// maps to feature-map continuous coord:
//   fx = px * (W_src - 1) / (W_norm - 1)
//   fy = py * (H_src - 1) / (H_norm - 1)
//
// When H_src == H_norm and W_src == W_norm (same-resolution sampling, e.g.
// heatmap ← keypoints), fx = px, fy = py — a direct lookup.
// When H_src == H_norm/8 (featsData ← full-res mkpts) the 8x down-scale is
// applied automatically.
// ---------------------------------------------------------------------------
class InterpolateSparse2D
{
public:
    enum class Mode { BILINEAR, NEAREST };

    explicit InterpolateSparse2D(const std::string& mode = "bilinear")
        : mode_(mode == "nearest" ? Mode::NEAREST : Mode::BILINEAR) {}

    // -------------------------------------------------------------------------
    // forward()
    //   feat      : flat [1, C, H_src, W_src] in C-major order
    //               index = c * H_src * W_src + y * W_src + x
    //   C, H_src, W_src : feat map dimensions
    //   kpts      : N keypoint positions (x, y) in [0, W_norm-1] x [0, H_norm-1]
    //   H_norm, W_norm  : normalization space (usually the full input resolution)
    //   out       : [N * C] — caller must pre-size or it is resized here
    // -------------------------------------------------------------------------
    void forward(
        const std::vector<float>& feat,
        int C, int H_src, int W_src,
        const std::vector<std::array<float, 2>>& kpts,
        int H_norm, int W_norm,
        std::vector<float>& out) const
    {
        const int N = static_cast<int>(kpts.size());
        out.assign(N * C, 0.0f);

        // Scale factors from norm-space to feat-space
        const float sx = (W_src > 1) ? static_cast<float>(W_src - 1) / static_cast<float>(W_norm - 1) : 0.0f;
        const float sy = (H_src > 1) ? static_cast<float>(H_src - 1) / static_cast<float>(H_norm - 1) : 0.0f;

        for (int i = 0; i < N; i++) {
            const float fx = kpts[i][0] * sx;   // continuous x in feat space
            const float fy = kpts[i][1] * sy;   // continuous y in feat space

            if (mode_ == Mode::BILINEAR) {
                const int x0 = static_cast<int>(std::floor(fx));
                const int y0 = static_cast<int>(std::floor(fy));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                const float dx = fx - static_cast<float>(x0);
                const float dy = fy - static_cast<float>(y0);

                // Helper: read with zeros padding for out-of-bounds
                auto get = [&](int y, int x, int c) -> float {
                    if (x < 0 || x >= W_src || y < 0 || y >= H_src) return 0.0f;
                    return feat[c * H_src * W_src + y * W_src + x];
                };

                for (int c = 0; c < C; c++) {
                    out[i * C + c] =
                        (1.0f - dx) * (1.0f - dy) * get(y0, x0, c) +
                               dx   * (1.0f - dy) * get(y0, x1, c) +
                        (1.0f - dx) *        dy   * get(y1, x0, c) +
                               dx   *        dy   * get(y1, x1, c);
                }
            } else {  // NEAREST
                // Nearest rounding (mirrors PyTorch's nearest mode in grid_sample)
                const int ix = static_cast<int>(std::round(fx));
                const int iy = static_cast<int>(std::round(fy));
                if (ix >= 0 && ix < W_src && iy >= 0 && iy < H_src) {
                    for (int c = 0; c < C; c++) {
                        out[i * C + c] = feat[c * H_src * W_src + iy * W_src + ix];
                    }
                }
                // else: remains 0 from assign()
            }
        }
    }

private:
    Mode mode_;
};

#endif  // INTER_SPARSE_2D_H