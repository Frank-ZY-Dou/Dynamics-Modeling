// Python binding for the exact top-8 SAD candidate builder (exact_candidates_cuda.cu).
#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> build_candidates_cuda(const at::Tensor& sites, int64_t W, int64_t H,
                                              double inv_sq, int64_t mode, int64_t radius_px,
                                              int64_t cell_px, int64_t tile_px, bool key_fp16,
                                              double cap_quantile, int64_t entries_per_site,
                                              bool coop_fallback);

std::vector<at::Tensor> build_candidates(const at::Tensor& sites, int64_t W, int64_t H,
                                         double inv_sq, int64_t mode, int64_t radius_px,
                                         int64_t cell_px, int64_t tile_px, bool key_fp16,
                                         double cap_quantile, int64_t entries_per_site,
                                         bool coop_fallback) {
    TORCH_CHECK(sites.is_cuda(), "sites must be a CUDA tensor");
    TORCH_CHECK(sites.scalar_type() == at::kFloat, "sites must be float32");
    TORCH_CHECK(sites.dim() == 3 && sites.size(2) == 10, "sites must be [F, N, 10]");
    TORCH_CHECK(sites.size(1) < (int64_t(1) << 24), "at most 2^24 - 1 sites");
    TORCH_CHECK(W > 0 && H > 0, "W and H must be positive");
    TORCH_CHECK(inv_sq > 0.0, "inv_sq must be positive");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (exact) or 1 (window)");
    TORCH_CHECK(radius_px >= 0, "radius_px must be >= 0");
    TORCH_CHECK(cell_px >= 1, "cell_px must be >= 1");
    TORCH_CHECK(tile_px == 8 || tile_px == 16, "tile_px must be 8 or 16");
    TORCH_CHECK(cap_quantile > 0.0 && cap_quantile <= 1.0, "cap_quantile must be in (0, 1]");
    TORCH_CHECK(entries_per_site >= 1, "entries_per_site must be >= 1");
    return build_candidates_cuda(sites.contiguous(), W, H, inv_sq, mode, radius_px, cell_px,
                                 tile_px, key_fp16, cap_quantile, entries_per_site, coop_fallback);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_candidates", &build_candidates,
          "Exact (mode 0) or fixed-window (mode 1) top-8 SAD candidates for F frames "
          "-> [cand0 [F,HW,4], cand1 [F,HW,4], stats [F,8]]; coop_fallback: exact mode "
          "rescans uncertified pixels by whole warps (else one thread each), same output");
}
