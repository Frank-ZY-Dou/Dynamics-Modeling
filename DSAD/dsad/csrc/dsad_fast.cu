// Fused CUDA kernels of the DSAD training step and decoder (--fast, dsad/fast_step.py).
//
// The kernels compute the forward pass of the eager PyTorch step and its
// gradients, written out by hand, in about 10 launches per iteration.
//
// Numerics against the eager step (tools/check_fast_ops.py):
//   * Everything that decides which terms are summed is bit-exact: the anchor
//     logits use the same IEEE operations, in the same order, as the eager
//     code (no FMA contraction, via __f*_rn intrinsics), and the top-K
//     selection breaks ties towards the lower anchor index, as torch.topk
//     does. Site packing (positions, colors, unit anisotropy directions), the
//     clamp + MSE gradient, the coarse-stage site gradients and the Adan and
//     Adam updates also reproduce the eager kernels bit for bit.
//   * Sums over the K selected anchors, the gradient reductions over sites
//     (scatter into anchors, ARAP neighbors) and the order in which several
//     gradient contributions are added differ from the eager kernels, so those
//     results differ in the last bits. The eager step is not deterministic at
//     that level either: the SAD render backward accumulates with atomics.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int MODE_VORONOI = 0;
constexpr int MODE_APOLLONIUS = 1;
constexpr int MODE_POWER = 2;
constexpr int MODE_ANISO = 3;
constexpr int TAB = 8;            // floats per anchor-table row
constexpr int MAX_KNOTS = 32;
constexpr int SMOOTH_SAMPLES = 32;   // one sample per lane in the temporal kernels

#define CHECK_CUDA_F32(t) TORCH_CHECK((t).is_cuda() && (t).scalar_type() == torch::kFloat32 && (t).is_contiguous(), #t " must be a contiguous float32 CUDA tensor")

inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }

inline float* fptr(const torch::Tensor& t) { return t.defined() && t.numel() > 0 ? t.data_ptr<float>() : nullptr; }

// ---------------------------------------------------------------------------
// Anchor table. One row of 8 floats per anchor, holding the per-anchor
// quantities exactly as the eager MotionSAD.basis computes them:
//   [mu0, mu1, u0, u1, ea, iea, r, s]
//   aniso       u = dir/(|dir|+1e-8), ea = exp(log_aspect), iea = 1/ea,
//               r = radius, s = exp(-log_sigma)
//   voronoi     s = exp(-log_sigma)
//   apollonius  r = radius, s = exp(-log_sigma)
//   power       r = weight, s = exp(-2 log_sigma)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void anchor_row(int mode, int j, const float* __restrict__ mu,
                                           const float* __restrict__ log_sigma,
                                           const float* __restrict__ radius,
                                           const float* __restrict__ log_aspect,
                                           const float* __restrict__ adir,
                                           const float* __restrict__ weight, float* row) {
    row[0] = mu[2 * j];
    row[1] = mu[2 * j + 1];
    row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
    const float ls = log_sigma[j];
    if (mode == MODE_POWER) {
        row[6] = weight[j];
        row[7] = expf(__fmul_rn(-2.0f, ls));
        return;
    }
    row[7] = expf(-ls);
    if (mode == MODE_APOLLONIUS || mode == MODE_ANISO) row[6] = radius[j];
    if (mode == MODE_ANISO) {
        const float a0 = adir[2 * j], a1 = adir[2 * j + 1];
        const float nrm = __fsqrt_rn(__fadd_rn(__fmul_rn(a0, a0), __fmul_rn(a1, a1)));
        const float den = __fadd_rn(nrm, 1e-8f);
        row[2] = __fdiv_rn(a0, den);
        row[3] = __fdiv_rn(a1, den);
        const float ea = expf(log_aspect[j]);
        row[4] = ea;
        row[5] = __fdiv_rn(1.0f, ea);
    }
}

// Logit of anchor `row` at canonical position x, bit-identical to the eager code.
template <int MODE>
__device__ __forceinline__ float anchor_logit(float x0, float x1, const float* row) {
    const float d0 = __fsub_rn(x0, row[0]);
    const float d1 = __fsub_rn(x1, row[1]);
    const float sq = __fadd_rn(__fmul_rn(d0, d0), __fmul_rn(d1, d1));
    if (MODE == MODE_POWER) return __fmul_rn(__fsub_rn(row[6], sq), row[7]);
    if (MODE == MODE_ANISO) {
        const float proj = __fadd_rn(__fmul_rn(d0, row[2]), __fmul_rn(d1, row[3]));
        const float perp = fmaxf(__fsub_rn(sq, __fmul_rn(proj, proj)), 0.0f);
        const float sqg = __fadd_rn(__fmul_rn(__fmul_rn(row[4], proj), proj), __fmul_rn(row[5], perp));
        const float dist = __fsqrt_rn(fmaxf(sqg, 1e-12f));
        return __fmul_rn(__fsub_rn(row[6], dist), row[7]);
    }
    const float dist = __fsqrt_rn(fmaxf(sq, 1e-12f));
    if (MODE == MODE_VORONOI) return __fmul_rn(-dist, row[7]);
    return __fmul_rn(__fsub_rn(row[6], dist), row[7]);
}

// Backward of anchor_logit: adds dL/dx to gx and returns the 8 per-anchor
// partials in the table layout ([mu0, mu1, u0, u1, ea, iea, r, s]).
template <int MODE>
__device__ __forceinline__ void anchor_logit_bwd(float x0, float x1, const float* row, float gl,
                                                 float& gx0, float& gx1, float* ga) {
    const float d0 = x0 - row[0];
    const float d1 = x1 - row[1];
    const float sq = d0 * d0 + d1 * d1;
    float gd0 = 0.f, gd1 = 0.f;
#pragma unroll
    for (int q = 0; q < TAB; ++q) ga[q] = 0.f;
    if (MODE == MODE_POWER) {
        // l = (w - sq) * s
        ga[6] = gl * row[7];
        ga[7] = gl * (row[6] - sq);
        const float gsq = -(gl * row[7]);
        gd0 = 2.f * d0 * gsq;
        gd1 = 2.f * d1 * gsq;
    } else if (MODE == MODE_ANISO) {
        const float proj = d0 * row[2] + d1 * row[3];
        const float perp_raw = sq - proj * proj;
        const float perp = fmaxf(perp_raw, 0.f);
        const float sqg = row[4] * proj * proj + row[5] * perp;
        const float dist = sqrtf(fmaxf(sqg, 1e-12f));
        // l = (r - dist) * s
        ga[6] = gl * row[7];
        ga[7] = gl * (row[6] - dist);
        const float gdist = -(gl * row[7]);
        const float gsqg = (sqg >= 1e-12f) ? gdist / (2.f * dist) : 0.f;
        ga[4] = gsqg * proj * proj;
        ga[5] = gsqg * perp;
        float gproj = gsqg * 2.f * row[4] * proj;
        const float gperp_raw = (perp_raw >= 0.f) ? gsqg * row[5] : 0.f;
        const float gsq = gperp_raw;
        gproj -= 2.f * proj * gperp_raw;
        gd0 = 2.f * d0 * gsq + row[2] * gproj;
        gd1 = 2.f * d1 * gsq + row[3] * gproj;
        ga[2] = d0 * gproj;
        ga[3] = d1 * gproj;
    } else {
        const float dist = sqrtf(fmaxf(sq, 1e-12f));
        float gdist;
        if (MODE == MODE_VORONOI) {
            // l = -dist * s
            ga[7] = -(gl * dist);
            gdist = -(gl * row[7]);
        } else {
            ga[6] = gl * row[7];
            ga[7] = gl * (row[6] - dist);
            gdist = -(gl * row[7]);
        }
        const float gsq = (sq >= 1e-12f) ? gdist / (2.f * dist) : 0.f;
        gd0 = 2.f * d0 * gsq;
        gd1 = 2.f * d1 * gsq;
    }
    gx0 += gd0;
    gx1 += gd1;
    ga[0] = -gd0;
    ga[1] = -gd1;
}

// Sites' unit anisotropy direction, u = D / (|D| + 1e-8), bit-identical to eager.
__device__ __forceinline__ void unit_dir(float a0, float a1, float& u0, float& u1) {
    const float nrm = __fsqrt_rn(__fadd_rn(__fmul_rn(a0, a0), __fmul_rn(a1, a1)));
    const float den = __fadd_rn(nrm, 1e-8f);
    u0 = __fdiv_rn(a0, den);
    u1 = __fdiv_rn(a1, den);
}

// Backward of unit_dir as autograd evaluates it (DivBackward0 + norm backward):
//   g_a = g_u / den + (sum_c -g_u_c * ((a_c / den) / den)) * (a / |a|)
__device__ __forceinline__ void unit_dir_bwd(float a0, float a1, float gu0, float gu1,
                                             float& ga0, float& ga1) {
    const float nrm = __fsqrt_rn(__fadd_rn(__fmul_rn(a0, a0), __fmul_rn(a1, a1)));
    const float den = __fadd_rn(nrm, 1e-8f);
    const float dir0 = __fdiv_rn(gu0, den);
    const float dir1 = __fdiv_rn(gu1, den);
    const float t0 = __fmul_rn(-gu0, __fdiv_rn(__fdiv_rn(a0, den), den));
    const float t1 = __fmul_rn(-gu1, __fdiv_rn(__fdiv_rn(a1, den), den));
    const float gden = __fadd_rn(t0, t1);
    float n0 = 0.f, n1 = 0.f;
    if (nrm != 0.f) {
        n0 = __fmul_rn(gden, __fdiv_rn(a0, nrm));
        n1 = __fmul_rn(gden, __fdiv_rn(a1, nrm));
    }
    ga0 = __fadd_rn(dir0, n0);
    ga1 = __fadd_rn(dir1, n1);
}

// aten::tanh_backward is one kernel computing g * (1 - y*y). nvcc contracts the
// inner expression there and here, so the two agree bit for bit.
__device__ __forceinline__ float tanh_bwd(float g, float y) { return g * (1.0f - y * y); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Adds v over the whole block into *dst (one atomic per warp). Every thread
// of the block must call it. Lanes are numbered by the linear thread index,
// so this also holds for 2D blocks.
__device__ __forceinline__ void block_atomic_add(float v, float* dst) {
    v = warp_sum(v);
    const unsigned lin = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    if ((lin & 31u) == 0 && v != 0.f) atomicAdd(dst, v);
}

// ---------------------------------------------------------------------------
// Temporal coefficients a_j(t): softmax over K knots of
//   -0.5 * (t - t_k)^2 * exp(-2 log_r_k), weighted sum of the knot values.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void knot_weights(float t, const float* tk, const float* lr, int K, float* w) {
    float mx = -INFINITY;
    for (int k = 0; k < K; ++k) {
        const float diff = __fsub_rn(t, tk[k]);
        const float l = __fmul_rn(__fmul_rn(-0.5f, __fmul_rn(diff, diff)), expf(__fmul_rn(-2.0f, lr[k])));
        w[k] = l;
        mx = fmaxf(mx, l);
    }
    float s = 0.f;
    for (int k = 0; k < K; ++k) {
        w[k] = expf(__fsub_rn(w[k], mx));
        s += w[k];
    }
    for (int k = 0; k < K; ++k) w[k] = __fdiv_rn(w[k], s);
}

struct CoefStack {
    const float* t;   // [M,K]
    const float* lr;  // [M,K]
    const float* v;   // [M,K,D]
    float* gt;        // grads (written)
    float* glr;
    float* gv;
    int D;
    int col;          // first column in the [M,C] coefficient table
};

constexpr unsigned FULL_MASK = 0xffffffffu;
static_assert(SMOOTH_SAMPLES == 32, "the temporal kernels map one smoothness sample to each lane");

// Softmax over one anchor's K knots at time t, one lane per knot (lanes >= K
// get weight 0). The butterfly sum adds the same terms in the same tree as
// torch's warp softmax for K = 8 or 16 (the extra lanes add zeros).
__device__ __forceinline__ float knot_weight_lane(float t, const float* tk, const float* lr, int K, int lane) {
    float l = -INFINITY;
    if (lane < K) {
        const float diff = __fsub_rn(t, tk[lane]);
        l = __fmul_rn(__fmul_rn(-0.5f, __fmul_rn(diff, diff)), expf(__fmul_rn(-2.0f, lr[lane])));
    }
    float mx = l;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL_MASK, mx, o));
    const float e = lane < K ? expf(__fsub_rn(l, mx)) : 0.f;
    const float s = warp_sum(e);
    return lane < K ? __fdiv_rn(e, s) : 0.f;
}

// Temporal smoothness curve of the xyz stack at this lane's sample time
// (lane i evaluates lin[i]); fills the lane's knot weights w[0..K).
__device__ __forceinline__ void smooth_curve_lane(const CoefStack& s, int j, int K, const float* lin,
                                                  int lane, float* w, float* cc) {
    const float* tk = s.t + j * K;
    const float* lr = s.lr + j * K;
    const float* v = s.v + j * K * 2;
    knot_weights(lin[lane], tk, lr, K, w);
    for (int c = 0; c < 2; ++c) {
        float a = 0.f;
        for (int k = 0; k < K; ++k) a += w[k] * v[k * 2 + c];
        cc[c] = a;
    }
}

// Row j of the [M,C] coefficient table at time t: a_j(t) of every stack (one warp).
__device__ __forceinline__ void table_row(float t, const CoefStack& s0, const CoefStack& s1, int nstacks, int j,
                                          int K, int C, int lane, float* __restrict__ table) {
    for (int si = 0; si < nstacks; ++si) {
        const CoefStack& s = si == 0 ? s0 : s1;
        const float w = knot_weight_lane(t, s.t + j * K, s.lr + j * K, K, lane);
        const float* v = s.v + j * K * s.D;
        for (int c = 0; c < s.D; ++c) {
            const float a = warp_sum(lane < K ? w * v[lane * s.D + c] : 0.f);
            if (lane == 0) table[j * C + s.col + c] = a;
        }
    }
}

// One warp per anchor: a(t) for every coefficient stack into the [M,C]
// table, and the temporal smoothness loss of stack 0 (xyz) at the 32-sample
// grid (one sample per lane).
__global__ void k_temporal_fwd(int M, int K, int C, float t, CoefStack s0, CoefStack s1, int nstacks,
                               const float* __restrict__ lin, int smooth, float* __restrict__ table,
                               float* __restrict__ stats) {
    const int lane = threadIdx.x & 31;
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (j >= M) return;  // whole warps exit together
    table_row(t, s0, s1, nstacks, j, K, C, lane, table);
    if (smooth) {
        float w[MAX_KNOTS], cc[2];
        smooth_curve_lane(s0, j, K, lin, lane, w, cc);
        float loss = 0.f;
        for (int c = 0; c < 2; ++c) {
            const float c1 = __shfl_down_sync(FULL_MASK, cc[c], 1);
            const float c2 = __shfl_down_sync(FULL_MASK, cc[c], 2);
            const float d2 = c2 - 2.f * c1 + cc[c];
            if (lane + 2 < SMOOTH_SAMPLES) loss += d2 * d2;
        }
        loss = warp_sum(loss);
        if (lane == 0) atomicAdd(stats + 2, loss);
    }
}

// The tables of the F frames of a GoP pass (grid y = frame f), frame f at
// t = (f + 1) / F rounded to float32, the time FusedStep.decode gives it.
__global__ void k_temporal_gop(int M, int K, int C, int F, CoefStack s0, CoefStack s1, int nstacks,
                               float* __restrict__ tables) {
    const int lane = threadIdx.x & 31;
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (j >= M) return;
    const int f = blockIdx.y;
    const float t = static_cast<float>(static_cast<double>(f + 1) / static_cast<double>(F));
    table_row(t, s0, s1, nstacks, j, K, C, lane, tables + static_cast<int64_t>(f) * M * C);
}

// ---------------------------------------------------------------------------
// ROM forward + site packing. One thread per site.
//   x = tanh(P); logits of all M anchors; top-K; softmax; d = sum_k phi_k A_k
//   y = tanh(P + d_xyz); sites = [((y+1)*0.5)*wh, log_tau, radius, F + d_feat, u, log_aniso]
// In the coarse stage (k_sites_fwd) d = 0 and y = tanh(P), as in forward_mse_loss.
// ---------------------------------------------------------------------------
struct SiteArgs {
    const float* P;     // [N,2]
    const float* F;     // [N,3]
    const float* LT;    // [N]
    const float* R;     // [N]
    const float* D;     // [N,2]
    const float* LA;    // [N]
    float W, H;
    float* sites;       // [N,10]
    float* y;           // [N,2] deformed tanh positions (may be null)
    float* gy_arap;     // [N,2] zeroed here for the ARAP kernel (may be null)
    float* sites12;     // [N,12] the same sites padded to 3 float4 (decoding's render; may be null)
};

struct AnchorArgs {
    const float* mu;
    const float* log_sigma;
    const float* radius;
    const float* log_aspect;
    const float* adir;
    const float* weight;
    int mode;
};

__device__ __forceinline__ void write_site(int n, const SiteArgs& a, float y0, float y1,
                                           const float* df) {
    float* s = a.sites + 10 * n;
    s[0] = __fmul_rn(__fmul_rn(__fadd_rn(y0, 1.0f), 0.5f), a.W);
    s[1] = __fmul_rn(__fmul_rn(__fadd_rn(y1, 1.0f), 0.5f), a.H);
    s[2] = a.LT[n];
    s[3] = a.R[n];
    for (int c = 0; c < 3; ++c) s[4 + c] = df ? __fadd_rn(a.F[3 * n + c], df[c]) : a.F[3 * n + c];
    float u0, u1;
    unit_dir(a.D[2 * n], a.D[2 * n + 1], u0, u1);
    s[7] = u0;
    s[8] = u1;
    s[9] = a.LA[n];
    if (a.sites12) {
        float4* p = reinterpret_cast<float4*>(a.sites12) + 3 * n;
        p[0] = make_float4(s[0], s[1], s[2], s[3]);
        p[1] = make_float4(s[4], s[5], s[6], s[7]);
        p[2] = make_float4(s[8], s[9], 0.0f, 0.0f);
    }
}

__global__ void k_sites_fwd(int N, SiteArgs a) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float y0 = tanhf(a.P[2 * n]), y1 = tanhf(a.P[2 * n + 1]);
    write_site(n, a, y0, y1, nullptr);
    if (a.y) { a.y[2 * n] = y0; a.y[2 * n + 1] = y1; }
}

// Insert (v, id) into the sorted top-K list (value desc, index asc).
template <int K>
__device__ __forceinline__ void topk_insert(float* tv, int* ti, float v, int id) {
#pragma unroll
    for (int q = 0; q < K; ++q) {
        const bool sw = (v > tv[q]) || (v == tv[q] && id < ti[q]);
        const float tq = tv[q];
        const int iq = ti[q];
        tv[q] = sw ? v : tq;
        ti[q] = sw ? id : iq;
        v = sw ? tq : v;
        id = sw ? iq : id;
    }
}

// Whether anchor j is in the top-K list.
template <int K>
__device__ __forceinline__ bool topk_listed(const int* ti, int j) {
    bool in = false;
#pragma unroll
    for (int q = 0; q < K; ++q) in |= ti[q] == j;
    return in;
}

// The top-K list is warm-started with the anchors in `warm` (the previous
// iteration's selection for this site, logits recomputed now), then all M
// anchors are scanned; an anchor already in the list is not inserted again.
// So each anchor counts once under the same total order, and the result does
// not depend on the warm start, which only makes later insertions rare. (A
// warm anchor that left the list ranks below its K-th entry, so its second
// visit does not insert it either.) Shared memory holds the anchor table only.
template <int MODE, int K>
__global__ void k_rom_fwd(int N, int M, int C, AnchorArgs an, const float* __restrict__ coef,
                          const int32_t* warm,
                          float* __restrict__ d_out, int32_t* topi_out,
                          float* __restrict__ phi_out, float* __restrict__ tab_out,
                          int pack, SiteArgs sa, float* __restrict__ stats) {
    extern __shared__ float smem[];
    float* tab = smem;                                            // [M, TAB]
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        float row[TAB];
        anchor_row(MODE, j, an.mu, an.log_sigma, an.radius, an.log_aspect, an.adir, an.weight, row);
#pragma unroll
        for (int q = 0; q < TAB; ++q) tab[TAB * j + q] = row[q];
        if (blockIdx.x == 0 && tab_out) {
#pragma unroll
            for (int q = 0; q < TAB; ++q) tab_out[TAB * j + q] = row[q];
        }
    }
    __syncthreads();
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    float move = 0.f;
    if (n < N) {
        const float p0 = sa.P[2 * n], p1 = sa.P[2 * n + 1];
        const float x0 = tanhf(p0), x1 = tanhf(p1);
        float tv[K];
        int ti[K];
#pragma unroll
        for (int k = 0; k < K; ++k) { tv[k] = -INFINITY; ti[k] = 0x7fffffff; }
        if (warm) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const int j = warm[K * n + k];
                if (j < 0 || j >= M || topk_listed<K>(ti, j)) continue;
                topk_insert<K>(tv, ti, anchor_logit<MODE>(x0, x1, tab + TAB * j), j);
            }
        }
        for (int j = 0; j < M; ++j) {
            const float l = anchor_logit<MODE>(x0, x1, tab + TAB * j);
            if ((l > tv[K - 1] || (l == tv[K - 1] && j < ti[K - 1])) && !topk_listed<K>(ti, j))
                topk_insert<K>(tv, ti, l, j);
        }
        float e[K];
        float s = 0.f;
        const float mx = tv[0];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            e[k] = expf(__fsub_rn(tv[k], mx));
            s += e[k];
        }
        float d[8];
        for (int c = 0; c < C; ++c) d[c] = 0.f;
        const bool valid = ti[K - 1] != 0x7fffffff && s == s;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float ph = valid ? __fdiv_rn(e[k], s) : NAN;
            const int j = valid ? ti[k] : 0;
            phi_out[K * n + k] = ph;
            topi_out[K * n + k] = j;
            // An explicit fused multiply-add, as in k_rom_apply, so that no toolchain can round
            // the two kernels differently.
            for (int c = 0; c < C; ++c) d[c] = __fmaf_rn(ph, coef[C * j + c], d[c]);
        }
        for (int c = 0; c < C; ++c) d_out[C * n + c] = d[c];
        if (pack) {
            const float y0 = tanhf(__fadd_rn(p0, d[0]));
            const float y1 = tanhf(__fadd_rn(p1, d[1]));
            write_site(n, sa, y0, y1, C >= 5 ? d + 2 : nullptr);
            if (sa.y) { sa.y[2 * n] = y0; sa.y[2 * n + 1] = y1; }
            if (sa.gy_arap) { sa.gy_arap[2 * n] = 0.f; sa.gy_arap[2 * n + 1] = 0.f; }
            move = d[0] * d[0] + d[1] * d[1];
        }
    }
    if (pack && stats) block_atomic_add(move, stats + 1);
}

// ROM apply, for decoding. The anchor basis (topi, phi) depends on the canonical sites and the
// anchors only, not on t, so after one k_rom_fwd on a GoP's first frame the later frames need just
// d = sum_k phi_k a_{topi_k}(t) and the packing. The arithmetic is k_rom_fwd's, in the same order
// and the same explicitly rounded operations (__fmaf_rn accumulation, the same packing), so the
// output is bit-identical to k_rom_fwd with that selection. Several frames per launch (grid y =
// frame f): frame f reads coef + f * coef_stride and writes sa.sites + f * site_stride (d_out, if
// given, + f * N * C; sa.y and sa.gy_arap only with one frame).
template <int K>
__global__ void k_rom_apply(int N, int C, const float* __restrict__ coef, const int32_t* __restrict__ topi,
                            const float* __restrict__ phi, float* __restrict__ d_out, int pack, SiteArgs sa,
                            int64_t coef_stride, int64_t site_stride) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int f = blockIdx.y;
    coef += f * coef_stride;
    if (pack) sa.sites += f * site_stride;
    if (pack && sa.sites12) sa.sites12 += static_cast<int64_t>(f) * N * 12;
    float d[8];
    for (int c = 0; c < C; ++c) d[c] = 0.f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const float ph = phi[K * n + k];
        const int j = topi[K * n + k];
        for (int c = 0; c < C; ++c) d[c] = __fmaf_rn(ph, coef[C * j + c], d[c]);  // as k_rom_fwd
    }
    if (d_out) {
        for (int c = 0; c < C; ++c) d_out[static_cast<int64_t>(f) * N * C + C * n + c] = d[c];
    }
    if (pack) {
        const float p0 = sa.P[2 * n], p1 = sa.P[2 * n + 1];
        const float y0 = tanhf(__fadd_rn(p0, d[0]));
        const float y1 = tanhf(__fadd_rn(p1, d[1]));
        write_site(n, sa, y0, y1, C >= 5 ? d + 2 : nullptr);
        if (sa.y) { sa.y[2 * n] = y0; sa.y[2 * n + 1] = y1; }
        if (sa.gy_arap) { sa.gy_arap[2 * n] = 0.f; sa.gy_arap[2 * n + 1] = 0.f; }
    }
}

// ---------------------------------------------------------------------------
// ARAP: loss = mean_{n,k} (|y_n - y_m| - |x_n - x_m|)^2, m = knn[n,k],
// x = tanh(P) (canonical, constant), y deformed. Writes dL/dy (scaled by
// lam) into gy via atomics; gy was zeroed by k_rom_fwd.
// ---------------------------------------------------------------------------
__global__ void k_arap(int N, int KN, const float* __restrict__ P, const float* __restrict__ y,
                       const int64_t* __restrict__ knn, float coef, float* __restrict__ gy,
                       float* __restrict__ stats) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    float loss = 0.f;
    if (n < N) {
        const float xn0 = tanhf(P[2 * n]), xn1 = tanhf(P[2 * n + 1]);
        const float yn0 = y[2 * n], yn1 = y[2 * n + 1];
        float g0 = 0.f, g1 = 0.f;
        for (int k = 0; k < KN; ++k) {
            const int m = static_cast<int>(knn[(int64_t)n * KN + k]);
            const float cx0 = __fsub_rn(xn0, tanhf(P[2 * m])), cx1 = __fsub_rn(xn1, tanhf(P[2 * m + 1]));
            const float dc = __fsqrt_rn(__fadd_rn(__fmul_rn(cx0, cx0), __fmul_rn(cx1, cx1)));
            const float v0 = __fsub_rn(yn0, y[2 * m]), v1 = __fsub_rn(yn1, y[2 * m + 1]);
            const float dd = __fsqrt_rn(__fadd_rn(__fmul_rn(v0, v0), __fmul_rn(v1, v1)));
            const float r = __fsub_rn(dd, dc);
            loss += r * r;
            if (dd != 0.f) {
                const float gdd = coef * r;
                const float w0 = gdd * (v0 / dd), w1 = gdd * (v1 / dd);
                g0 += w0;
                g1 += w1;
                atomicAdd(gy + 2 * m, -w0);
                atomicAdd(gy + 2 * m + 1, -w1);
            }
        }
        atomicAdd(gy + 2 * n, g0);
        atomicAdd(gy + 2 * n + 1, g1);
    }
    block_atomic_add(loss, stats + 3);
}

// ---------------------------------------------------------------------------
// Clamp + MSE: grad = (0 <= r <= 1) ? (alpha * (clamp(r) - gt)) * 1 : 0,
// bit-identical to MseLossBackward + ClampBackward1; also sums the squared error.
// ---------------------------------------------------------------------------
__global__ void k_mse(int64_t n, const float* __restrict__ r, const float* __restrict__ gt,
                      float alpha, float* __restrict__ grad, float* __restrict__ stats) {
    float acc = 0.f;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const float v = r[i];
        const float rc = fminf(fmaxf(v, 0.f), 1.f);
        const float diff = __fsub_rn(rc, gt[i]);
        acc += diff * diff;
        grad[i] = (v >= 0.f && v <= 1.f) ? __fmul_rn(__fmul_rn(alpha, diff), 1.0f) : 0.f;
    }
    block_atomic_add(acc, stats + 0);
}

// ---------------------------------------------------------------------------
// Site backward (+ ROM backward in the fine stage). One thread per site.
// Inputs: gs = dL/dsites from the SAD render backward.
// ---------------------------------------------------------------------------
struct SiteGrads {
    float* gP; float* gF; float* gLT; float* gR; float* gD; float* gLA;
};

__global__ void k_sites_bwd(int N, SiteArgs a, const float* __restrict__ gs, SiteGrads g) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* q = gs + 10 * n;
    const float x0 = tanhf(a.P[2 * n]), x1 = tanhf(a.P[2 * n + 1]);
    g.gP[2 * n] = tanh_bwd(__fmul_rn(__fmul_rn(q[0], a.W), 0.5f), x0);
    g.gP[2 * n + 1] = tanh_bwd(__fmul_rn(__fmul_rn(q[1], a.H), 0.5f), x1);
    g.gLT[n] = q[2];
    g.gR[n] = q[3];
    for (int c = 0; c < 3; ++c) g.gF[3 * n + c] = q[4 + c];
    float g0, g1;
    unit_dir_bwd(a.D[2 * n], a.D[2 * n + 1], q[7], q[8], g0, g1);
    g.gD[2 * n] = g0;
    g.gD[2 * n + 1] = g1;
    g.gLA[n] = q[9];
}

template <int MODE, int K, bool SHARED_ACC>
__global__ void k_rom_bwd(int N, int M, int C, SiteArgs a, const float* __restrict__ gs,
                          SiteGrads g, const float* __restrict__ coef, const float* __restrict__ tab,
                          const float* __restrict__ d, const int32_t* __restrict__ topi,
                          const float* __restrict__ phi, float move_coef,
                          float* __restrict__ gacc, float* __restrict__ gcoef) {
    extern __shared__ float smem[];
    const int stride = TAB + C;
    float* acc = smem;  // [M, TAB + C] when SHARED_ACC
    if (SHARED_ACC) {
        for (int i = threadIdx.x; i < M * stride; i += blockDim.x) acc[i] = 0.f;
        __syncthreads();
    }
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        const float* q = gs + 10 * n;
        const float y0 = a.y[2 * n], y1 = a.y[2 * n + 1];
        // Render path: pos_px = ((y + 1) * 0.5) * wh, y = tanh(P + d_xyz).
        const float r0 = tanh_bwd(__fmul_rn(__fmul_rn(q[0], a.W), 0.5f), y0);
        const float r1 = tanh_bwd(__fmul_rn(__fmul_rn(q[1], a.H), 0.5f), y1);
        // ARAP path (a separate tanh node in the eager graph).
        float a0 = 0.f, a1 = 0.f;
        if (a.gy_arap) {
            a0 = tanh_bwd(a.gy_arap[2 * n], y0);
            a1 = tanh_bwd(a.gy_arap[2 * n + 1], y1);
        }
        float gd[8];
        gd[0] = r0 + a0;
        gd[1] = r1 + a1;
        if (move_coef != 0.f) {
            // movement penalty mean_n |d|^2: d/dd = (g/N) d + (g/N) d
            const float t0 = move_coef * d[C * n], t1 = move_coef * d[C * n + 1];
            gd[0] += t0 + t0;
            gd[1] += t1 + t1;
        }
        for (int c = 2; c < C; ++c) gd[c] = q[2 + c];  // d_feat gets dL/dcolor
        float gP0 = r0 + a0, gP1 = r1 + a1;
        g.gLT[n] = q[2];
        g.gR[n] = q[3];
        for (int c = 0; c < 3; ++c) g.gF[3 * n + c] = q[4 + c];
        float u0, u1;
        unit_dir_bwd(a.D[2 * n], a.D[2 * n + 1], q[7], q[8], u0, u1);
        g.gD[2 * n] = u0;
        g.gD[2 * n + 1] = u1;
        g.gLA[n] = q[9];

        // ROM backward: d = sum_k phi_k A[topi_k]; phi = softmax(top-K logits).
        const float x0 = tanhf(a.P[2 * n]), x1 = tanhf(a.P[2 * n + 1]);
        float gphi[K], ph[K];
        int ti[K];
        float sdot = 0.f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            ti[k] = topi[K * n + k];
            ph[k] = phi[K * n + k];
            float acc_k = 0.f;
            for (int c = 0; c < C; ++c) acc_k += gd[c] * coef[C * ti[k] + c];
            gphi[k] = acc_k;
            sdot += ph[k] * acc_k;
        }
        float gx0 = 0.f, gx1 = 0.f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int j = ti[k];
            const float gl = ph[k] * (gphi[k] - sdot);
            float ga[TAB];
            anchor_logit_bwd<MODE>(x0, x1, tab + TAB * j, gl, gx0, gx1, ga);
            float* dst = SHARED_ACC ? acc + stride * j : nullptr;
#pragma unroll
            for (int q2 = 0; q2 < TAB; ++q2) {
                if (ga[q2] != 0.f) {
                    if (SHARED_ACC) atomicAdd(dst + q2, ga[q2]);
                    else atomicAdd(gacc + TAB * j + q2, ga[q2]);
                }
            }
            for (int c = 0; c < C; ++c) {
                const float v = ph[k] * gd[c];
                if (SHARED_ACC) atomicAdd(dst + TAB + c, v);
                else atomicAdd(gcoef + C * j + c, v);
            }
        }
        gP0 += tanh_bwd(gx0, x0);
        gP1 += tanh_bwd(gx1, x1);
        g.gP[2 * n] = gP0;
        g.gP[2 * n + 1] = gP1;
    }
    if (SHARED_ACC) {
        __syncthreads();
        for (int i = threadIdx.x; i < M * stride; i += blockDim.x) {
            const float v = acc[i];
            if (v != 0.f) {
                const int j = i / stride, q2 = i % stride;
                if (q2 < TAB) atomicAdd(gacc + TAB * j + q2, v);
                else atomicAdd(gcoef + C * j + (q2 - TAB), v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-anchor backward: temporal coefficients (current time + smoothness) and
// the anchor parameters from the accumulated table partials. Zeroes the
// accumulators for the next iteration.
// ---------------------------------------------------------------------------
struct AnchorGrads {
    float* gmu; float* glog_sigma; float* gradius; float* glog_aspect; float* gadir; float* gweight;
};

// One warp per anchor. Lane k owns knot k for the gradient write-out; for the
// smoothness term lane i evaluates sample time lin[i].
__global__ void k_anchor_bwd(int M, int K, int C, float t, CoefStack s0, CoefStack s1, int nstacks,
                             const float* __restrict__ lin, float smooth_coef,
                             AnchorArgs an, AnchorGrads ag, const float* __restrict__ tab,
                             float* __restrict__ gacc, float* __restrict__ gcoef) {
    const int lane = threadIdx.x & 31;
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (j >= M) return;  // whole warps exit together
    for (int si = 0; si < nstacks; ++si) {
        const CoefStack& s = si == 0 ? s0 : s1;
        const float* tk = s.t + j * K;
        const float* lr = s.lr + j * K;
        const float* v = s.v + j * K * s.D;
        float ga[3] = {0.f, 0.f, 0.f};
        for (int c = 0; c < s.D; ++c) ga[c] = gcoef[C * j + s.col + c];
        // a(t) = sum_k w_k v_k at the current time; lane k = knot k.
        float acc_t = 0.f, acc_lr = 0.f, acc_v[3] = {0.f, 0.f, 0.f};
        {
            const float w = knot_weight_lane(t, tk, lr, K, lane);
            float gw = 0.f;
            if (lane < K)
                for (int c = 0; c < s.D; ++c) gw += ga[c] * v[lane * s.D + c];
            const float sdot = warp_sum(w * gw);
            if (lane < K) {
                const float gl = w * (gw - sdot);
                const float diff = t - tk[lane];
                const float inv = expf(-2.0f * lr[lane]);
                acc_t = gl * inv * diff;
                acc_lr = gl * (-0.5f * diff * diff) * inv * -2.0f;
                for (int c = 0; c < s.D; ++c) acc_v[c] = w * ga[c];
            }
        }
        if (si == 0 && smooth_coef != 0.f) {
            // L += lam_t * mean_{i,j,c} d2^2,  d2_i = c_{i+2} - 2 c_{i+1} + c_i.
            float w[MAX_KNOTS], cc[2], gc[2];
            smooth_curve_lane(s, j, K, lin, lane, w, cc);
            for (int c = 0; c < 2; ++c) {
                const float c1 = __shfl_down_sync(FULL_MASK, cc[c], 1);
                const float c2 = __shfl_down_sync(FULL_MASK, cc[c], 2);
                const float d2 = c2 - 2.f * c1 + cc[c];
                const float h = smooth_coef * d2;
                const float gd2 = (lane + 2 < SMOOTH_SAMPLES) ? h + h : 0.f;
                const float gm1 = __shfl_up_sync(FULL_MASK, gd2, 1);
                const float gm2 = __shfl_up_sync(FULL_MASK, gd2, 2);
                gc[c] = gd2 + (lane >= 1 ? -(gm1 * 2.f) : 0.f) + (lane >= 2 ? gm2 : 0.f);
            }
            // Backward of this lane's curve point, reduced over lanes per knot.
            float sdot = 0.f, gwk[MAX_KNOTS];
            for (int k = 0; k < K; ++k) {
                gwk[k] = gc[0] * v[k * 2] + gc[1] * v[k * 2 + 1];
                sdot += w[k] * gwk[k];
            }
            const float si_t = lin[lane];
            for (int k = 0; k < K; ++k) {
                const float gl = w[k] * (gwk[k] - sdot);
                const float diff = si_t - tk[k];
                const float inv = expf(-2.0f * lr[k]);
                const float rt = warp_sum(gl * inv * diff);
                const float rlr = warp_sum(gl * (-0.5f * diff * diff) * inv * -2.0f);
                const float rv0 = warp_sum(w[k] * gc[0]);
                const float rv1 = warp_sum(w[k] * gc[1]);
                if (lane == k) {
                    acc_t += rt;
                    acc_lr += rlr;
                    acc_v[0] += rv0;
                    acc_v[1] += rv1;
                }
            }
        }
        if (lane < K) {
            s.gt[j * K + lane] = acc_t;
            s.glr[j * K + lane] = acc_lr;
            for (int c = 0; c < s.D; ++c) s.gv[(j * K + lane) * s.D + c] = acc_v[c];
        }
    }
    __syncwarp();
    for (int c = lane; c < C; c += 32) gcoef[C * j + c] = 0.f;
    if (lane != 0) return;
    // Anchor parameters from the [mu0, mu1, u0, u1, ea, iea, r, s] partials.
    float acc[TAB];
    for (int q = 0; q < TAB; ++q) { acc[q] = gacc[TAB * j + q]; gacc[TAB * j + q] = 0.f; }
    const float* row = tab + TAB * j;
    ag.gmu[2 * j] = acc[0];
    ag.gmu[2 * j + 1] = acc[1];
    if (an.mode == MODE_POWER) {
        ag.gweight[j] = acc[6];
        ag.glog_sigma[j] = (acc[7] * row[7]) * -2.0f;
        return;
    }
    ag.glog_sigma[j] = -(acc[7] * row[7]);
    if (an.mode == MODE_APOLLONIUS || an.mode == MODE_ANISO) ag.gradius[j] = acc[6];
    if (an.mode == MODE_ANISO) {
        // iea = reciprocal(ea) * 1.0 ; ea = exp(log_aspect)
        const float gea = acc[4] + -acc[5] * (row[5] * row[5]);
        ag.glog_aspect[j] = gea * row[4];
        float g0, g1;
        unit_dir_bwd(an.adir[2 * j], an.adir[2 * j + 1], acc[2], acc[3], g0, g1);
        ag.gadir[2 * j] = g0;
        ag.gadir[2 * j + 1] = g1;
    }
}

// ---------------------------------------------------------------------------
// Fused optimizers. Each reproduces, element by element and with the same
// roundings, the multi-tensor (foreach) implementation it replaces: Adan
// (_multi_tensor_adan of baselines/D2GV/optimizer.py, clip factor 1, no weight
// decay; Apache-2.0) and torch.optim.Adam (_multi_tensor_adam, no amsgrad /
// weight decay / maximize; BSD-3-Clause). See THIRD_PARTY_NOTICES.md.
// ---------------------------------------------------------------------------
constexpr int MAX_OPT_TENSORS = 12;

struct AdanTensor {
    float* p; const float* g; float* m; float* v; float* dif; float* npg; int64_t n;
    float b1, b2, b3, omb1, omb2, omb3, inv_bc3s, eps, nss, nssd;
    int init;
};
struct AdanBatch { AdanTensor t[MAX_OPT_TENSORS]; int64_t start[MAX_OPT_TENSORS + 1]; int count; };

__global__ void k_adan(AdanBatch b) {
    const int64_t total = b.start[b.count];
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x) {
        int ti = 0;
        while (i >= b.start[ti + 1]) ++ti;
        const AdanTensor& T = b.t[ti];
        const int64_t e = i - b.start[ti];
        const float g = T.g[e];
        float n = T.init ? __fmul_rn(g, -1.0f) : T.npg[e];
        n = __fadd_rn(n, g);
        float m = __fmaf_rn(T.omb1, g, __fmul_rn(T.m[e], T.b1));
        float dif = __fmaf_rn(T.omb2, n, __fmul_rn(T.dif[e], T.b2));
        n = __fadd_rn(__fmul_rn(n, T.b2), g);
        float v = __fmaf_rn(T.omb3, __fmul_rn(n, n), __fmul_rn(T.v[e], T.b3));
        // _foreach_div_(denom, scalar) multiplies by the reciprocal (double, rounded to float).
        float den = __fadd_rn(__fmul_rn(__fsqrt_rn(v), T.inv_bc3s), T.eps);
        float p = T.p[e];
        p = __fmaf_rn(T.nss, __fdiv_rn(m, den), p);
        p = __fmaf_rn(T.nssd, __fdiv_rn(dif, den), p);
        T.p[e] = p;
        T.m[e] = m;
        T.dif[e] = dif;
        T.v[e] = v;
        T.npg[e] = __fmaf_rn(-1.0f, g, 0.0f);
    }
}

struct AdamTensor {
    float* p; const float* g; float* m; float* v; int64_t n;
    float w, b2, omb2, bc2s, eps, ss;
};
struct AdamBatch { AdamTensor t[MAX_OPT_TENSORS]; int64_t start[MAX_OPT_TENSORS + 1]; int count; };

__global__ void k_adam(AdamBatch b) {
    const int64_t total = b.start[b.count];
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x) {
        int ti = 0;
        while (i >= b.start[ti + 1]) ++ti;
        const AdamTensor& T = b.t[ti];
        const int64_t e = i - b.start[ti];
        const float g = T.g[e];
        float m = T.m[e];
        // lerp(m, g, w) with |w| < 0.5: m + w * (g - m)
        m = __fmaf_rn(T.w, __fsub_rn(g, m), m);
        float v = __fmaf_rn(T.omb2, __fmul_rn(g, g), __fmul_rn(T.v[e], T.b2));
        const float den = __fadd_rn(__fdiv_rn(__fsqrt_rn(v), T.bc2s), T.eps);
        T.p[e] = __fmaf_rn(T.ss, __fdiv_rn(m, den), T.p[e]);
        T.m[e] = m;
        T.v[e] = v;
    }
}

// Per-iteration loss values from the summed terms, and float64 running sums
// for the periodic diagnostic line. Resets the sums.
__global__ void k_stats(float* stats, const float* scale, const float* lam, float* cur, double* running) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float total = 0.f;
    for (int i = 0; i < 4; ++i) {
        const float v = stats[i] * scale[i];
        cur[i] = v;
        running[i] += static_cast<double>(v);
        total += lam[i] * v;
        stats[i] = 0.f;
    }
    cur[4] = total;
}

int grid_for(int64_t n, int block) { return static_cast<int>((n + block - 1) / block); }

// ---------------------------------------------------------------------------
// Render + clamp/MSE + render backward in one pass (training only).
//
// The code in namespace sadref comes from SAD (MIT License, Copyright (c) 2026
// LuckyIYI; the full license text is in THIRD_PARTY_NOTICES.md):
//   * Site, dot, site_color, site_aniso_dir and voronoi_dmix2 are copied from
//     backends/cuda/src/sad_common.cuh;
//   * candidate_grad_terms is the per-candidate body of render_sad_backward_kernel
//     (backends/pytorch/torch_ext/cuda/sad_bindings.cu);
//   * render_weights merges renderVoronoiKernel's color (backends/cuda/src/render.cu)
//     and compute_candidate_weights_raw's weights and prediction (sad_bindings.cu)
//     into one pass over the candidates, with their expressions and each one's
//     validity rules.
// Compiled with the same nvcc and flags, every per-pixel weight, rendered color
// and per-candidate gradient term is the one render_sad / render_sad_backward
// compute. Only the accumulation differs: the lanes of a warp that hold the same
// site in the same candidate slot are summed first, then one atomic per value,
// in place of the SAD kernel's 80 contended atomics per pixel.
// ---------------------------------------------------------------------------
namespace sadref {

#define SAD_NUM_CANDIDATES 8

struct Site {
    float2 position;
    float log_tau;
    float radius;
    float color_r;
    float color_g;
    float color_b;
    float aniso_dir_x;
    float aniso_dir_y;
    float log_aniso;
};

__device__ __forceinline__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ __forceinline__ float3 site_color(const Site &s) {
    return make_float3(s.color_r, s.color_g, s.color_b);
}

__host__ __device__ __forceinline__ float2 site_aniso_dir(const Site &s) {
    return make_float2(s.aniso_dir_x, s.aniso_dir_y);
}

__device__ __forceinline__ float voronoi_dmix2(
    const Site &site,
    float2 uv,
    float inv_scale_sq
) {
    float2 diff = make_float2(uv.x - site.position.x, uv.y - site.position.y);

    // Anisotropic metric (SPD, det=1):
    // G = e^a * uu^T + e^-a * (I - uu^T)
    float diff2 = dot(diff, diff);
    float proj = dot(site_aniso_dir(site), diff);
    float proj2 = proj * proj;
    float perp2 = fmaxf(diff2 - proj2, 0.0f);

    float l1 = expf(site.log_aniso);
    float l2 = 1.0f / l1;
    float d2_aniso = l1 * proj2 + l2 * perp2;

    float d2_norm = d2_aniso * inv_scale_sq;
    float d2_safe = fmaxf(d2_norm, 1e-8f);
    float inv_scale = sqrtf(inv_scale_sq);
    float r_norm = site.radius * inv_scale;

    return (sqrtf(d2_safe) - r_norm);
}

// renderVoronoiKernel's color and compute_candidate_weights_raw's weights and
// prediction from one pass over the candidates. Both functions compute the
// same logits, exponentials, sum and 1/sum with the same expressions; this
// evaluates them once and applies each function's own validity rules, so
// every output value equals the one the corresponding function returns.
__device__ __forceinline__ bool render_weights(const uint32_t cand_ids[SAD_NUM_CANDIDATES],
                                               const Site* __restrict__ sites, uint32_t site_count,
                                               float2 uv, float inv_scale_sq, float3& color,
                                               float weights[SAD_NUM_CANDIDATES], float3& pred) {
    float logits[SAD_NUM_CANDIDATES];
    float3 col[SAD_NUM_CANDIDATES];
    bool cvalid[SAD_NUM_CANDIDATES];
    float max_logit = -INFINITY;
    bool any = false;
    #pragma unroll
    for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
        uint32_t idx = cand_ids[i];
        cvalid[i] = false;
        col[i] = make_float3(0.0f, 0.0f, 0.0f);
        if (idx >= site_count) {
            logits[i] = -INFINITY;
            continue;
        }
        Site site = sites[idx];
        cvalid[i] = site.position.x >= 0.0f;   // color loops: idx < count && x >= 0
        col[i] = site_color(site);
        if (site.position.x < 0.0f) {
            logits[i] = -INFINITY;
            continue;
        }
        float tau = expf(site.log_tau);
        float dmix2 = voronoi_dmix2(site, uv, inv_scale_sq);
        logits[i] = -tau * dmix2;
        max_logit = fmaxf(max_logit, logits[i]);
        any = true;
    }

    // renderVoronoiKernel: zero color when every candidate is invalid.
    color = make_float3(0.0f, 0.0f, 0.0f);
    float inv_sum = 0.0f;
    if (!(isinf(max_logit) && max_logit < 0.0f)) {
        float sum_w = 0.0f;
        #pragma unroll
        for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
            weights[i] = expf(logits[i] - max_logit);
            sum_w += weights[i];
        }
        inv_sum = 1.0f / fmaxf(sum_w, 1e-8f);
        #pragma unroll
        for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
            float w = weights[i] * inv_sum;
            if (!isnan(w) && !isinf(w) && cvalid[i]) {
                color.x += w * col[i].x;
                color.y += w * col[i].y;
                color.z += w * col[i].z;
            }
        }
    }

    // compute_candidate_weights_raw: no gradient unless some candidate is valid
    // and the largest logit is finite.
    if (!any || isinf(max_logit)) {
        return false;
    }
    pred = make_float3(0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
        weights[i] *= inv_sum;
        if (cvalid[i]) {
            pred.x += weights[i] * col[i].x;
            pred.y += weights[i] * col[i].y;
            pred.z += weights[i] * col[i].z;
        }
    }
    return true;
}

// Site loads for render_color: SAD's [N, 10] layout, or the same values padded to
// [N, 12] (three 16-byte loads instead of five 8-byte ones).
struct SiteLoad10 {
    const Site* __restrict__ s;
    __device__ __forceinline__ Site operator()(uint32_t i) const { return s[i]; }
};
struct SiteLoad12 {
    const float4* __restrict__ s;
    __device__ __forceinline__ Site operator()(uint32_t i) const {
        const float4 a = s[3 * i], b = s[3 * i + 1], c = s[3 * i + 2];
        Site site;
        site.position = make_float2(a.x, a.y);
        site.log_tau = a.z;
        site.radius = a.w;
        site.color_r = b.x;
        site.color_g = b.y;
        site.color_b = b.z;
        site.aniso_dir_x = b.w;
        site.aniso_dir_y = c.x;
        site.log_aniso = c.y;
        return site;
    }
};

// renderVoronoiKernel's color alone: render_weights without the backward's
// weights and prediction (the same expressions, so the same color).
template <class Load>
__device__ __forceinline__ float3 render_color(const uint32_t cand_ids[SAD_NUM_CANDIDATES], Load sites,
                                               uint32_t site_count, float2 uv, float inv_scale_sq) {
    float logits[SAD_NUM_CANDIDATES];
    float3 col[SAD_NUM_CANDIDATES];
    bool cvalid[SAD_NUM_CANDIDATES];
    float max_logit = -INFINITY;
    #pragma unroll
    for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
        uint32_t idx = cand_ids[i];
        cvalid[i] = false;
        col[i] = make_float3(0.0f, 0.0f, 0.0f);
        if (idx >= site_count) {
            logits[i] = -INFINITY;
            continue;
        }
        Site site = sites(idx);
        cvalid[i] = site.position.x >= 0.0f;
        col[i] = site_color(site);
        if (site.position.x < 0.0f) {
            logits[i] = -INFINITY;
            continue;
        }
        float tau = expf(site.log_tau);
        float dmix2 = voronoi_dmix2(site, uv, inv_scale_sq);
        logits[i] = -tau * dmix2;
        max_logit = fmaxf(max_logit, logits[i]);
    }
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    if (!(isinf(max_logit) && max_logit < 0.0f)) {
        float weights[SAD_NUM_CANDIDATES];
        float sum_w = 0.0f;
        #pragma unroll
        for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
            weights[i] = expf(logits[i] - max_logit);
            sum_w += weights[i];
        }
        float inv_sum = 1.0f / fmaxf(sum_w, 1e-8f);
        #pragma unroll
        for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
            float w = weights[i] * inv_sum;
            if (!isnan(w) && !isinf(w) && cvalid[i]) {
                color.x += w * col[i].x;
                color.y += w * col[i].y;
                color.z += w * col[i].z;
            }
        }
    }
    return color;
}

// The 10 gradient terms render_sad_backward_kernel adds for candidate `site`.
__device__ __forceinline__ void candidate_grad_terms(const Site& site, float w, float3 dL_dpred, float3 pred,
                                                     float2 uv, float inv_scale_sq, float* out) {
        float3 dL_dcolor = make_float3(
            w * dL_dpred.x,
            w * dL_dpred.y,
            w * dL_dpred.z);
        float3 site_col = site_color(site);
        float3 diff_col = make_float3(site_col.x - pred.x,
                                      site_col.y - pred.y,
                                      site_col.z - pred.z);
        float dL_dlogit = dL_dpred.x * (w * diff_col.x)
                        + dL_dpred.y * (w * diff_col.y)
                        + dL_dpred.z * (w * diff_col.z);

        float tau = expf(site.log_tau);
        float2 diff = make_float2(uv.x - site.position.x, uv.y - site.position.y);
        float diff2 = dot(diff, diff);
        float proj = dot(site_aniso_dir(site), diff);
        float proj2 = proj * proj;
        float perp2 = fmaxf(diff2 - proj2, 0.0f);
        float l1 = expf(site.log_aniso);
        float l2 = 1.0f / l1;

        float d2_norm = (l1 * proj2 + l2 * perp2) * inv_scale_sq;
        float d2_safe = fmaxf(d2_norm, 1e-8f);
        float inv_sqrt_d2 = rsqrtf(d2_safe);
        float inv_scale = sqrtf(inv_scale_sq);
        float r_norm = site.radius * inv_scale;
        float dmix2 = (sqrtf(d2_safe) - r_norm);

        float2 g_diff = make_float2(
            l2 * diff.x + (l1 - l2) * site.aniso_dir_x * proj,
            l2 * diff.y + (l1 - l2) * site.aniso_dir_y * proj);
        float2 dL_dpos = make_float2(
            dL_dlogit * (tau * inv_scale_sq * inv_sqrt_d2) * g_diff.x,
            dL_dlogit * (tau * inv_scale_sq * inv_sqrt_d2) * g_diff.y);

        float dL_dlog_tau = dL_dlogit * (-dmix2) * tau;
        float dL_dradius = dL_dlogit * tau * inv_scale;

        float d2_dlog_aniso = (l1 * proj2 - l2 * perp2) * inv_scale_sq;
        float dL_dlog_aniso = dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_dlog_aniso;

        float2 d2_ddir = make_float2(
            (2.0f * (l1 - l2) * proj * inv_scale_sq) * diff.x,
            (2.0f * (l1 - l2) * proj * inv_scale_sq) * diff.y);
        float2 dL_ddir = make_float2(
            dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_ddir.x,
            dL_dlogit * (-tau) * (0.5f * inv_sqrt_d2) * d2_ddir.y);

        out[0] = dL_dpos.x;
        out[1] = dL_dpos.y;
        out[2] = dL_dlog_tau;
        out[3] = dL_dradius;
        out[4] = dL_dcolor.x;
        out[5] = dL_dcolor.y;
        out[6] = dL_dcolor.z;
        out[7] = dL_ddir.x;
        out[8] = dL_ddir.y;
        out[9] = dL_dlog_aniso;
}

}  // namespace sadref

// One 16x16 block of pixels per CUDA block (as the SAD kernels): rendered
// color, clamp + MSE gradient (bit-identical to MseLossBackward +
// ClampBackward1 of render_sad's output), squared-error sum, then the SAD
// backward terms with warp-aggregated atomics into grad_sites (zeroed).
// out_color (optional) receives the rendered image.
__global__ void k_render_mse_bwd(const uint32_t* __restrict__ cand0, const uint32_t* __restrict__ cand1,
                                 const sadref::Site* __restrict__ sites, uint32_t site_count,
                                 const float* __restrict__ gt, float alpha, float inv_scale_sq,
                                 int width, int height, float* __restrict__ grad_sites,
                                 float* __restrict__ stats, float* __restrict__ out_color) {
    __shared__ float stage[256][10];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane = tid & 31;
    const bool inside = x < width && y < height;
    const int pixel_idx = y * width + x;
    const float2 uv = make_float2(float(x), float(y));

    uint32_t cand_ids[SAD_NUM_CANDIDATES];
    float3 dL_dpred = make_float3(0.f, 0.f, 0.f);
    float weights[SAD_NUM_CANDIDATES];
    float3 pred = make_float3(0.f, 0.f, 0.f);
    bool ok = false;
    float sq = 0.f;
    if (inside) {
        // SAD's candidate_index_for_pixel with candW = width, candH = height.
        const uint4 c0 = reinterpret_cast<const uint4*>(cand0)[pixel_idx];
        const uint4 c1 = reinterpret_cast<const uint4*>(cand1)[pixel_idx];
        cand_ids[0] = c0.x; cand_ids[1] = c0.y; cand_ids[2] = c0.z; cand_ids[3] = c0.w;
        cand_ids[4] = c1.x; cand_ids[5] = c1.y; cand_ids[6] = c1.z; cand_ids[7] = c1.w;
        float3 col;
        ok = sadref::render_weights(cand_ids, sites, site_count, uv, inv_scale_sq, col, weights, pred);
        if (out_color) {
            out_color[pixel_idx * 3 + 0] = col.x;
            out_color[pixel_idx * 3 + 1] = col.y;
            out_color[pixel_idx * 3 + 2] = col.z;
        }
        const float cv[3] = {col.x, col.y, col.z};
        float g[3];
        for (int c = 0; c < 3; ++c) {
            const float v = cv[c];
            const float rc = fminf(fmaxf(v, 0.f), 1.f);
            const float diff = __fsub_rn(rc, gt[pixel_idx * 3 + c]);
            sq += diff * diff;
            g[c] = (v >= 0.f && v <= 1.f) ? __fmul_rn(__fmul_rn(alpha, diff), 1.0f) : 0.f;
        }
        dL_dpred = make_float3(g[0], g[1], g[2]);
    } else {
        for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
            cand_ids[i] = 0xffffffffu;
            weights[i] = 0.f;
        }
    }
    block_atomic_add(sq, stats);

    // The slots one at a time. The two arrays shift down by one per slot, so the
    // current slot is always entry 0 and they stay in registers (an index that
    // varies in this loop would put them in local memory).
#pragma unroll 1
    for (int i = 0; i < SAD_NUM_CANDIDATES; ++i) {
        const uint32_t idx = cand_ids[0];
        const float w = weights[0];
#pragma unroll
        for (int k = 0; k < SAD_NUM_CANDIDATES - 1; ++k) {
            cand_ids[k] = cand_ids[k + 1];
            weights[k] = weights[k + 1];
        }
        uint32_t id = 0xffffffffu;
        float terms[10];
#pragma unroll
        for (int q = 0; q < 10; ++q) terms[q] = 0.f;
        if (ok && idx < site_count) {
            const sadref::Site site = sites[idx];
            if (!(site.position.x < 0.0f)) {
                sadref::candidate_grad_terms(site, w, dL_dpred, pred, uv, inv_scale_sq, terms);
                id = idx;
            }
        }
        // Lanes holding the same site in this slot are summed, one value index
        // per lane of the group (in increasing lane order, from 0), then one
        // atomic each. A site that no other lane holds adds its own terms: the
        // same values (0 + t is t, and a zero, even -0, is skipped either way),
        // without the staging.
        const unsigned mask = __match_any_sync(FULL_MASK, id);
        const int gsize = __popc(mask);
        if (gsize == 1) {
            if (id != 0xffffffffu) {
#pragma unroll
                for (int q = 0; q < 10; ++q) {
                    if (terms[q] != 0.f) atomicAdd(grad_sites + (size_t)id * 10u + q, terms[q]);
                }
            }
        } else {
#pragma unroll
            for (int q = 0; q < 10; ++q) stage[tid][q] = terms[q];
        }
        __syncwarp();
        if (id != 0xffffffffu && gsize > 1) {
            const int rank = __popc(mask & ((1u << lane) - 1u));
            const int base = tid - lane;
            for (int q = rank; q < 10; q += gsize) {
                float acc = 0.f;
                unsigned m = mask;
                while (m) {
                    const int src = __ffs(m) - 1;
                    m &= m - 1u;
                    acc += stage[base + src][q];
                }
                if (acc != 0.f) atomicAdd(grad_sites + (size_t)id * 10u + q, acc);
            }
        }
        __syncwarp();
    }
}

// Decoding: render_sad(cand0, cand1, sites, ...).clamp_(0, 1) in one kernel, for
// F frames (grid z): frame f reads sites + f * site_stride, cand* + f * W * H * 4
// and writes out + f * W * H * 3. The color is renderVoronoiKernel's; the clamp
// is torch's (a NaN stays NaN). No zero-fill: every pixel is written. PADDED:
// the sites in the [N, 12] layout (SiteArgs.sites12).
template <bool PADDED>
__global__ void k_render_clamp(const uint32_t* __restrict__ cand0, const uint32_t* __restrict__ cand1,
                               const float* __restrict__ sites, int64_t site_stride, uint32_t site_count,
                               float inv_scale_sq, int width, int height, float* __restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int64_t frame_px = static_cast<int64_t>(blockIdx.z) * width * height;
    const int64_t p = frame_px + static_cast<int64_t>(y) * width + x;
    const uint4 c0 = reinterpret_cast<const uint4*>(cand0)[p];
    const uint4 c1 = reinterpret_cast<const uint4*>(cand1)[p];
    const uint32_t ids[SAD_NUM_CANDIDATES] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    const float* fs = sites + blockIdx.z * site_stride;
    const float2 uv = make_float2(float(x), float(y));
    const float3 c = PADDED
        ? sadref::render_color(ids, sadref::SiteLoad12{reinterpret_cast<const float4*>(fs)}, site_count, uv, inv_scale_sq)
        : sadref::render_color(ids, sadref::SiteLoad10{reinterpret_cast<const sadref::Site*>(fs)}, site_count, uv,
                               inv_scale_sq);
    const float v[3] = {c.x, c.y, c.z};
#pragma unroll
    for (int k = 0; k < 3; ++k) out[p * 3 + k] = isnan(v[k]) ? v[k] : fminf(fmaxf(v[k], 0.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// VPT candidate update: one refinement pass of a frame's top-8 candidate
// buffers, the pass SADAppearance_V3._vpt_step runs with SAD's
// pack_candidate_sites + update_candidates_compact.
//
// It computes SAD's updateCandidatesKernel (backends/cuda/src/jfa.cu; MIT, see
// the note on namespace sadref) for the settings _vpt_step passes (no radius or
// Hilbert probes, the candidate grid is the frame): each pixel merges its own 8
// candidates, then those of its 4 neighbors at the pass's jump distance, then
// `inject` pseudo-random sites, into a sorted list of the 8 smallest keys.
// The output is the same bit for bit; two changes make it faster:
//   * the per-site constants of the key are computed once per pass (k_vpt_prep):
//     the half-precision packing of pack_candidate_sites, exp(log_aniso) and its
//     reciprocal, radius * sqrt(inv_sq) and max(exp(log_tau), 1e-4), from the
//     packed values, as candidatePackedDistance computes them per candidate. The
//     position and axis stay half-precision, which keeps a site at 24 bytes;
//   * the sorted insertion is a branch-free network. With c[t] = key < key[t]
//     (strict: a key equal to an entry goes after it, as in SAD's insertion),
//     entry t takes entry t-1 if c[t-1], else the new site if c[t], else stays:
//     the position SAD's branch tree finds, since the keys are sorted.
// The key keeps the operations nvcc emits for candidatePackedDistance (fma(l1,
// proj^2, l2 * perp^2); the radius term rounded on its own). Only that
// multiply-add is pinned (__fmaf_rn). The rest (|d|^2, the projection, the
// perpendicular part) is left to nvcc's contraction, as in SAD's own build. The
// output is bit-identical with nvcc 12.4 for sm_75; another compiler or
// architecture may round this kernel and SAD's differently. Check a build with
// `python tools/check_fast_ops.py --only vpt` (exit status 0: bit for bit).
// ---------------------------------------------------------------------------
// ph = (px, py, ux, uy) of the packed site as halves (px = -1: inactive, as packed),
// pb = (l1, 1 / l1, radius * sqrt(inv_sq), max(exp(log_tau), 1e-4)).
__global__ void k_vpt_prep(int N, const float* __restrict__ sites, float inv_sq, uint2* __restrict__ ph,
                           float4* __restrict__ pb) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    const float* s = sites + 10 * (size_t)j;
    __half2 pos = __floats2half2_rn(-1.f, -1.f), dir = __floats2half2_rn(0.f, 0.f);
    float lt = 0.f, rad = 0.f, la = 0.f;
    if (!(s[0] < 0.0f)) {   // packCandidateSitesKernel: position.x < 0 packs an inactive site
        pos = __halves2half2(__float2half(s[0]), __float2half(s[1]));
        dir = __halves2half2(__float2half(s[7]), __float2half(s[8]));
        lt = __half2float(__float2half(s[2]));
        rad = __half2float(__float2half(s[3]));
        la = __half2float(__float2half(s[9]));
    }
    const float l1 = expf(la);
    const float l2 = 1.0f / l1;
    const float rnorm = rad * sqrtf(inv_sq);
    const float tau = fmaxf(expf(lt), 1e-4f);
    ph[j] = make_uint2(*reinterpret_cast<const uint32_t*>(&pos), *reinterpret_cast<const uint32_t*>(&dir));
    pb[j] = make_float4(l1, l2, rnorm, tau);
}

// candidatePackedDistance from the prepared constants.
__device__ __forceinline__ float vpt_key(const uint2 h, const float4 b, float uvx, float uvy, float inv_sq) {
    const float2 p = __half22float2(*reinterpret_cast<const __half2*>(&h.x));
    const float2 u = __half22float2(*reinterpret_cast<const __half2*>(&h.y));
    if (p.x < 0.0f) return INFINITY;
    const float dx = uvx - p.x;
    const float dy = uvy - p.y;
    const float diff2 = dx * dx + dy * dy;
    const float proj = u.x * dx + u.y * dy;
    const float proj2 = proj * proj;
    const float perp2 = fmaxf(diff2 - proj2, 0.0f);
    const float d2_aniso = __fmaf_rn(b.x, proj2, __fmul_rn(b.y, perp2));
    const float d2_safe = fmaxf(d2_aniso * inv_sq, 1e-8f);
    return (sqrtf(d2_safe) - b.z) * b.w;
}

// insertClosest8Packed: skip an out-of-range index or a site already listed, else insert by key.
__device__ __forceinline__ void vpt_insert(uint32_t idx[8], float key[8], uint32_t cand, float uvx, float uvy,
                                           const uint2* __restrict__ ph, const float4* __restrict__ pb,
                                           uint32_t site_count, float inv_sq) {
    if (cand >= site_count) return;
    bool dup = false;
#pragma unroll
    for (int t = 0; t < 8; ++t) dup |= idx[t] == cand;
    if (dup) return;
    const float k = vpt_key(ph[cand], pb[cand], uvx, uvy, inv_sq);
    if (!(k < key[7])) return;
    bool c[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) c[t] = k < key[t];
#pragma unroll
    for (int t = 7; t > 0; --t) {
        key[t] = c[t - 1] ? key[t - 1] : (c[t] ? k : key[t]);
        idx[t] = c[t - 1] ? idx[t - 1] : (c[t] ? cand : idx[t]);
    }
    key[0] = c[0] ? k : key[0];
    idx[0] = c[0] ? cand : idx[0];
}

__device__ __forceinline__ uint32_t vpt_xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__global__ void __launch_bounds__(256) k_vpt_update(const uint32_t* __restrict__ in0, const uint32_t* __restrict__ in1,
                                                    uint32_t* __restrict__ out0, uint32_t* __restrict__ out1,
                                                    const uint2* __restrict__ ph, const float4* __restrict__ pb,
                                                    uint32_t site_count, uint32_t step, uint32_t step_high,
                                                    float inv_sq, uint32_t inject, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const uint32_t step_index = step & 0xFFFFu;
    uint32_t jump = step >> 16;
    if (jump == 0u) jump = 1u;
    const uint32_t full_step = (step_high << 16) | step_index;
    const float uvx = static_cast<float>(x), uvy = static_cast<float>(y);
    uint32_t idx[8];
    float key[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) { idx[t] = 0xFFFFFFFFu; key[t] = INFINITY; }
    // The pixel itself, then its neighbors at -x, +x, -y, +y (clamped to the frame).
    const int dxs[5] = {0, -1, 1, 0, 0}, dys[5] = {0, 0, 0, -1, 1};
#pragma unroll 1
    for (int s = 0; s < 5; ++s) {
        const int nx = min(max(x + dxs[s] * static_cast<int>(jump), 0), width - 1);
        const int ny = min(max(y + dys[s] * static_cast<int>(jump), 0), height - 1);
        const int n = ny * width + nx;
        const uint4 c0 = reinterpret_cast<const uint4*>(in0)[n];
        const uint4 c1 = reinterpret_cast<const uint4*>(in1)[n];
        const uint32_t cs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int k = 0; k < 8; ++k) vpt_insert(idx, key, cs[k], uvx, uvy, ph, pb, site_count, inv_sq);
    }
    // Deterministic per-pixel injection (SAD's generator and order).
    uint32_t state = (static_cast<uint32_t>(x) * 1664525u) ^ (static_cast<uint32_t>(y) * 1013904223u) ^
                     (full_step * 374761393u);
    for (uint32_t i = 0; i < inject; ++i) {
        state = vpt_xorshift32(state + i);
        vpt_insert(idx, key, state % site_count, uvx, uvy, ph, pb, site_count, inv_sq);
    }
    const int p = y * width + x;
    reinterpret_cast<uint4*>(out0)[p] = make_uint4(idx[0], idx[1], idx[2], idx[3]);
    reinterpret_cast<uint4*>(out1)[p] = make_uint4(idx[4], idx[5], idx[6], idx[7]);
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------
static AnchorArgs make_anchor_args(int64_t mode, const torch::Tensor& mu, const torch::Tensor& log_sigma,
                                   const torch::Tensor& radius, const torch::Tensor& log_aspect,
                                   const torch::Tensor& adir, const torch::Tensor& weight) {
    AnchorArgs a;
    a.mu = fptr(mu);
    a.log_sigma = fptr(log_sigma);
    a.radius = fptr(radius);
    a.log_aspect = fptr(log_aspect);
    a.adir = fptr(adir);
    a.weight = fptr(weight);
    a.mode = static_cast<int>(mode);
    return a;
}

static SiteArgs make_site_args(const std::vector<torch::Tensor>& site, double W, double H,
                               const torch::Tensor& sites_out, const torch::Tensor& y_out,
                               const torch::Tensor& gy_arap) {
    TORCH_CHECK(site.size() == 6, "site params: [P, F, log_tau, radius, dir, log_aniso]");
    for (auto& t : site) CHECK_CUDA_F32(t);
    SiteArgs a;
    a.P = fptr(site[0]); a.F = fptr(site[1]); a.LT = fptr(site[2]);
    a.R = fptr(site[3]); a.D = fptr(site[4]); a.LA = fptr(site[5]);
    a.W = static_cast<float>(W);
    a.H = static_cast<float>(H);
    a.sites = fptr(sites_out);
    a.y = fptr(y_out);
    a.gy_arap = fptr(gy_arap);
    a.sites12 = nullptr;
    return a;
}

static CoefStack make_stack(const std::vector<torch::Tensor>& p, const std::vector<torch::Tensor>& g, int col) {
    CoefStack s;
    s.t = fptr(p[0]); s.lr = fptr(p[1]); s.v = fptr(p[2]);
    s.gt = g.empty() ? nullptr : fptr(g[0]);
    s.glr = g.empty() ? nullptr : fptr(g[1]);
    s.gv = g.empty() ? nullptr : fptr(g[2]);
    s.D = static_cast<int>(p[2].size(2));
    s.col = col;
    return s;
}

void temporal_forward(double t, std::vector<torch::Tensor> stack0, std::vector<torch::Tensor> stack1,
                      torch::Tensor lin, bool smooth, torch::Tensor table, torch::Tensor stats) {
    const int M = stack0[0].size(0), K = stack0[0].size(1), C = table.size(1);
    TORCH_CHECK(K <= MAX_KNOTS, "num_knots > ", MAX_KNOTS);
    CoefStack s0 = make_stack(stack0, {}, 0);
    CoefStack s1 = stack1.empty() ? s0 : make_stack(stack1, {}, s0.D);
    const int ns = stack1.empty() ? 1 : 2;
    k_temporal_fwd<<<grid_for((int64_t)M * 32, 128), 128, 0, stream()>>>(M, K, C, static_cast<float>(t), s0, s1, ns,
                                                         fptr(lin), smooth ? 1 : 0, fptr(table), fptr(stats));
}

constexpr int ROM_FWD_BLOCK = 128;
// k_rom_fwd's block: 8 warps share one copy of the anchor table in shared memory, the occupancy limit
// at the larger anchor counts (the result does not depend on it: each warp is 32 consecutive sites).
constexpr int ROM_FWD_THREADS = 256;

static size_t rom_fwd_smem(int64_t M) {
    return sizeof(float) * (size_t)M * TAB;
}

template <int MODE, int K>
static void launch_rom_fwd(int N, int M, int C, AnchorArgs an, const float* coef, const int32_t* warm,
                           float* d, int32_t* topi, float* phi, float* tab, int pack, SiteArgs sa, float* stats) {
    k_rom_fwd<MODE, K><<<grid_for(N, ROM_FWD_THREADS), ROM_FWD_THREADS, rom_fwd_smem(M), stream()>>>(
        N, M, C, an, coef, warm, d, topi, phi, tab, pack, sa, stats);
}

template <int MODE>
static void dispatch_rom_fwd_k(int K, int N, int M, int C, AnchorArgs an, const float* coef, const int32_t* warm,
                               float* d, int32_t* topi, float* phi, float* tab, int pack, SiteArgs sa, float* stats) {
    switch (K) {
        case 8: launch_rom_fwd<MODE, 8>(N, M, C, an, coef, warm, d, topi, phi, tab, pack, sa, stats); break;
        case 12: launch_rom_fwd<MODE, 12>(N, M, C, an, coef, warm, d, topi, phi, tab, pack, sa, stats); break;
        case 16: launch_rom_fwd<MODE, 16>(N, M, C, an, coef, warm, d, topi, phi, tab, pack, sa, stats); break;
        default: TORCH_CHECK(false, "unsupported anchor top-K ", K);
    }
}

bool rom_supported(int64_t M, int64_t K, int64_t C) {
    const bool k_ok = K == 8 || K == 12 || K == 16;
    return k_ok && K < M && M <= 1024 && C <= 8 && rom_fwd_smem(M) <= 48 * 1024;
}

// ROM forward. pack: also write sites/y (fine-stage site packing) and zero gy_arap.
void rom_forward(torch::Tensor P, int64_t mode, int64_t K, torch::Tensor mu, torch::Tensor log_sigma,
                 torch::Tensor radius, torch::Tensor log_aspect, torch::Tensor adir, torch::Tensor weight,
                 torch::Tensor coef, bool warm_start, torch::Tensor d, torch::Tensor topi, torch::Tensor phi,
                 torch::Tensor tab,
                 bool pack, std::vector<torch::Tensor> site, double W, double H, torch::Tensor sites_out,
                 torch::Tensor y_out, torch::Tensor gy_arap, torch::Tensor stats) {
    CHECK_CUDA_F32(P); CHECK_CUDA_F32(coef); CHECK_CUDA_F32(d); CHECK_CUDA_F32(phi);
    const int N = P.size(0), M = mu.size(0), C = coef.size(1);
    TORCH_CHECK(rom_supported(M, K, C), "rom_forward: unsupported M/K/C");
    AnchorArgs an = make_anchor_args(mode, mu, log_sigma, radius, log_aspect, adir, weight);
    SiteArgs sa;
    if (pack) {
        sa = make_site_args(site, W, H, sites_out, y_out, gy_arap);
    } else {
        sa = SiteArgs{};
        sa.P = fptr(P);
    }
    sa.P = fptr(P);
    float* st = fptr(stats);
    TORCH_CHECK(topi.scalar_type() == torch::kInt32 && topi.is_contiguous(), "topi must be contiguous int32");
    // warm_start: topi holds the previous selection; each thread reads its row, then overwrites it.
    const int32_t* warm = warm_start ? topi.data_ptr<int32_t>() : nullptr;
    switch (mode) {
        case MODE_VORONOI: dispatch_rom_fwd_k<MODE_VORONOI>(K, N, M, C, an, fptr(coef), warm, fptr(d), topi.data_ptr<int32_t>(), fptr(phi), fptr(tab), pack, sa, st); break;
        case MODE_APOLLONIUS: dispatch_rom_fwd_k<MODE_APOLLONIUS>(K, N, M, C, an, fptr(coef), warm, fptr(d), topi.data_ptr<int32_t>(), fptr(phi), fptr(tab), pack, sa, st); break;
        case MODE_POWER: dispatch_rom_fwd_k<MODE_POWER>(K, N, M, C, an, fptr(coef), warm, fptr(d), topi.data_ptr<int32_t>(), fptr(phi), fptr(tab), pack, sa, st); break;
        case MODE_ANISO: dispatch_rom_fwd_k<MODE_ANISO>(K, N, M, C, an, fptr(coef), warm, fptr(d), topi.data_ptr<int32_t>(), fptr(phi), fptr(tab), pack, sa, st); break;
        default: TORCH_CHECK(false, "bad mode");
    }
}

static void launch_rom_apply(int64_t K, int N, int C, int F, const float* coef, int64_t coef_stride,
                             const int32_t* topi, const float* phi, float* d, bool pack, const SiteArgs& sa,
                             int64_t site_stride) {
    const dim3 g(grid_for(N, ROM_FWD_BLOCK), F);
    switch (K) {
        case 8: k_rom_apply<8><<<g, ROM_FWD_BLOCK, 0, stream()>>>(N, C, coef, topi, phi, d, pack, sa, coef_stride, site_stride); break;
        case 12: k_rom_apply<12><<<g, ROM_FWD_BLOCK, 0, stream()>>>(N, C, coef, topi, phi, d, pack, sa, coef_stride, site_stride); break;
        case 16: k_rom_apply<16><<<g, ROM_FWD_BLOCK, 0, stream()>>>(N, C, coef, topi, phi, d, pack, sa, coef_stride, site_stride); break;
        default: TORCH_CHECK(false, "unsupported anchor top-K ", K);
    }
}

// ROM apply (decoding): d and the packed sites from a stored selection (topi, phi) of rom_forward.
void rom_apply(torch::Tensor P, int64_t K, torch::Tensor coef, torch::Tensor topi, torch::Tensor phi,
               torch::Tensor d, bool pack, std::vector<torch::Tensor> site, double W, double H,
               torch::Tensor sites_out, torch::Tensor y_out) {
    CHECK_CUDA_F32(P); CHECK_CUDA_F32(coef); CHECK_CUDA_F32(d); CHECK_CUDA_F32(phi);
    TORCH_CHECK(topi.scalar_type() == torch::kInt32 && topi.is_contiguous(), "topi must be contiguous int32");
    const int N = P.size(0), C = coef.size(1);
    TORCH_CHECK(C <= 8, "rom_apply: C > 8");
    TORCH_CHECK(topi.size(0) == N && topi.size(1) == K && phi.size(0) == N && phi.size(1) == K,
                "rom_apply: topi/phi must be [N, K]");
    TORCH_CHECK(d.size(0) == N && d.size(1) == C, "rom_apply: d must be [N, C]");
    SiteArgs sa;
    if (pack) {
        sa = make_site_args(site, W, H, sites_out, y_out, torch::Tensor());
    } else {
        sa = SiteArgs{};
    }
    sa.P = fptr(P);
    launch_rom_apply(K, N, C, 1, fptr(coef), 0, topi.data_ptr<int32_t>(), fptr(phi), fptr(d), pack, sa, 0);
}

// Decoding a GoP pass: the packed sites of F frames ([F, N, 10]) from their coefficient tables
// ([F, M, C], temporal_gop) and one anchor basis, in one launch; frame by frame, rom_apply.
// sites12_out (optional, [F, N, 12]): the same sites padded for render_decode.
void rom_apply_gop(torch::Tensor P, int64_t K, torch::Tensor tables, torch::Tensor topi, torch::Tensor phi,
                   std::vector<torch::Tensor> site, double W, double H, torch::Tensor sites_out,
                   torch::Tensor sites12_out) {
    CHECK_CUDA_F32(P); CHECK_CUDA_F32(tables); CHECK_CUDA_F32(phi); CHECK_CUDA_F32(sites_out);
    TORCH_CHECK(topi.scalar_type() == torch::kInt32 && topi.is_contiguous(), "topi must be contiguous int32");
    TORCH_CHECK(tables.dim() == 3 && sites_out.dim() == 3 && sites_out.size(0) == tables.size(0),
                "rom_apply_gop: tables [F, M, C], sites_out [F, N, 10]");
    const int F = tables.size(0), N = P.size(0), C = tables.size(2);
    TORCH_CHECK(C <= 8, "rom_apply_gop: C > 8");
    TORCH_CHECK(topi.size(0) == N && topi.size(1) == K && phi.size(0) == N && phi.size(1) == K,
                "rom_apply_gop: topi/phi must be [N, K]");
    TORCH_CHECK(sites_out.size(1) == N && sites_out.size(2) == 10, "rom_apply_gop: sites_out [F, N, 10]");
    if (F == 0 || N == 0) return;
    SiteArgs sa = make_site_args(site, W, H, sites_out, torch::Tensor(), torch::Tensor());
    sa.P = fptr(P);
    if (sites12_out.defined() && sites12_out.numel() > 0) {
        CHECK_CUDA_F32(sites12_out);
        TORCH_CHECK(sites12_out.numel() == static_cast<int64_t>(F) * N * 12, "rom_apply_gop: sites12_out [F, N, 12]");
        sa.sites12 = fptr(sites12_out);
    }
    launch_rom_apply(K, N, C, F, fptr(tables), tables.stride(0), topi.data_ptr<int32_t>(), fptr(phi), nullptr,
                     true, sa, static_cast<int64_t>(N) * 10);
}

// Decoding a GoP pass: the coefficient tables of its F frames ([F, M, C]), frame f at
// t = (f + 1) / F as FusedStep.decode computes it; frame by frame, temporal_forward.
void temporal_gop(std::vector<torch::Tensor> stack0, std::vector<torch::Tensor> stack1, torch::Tensor tables) {
    CHECK_CUDA_F32(tables);
    TORCH_CHECK(tables.dim() == 3, "temporal_gop: tables [F, M, C]");
    const int F = tables.size(0), M = stack0[0].size(0), K = stack0[0].size(1), C = tables.size(2);
    TORCH_CHECK(K <= MAX_KNOTS, "num_knots > ", MAX_KNOTS);
    TORCH_CHECK(tables.size(1) == M, "temporal_gop: tables [F, M, C]");
    if (F == 0) return;
    CoefStack s0 = make_stack(stack0, {}, 0);
    CoefStack s1 = stack1.empty() ? s0 : make_stack(stack1, {}, s0.D);
    const int ns = stack1.empty() ? 1 : 2;
    const dim3 g(grid_for((int64_t)M * 32, 128), F);
    k_temporal_gop<<<g, 128, 0, stream()>>>(M, K, C, F, s0, s1, ns, fptr(tables));
}

void sites_forward(std::vector<torch::Tensor> site, double W, double H, torch::Tensor sites_out, torch::Tensor y_out) {
    SiteArgs a = make_site_args(site, W, H, sites_out, y_out, torch::Tensor());
    const int N = site[0].size(0);
    k_sites_fwd<<<grid_for(N, 256), 256, 0, stream()>>>(N, a);
}

void arap_fb(torch::Tensor P, torch::Tensor y, torch::Tensor knn, double coef, torch::Tensor gy, torch::Tensor stats) {
    CHECK_CUDA_F32(P); CHECK_CUDA_F32(y); CHECK_CUDA_F32(gy);
    TORCH_CHECK(knn.scalar_type() == torch::kInt64 && knn.is_contiguous(), "knn must be contiguous int64");
    const int N = P.size(0), KN = knn.size(1);
    k_arap<<<grid_for(N, 256), 256, 0, stream()>>>(N, KN, fptr(P), fptr(y), knn.data_ptr<int64_t>(),
                                                    static_cast<float>(coef), fptr(gy), fptr(stats));
}

void mse_fb(torch::Tensor rendered, torch::Tensor gt, double alpha, torch::Tensor grad, torch::Tensor stats) {
    CHECK_CUDA_F32(rendered); CHECK_CUDA_F32(gt); CHECK_CUDA_F32(grad);
    TORCH_CHECK(rendered.numel() == gt.numel() && grad.numel() == gt.numel(), "mse_fb: size mismatch");
    const int64_t n = rendered.numel();
    const int block = 256;
    const int grid = static_cast<int>(std::min<int64_t>(grid_for(n, block), 68 * 16));
    k_mse<<<grid, block, 0, stream()>>>(n, fptr(rendered), fptr(gt), static_cast<float>(alpha), fptr(grad), fptr(stats));
}

static SiteGrads make_site_grads(const std::vector<torch::Tensor>& g) {
    TORCH_CHECK(g.size() == 6, "site grads: 6 tensors");
    for (auto& t : g) CHECK_CUDA_F32(t);
    return SiteGrads{fptr(g[0]), fptr(g[1]), fptr(g[2]), fptr(g[3]), fptr(g[4]), fptr(g[5])};
}

void sites_backward(std::vector<torch::Tensor> site, double W, double H, torch::Tensor gs, std::vector<torch::Tensor> grads) {
    CHECK_CUDA_F32(gs);
    SiteArgs a = make_site_args(site, W, H, torch::Tensor(), torch::Tensor(), torch::Tensor());
    SiteGrads g = make_site_grads(grads);
    const int N = site[0].size(0);
    k_sites_bwd<<<grid_for(N, 256), 256, 0, stream()>>>(N, a, fptr(gs), g);
}

template <int MODE, int K>
static void launch_rom_bwd(int N, int M, int C, SiteArgs a, const float* gs, SiteGrads g, const float* coef,
                           const float* tab, const float* d, const int32_t* topi, const float* phi,
                           float move_coef, float* gacc, float* gcoef) {
    const int block = 128;
    const size_t sm = sizeof(float) * (size_t)M * (TAB + C);
    if (sm <= 32 * 1024) {
        k_rom_bwd<MODE, K, true><<<grid_for(N, block), block, sm, stream()>>>(
            N, M, C, a, gs, g, coef, tab, d, topi, phi, move_coef, gacc, gcoef);
    } else {
        k_rom_bwd<MODE, K, false><<<grid_for(N, block), block, 0, stream()>>>(
            N, M, C, a, gs, g, coef, tab, d, topi, phi, move_coef, gacc, gcoef);
    }
}

template <int MODE>
static void dispatch_rom_bwd_k(int K, int N, int M, int C, SiteArgs a, const float* gs, SiteGrads g,
                               const float* coef, const float* tab, const float* d, const int32_t* topi,
                               const float* phi, float move_coef, float* gacc, float* gcoef) {
    switch (K) {
        case 8: launch_rom_bwd<MODE, 8>(N, M, C, a, gs, g, coef, tab, d, topi, phi, move_coef, gacc, gcoef); break;
        case 12: launch_rom_bwd<MODE, 12>(N, M, C, a, gs, g, coef, tab, d, topi, phi, move_coef, gacc, gcoef); break;
        case 16: launch_rom_bwd<MODE, 16>(N, M, C, a, gs, g, coef, tab, d, topi, phi, move_coef, gacc, gcoef); break;
        default: TORCH_CHECK(false, "unsupported anchor top-K ", K);
    }
}

// Fine-stage site + ROM backward. gy_arap: dL/dy from ARAP (or undefined).
void rom_backward(std::vector<torch::Tensor> site, double W, double H, torch::Tensor gs, std::vector<torch::Tensor> grads,
                  int64_t mode, int64_t K, torch::Tensor coef, torch::Tensor tab, torch::Tensor d, torch::Tensor topi,
                  torch::Tensor phi, torch::Tensor y, torch::Tensor gy_arap, double move_coef,
                  torch::Tensor gacc, torch::Tensor gcoef) {
    CHECK_CUDA_F32(gs); CHECK_CUDA_F32(coef); CHECK_CUDA_F32(tab); CHECK_CUDA_F32(d); CHECK_CUDA_F32(phi);
    CHECK_CUDA_F32(y); CHECK_CUDA_F32(gacc); CHECK_CUDA_F32(gcoef);
    SiteArgs a = make_site_args(site, W, H, torch::Tensor(), y, gy_arap);
    SiteGrads g = make_site_grads(grads);
    const int N = site[0].size(0), M = tab.size(0), C = coef.size(1);
    const float mc = static_cast<float>(move_coef);
    switch (mode) {
        case MODE_VORONOI: dispatch_rom_bwd_k<MODE_VORONOI>(K, N, M, C, a, fptr(gs), g, fptr(coef), fptr(tab), fptr(d), topi.data_ptr<int32_t>(), fptr(phi), mc, fptr(gacc), fptr(gcoef)); break;
        case MODE_APOLLONIUS: dispatch_rom_bwd_k<MODE_APOLLONIUS>(K, N, M, C, a, fptr(gs), g, fptr(coef), fptr(tab), fptr(d), topi.data_ptr<int32_t>(), fptr(phi), mc, fptr(gacc), fptr(gcoef)); break;
        case MODE_POWER: dispatch_rom_bwd_k<MODE_POWER>(K, N, M, C, a, fptr(gs), g, fptr(coef), fptr(tab), fptr(d), topi.data_ptr<int32_t>(), fptr(phi), mc, fptr(gacc), fptr(gcoef)); break;
        case MODE_ANISO: dispatch_rom_bwd_k<MODE_ANISO>(K, N, M, C, a, fptr(gs), g, fptr(coef), fptr(tab), fptr(d), topi.data_ptr<int32_t>(), fptr(phi), mc, fptr(gacc), fptr(gcoef)); break;
        default: TORCH_CHECK(false, "bad mode");
    }
}

void anchor_backward(double t, std::vector<torch::Tensor> stack0, std::vector<torch::Tensor> gstack0,
                     std::vector<torch::Tensor> stack1, std::vector<torch::Tensor> gstack1,
                     torch::Tensor lin, double smooth_coef, int64_t mode, torch::Tensor mu, torch::Tensor log_sigma,
                     torch::Tensor radius, torch::Tensor log_aspect, torch::Tensor adir, torch::Tensor weight,
                     std::vector<torch::Tensor> ganchor, torch::Tensor tab, torch::Tensor gacc, torch::Tensor gcoef) {
    const int M = stack0[0].size(0), K = stack0[0].size(1), C = gcoef.size(1);
    TORCH_CHECK(K <= MAX_KNOTS, "num_knots > ", MAX_KNOTS);
    CoefStack s0 = make_stack(stack0, gstack0, 0);
    CoefStack s1 = stack1.empty() ? s0 : make_stack(stack1, gstack1, s0.D);
    const int ns = stack1.empty() ? 1 : 2;
    AnchorArgs an = make_anchor_args(mode, mu, log_sigma, radius, log_aspect, adir, weight);
    TORCH_CHECK(ganchor.size() == 6, "anchor grads: [mu, log_sigma, radius, log_aspect, dir, weight]");
    AnchorGrads ag{fptr(ganchor[0]), fptr(ganchor[1]), fptr(ganchor[2]), fptr(ganchor[3]), fptr(ganchor[4]), fptr(ganchor[5])};
    k_anchor_bwd<<<grid_for((int64_t)M * 32, 128), 128, 0, stream()>>>(M, K, C, static_cast<float>(t), s0, s1, ns, fptr(lin),
                                                       static_cast<float>(smooth_coef), an, ag, fptr(tab),
                                                       fptr(gacc), fptr(gcoef));
}

void adan_step(std::vector<torch::Tensor> p, std::vector<torch::Tensor> g, std::vector<torch::Tensor> m,
               std::vector<torch::Tensor> v, std::vector<torch::Tensor> dif, std::vector<torch::Tensor> npg,
               std::vector<double> scal, bool init) {
    // scal per tensor: b1, b2, b3, 1-b1, 1-b2, 1-b3, float32(1/sqrt(bc3)) (reciprocal in double), eps, -step_size, -step_size_diff
    const int cnt = p.size();
    TORCH_CHECK(cnt <= MAX_OPT_TENSORS && (int)scal.size() == 10 * cnt, "adan_step: bad args");
    AdanBatch b;
    b.count = cnt;
    b.start[0] = 0;
    for (int i = 0; i < cnt; ++i) {
        AdanTensor& T = b.t[i];
        T.p = fptr(p[i]); T.g = fptr(g[i]); T.m = fptr(m[i]); T.v = fptr(v[i]);
        T.dif = fptr(dif[i]); T.npg = fptr(npg[i]); T.n = p[i].numel();
        const double* s = scal.data() + 10 * i;
        T.b1 = s[0]; T.b2 = s[1]; T.b3 = s[2]; T.omb1 = s[3]; T.omb2 = s[4]; T.omb3 = s[5];
        T.inv_bc3s = s[6]; T.eps = s[7]; T.nss = s[8]; T.nssd = s[9];
        T.init = init ? 1 : 0;
        b.start[i + 1] = b.start[i] + T.n;
    }
    const int block = 256;
    const int grid = static_cast<int>(std::min<int64_t>(grid_for(b.start[cnt], block), 68 * 8));
    k_adan<<<grid, block, 0, stream()>>>(b);
}

void adam_step(std::vector<torch::Tensor> p, std::vector<torch::Tensor> g, std::vector<torch::Tensor> m,
               std::vector<torch::Tensor> v, std::vector<double> scal) {
    // scal per tensor: 1-b1, b2, 1-b2, sqrt(bc2), eps, -(lr/bc1)
    const int cnt = p.size();
    TORCH_CHECK(cnt <= MAX_OPT_TENSORS && (int)scal.size() == 6 * cnt, "adam_step: bad args");
    AdamBatch b;
    b.count = cnt;
    b.start[0] = 0;
    for (int i = 0; i < cnt; ++i) {
        AdamTensor& T = b.t[i];
        T.p = fptr(p[i]); T.g = fptr(g[i]); T.m = fptr(m[i]); T.v = fptr(v[i]); T.n = p[i].numel();
        const double* s = scal.data() + 6 * i;
        T.w = s[0]; T.b2 = s[1]; T.omb2 = s[2]; T.bc2s = s[3]; T.eps = s[4]; T.ss = s[5];
        b.start[i + 1] = b.start[i] + T.n;
    }
    const int block = 256;
    const int grid = static_cast<int>(std::min<int64_t>(grid_for(b.start[cnt], block), 68 * 8));
    k_adam<<<grid, block, 0, stream()>>>(b);
}

// Fused render + clamp/MSE + render backward. Zeroes and fills grad_sites
// [N,10]; adds the squared-error sum to stats[0]. color_out (optional, [H,W,3])
// receives the rendered image.
void render_mse_backward(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites, torch::Tensor gt,
                         double alpha, double inv_scale_sq, int64_t width, int64_t height,
                         torch::Tensor grad_sites, torch::Tensor stats, torch::Tensor color_out) {
    TORCH_CHECK(cand0.scalar_type() == torch::kInt32 && cand1.scalar_type() == torch::kInt32, "candidates must be int32");
    TORCH_CHECK(cand0.is_contiguous() && cand1.is_contiguous(), "candidates must be contiguous");
    TORCH_CHECK(cand0.size(0) == width * height && cand1.size(0) == width * height, "candidate size");
    CHECK_CUDA_F32(sites); CHECK_CUDA_F32(gt); CHECK_CUDA_F32(grad_sites); CHECK_CUDA_F32(stats);
    TORCH_CHECK(sites.size(1) == 10 && grad_sites.sizes() == sites.sizes(), "sites / grad_sites must be [N,10]");
    TORCH_CHECK(gt.numel() == width * height * 3, "gt must be [H,W,3]");
    grad_sites.zero_();
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    k_render_mse_bwd<<<grid, block, 0, stream()>>>(
        reinterpret_cast<const uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(cand1.data_ptr<int32_t>()),
        reinterpret_cast<const sadref::Site*>(sites.data_ptr<float>()), static_cast<uint32_t>(sites.size(0)),
        fptr(gt), static_cast<float>(alpha), static_cast<float>(inv_scale_sq),
        static_cast<int>(width), static_cast<int>(height), fptr(grad_sites), fptr(stats), fptr(color_out));
}

void stats_step(torch::Tensor stats, torch::Tensor scale, torch::Tensor lam, torch::Tensor cur, torch::Tensor running) {
    k_stats<<<1, 32, 0, stream()>>>(fptr(stats), fptr(scale), fptr(lam), fptr(cur), running.data_ptr<double>());
}

// Decoding: out = render_sad(cand0, cand1, sites, inv_sq, W, H).clamp_(0, 1), bit for bit.
// sites [N, 10] with cand* [W*H, 4] and out [H, W, 3], or F frames at once: sites [F, N, 10],
// cand* [F, W*H, 4], out [F, H, W, 3]. Sites padded to 12 numbers (rom_apply_gop's sites12)
// render faster.
void render_decode(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites, double inv_scale_sq,
                   int64_t width, int64_t height, torch::Tensor out) {
    CHECK_CUDA_F32(sites); CHECK_CUDA_F32(out);
    const bool batch = sites.dim() == 3;
    const int64_t F = batch ? sites.size(0) : 1;
    const int64_t N = sites.size(sites.dim() - 2);
    const int64_t S = sites.size(sites.dim() - 1);
    TORCH_CHECK(S == 10 || S == 12, "sites must be [(F,) N, 10], or padded to [(F,) N, 12]");
    for (const auto* c : {&cand0, &cand1}) {
        TORCH_CHECK(c->is_cuda() && c->scalar_type() == torch::kInt32 && c->is_contiguous() &&
                    c->numel() == F * width * height * 4, "candidates must be contiguous int32 [(F,) W*H, 4]");
    }
    TORCH_CHECK(out.numel() == F * width * height * 3, "out must be [(F,) H, W, 3]");
    // The loads' alignment: float4 for padded sites, Site (a float2 first) for packed ones.
    const uintptr_t align = S == 12 ? 16 : 8;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(sites.data_ptr<float>()) % align == 0,
                S == 12 ? "padded sites [(F,) N, 12] must be 16-byte aligned (read as float4)"
                        : "sites [(F,) N, 10] must be 8-byte aligned (read as Site)");
    if (F == 0) return;
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16, F);
    auto kernel = S == 12 ? k_render_clamp<true> : k_render_clamp<false>;
    kernel<<<grid, block, 0, stream()>>>(
        reinterpret_cast<const uint32_t*>(cand0.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(cand1.data_ptr<int32_t>()),
        sites.data_ptr<float>(), N * S, static_cast<uint32_t>(N),
        static_cast<float>(inv_scale_sq), static_cast<int>(width), static_cast<int>(height), fptr(out));
}

// VPT pass, step 1: the per-site key constants of packed sites [N, 10]: ph int32 [N, 2]
// (position and axis as halves), pb float32 [N, 4].
void vpt_prep(torch::Tensor sites, double inv_sq, torch::Tensor ph, torch::Tensor pb) {
    CHECK_CUDA_F32(sites); CHECK_CUDA_F32(pb);
    const int N = sites.size(0);
    TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
    TORCH_CHECK(ph.is_cuda() && ph.scalar_type() == torch::kInt32 && ph.is_contiguous() && ph.numel() == 2 * (int64_t)N,
                "ph must be a contiguous int32 CUDA [N,2]");
    TORCH_CHECK(pb.numel() == 4 * (int64_t)N, "pb must be [N,4]");
    if (N == 0) return;
    k_vpt_prep<<<grid_for(N, 256), 256, 0, stream()>>>(N, fptr(sites), static_cast<float>(inv_sq),
                                                       reinterpret_cast<uint2*>(ph.data_ptr<int32_t>()),
                                                       reinterpret_cast<float4*>(fptr(pb)));
}

// VPT pass, step 2: update_candidates_compact(cand0_in, cand1_in, cand0_out, cand1_out, packed, ...,
// inv_sq, N, step, step_high, radius_scale, radius_probes = 0, inject, hilbert_probes = 0,
// hilbert_window = 0, cand_downscale = 1, W, H, W, H), with ph / pb of vpt_prep.
void vpt_update(torch::Tensor cand0_in, torch::Tensor cand1_in, torch::Tensor cand0_out, torch::Tensor cand1_out,
                torch::Tensor ph, torch::Tensor pb, double inv_sq, int64_t step, int64_t step_high, int64_t inject,
                int64_t width, int64_t height) {
    for (const auto* t : {&cand0_in, &cand1_in, &cand0_out, &cand1_out}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kInt32 && t->is_contiguous() &&
                    t->dim() == 2 && t->size(0) == width * height && t->size(1) == 4,
                    "candidate buffers must be contiguous int32 CUDA [W*H, 4]");
    }
    CHECK_CUDA_F32(pb);
    TORCH_CHECK(ph.is_cuda() && ph.scalar_type() == torch::kInt32 && ph.is_contiguous(), "ph must be contiguous int32");
    const int64_t N = pb.size(0);
    TORCH_CHECK(N > 0 && ph.size(0) == N, "vpt_update: no sites");
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    k_vpt_update<<<grid, block, 0, stream()>>>(
        reinterpret_cast<const uint32_t*>(cand0_in.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(cand1_in.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand0_out.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(cand1_out.data_ptr<int32_t>()),
        reinterpret_cast<const uint2*>(ph.data_ptr<int32_t>()), reinterpret_cast<const float4*>(fptr(pb)),
        static_cast<uint32_t>(N), static_cast<uint32_t>(step), static_cast<uint32_t>(step_high),
        static_cast<float>(inv_sq), static_cast<uint32_t>(inject), static_cast<int>(width), static_cast<int>(height));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rom_supported", &rom_supported);
    m.def("temporal_forward", &temporal_forward);
    m.def("rom_forward", &rom_forward);
    m.def("rom_apply", &rom_apply);
    m.def("rom_apply_gop", &rom_apply_gop);
    m.def("temporal_gop", &temporal_gop);
    m.def("sites_forward", &sites_forward);
    m.def("arap_fb", &arap_fb);
    m.def("mse_fb", &mse_fb);
    m.def("sites_backward", &sites_backward);
    m.def("rom_backward", &rom_backward);
    m.def("anchor_backward", &anchor_backward);
    m.def("adan_step", &adan_step);
    m.def("adam_step", &adam_step);
    m.def("stats_step", &stats_step);
    m.def("render_mse_backward", &render_mse_backward);
    m.def("render_decode", &render_decode);
    m.def("vpt_prep", &vpt_prep);
    m.def("vpt_update", &vpt_update);
}
