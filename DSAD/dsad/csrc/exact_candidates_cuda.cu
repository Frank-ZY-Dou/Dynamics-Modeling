// Exact top-8 SAD candidate builder: the per-pixel candidate buffers of the SAD
// renderer, built directly from the sites.
//
// For every pixel x the SAD renderer blends the 8 sites with the smallest key
//
//     key_i(x) = max(exp(log_tau_i), 1e-4) *
//                ( sqrt(max(d2a_i(x) * inv_sq, 1e-8)) - radius_i * sqrt(inv_sq) ),
//     d2a_i(x) = e^{a_i} proj^2 + e^{-a_i} max(|x - p_i|^2 - proj^2, 0),
//     proj     = (x - p_i) . u_i,
//
// with x = (float(px), float(py)): integer pixel coordinates, no +0.5, as SAD's
// updateCandidatesKernel and renderVoronoiKernel form `uv`.
//
// The minimizers can be far away (thin anisotropic sites with low tau reach
// hundreds of pixels along their axis), so a fixed search radius is not exact.
// "exact" mode bins every site into the tiles it can reach (one CUDA block per
// TS x TS tile, one thread per pixel, each warp a strip of 32/TS rows):
//
//   1. prep      per-site constants; counting sort of the sites into a cell grid.
//   2. pass A    per pixel, top-8 keys over a small cell window. U_w, the max of
//                the 8th key over a warp strip, bounds the true 8th key of each
//                of its pixels; U_T = max_w U_w for the tile.
//   3. cap       Kcap = a high quantile of U_T (radix select); every bound is
//                capped at Kcap (U1 = min(U, Kcap)). A few windows see only
//                steep nearby sites and give useless bounds; the cap keeps them
//                from inflating step 4.
//   4. footprint site j can beat a bound U1 at some pixel of a box only if the
//                ellipse {x : Q_j(x - p_j) <= rho^2}, rho = U1 / (tau_j s) + r_j,
//                meets the box (Q_j is a quadratic lower bound of d2a_j). The
//                tile rows that each site's ellipse spans at the largest bound
//                become compacted work items; one thread per item walks the
//                row's x-range of the ellipse and tests each tile exactly
//                against U1_T (a count pass, an offset scan, an emit pass).
//   5. pass C    per pixel, top-8 over its tile's list. Each entry is tested
//                against every warp strip at the strip's own U1_w, and a warp
//                takes only the entries that can reach it. If the pixel's 8th
//                key is <= U1_w, every site not offered has a larger key: the
//                pixel is certified exact. Uncapped strips always certify
//                (pass A's 8 sites are offered). A pixel that fails (true 8th
//                key above Kcap) rescans all sites, and so does every pixel of
//                a tile whose list did not fit the preallocated capacity. The
//                rescan is cooperative by default (coop_rescan): a whole warp
//                takes each failed pixel, and its 32 lanes split the sites and
//                merge their top-8 lists in the same (key, index) order. A tile
//                with many failed pixels uses the per-thread rescan, which
//                coop_fallback = false uses for every tile.
//
// The result equals a brute-force scan over all sites (ties broken by index).
// F frames are built per call (grid y / z = frame). The build is a fixed
// sequence of kernel launches, without host synchronization.
// "window" mode scans only the cells within radius R, which is not exact.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

constexpr unsigned kEmpty = 0xFFFFFFFFu;    // SAD's "no candidate" marker
constexpr int kMaxWindowRows = 512;         // cell rows a window may span
constexpr float kRhoUnbounded = 1e7f;       // footprint radius treated as "whole image"

// stats[] slots (int32, device, per frame)
constexpr int kStatEntries = 0;     // total (site, tile) list entries
constexpr int kStatKmax = 1;        // ordered bits of max_T U_T (pass A)
constexpr int kStatKcap = 2;        // ordered bits of Kcap
constexpr int kStatCapped = 3;      // tiles with U_T > Kcap
constexpr int kStatFallback = 4;    // pixels rescanned against all sites
constexpr int kStatOverflow = 5;    // tiles whose list exceeded the capacity
constexpr int kStatItems = 6;       // (site, tile row) work items
constexpr int kNumStats = 8;

// Per-frame sizes.
struct Dims {
    int n, W, H, cell, gx, gy, ncell;
    int TS, ntx, nty, ntile, nwarp;
    int capacity;            // tile-list entries per frame
    float inv_sq, inv_scale;
};

// Device buffers; every array is frame-major ([F][per-frame size]).
struct Bufs {
    const float* sites;                              // [F][n][10]
    float4 *pa, *pb, *pbound, *prt, *pg0, *pg1;      // [F][n]
    float* tile_U;                                   // [F][ntile]
    float* warp_U;                                   // [F][ntile * nwarp]
    int* stats;                                      // [F][kNumStats]
    int *cell_count, *cell_cursor;                   // [F][ncell]
    int *tile_count, *tile_cursor;                   // [F][ntile]
    int *cell_of, *sorted_idx;                       // [F][n]
    int* cell_start;                                 // [F][ncell + 1]
    int* tile_off;                                   // [F][ntile + 1]
    unsigned* tile_sites;                            // [F][capacity]: site indices
    int *row_r0, *row_cnt;                           // [F][n]
    int* item_off;                                   // [F][n + 1]
    int* items;                                      // [F][n * nty]
    int *cand0, *cand1;                              // [F][W*H][4]
};

__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__host__ __device__ __forceinline__ int floor_div(int a, int b) {
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

__device__ __forceinline__ float half_round(float v) {
    return __half2float(__float2half(v));
}

// Order-preserving maps between float and uint (atomicMax / radix select on signed keys).
__device__ __forceinline__ unsigned float_to_ordered(float f) {
    unsigned b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float ordered_to_float(unsigned u) {
    unsigned b = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    return __uint_as_float(b);
}

// Safety margin for footprint radii. The fp32 key of a site is computed with a
// rounding error that the bounds must absorb: a relative part, plus the
// cancellation in d2 - proj^2, which the large perpendicular coefficient B of a
// thin site amplifies to ~2 eps B |x - p| / sqrt(min(A, B)) in units of rho.
// kappa = kErrScale * B / sqrt(min(A, B)) is precomputed per site and `dist`
// bounds |x - p| over the pixels concerned.
constexpr float kErrScale = 2e-6f;   // ~8x the worst-case accumulated eps

__device__ __forceinline__ float inflate_rho(float rho, float kappa, float dist) {
    return rho * 1.001f + 0.25f + kappa * dist;
}

// The key below reproduces expressions of SAD (https://github.com/LuckyIYI/SAD; MIT
// License, Copyright (c) 2026 LuckyIYI, its text in THIRD_PARTY_NOTICES.md): voronoi_dmix2
// (backends/cuda/src/sad_common.cuh) and candidatePackedDistance (backends/cuda/src/jfa.cu).
//
// Normalized squared anisotropic distance max(d2a * inv_sq, 1e-8): the
// expression sequence of SAD's voronoi_dmix2 (sad_common.cuh) with the per-site
// constants hoisted, a = (px, py, ux, uy), b = (exp(a), 1/exp(a), r*sqrt(inv_sq), tau).
__device__ __forceinline__ float sad_d2safe(const float4 a, const float4 b,
                                            float uvx, float uvy, float inv_sq) {
    float dx = uvx - a.x;
    float dy = uvy - a.y;
    float diff2 = dx * dx + dy * dy;
    float proj = a.z * dx + a.w * dy;
    float proj2 = proj * proj;
    float perp2 = fmaxf(diff2 - proj2, 0.0f);
    // Same contraction as SAD's compiled VPT key, so rankings agree bit for bit.
    float d2_aniso = __fmaf_rn(b.x, proj2, __fmul_rn(b.y, perp2));
    float d2_norm = d2_aniso * inv_sq;
    return fmaxf(d2_norm, 1e-8f);
}

// sqrtf for x in [2^-101, FLT_MAX]: the sequence nvcc emits for the IEEE
// square root (sqrt.rn.f32), without its branch to the special-case path.
// sqrt.rn is correctly rounded, so on this range the result is bitwise
// sqrtf(x). The argument, d2_safe, is >= 1e-8.
__device__ __forceinline__ float sqrt_rn_normal(float x) {
    float r, y, h;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(r));
    asm("mul.rn.ftz.f32 %0, %1, 0f3F000000;" : "=f"(h) : "f"(r));
    const float e = __fmaf_rn(-y, y, x);
    return __fmaf_rn(e, h, y);
}

// The SAD candidate key, tau * (sqrt(d2_safe) - r_norm), as in candidatePackedDistance.
__device__ __forceinline__ float sad_key(const float4 a, const float4 b,
                                         float uvx, float uvy, float inv_sq) {
    float dmix = sqrt_rn_normal(sad_d2safe(a, b, uvx, uvy, inv_sq)) - b.z;
    return b.w * dmix;
}

// Register-resident sorted list of the 8 smallest (key, index) pairs, each
// packed as (order-preserving key bits << 32 | index): ties on the key go to
// the smaller site index, so the visiting order does not matter. Keys of
// active sites are finite. Insertion is a branch-free network.
struct Top8 {
    unsigned long long v[8];

    __device__ __forceinline__ void reset() {
#pragma unroll
        for (int t = 0; t < 8; ++t) v[t] = ~0ull;    // index part = kEmpty
    }

    __device__ __forceinline__ void push(float key, unsigned idx) {
        const unsigned long long x =
            (static_cast<unsigned long long>(float_to_ordered(key)) << 32) | idx;
        if (!(x < v[7])) return;
        bool c[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) c[t] = x < v[t];   // monotone in t
#pragma unroll
        for (int t = 7; t > 0; --t) v[t] = c[t - 1] ? v[t - 1] : (c[t] ? x : v[t]);
        v[0] = c[0] ? x : v[0];
    }

    // Drop the smallest entry if `pop`: the list shifts down, an empty slot enters at the end.
    __device__ __forceinline__ void pop_front(bool pop) {
#pragma unroll
        for (int t = 0; t < 7; ++t) v[t] = pop ? v[t + 1] : v[t];
        v[7] = pop ? ~0ull : v[7];
    }

    __device__ __forceinline__ unsigned idx(int t) const {
        return static_cast<unsigned>(v[t] & 0xFFFFFFFFull);
    }

    __device__ __forceinline__ bool full() const { return idx(7) != kEmpty; }

    // 8th-smallest key (+inf while fewer than 8 sites were seen).
    __device__ __forceinline__ float kth() const {
        return full() ? ordered_to_float(static_cast<unsigned>(v[7] >> 32)) : INFINITY;
    }
};

// Keys-only variant for pass A, which needs just the 8th-smallest key value.
struct Top8Keys {
    float k[8];

    __device__ __forceinline__ void reset() {
#pragma unroll
        for (int t = 0; t < 8; ++t) k[t] = INFINITY;
    }

    __device__ __forceinline__ void push(float key, unsigned) {
        if (!(key < k[7])) return;
        bool c[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) c[t] = key < k[t];
#pragma unroll
        for (int t = 7; t > 0; --t) k[t] = c[t - 1] ? k[t - 1] : (c[t] ? key : k[t]);
        k[0] = c[0] ? key : k[0];
    }

    __device__ __forceinline__ bool full() const { return k[7] < INFINITY; }
    __device__ __forceinline__ float kth() const { return k[7]; }
};

// Shared-memory staging of one chunk of sites.
template <int NT>
struct SiteChunk {
    float4 a[NT];
    float4 b[NT];
    unsigned j[NT];
    unsigned m[NT];     // warp mask (pass C)
};

__device__ __forceinline__ void write_cands(const Top8& best, int* cand0, int* cand1, size_t p) {
    reinterpret_cast<int4*>(cand0)[p] = make_int4(best.idx(0), best.idx(1), best.idx(2), best.idx(3));
    reinterpret_cast<int4*>(cand1)[p] = make_int4(best.idx(4), best.idx(5), best.idx(6), best.idx(7));
}
__device__ __forceinline__ void write_cands(const Top8Keys&, int*, int*, size_t) {}

// --------------------------------------------------------------------------
// 1. Per-site preparation and cell sort
// --------------------------------------------------------------------------
// pa     = (px, py, ux, uy)          px = -1 marks an inactive site
// pb     = (l1, l2, rnorm, tau)      key constants
// pbound = (A, B, hx, hy)            Q_lb(v) = A (h.v)^2 + B (h_perp.v)^2 <= d2a(v), |h| = 1
// prt    = (radius_px, tau * s, kappa, 0)   kappa: see inflate_rho
// pg0    = (Gxx, Gyy, Gxy, det)      Q_lb in xy form
// pg1    = (-Gxy/Gyy, -Gxy/Gxx, sqrt(Gxx/det), -Gxy/sqrt(Gyy det))
__global__ void prep_kernel(Bufs b, Dims d, int key_fp16) {
    const int f = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d.n) return;
    const size_t fj = static_cast<size_t>(f) * d.n + j;
    const float* s = b.sites + fj * 10;
    float px = s[0], py = s[1], lt = s[2], rad = s[3];
    float ux = s[7], uy = s[8], la = s[9];
    // SAD treats position.x < 0 as an inactive site; a NaN/inf parameter gives a
    // key that never wins, so such a site is dropped as well.
    bool active = px >= 0.0f && isfinite(px) && isfinite(py) && isfinite(lt) &&
                  isfinite(rad) && isfinite(ux) && isfinite(uy) && isfinite(la);
    if (key_fp16) {
        // Mirror pack_candidate_sites: VPT ranks with half-precision site parameters.
        px = half_round(px); py = half_round(py);
        lt = half_round(lt); rad = half_round(rad);
        ux = half_round(ux); uy = half_round(uy);
        la = half_round(la);
    }
    const float l1 = expf(la);
    const float l2 = 1.0f / l1;
    const float tau = fmaxf(expf(lt), 1e-4f);
    const float rnorm = rad * d.inv_scale;
    active = active && isfinite(l1) && l1 > 0.0f && isfinite(l2) && l2 > 0.0f && isfinite(tau);

    b.pa[fj] = make_float4(active ? px : -1.0f, py, ux, uy);
    b.pb[fj] = make_float4(l1, l2, rnorm, tau);

    // Quadratic lower bound of d2a in the eigenbasis of h = u/|u|. With n2 = |u|^2,
    // w = h.v, q = h_perp.v:  d2a = l1 n2 w^2 + l2 max(q^2 + (1 - n2) w^2, 0).
    const float n2 = ux * ux + uy * uy;
    float hx = 1.0f, hy = 0.0f;
    if (n2 > 0.0f) {
        const float inv = rsqrtf(n2);
        hx = ux * inv;
        hy = uy * inv;
    }
    float A, B;
    if (n2 <= 1.0f) {
        A = l1 * n2 + l2 * (1.0f - n2);
        B = l2;
    } else {
        // max(z, 0) >= theta z for theta in [0, 1]; keep the w^2 coefficient positive.
        const float excess = n2 - 1.0f;
        const float theta = fminf(1.0f, 0.5f * l1 * n2 / (l2 * excess));
        A = l1 * n2 - theta * l2 * excess;
        B = theta * l2;
    }
    b.pbound[fj] = make_float4(A, B, hx, hy);
    // Margin also bounds the position-rounding error along both axes, and a
    // non-finite slope (theta/B underflow) is treated as reaching everything.
    float kappa = kErrScale * B * rsqrtf(fminf(A, B)) + kErrScale * (sqrtf(A) + sqrtf(B));
    if (!(kappa < 1e30f)) kappa = 1e30f;
    b.prt[fj] = make_float4(rad, tau * d.inv_scale, kappa, 0.0f);
    // xy form; sums of positive terms (no cancellation), det = A B exactly.
    const float Gxx = A * hx * hx + B * hy * hy;
    const float Gyy = A * hy * hy + B * hx * hx;
    const float Gxy = (A - B) * hx * hy;
    const float det = A * B;
    b.pg0[fj] = make_float4(Gxx, Gyy, Gxy, det);
    b.pg1[fj] = make_float4(-Gxy / Gyy, -Gxy / Gxx, sqrtf(Gxx / det), -Gxy / sqrtf(Gyy * det));

    int c = -1;
    if (active) {
        const int cx = clampi(static_cast<int>(floorf(px / static_cast<float>(d.cell))), 0, d.gx - 1);
        const int cy = clampi(static_cast<int>(floorf(py / static_cast<float>(d.cell))), 0, d.gy - 1);
        c = cy * d.gx + cx;
        atomicAdd(&b.cell_count[static_cast<size_t>(f) * d.ncell + c], 1);
    }
    b.cell_of[fj] = c;
}

// One block per frame: exclusive scan out[0..n] (out[n] = total) of in[0..n).
// Sums accumulate in 64 bits and every output is clamped to `cap`, so a frame
// whose counts exceed 2^31 cannot wrap negative. For tile lists cap is
// capacity + 1, so any tile starting past the list capacity reads as
// overflowed and is rescanned exactly. The input is staged in chunks of 4096;
// each thread sums 4 consecutive values, warp shuffles scan the thread sums
// and one warp the warp sums (integer sums: the same outputs in any order).
__global__ void __launch_bounds__(1024) exclusive_scan_kernel(const int* __restrict__ in, int in_stride,
                                                              int* __restrict__ out, int out_stride, int n,
                                                              int* __restrict__ total_out, int total_stride,
                                                              long long cap) {
    constexpr int PER = 4, CHUNK = 1024 * PER;
    __shared__ int s_in[CHUNK];
    __shared__ long long s_warp[32];
    const int f = blockIdx.x;
    in += static_cast<size_t>(f) * in_stride;
    out += static_cast<size_t>(f) * out_stride;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    long long base = 0;                                  // the total of the chunks done so far
    for (int c0 = 0; c0 < n; c0 += CHUNK) {
        const int cnt = min(CHUNK, n - c0);
        for (int i = tid; i < CHUNK; i += 1024) s_in[i] = i < cnt ? in[c0 + i] : 0;
        __syncthreads();
        long long v[PER], tsum = 0;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            v[k] = s_in[tid * PER + k];
            tsum += v[k];
        }
        long long x = tsum;                              // inclusive scan over the warp's threads
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const long long y = __shfl_up_sync(0xFFFFFFFFu, x, off);
            if (lane >= off) x += y;
        }
        if (lane == 31) s_warp[warp] = x;
        __syncthreads();
        if (warp == 0) {                                 // inclusive scan over the 32 warp sums
            long long w = s_warp[lane];
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const long long y = __shfl_up_sync(0xFFFFFFFFu, w, off);
                if (lane >= off) w += y;
            }
            s_warp[lane] = w;
        }
        __syncthreads();
        long long run = base + (warp > 0 ? s_warp[warp - 1] : 0) + (x - tsum);
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int i = c0 + tid * PER + k;
            if (i < n) out[i] = static_cast<int>(min(run, cap));
            run += v[k];
        }
        base += s_warp[31];
        __syncthreads();                                 // s_in and s_warp are reused
    }
    if (tid == 0) {
        const int total = static_cast<int>(min(base, cap));
        out[n] = total;
        if (total_out != nullptr) total_out[static_cast<size_t>(f) * total_stride] = total;
    }
}

__global__ void cell_scatter_kernel(Bufs b, Dims d) {
    const int f = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d.n) return;
    const int c = b.cell_of[static_cast<size_t>(f) * d.n + j];
    if (c < 0) return;
    const size_t fc = static_cast<size_t>(f) * d.ncell + c;
    const int pos = b.cell_start[static_cast<size_t>(f) * (d.ncell + 1) + c] +
                    atomicAdd(&b.cell_cursor[fc], 1);
    b.sorted_idx[static_cast<size_t>(f) * d.n + pos] = j;
}

// --------------------------------------------------------------------------
// 2. Window scan (pass A of exact mode, or the whole of window mode)
// --------------------------------------------------------------------------
// One block per TS x TS tile. The block stages the sites of every cell within
// `radius` pixels of the tile (L-infinity) in shared memory; each thread keeps
// its pixel's top-8. If a pixel of the tile has seen fewer than 8 sites the
// window is enlarged and rescanned.
template <int TS, bool WRITE_CANDS>
__global__ void __launch_bounds__(TS * TS)
window_topk_kernel(Bufs b, Dims d, int radius) {
    constexpr int NT = TS * TS;
    constexpr int NWARP = NT / 32;
    __shared__ SiteChunk<NT> sm;
    __shared__ int s_rowoff[kMaxWindowRows + 1];
    __shared__ int s_rowbeg[kMaxWindowRows];
    __shared__ float s_red[NWARP];

    const int f = blockIdx.z;
    const float4* __restrict__ pa = b.pa + static_cast<size_t>(f) * d.n;
    const float4* __restrict__ pb = b.pb + static_cast<size_t>(f) * d.n;
    const int* __restrict__ sorted_idx = b.sorted_idx + static_cast<size_t>(f) * d.n;
    const int* __restrict__ cell_start = b.cell_start + static_cast<size_t>(f) * (d.ncell + 1);
    const int tid = threadIdx.y * TS + threadIdx.x;
    const int tx0 = blockIdx.x * TS, ty0 = blockIdx.y * TS;
    const int x = tx0 + threadIdx.x, y = ty0 + threadIdx.y;
    const bool valid = x < d.W && y < d.H;
    const float uvx = static_cast<float>(x), uvy = static_cast<float>(y);
    const int tx1 = min(tx0 + TS - 1, d.W - 1), ty1 = min(ty0 + TS - 1, d.H - 1);

    typename std::conditional<WRITE_CANDS, Top8, Top8Keys>::type best;
    int R = radius;
    while (true) {
        best.reset();
        const int cx0 = clampi(floor_div(tx0 - R, d.cell), 0, d.gx - 1);
        const int cx1 = clampi(floor_div(tx1 + R, d.cell), 0, d.gx - 1);
        const int cy0 = clampi(floor_div(ty0 - R, d.cell), 0, d.gy - 1);
        const int cy1 = clampi(floor_div(ty1 + R, d.cell), 0, d.gy - 1);
        const int nrows = cy1 - cy0 + 1;
        if (tid == 0) {
            int acc = 0;
            for (int r = 0; r < nrows; ++r) {
                const int row = (cy0 + r) * d.gx;
                const int beg = cell_start[row + cx0];
                const int end = cell_start[row + cx1 + 1];
                s_rowbeg[r] = beg;
                s_rowoff[r] = acc;
                acc += end - beg;
            }
            s_rowoff[nrows] = acc;
        }
        __syncthreads();
        const int total = s_rowoff[nrows];
        for (int base = 0; base < total; base += NT) {
            const int i = base + tid;
            if (i < total) {
                int lo = 0, hi = nrows - 1;          // last row with s_rowoff[row] <= i
                while (lo < hi) {
                    const int mid = (lo + hi + 1) >> 1;
                    if (s_rowoff[mid] <= i) lo = mid; else hi = mid - 1;
                }
                const int j = sorted_idx[s_rowbeg[lo] + (i - s_rowoff[lo])];
                sm.a[tid] = pa[j];
                sm.b[tid] = pb[j];
                sm.j[tid] = static_cast<unsigned>(j);
            }
            __syncthreads();
            const int cnt = min(NT, total - base);
            if (valid) {
                for (int k = 0; k < cnt; ++k) {
                    best.push(sad_key(sm.a[k], sm.b[k], uvx, uvy, d.inv_sq), sm.j[k]);
                }
            }
            __syncthreads();
        }
        const bool full = cx0 == 0 && cy0 == 0 && cx1 == d.gx - 1 && cy1 == d.gy - 1;
        const bool ok = !valid || best.full();
        if (__syncthreads_and(ok) || full) break;
        R = 2 * R + d.cell;
    }

    if (WRITE_CANDS) {
        if (valid) {
            const size_t frame_px = static_cast<size_t>(f) * d.W * d.H;
            write_cands(best, b.cand0, b.cand1, frame_px + static_cast<size_t>(y) * d.W + x);
        }
        return;
    }
    // U_w = max over the warp strip of the 8th key: an upper bound on the true
    // 8th key of each of its pixels (these are 8 actual sites); U_T = max_w U_w.
    float u = valid ? best.kth() : -INFINITY;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) u = fmaxf(u, __shfl_xor_sync(0xFFFFFFFFu, u, off));
    const int tile = blockIdx.y * d.ntx + blockIdx.x;
    if ((tid & 31) == 0) {
        s_red[tid >> 5] = u;
        b.warp_U[(static_cast<size_t>(f) * d.ntile + tile) * NWARP + (tid >> 5)] = u;
    }
    __syncthreads();
    if (tid == 0) {
        float m = s_red[0];
#pragma unroll
        for (int k = 1; k < NWARP; ++k) m = fmaxf(m, s_red[k]);
        b.tile_U[static_cast<size_t>(f) * d.ntile + tile] = m;
        atomicMax(reinterpret_cast<unsigned*>(b.stats + f * kNumStats + kStatKmax),
                  float_to_ordered(m));
    }
}

// --------------------------------------------------------------------------
// 3. Cap: Kcap = q-quantile of the tile bounds (radix select on ordered bits)
// --------------------------------------------------------------------------
__device__ __forceinline__ int block_sum_1024(int v, int* s_warp) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = v;
    __syncthreads();
    int total = 0;
#pragma unroll
    for (int w = 0; w < 32; ++w) total += s_warp[w];
    __syncthreads();
    return total;
}

__global__ void __launch_bounds__(1024) cap_kernel(Bufs b, Dims d, float q) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_bins[8];
    __shared__ unsigned s_prefix, s_rank;
    __shared__ int s_warp[32];
    const int f = blockIdx.x;
    const int n = d.ntile;
    const float* __restrict__ U = b.tile_U + static_cast<size_t>(f) * n;
    // 0-based rank of the q-quantile: smallest u with #{U_T <= u} >= ceil(q n).
    unsigned rank = static_cast<unsigned>(max(1, min(n, static_cast<int>(ceilf(q * n)))) - 1);
    unsigned prefix = 0u, mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        for (int t = threadIdx.x; t < n; t += blockDim.x) {
            const unsigned u = float_to_ordered(U[t]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        // The bin holding the rank-th value: the one whose count before it is <= rank < the count up
        // to and including it (prefix sums of the 256 bins by warps 0-7).
        unsigned h = 0u, incl = 0u;
        if (threadIdx.x < 256) {
            const int lane = threadIdx.x & 31;
            h = hist[threadIdx.x];
            incl = h;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const unsigned y = __shfl_up_sync(0xFFFFFFFFu, incl, off);
                if (lane >= off) incl += y;
            }
            if (lane == 31) s_bins[threadIdx.x >> 5] = incl;
        }
        __syncthreads();
        if (threadIdx.x < 256) {
            for (int w = 0; w < (threadIdx.x >> 5); ++w) incl += s_bins[w];
            const unsigned before = incl - h;
            if (before <= rank && rank < incl) {
                s_prefix = prefix | (static_cast<unsigned>(threadIdx.x) << shift);
                s_rank = rank - before;
            }
        }
        __syncthreads();
        prefix = s_prefix;
        rank = s_rank;
        mask |= 255u << shift;
        __syncthreads();
    }
    const float kcap = ordered_to_float(prefix);
    int capped = 0;
    for (int t = threadIdx.x; t < n; t += blockDim.x) capped += U[t] > kcap;
    capped = block_sum_1024(capped, s_warp);
    if (threadIdx.x == 0) {
        b.stats[f * kNumStats + kStatKcap] = static_cast<int>(prefix);
        b.stats[f * kNumStats + kStatCapped] = capped;
    }
}

// --------------------------------------------------------------------------
// 4. Footprint binning
// --------------------------------------------------------------------------
__device__ __forceinline__ float qform(float A, float B, float hx, float hy, float vx, float vy) {
    const float w = hx * vx + hy * vy, q = hx * vy - hy * vx;
    return A * w * w + B * q * q;
}

// min over the box [vx0, vx1] x [vy0, vy1] (relative to the site) of Q, Q
// convex: 0 if the site is inside, else the minimum over the 4 edges. On an
// edge vx = const, Q is smallest at vy = rv vx (rv = -Gxy/Gyy); on vy = const
// at vx = rh vy (rh = -Gxy/Gxx).
__device__ __forceinline__ float box_min_q(float vx0, float vx1, float vy0, float vy1,
                                           const float4 bd, float rv, float rh) {
    if (vx0 <= 0.0f && vx1 >= 0.0f && vy0 <= 0.0f && vy1 >= 0.0f) return 0.0f;
    const float A = bd.x, B = bd.y, hx = bd.z, hy = bd.w;
    float q = qform(A, B, hx, hy, vx0, fminf(fmaxf(rv * vx0, vy0), vy1));
    q = fminf(q, qform(A, B, hx, hy, vx1, fminf(fmaxf(rv * vx1, vy0), vy1)));
    q = fminf(q, qform(A, B, hx, hy, fminf(fmaxf(rh * vy0, vx0), vx1), vy0));
    q = fminf(q, qform(A, B, hx, hy, fminf(fmaxf(rh * vy1, vx0), vx1), vy1));
    return q;
}

// Extent [lo_out, hi_out] of the ellipse Q(v) <= rho^2 along one axis, within the
// band [b0, b1] of the other axis (all relative to the site, conservative).
// With Q = Gaa a^2 + 2 Gab a b + Gbb b^2 (a: the extent axis, b: the band axis):
// half-width along b is rho sqrt(Gaa/det) (`hb`), the extreme points of a sit
// at b = +-rho * `ext_b` (ext_b = -Gab / sqrt(Gbb det) for the max), and the
// boundary is a(b) = (-Gab b +- sqrt(Gaa rho^2 - det b^2)) / Gaa (upper part
// concave, lower convex). False if the band misses the ellipse.
__device__ __forceinline__ bool ellipse_band_extent(float Gaa, float Gab, float det, float hb,
                                                    float ext_b, float rho, float b0, float b1,
                                                    float& lo_out, float& hi_out) {
    const float bmax = rho * hb * 1.001f + 1.0f;
    const float lo = fmaxf(b0, -bmax), hi = fminf(b1, bmax);
    if (lo > hi) return false;
    const float dR = rho * ext_b;
    const float rr = rho * rho * Gaa;
    float t = fminf(fmaxf(dR, lo), hi);
    const float ahi = (-Gab * t + sqrtf(fmaxf(rr - det * t * t, 0.0f))) / Gaa;
    t = fminf(fmaxf(-dR, lo), hi);
    const float alo = (-Gab * t - sqrtf(fmaxf(rr - det * t * t, 0.0f))) / Gaa;
    lo_out = alo - (1.0f + 1e-3f * fabsf(alo));
    hi_out = ahi + (1.0f + 1e-3f * fabsf(ahi));
    return true;
}

// Coarse ellipse (at the frame's largest bound) of a site: rho and margin data.
__device__ __forceinline__ float coarse_rho(const Bufs& b, const Dims& d, int f, size_t fj,
                                            float4& a, float4& rt) {
    a = b.pa[fj];
    if (!(a.x >= 0.0f)) return -1.0f;
    rt = b.prt[fj];
    const int* st = b.stats + f * kNumStats;
    const float kcap = ordered_to_float(static_cast<unsigned>(st[kStatKcap]));
    const float kglob = fminf(ordered_to_float(static_cast<unsigned>(st[kStatKmax])), kcap);
    // |x - p| <= L1 distance to the farthest image corner.
    const float far = fabsf(fmaxf(a.x, static_cast<float>(d.W - 1) - a.x) +
                            fmaxf(a.y, static_cast<float>(d.H - 1) - a.y));
    return inflate_rho(kglob / rt.y + rt.x, rt.z, far);
}

// Per (frame, site): first tile row and number of rows the coarse ellipse spans.
__global__ void site_rows_kernel(Bufs b, Dims d) {
    const int f = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d.n) return;
    const size_t fj = static_cast<size_t>(f) * d.n + j;
    float4 a, rt;
    const float rho_g = coarse_rho(b, d, f, fj, a, rt);
    int r0 = 0, cnt = 0;
    if (rho_g > 0.0f) {                     // else key_j exceeds every bound everywhere
        int r1 = d.nty - 1;
        if (rho_g < kRhoUnbounded) {
            const float ymax = rho_g * b.pg1[fj].z * 1.001f + 1.0f;
            r0 = clampi(static_cast<int>(floorf((a.y - ymax) / d.TS)), 0, d.nty - 1);
            r1 = clampi(static_cast<int>(floorf((a.y + ymax) / d.TS)), 0, d.nty - 1);
        }
        cnt = r1 - r0 + 1;
    }
    b.row_r0[fj] = r0;
    b.row_cnt[fj] = cnt;
}

// Expand the per-site row counts into a compact list of (site, row) items.
__global__ void site_items_kernel(Bufs b, Dims d) {
    const int f = blockIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d.n) return;
    const size_t fj = static_cast<size_t>(f) * d.n + j;
    const int cnt = b.row_cnt[fj];
    int* out = b.items + static_cast<size_t>(f) * d.n * d.nty +
               b.item_off[static_cast<size_t>(f) * (d.n + 1) + j];
    for (int k = 0; k < cnt; ++k) out[k] = j;
}

// One thread per (site, tile row) item (grid-stride). COUNT pass: tile_count
// += 1 per (site, tile) pair. EMIT pass: append the site index to the tile's
// slice (entries beyond the capacity are dropped; pass C rescans such tiles).
template <bool EMIT, int TS>
__global__ void footprint_kernel(Bufs b, Dims d) {
    const int f = blockIdx.y;
    const int nitems = b.stats[f * kNumStats + kStatItems];
    const float kcap = ordered_to_float(static_cast<unsigned>(b.stats[f * kNumStats + kStatKcap]));
    const int* __restrict__ items = b.items + static_cast<size_t>(f) * d.n * d.nty;
    const int* __restrict__ item_off = b.item_off + static_cast<size_t>(f) * (d.n + 1);
    const float* __restrict__ tU = b.tile_U + static_cast<size_t>(f) * d.ntile;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nitems; i += gridDim.x * blockDim.x) {
        const int j = items[i];
        const size_t fj = static_cast<size_t>(f) * d.n + j;
        const int row = b.row_r0[fj] + (i - item_off[j]);
        float4 a, rt;
        const float rho_g = coarse_rho(b, d, f, fj, a, rt);
        const float rad = rt.x, ts = rt.y, kappa = rt.z;   // radius (px), tau * s, error slope
        const float4 g0 = b.pg0[fj], g1 = b.pg1[fj];
        const float y0 = static_cast<float>(row * TS);
        const float y1 = static_cast<float>(min(row * TS + TS - 1, d.H - 1));
        const float vy0 = y0 - a.y, vy1 = y1 - a.y;
        int c0 = 0, c1 = d.ntx - 1;
        if (rho_g < kRhoUnbounded) {
            float xlo, xhi;
            // x-extent in the row band: a = x (Gxx), band along y.
            if (!ellipse_band_extent(g0.x, g0.z, g0.w, g1.z, g1.w, rho_g, vy0, vy1, xlo, xhi)) {
                continue;
            }
            if (a.x + xhi < 0.0f || a.x + xlo > static_cast<float>(d.W - 1)) continue;
            c0 = clampi(static_cast<int>(floorf((a.x + xlo) / TS)), 0, d.ntx - 1);
            c1 = clampi(static_cast<int>(floorf((a.x + xhi) / TS)), 0, d.ntx - 1);
        }
        const float4 bd = b.pbound[fj];
        const float fy = fmaxf(fabsf(vy0), fabsf(vy1));
        for (int col = c0; col <= c1; ++col) {
            const int t = row * d.ntx + col;
            const float vx0 = static_cast<float>(col * TS) - a.x;
            const float vx1 = static_cast<float>(min(col * TS + TS - 1, d.W - 1)) - a.x;
            const float fx = fmaxf(fabsf(vx0), fabsf(vx1));   // L1 to the farthest corner
            const float rho = inflate_rho(fminf(tU[t], kcap) / ts + rad, kappa, fx + fy);
            if (!(rho > 0.0f)) continue;
            const bool bounded = rho < kRhoUnbounded;
            if (bounded && box_min_q(vx0, vx1, vy0, vy1, bd, g1.x, g1.y) > rho * rho) continue;
            if (!EMIT) {
                atomicAdd(&b.tile_count[static_cast<size_t>(f) * d.ntile + t], 1);
                continue;
            }
            const int slot = b.tile_off[static_cast<size_t>(f) * (d.ntile + 1) + t] +
                             atomicAdd(&b.tile_cursor[static_cast<size_t>(f) * d.ntile + t], 1);
            if (slot < d.capacity) b.tile_sites[static_cast<size_t>(f) * d.capacity + slot] = j;
        }
    }
}

// --------------------------------------------------------------------------
// 5. Pass C: top-8 over each tile's list, certified; full rescan if not
// --------------------------------------------------------------------------
// Per-thread rescan (called by every thread of the block): the block stages all
// sites in chunks of NT and each bad pixel's own thread pushes every site. The
// same code as list_topk_kernel's own (COOP = false), for coop_rescan's
// tiles with many bad pixels.
template <int NT>
__device__ __forceinline__ void thread_rescan(SiteChunk<NT>& sm, const float4* __restrict__ pa,
                                              const float4* __restrict__ pb,
                                              const int* __restrict__ sorted_idx, int total,
                                              int tid, float uvx, float uvy, float inv_sq,
                                              bool bad, Top8& best) {
    if (bad) best.reset();
    for (int base = 0; base < total; base += NT) {
        const int i = base + tid;
        if (i < total) {
            const int j = sorted_idx[i];
            sm.a[tid] = pa[j];
            sm.b[tid] = pb[j];
            sm.j[tid] = static_cast<unsigned>(j);
        }
        __syncthreads();
        const int cnt = min(NT, total - base);
        if (bad) {
            for (int k = 0; k < cnt; ++k) {
                best.push(sad_key(sm.a[k], sm.b[k], uvx, uvy, inv_sq), sm.j[k]);
            }
        }
        __syncthreads();
    }
}

// Cooperative rescan (called by every thread of the block; `best` is used only
// by the per-thread branch). The bad pixels are listed in thread order (ballot
// compaction); in rounds of NWARP, warp w takes the round's w-th one. The block
// stages all sites in chunks as thread_rescan does, but one chunk ahead, so that
// the global loads overlap the pushes. The warp's lanes split each chunk
// (lane-strided), each lane keeps a Top8 of its sites, and the 32 lists are
// merged by extracting the warp minimum 8 times. Every site is pushed by exactly
// one lane, so the packed values are distinct (empty slots aside) and the merge
// yields the 8 smallest under Top8's total order: the list thread_rescan's scan
// holds. Lane 0 writes it. A round costs ~total / 32 pushes per lane, but every
// round stages all sites again. A tile with more than MAX_ROUNDS rounds of bad
// pixels (e.g. an overflowed tile) therefore uses thread_rescan, whose cost
// does not depend on the number of bad pixels.
template <int TS>
__device__ __forceinline__ void coop_rescan(SiteChunk<TS * TS>& sm, const float4* __restrict__ pa,
                                            const float4* __restrict__ pb,
                                            const int* __restrict__ sorted_idx, int total,
                                            int tid, float uvx, float uvy, float inv_sq,
                                            bool bad, Top8& best, int* cand0, int* cand1,
                                            size_t frame_px, int W, size_t own) {
    constexpr int NT = TS * TS;
    constexpr int NWARP = NT / 32;
    constexpr int MAX_ROUNDS = TS == 16 ? 12 : 4;
    __shared__ int s_cnt[NWARP];
    __shared__ int s_bad[NT];                          // bad pixels (thread index)
    const int lane = tid & 31, warp = tid >> 5;
    const unsigned ballot = __ballot_sync(0xFFFFFFFFu, bad);
    if (lane == 0) s_cnt[warp] = __popc(ballot);
    __syncthreads();
    int nbad = 0, rank = __popc(ballot & ((1u << lane) - 1u));
#pragma unroll
    for (int w = 0; w < NWARP; ++w) {
        rank += w < warp ? s_cnt[w] : 0;
        nbad += s_cnt[w];
    }
    if (nbad > NWARP * MAX_ROUNDS) {                   // block-uniform
        thread_rescan<NT>(sm, pa, pb, sorted_idx, total, tid, uvx, uvy, inv_sq, bad, best);
        if (bad) write_cands(best, cand0, cand1, own);
        return;
    }
    if (bad) s_bad[rank] = tid;
    __syncthreads();
    for (int r0 = 0; r0 < nbad; r0 += NWARP) {
        const bool mine = r0 + warp < nbad;            // warp-uniform
        const int p = s_bad[mine ? r0 + warp : 0];
        const int px = blockIdx.x * TS + p % TS, py = blockIdx.y * TS + p / TS;
        const float qx = static_cast<float>(px), qy = static_cast<float>(py);
        Top8 part;
        part.reset();
        // (j1, a1, b1): this thread's entry of the next chunk; j2: its index in the one after.
        int j1 = tid < total ? sorted_idx[tid] : 0;
        float4 a1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), b1 = a1;
        if (tid < total) {
            a1 = pa[j1];
            b1 = pb[j1];
        }
        int j2 = tid + NT < total ? sorted_idx[tid + NT] : 0;
        for (int base = 0; base < total; base += NT) {
            if (base + tid < total) {
                sm.a[tid] = a1;
                sm.b[tid] = b1;
                sm.j[tid] = static_cast<unsigned>(j1);
            }
            __syncthreads();
            const int i = base + NT + tid;
            if (i < total) {
                j1 = j2;
                a1 = pa[j1];
                b1 = pb[j1];
                if (i + NT < total) j2 = sorted_idx[i + NT];
            }
            const int cnt = min(NT, total - base);
            if (mine) {
#pragma unroll
                for (int u = 0; u < NT / 32; ++u) {
                    const int k = lane + 32 * u;
                    if (k < cnt) part.push(sad_key(sm.a[k], sm.b[k], qx, qy, inv_sq), sm.j[k]);
                }
            }
            __syncthreads();
        }
        if (mine) {
            Top8 res;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                unsigned long long m = part.v[0];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, m, off);
                    m = o < m ? o : m;
                }
                res.v[t] = m;
                part.pop_front(part.v[0] == m);    // the owner (all lanes once only empties remain)
            }
            if (lane == 0) write_cands(res, cand0, cand1, frame_px + static_cast<size_t>(py) * W + px);
        }
    }
}

// COOP: rescan the bad pixels with coop_rescan; else each with its own thread.
template <int TS, bool COOP>
__global__ void __launch_bounds__(TS * TS) list_topk_kernel(Bufs b, Dims d) {
    constexpr int NT = TS * TS;
    constexpr int NWARP = NT / 32;
    constexpr int WROWS = 32 / TS;
    __shared__ SiteChunk<NT> sm;
    __shared__ float s_wu[NWARP];
    __shared__ uint8_t s_mine[NWARP][NT];      // each warp's entries of the staged chunk

    const int f = blockIdx.z;
    const size_t fn = static_cast<size_t>(f) * d.n;
    const float4* __restrict__ pa = b.pa + fn;
    const float4* __restrict__ pb = b.pb + fn;
    const int tid = threadIdx.y * TS + threadIdx.x;
    const unsigned wbit = 1u << (tid >> 5);
    const int x = blockIdx.x * TS + threadIdx.x, y = blockIdx.y * TS + threadIdx.y;
    const bool valid = x < d.W && y < d.H;
    const float uvx = static_cast<float>(x), uvy = static_cast<float>(y);
    const int tile = blockIdx.y * d.ntx + blockIdx.x;
    const int* off = b.tile_off + static_cast<size_t>(f) * (d.ntile + 1);
    const int begin = off[tile], end = off[tile + 1];
    const bool overflow = end > d.capacity;     // list incomplete: rescan everything
    const unsigned* __restrict__ list = b.tile_sites + static_cast<size_t>(f) * d.capacity;
    int* st = b.stats + f * kNumStats;
    const float kcap = ordered_to_float(static_cast<unsigned>(st[kStatKcap]));
    if (tid < NWARP) s_wu[tid] = fminf(b.warp_U[(static_cast<size_t>(f) * d.ntile + tile) * NWARP + tid], kcap);
    __syncthreads();
    // The tile box, for the per-entry warp-strip tests.
    const int ty0 = blockIdx.y * TS, ty1 = min(ty0 + TS - 1, d.H - 1);
    const float bx0 = static_cast<float>(blockIdx.x * TS);
    const float bx1 = static_cast<float>(min(blockIdx.x * TS + TS - 1, d.W - 1));

    Top8 best;
    best.reset();
    if (!overflow) {
        for (int base = begin; base < end; base += NT) {
            const int i = base + tid;
            if (i < end) {
                const int j = static_cast<int>(list[i]);
                const float4 a = pa[j];
                sm.a[tid] = a;
                sm.b[tid] = pb[j];
                sm.j[tid] = static_cast<unsigned>(j);
                // Warp strips (WROWS rows each) the site can reach at their own
                // bounds U1_w (the entry passed the tile test at U1_T >= U1_w).
                const float4 rt = b.prt[fn + j], bd = b.pbound[fn + j], g1 = b.pg1[fn + j];
                const float vx0 = bx0 - a.x, vx1 = bx1 - a.x;
                const float fx = fmaxf(fabsf(vx0), fabsf(vx1));
                unsigned mask = 0u;
#pragma unroll
                for (int w = 0; w < NWARP; ++w) {
                    const int r0 = ty0 + w * WROWS;
                    if (r0 > ty1) break;                 // strip below the frame
                    const float wy0 = static_cast<float>(r0) - a.y;
                    const float wy1 = static_cast<float>(min(r0 + WROWS - 1, ty1)) - a.y;
                    const float rw = inflate_rho(s_wu[w] / rt.y + rt.x, rt.z,
                                                 fx + fmaxf(fabsf(wy0), fabsf(wy1)));
                    if (!(rw > 0.0f)) continue;
                    if (rw < kRhoUnbounded &&
                        box_min_q(vx0, vx1, wy0, wy1, bd, g1.x, g1.y) > rw * rw) {
                        continue;
                    }
                    mask |= 1u << w;
                }
                sm.m[tid] = mask;
            }
            __syncthreads();
            const int cnt = min(NT, end - base);
            // The entries that reach this warp's strip are compacted (every lane takes part)
            // and pushed four keys at a time. The four keys are independent, so they are in
            // flight together, and Top8 keeps the 8 smallest (key, index) pairs in any order.
            uint8_t* mine = s_mine[tid >> 5];
            int nm = 0;
            for (int k0 = 0; k0 < cnt; k0 += 32) {
                const int k = k0 + (tid & 31);
                const bool has = k < cnt && (sm.m[k] & wbit);
                const unsigned bal = __ballot_sync(0xFFFFFFFFu, has);
                if (has) mine[nm + __popc(bal & ((1u << (tid & 31)) - 1u))] = static_cast<uint8_t>(k);
                nm += __popc(bal);
            }
            __syncwarp();
            if (valid) {
                int i = 0;
                for (; i + 4 <= nm; i += 4) {
                    float key[4];
                    unsigned idx[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int k = mine[i + u];
                        key[u] = sad_key(sm.a[k], sm.b[k], uvx, uvy, d.inv_sq);
                        idx[u] = sm.j[k];
                    }
#pragma unroll
                    for (int u = 0; u < 4; ++u) best.push(key[u], idx[u]);
                }
                for (; i < nm; ++i) {
                    const int k = mine[i];
                    best.push(sad_key(sm.a[k], sm.b[k], uvx, uvy, d.inv_sq), sm.j[k]);
                }
            }
            __syncthreads();
        }
    }

    // Certificate: every site not offered to this pixel has key > U1_w at it.
    const float u1 = s_wu[tid >> 5];
    const bool bad = valid && (overflow || !(best.full() && best.kth() <= u1));
    if (COOP) {
        // A certified pixel's list is final: written now, so `best` is not live
        // across the cooperative rounds (coop_rescan writes the rescanned ones).
        const size_t frame_px = static_cast<size_t>(f) * d.W * d.H;
        const size_t own = frame_px + static_cast<size_t>(y) * d.W + x;
        if (valid && !bad) write_cands(best, b.cand0, b.cand1, own);
        if (__syncthreads_or(bad)) {
            const int* __restrict__ sorted_idx = b.sorted_idx + static_cast<size_t>(f) * d.n;
            const int total = b.cell_start[static_cast<size_t>(f) * (d.ncell + 1) + d.ncell];
            coop_rescan<TS>(sm, pa, pb, sorted_idx, total, tid, uvx, uvy, d.inv_sq, bad, best,
                            b.cand0, b.cand1, frame_px, d.W, own);
            if (bad) atomicAdd(st + kStatFallback, 1);
            if (overflow && tid == 0) atomicAdd(st + kStatOverflow, 1);
        }
        return;
    }
    if (__syncthreads_or(bad)) {
        if (bad) best.reset();
        const int* __restrict__ sorted_idx = b.sorted_idx + static_cast<size_t>(f) * d.n;
        const int total = b.cell_start[static_cast<size_t>(f) * (d.ncell + 1) + d.ncell];
        for (int base = 0; base < total; base += NT) {
            const int i = base + tid;
            if (i < total) {
                const int j = sorted_idx[i];
                sm.a[tid] = pa[j];
                sm.b[tid] = pb[j];
                sm.j[tid] = static_cast<unsigned>(j);
            }
            __syncthreads();
            const int cnt = min(NT, total - base);
            if (bad) {
                for (int k = 0; k < cnt; ++k) {
                    best.push(sad_key(sm.a[k], sm.b[k], uvx, uvy, d.inv_sq), sm.j[k]);
                }
            }
            __syncthreads();
        }
        if (bad) atomicAdd(st + kStatFallback, 1);
        if (overflow && tid == 0) atomicAdd(st + kStatOverflow, 1);
    }
    if (valid) {
        const size_t frame_px = static_cast<size_t>(f) * d.W * d.H;
        write_cands(best, b.cand0, b.cand1, frame_px + static_cast<size_t>(y) * d.W + x);
    }
}

template <int TS>
void launch_window(bool write_cands, dim3 grid, cudaStream_t stream, const Bufs& b,
                   const Dims& d, int radius) {
    const dim3 block(TS, TS);
    if (write_cands) {
        window_topk_kernel<TS, true><<<grid, block, 0, stream>>>(b, d, radius);
    } else {
        window_topk_kernel<TS, false><<<grid, block, 0, stream>>>(b, d, radius);
    }
}

template <int TS>
void launch_footprint_and_c(dim3 tgrid, int F, int item_blocks, cudaStream_t stream,
                            const Bufs& b, const Dims& d, bool coop) {
    const dim3 fgrid(item_blocks, F);
    footprint_kernel<false, TS><<<fgrid, 256, 0, stream>>>(b, d);
    exclusive_scan_kernel<<<F, 1024, 0, stream>>>(b.tile_count, d.ntile, b.tile_off,
                                                  d.ntile + 1, d.ntile, b.stats + kStatEntries,
                                                  kNumStats, static_cast<long long>(d.capacity) + 1);
    footprint_kernel<true, TS><<<fgrid, 256, 0, stream>>>(b, d);
    if (coop) {
        list_topk_kernel<TS, true><<<tgrid, dim3(TS, TS), 0, stream>>>(b, d);
    } else {
        list_topk_kernel<TS, false><<<tgrid, dim3(TS, TS), 0, stream>>>(b, d);
    }
}

}  // namespace

// sites: [F, N, 10] float32 (F frames of the same N sites).
// mode 0 = exact (footprint binning; radius_px = pass-A window radius)
// mode 1 = window (scan every cell within radius_px of the tile)
// coop_fallback: exact mode's rescan of uncertified pixels by whole warps
// (coop_rescan) rather than one thread per pixel; the same output either way.
// Returns {cand0, cand1, stats}: cand* [F, H*W, 4] int32, stats [F, 8] int32
// (see kStat* slots).
std::vector<at::Tensor> build_candidates_cuda(const at::Tensor& sites, int64_t W64, int64_t H64,
                                              double inv_sq_d, int64_t mode, int64_t radius_px,
                                              int64_t cell_px, int64_t tile_px, bool key_fp16,
                                              double cap_quantile, int64_t entries_per_site,
                                              bool coop_fallback) {
    const at::cuda::CUDAGuard guard(sites.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int F = static_cast<int>(sites.size(0));
    Dims d;
    d.n = static_cast<int>(sites.size(1));
    d.W = static_cast<int>(W64);
    d.H = static_cast<int>(H64);
    d.cell = static_cast<int>(cell_px);
    d.gx = (d.W + d.cell - 1) / d.cell;
    d.gy = (d.H + d.cell - 1) / d.cell;
    d.ncell = d.gx * d.gy;
    d.TS = static_cast<int>(tile_px);
    d.ntx = (d.W + d.TS - 1) / d.TS;
    d.nty = (d.H + d.TS - 1) / d.TS;
    d.ntile = d.ntx * d.nty;
    d.nwarp = d.TS * d.TS / 32;
    d.inv_sq = static_cast<float>(inv_sq_d);
    d.inv_scale = std::sqrt(d.inv_sq);   // correctly rounded, = device sqrtf(inv_sq)
    TORCH_CHECK(d.gy <= kMaxWindowRows, "cell grid has too many rows; increase cell_px");
    TORCH_CHECK(d.n < (1 << 24), "at most 2^24 - 1 sites");
    TORCH_CHECK(static_cast<long long>(d.n) * d.nty < (1LL << 31), "too many (site, row) items");
    const bool window_mode = mode == 1;
    // List capacity: tiles that do not fit are rescanned (exact, just slower).
    // Upper bound leaves room for slot = tile_off + cursor (cursor <= n < 2^24)
    // without int overflow; tile_off itself is clamped to capacity + 1.
    const long long cap_ll = std::min<long long>(
        static_cast<long long>(entries_per_site) * std::max(d.n, 1) * (16 / d.TS) * (16 / d.TS) +
            16LL * d.ntile, (1LL << 31) - 1 - (1LL << 24) - 2);
    d.capacity = window_mode ? 1 : static_cast<int>(cap_ll);
    const int64_t nF = F, nn = d.n;

    // Float workspace (frame-major): 6 float4 site arrays, then tile and warp bounds.
    at::Tensor fws = at::empty({nF * (6 * nn * 4 + d.ntile + static_cast<int64_t>(d.ntile) * d.nwarp) + 4},
                               sites.options().dtype(at::kFloat));
    Bufs b;
    b.sites = sites.data_ptr<float>();
    b.pa = reinterpret_cast<float4*>(fws.data_ptr<float>());
    b.pb = b.pa + nF * nn;
    b.pbound = b.pb + nF * nn;
    b.prt = b.pbound + nF * nn;
    b.pg0 = b.prt + nF * nn;
    b.pg1 = b.pg0 + nF * nn;
    b.tile_U = reinterpret_cast<float*>(b.pg1 + nF * nn);
    b.warp_U = b.tile_U + nF * d.ntile;
    // Int workspace: zeroed [stats | cell_count | cell_cursor | tile_count | tile_cursor],
    // then [cell_of | sorted_idx | cell_start | tile_off | tile_sites | row_r0 | row_cnt |
    //       item_off | items].
    const int64_t nzero = nF * (kNumStats + 2 * d.ncell + 2 * d.ntile);
    const int64_t items_per_frame = window_mode ? 1 : nn * d.nty;
    const int64_t nrest = nF * (2 * nn + (d.ncell + 1) + (d.ntile + 1) + d.capacity + 2 * nn +
                                (nn + 1) + items_per_frame);
    at::Tensor iws = at::empty({nzero + nrest}, sites.options().dtype(at::kInt));
    int* ip = iws.data_ptr<int>();
    b.stats = ip;
    b.cell_count = b.stats + nF * kNumStats;
    b.cell_cursor = b.cell_count + nF * d.ncell;
    b.tile_count = b.cell_cursor + nF * d.ncell;
    b.tile_cursor = b.tile_count + nF * d.ntile;
    b.cell_of = ip + nzero;
    b.sorted_idx = b.cell_of + nF * nn;
    b.cell_start = b.sorted_idx + nF * nn;
    b.tile_off = b.cell_start + nF * (d.ncell + 1);
    b.tile_sites = reinterpret_cast<unsigned*>(b.tile_off + nF * (d.ntile + 1));
    b.row_r0 = reinterpret_cast<int*>(b.tile_sites + nF * d.capacity);
    b.row_cnt = b.row_r0 + nF * nn;
    b.item_off = b.row_cnt + nF * nn;
    b.items = b.item_off + nF * (nn + 1);
    C10_CUDA_CHECK(cudaMemsetAsync(ip, 0, nzero * sizeof(int), stream));

    auto iopt = sites.options().dtype(at::kInt);
    at::Tensor cand0 = at::empty({nF, static_cast<int64_t>(d.W) * d.H, 4}, iopt);
    at::Tensor cand1 = at::empty({nF, static_cast<int64_t>(d.W) * d.H, 4}, iopt);
    b.cand0 = cand0.data_ptr<int>();
    b.cand1 = cand1.data_ptr<int>();
    at::Tensor stats = iws.narrow(0, 0, nF * kNumStats).view({nF, kNumStats});
    if (F == 0) return {cand0, cand1, stats};

    const int threads = 256;
    const dim3 sgrid((d.n + threads - 1) / threads, F);
    if (d.n > 0) prep_kernel<<<sgrid, threads, 0, stream>>>(b, d, key_fp16 ? 1 : 0);
    exclusive_scan_kernel<<<F, 1024, 0, stream>>>(b.cell_count, d.ncell, b.cell_start,
                                                  d.ncell + 1, d.ncell, nullptr, 0, INT_MAX);
    if (d.n > 0) cell_scatter_kernel<<<sgrid, threads, 0, stream>>>(b, d);

    const dim3 tgrid(d.ntx, d.nty, F);
    TORCH_CHECK(radius_px >= 0 && radius_px <= (1 << 20), "radius_px out of range [0, 2^20]");
    const int radius = static_cast<int>(radius_px);
    if (window_mode) {
        if (d.TS == 16) launch_window<16>(true, tgrid, stream, b, d, radius);
        else launch_window<8>(true, tgrid, stream, b, d, radius);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {cand0, cand1, stats};
    }

    // In order: pass A bounds, cap, (site, row) items, footprint count, tile
    // offsets, emit, pass C.
    if (d.TS == 16) launch_window<16>(false, tgrid, stream, b, d, radius);
    else launch_window<8>(false, tgrid, stream, b, d, radius);
    cap_kernel<<<F, 1024, 0, stream>>>(b, d, static_cast<float>(cap_quantile));
    if (d.n > 0) {
        site_rows_kernel<<<sgrid, threads, 0, stream>>>(b, d);
        exclusive_scan_kernel<<<F, 1024, 0, stream>>>(b.row_cnt, d.n, b.item_off, d.n + 1, d.n,
                                                      b.stats + kStatItems, kNumStats, INT_MAX);
        site_items_kernel<<<sgrid, threads, 0, stream>>>(b, d);
    }
    // Grid-stride over the items: ~8 rows per site is typical.
    const int item_blocks = std::max(1, std::min((d.n * 8 + threads - 1) / threads, 65535));
    if (d.TS == 16) launch_footprint_and_c<16>(tgrid, F, item_blocks, stream, b, d, coop_fallback);
    else launch_footprint_and_c<8>(tgrid, F, item_blocks, stream, b, d, coop_fallback);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cand0, cand1, stats};
}
