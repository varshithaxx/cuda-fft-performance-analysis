// optimised CUDA FFT: shared memory tiles, fused radix 2^2 passes, race check and sweep
#pragma once

#include "gpu_basic.cuh"

HD float2 mul_minus_i(float2 a) { return make_float2(a.y, -a.x); }

// same idea for two fused stages: thread t handles i0, i0+h, i0+2h, i0+3h
HD void radix4_index(unsigned t, unsigned h, unsigned& i0, unsigned& k) {
    k = t & (h - 1);
    i0 = ((t - k) << 2) + k;
}

// two radix 2 stages done in registers, so memory is read/written once.
// the second stage needs W_4h^(k+h) = W_4h^k * (-i), so no extra table read
HD void butterfly22(float2* a, unsigned i0, unsigned h, float2 w1, float2 w2, bool inv) {
    float2 x0 = a[i0], x1 = a[i0 + h], x2 = a[i0 + 2 * h], x3 = a[i0 + 3 * h];
    float2 t1 = cmul(x1, w1);
    float2 y0 = cadd(x0, t1), y1 = csub(x0, t1);
    float2 t3 = cmul(x3, w1);
    float2 y2 = cadd(x2, t3), y3 = csub(x2, t3);
    float2 w3 = inv ? make_float2(-w2.y, w2.x) : mul_minus_i(w2);
    float2 u = cmul(y2, w2);
    float2 v = cmul(y3, w3);
    a[i0]         = cadd(y0, u);
    a[i0 + 2 * h] = csub(y0, u);
    a[i0 + h]     = cadd(y1, v);
    a[i0 + 3 * h] = csub(y1, v);
}

HD void stage4_table_thread(float2* a, const float2* W, unsigned t, int s, int logn, bool inv) {
    unsigned h = 1u << (s - 1), i0, k;
    radix4_index(t, h, i0, k);
    float dir = inv ? -1.0f : 1.0f;
    float2 w1 = cdir(W[k << (logn - s)], dir);
    float2 w2 = cdir(W[k << (logn - s - 1)], dir);
    butterfly22(a, i0, h, w1, w2, inv);
}

HD void shared_load_elem(float2* sh, const float2* in, unsigned base, unsigned e, int logn) {
    sh[e] = in[bitrev(base + e, logn)];  // bit reversal fused into the load
}
HD void shared_stage_thread(float2* sh, const float2* W, unsigned t, int s, int logn, float dir) {
    unsigned h = 1u << (s - 1), i, k;
    radix2_index(t, h, i, k);
    butterfly2(sh, i, h, cdir(W[k << (logn - s)], dir));
}

inline int effective_logB(int logn, const GpuConfig& c) { return c.logB < logn ? c.logB : logn; }

// v3/v4: one shared memory pass for the first logB stages, then the rest in global memory
std::vector<Step> make_schedule_optimized(GpuImpl impl, int logn, const GpuConfig& cfg) {
    std::vector<Step> st;
    switch (impl) {
        case GpuImpl::Shared: {
            int lb = effective_logB(logn, cfg);
            st.push_back({Step::SharedBlock, lb});
            for (int s = lb + 1; s <= logn; ++s) st.push_back({Step::Radix2Table, s});
            break;
        }
        case GpuImpl::SharedR4: {
            int lb = effective_logB(logn, cfg);
            st.push_back({Step::SharedBlock, lb});
            int s = lb + 1;
            for (; s + 1 <= logn; s += 2) st.push_back({Step::Radix4Table, s});
            if (s <= logn) st.push_back({Step::Radix2Table, s});
            break;
        }
        default:
            break;
    }
    return st;
}

__global__ void k_radix4_table(float2* a, const float2* __restrict__ W,
                               unsigned quarter_n, int s, int logn, bool inv) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < quarter_n) stage4_table_thread(a, W, t, s, logn, inv);
}

// after bit reversal the first logB stages only mix points inside a tile of
// 2^logB, so the whole tile can stay in shared memory for those stages
__global__ void k_shared_block(const float2* __restrict__ in, float2* __restrict__ out,
                               const float2* __restrict__ W, int logn, int logB, float dir) {
    extern __shared__ float2 sh[];
    const unsigned B = 1u << logB;
    const unsigned base = blockIdx.x * B;

    for (unsigned e = threadIdx.x; e < B; e += blockDim.x) shared_load_elem(sh, in, base, e, logn);
    __syncthreads();

    for (int s = 1; s <= logB; ++s) {
        for (unsigned t = threadIdx.x; t < B / 2; t += blockDim.x) shared_stage_thread(sh, W, t, s, logn, dir);
        __syncthreads();
    }

    for (unsigned e = threadIdx.x; e < B; e += blockDim.x) out[base + e] = sh[e];
}

void launch_shared_block(const float2* d_in, float2* d_out, const float2* d_W, unsigned n, int logn,
                         int logB, int sharedThreads, float dir, cudaStream_t st) {
    const unsigned B = 1u << logB;
    int thr = sharedThreads;
    if ((unsigned)thr > B / 2) thr = (int)(B / 2);
    if (thr < 1) thr = 1;
    k_shared_block<<<n >> logB, thr, B * sizeof(float2), st>>>(d_in, d_out, d_W, logn, logB, dir);
}

void launch_radix4(float2* d_a, const float2* d_W, unsigned n, int logn, int s, int threads, bool inv,
                   cudaStream_t st) {
    k_radix4_table<<<grid_for(n / 4, threads), threads, 0, st>>>(d_a, d_W, n / 4, s, logn, inv);
}

namespace sim {

static int g_fail = 0;

static void sim_run(GpuImpl impl, const GpuConfig& cfg_in, const std::vector<float2>& in,
                    std::vector<float2>& out, bool inv) {
    const unsigned n = (unsigned)in.size();
    const int logn = ilog2(n);
    GpuConfig cfg = cfg_in;
    if (cfg.logB > 12) cfg.logB = 12;
    std::vector<float2> W = make_twiddle_table(n);
    const float dir = inv ? -1.0f : 1.0f;
    out.assign(n, make_float2(0.f, 0.f));

    for (const Step& step : make_schedule(impl, logn, cfg)) {
        switch (step.kind) {
            case Step::BitRev:
                for (unsigned i = 0; i < n; ++i) out[bitrev(i, logn)] = in[i];
                break;
            case Step::SharedBlock: {
                const int lb = step.s;
                const unsigned B = 1u << lb;
                unsigned thr = (unsigned)cfg.sharedThreads;
                if (thr > B / 2) thr = B / 2;
                if (thr < 1) thr = 1;
                std::vector<float2> sh(B);
                for (unsigned blk = 0; blk < (n >> lb); ++blk) {
                    const unsigned base = blk * B;
                    for (unsigned tid = 0; tid < thr; ++tid)
                        for (unsigned e = tid; e < B; e += thr) shared_load_elem(sh.data(), in.data(), base, e, logn);
                    for (int s = 1; s <= lb; ++s)
                        for (unsigned tid = 0; tid < thr; ++tid)
                            for (unsigned t = tid; t < B / 2; t += thr)
                                shared_stage_thread(sh.data(), W.data(), t, s, logn, dir);
                    for (unsigned e = 0; e < B; ++e) out[base + e] = sh[e];
                }
                break;
            }
            case Step::Radix2OnFly:
                for (unsigned t = 0; t < n / 2; ++t) stage2_onfly_thread(out.data(), t, step.s, dir);
                break;
            case Step::Radix2Table:
                for (unsigned t = 0; t < n / 2; ++t) stage2_table_thread(out.data(), W.data(), t, step.s, logn, dir);
                break;
            case Step::Radix4Table:
                for (unsigned t = 0; t < n / 4; ++t) stage4_table_thread(out.data(), W.data(), t, step.s, logn, inv);
                break;
        }
    }
    if (inv)
        for (auto& v : out) { v.x /= float(n); v.y /= float(n); }
}

// running threads one after another cannot show a race, so we also check that
// every element is written by exactly one thread in each kernel
static bool check_partition_radix2(unsigned n, int s) {
    std::vector<unsigned char> hit(n, 0);
    unsigned h = 1u << (s - 1);
    for (unsigned t = 0; t < n / 2; ++t) {
        unsigned i, k;
        radix2_index(t, h, i, k);
        if (i + h >= n || ++hit[i] > 1 || ++hit[i + h] > 1) return false;
    }
    for (unsigned i = 0; i < n; ++i) if (hit[i] != 1) return false;
    return true;
}
static bool check_partition_radix4(unsigned n, int s) {
    std::vector<unsigned char> hit(n, 0);
    unsigned h = 1u << (s - 1);
    for (unsigned t = 0; t < n / 4; ++t) {
        unsigned i0, k;
        radix4_index(t, h, i0, k);
        if (i0 + 3 * h >= n) return false;
        for (unsigned q = 0; q < 4; ++q) if (++hit[i0 + q * h] > 1) return false;
    }
    for (unsigned i = 0; i < n; ++i) if (hit[i] != 1) return false;
    return true;
}
static bool check_bitrev_permutation(unsigned n, int logn) {
    std::vector<unsigned char> hit(n, 0);
    for (unsigned i = 0; i < n; ++i) {
        unsigned r = bitrev(i, logn);
        if (r >= n || ++hit[r] > 1) return false;
        if (bitrev(r, logn) != i) return false;
    }
    return true;
}

int run(int argc, char** argv) {
    const int maxlog = (int)arg_int(argc, argv, "--maxlog", 16);

    std::printf("== Index map checks (race freedom) ==\n");
    int bad = 0;
    for (int lg = 1; lg <= 20; ++lg) {
        unsigned n = 1u << lg;
        if (!check_bitrev_permutation(n, lg)) { ++bad; std::printf("  bitrev FAIL N=%u\n", n); }
        for (int s = 1; s <= lg; ++s)
            if (!check_partition_radix2(n, s)) { ++bad; std::printf("  radix2 FAIL N=%u s=%d\n", n, s); }
        for (int s = 1; s + 1 <= lg; ++s)
            if (!check_partition_radix4(n, s)) { ++bad; std::printf("  radix4 FAIL N=%u s=%d\n", n, s); }
    }
    std::printf("  [%s] bit reversal, radix 2 and radix 2^2 maps partition [0,N) for N = 2 .. 2^20\n",
                bad ? "FAIL" : "PASS");
    g_fail += bad;

    std::printf("== Simulated kernels vs CPU double FFT ==\n");
    const GpuImpl impls[] = {GpuImpl::Basic, GpuImpl::Twiddle, GpuImpl::Shared, GpuImpl::SharedR4};
    const SignalType sigs[] = {SignalType::Impulse, SignalType::Sine, SignalType::MultiSine, SignalType::Random};

    std::vector<GpuConfig> cfgs(3);
    cfgs[0].logB = 11; cfgs[0].sharedThreads = 512;
    cfgs[1].logB = 4;  cfgs[1].sharedThreads = 3;
    cfgs[2].logB = 7;  cfgs[2].sharedThreads = 1024; // more threads than butterflies

    int checks = 0;
    double worst = 0.0;
    for (int lg = 1; lg <= maxlog; ++lg) {
        unsigned n = 1u << lg;
        double tol = 1e-6 * (lg + 2);
        for (SignalType sg : sigs) {
            auto x = generate_signal(sg, n);
            std::vector<cdouble> ref = x;
            fft_cpu(ref.data(), n, false);
            std::vector<float2> xin(n);
            for (unsigned i = 0; i < n; ++i) xin[i] = make_float2((float)x[i].real(), (float)x[i].imag());

            for (GpuImpl im : impls) {
                for (size_t c = 0; c < cfgs.size(); ++c) {
                    if (c > 0 && (im == GpuImpl::Basic || im == GpuImpl::Twiddle)) continue;
                    std::vector<float2> y, z;
                    sim_run(im, cfgs[c], xin, y, false);
                    double e = rel_l2_error(reinterpret_cast<const cfloat*>(y.data()), ref.data(), n);
                    sim_run(im, cfgs[c], y, z, true);
                    double r = rel_l2_error(reinterpret_cast<const cfloat*>(z.data()), x.data(), n);
                    ++checks;
                    worst = std::max(worst, std::max(e, r));
                    if (!(e < tol && r < tol)) {
                        ++g_fail;
                        std::printf("  [FAIL] %-14s cfg%zu N=%-7u %-9s fwd_err=%.2e roundtrip_err=%.2e\n",
                                    impl_name(im), c, n, signal_name(sg), e, r);
                    }
                }
            }
        }
        std::printf("  N=2^%-2d done\n", lg);
    }
    std::printf("  %d forward+inverse checks, worst relative error %.2e\n", checks, worst);
    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL SIMULATION TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}

}  // namespace sim

namespace sweep {

int run(int argc, char** argv) {
    const std::string out = arg_str(argc, argv, "--out", "results/sweep.csv");
    const std::string logs = arg_str(argc, argv, "--sweeplogs", "16,20,24");
    const int reps = (int)arg_int(argc, argv, "--reps", 9);

    std::vector<int> lgs;
    { std::stringstream ss(logs); std::string tok; while (std::getline(ss, tok, ',')) lgs.push_back(std::atoi(tok.c_str())); }

    std::ofstream csv(out);
    if (!csv) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    csv << "sweep,impl,n,log2n,threads,logB,shared_threads,kernel_ms,gflops\n";

    int maxlg = 0; for (int l : lgs) maxlg = std::max(maxlg, l);
    const unsigned NMAX = 1u << maxlg;
    float2 *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, NMAX * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_out, NMAX * sizeof(float2)));

    for (int lg : lgs) {
        const unsigned n = 1u << lg;
        auto xf = to_precision<float>(generate_signal(SignalType::Random, n));
        CUDA_CHECK(cudaMemcpy(d_in, xf.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
        auto emit = [&](const char* sweep, GpuImpl im, const GpuConfig& c) {
            GpuFFT f(n, im, c);
            double ms = time_gpu([&] { f.forward(d_in, d_out); }, n, reps);
            double gf = fft_flops(n) / (ms * 1e-3) / 1e9;
            csv << sweep << ',' << impl_name(im) << ',' << n << ',' << lg << ',' << c.threads << ',' << c.logB
                << ',' << c.sharedThreads << ',' << ms << ',' << gf << '\n';
            csv.flush();
            std::printf("  %-8s %-14s N=2^%-2d threads=%-4d logB=%-2d sharedThreads=%-4d %9.4f ms %7.1f GFLOP/s\n",
                        sweep, impl_name(im), lg, c.threads, c.logB, c.sharedThreads, ms, gf);
            std::fflush(stdout);
        };

        std::printf("== sweep N = 2^%d ==\n", lg);
        for (GpuImpl im : {GpuImpl::Basic, GpuImpl::Twiddle, GpuImpl::SharedR4})
            for (int t : {32, 64, 128, 256, 512, 1024}) {
                GpuConfig c; c.threads = t;
                emit("threads", im, c);
            }
        for (GpuImpl im : {GpuImpl::Shared, GpuImpl::SharedR4})
            for (int lb = 6; lb <= 12; ++lb)
                for (int st : {32, 64, 128, 256, 512, 1024}) {
                    if (st > (1 << lb) / 2) continue;
                    GpuConfig c; c.logB = lb; c.sharedThreads = st;
                    emit("tile", im, c);
                }
    }
    cudaFree(d_in); cudaFree(d_out);
    std::printf("wrote %s\n", out.c_str());
    return 0;
}

}  // namespace sweep
