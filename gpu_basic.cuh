// basic CUDA FFT: v1 and v2 kernels, GpuFFT, cuFFT wrapper and GPU tests
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include "cpu_fft.cuh"

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

#ifndef FFT_PI
#define FFT_PI 3.14159265358979323846
#endif

HD float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
HD float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
HD float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
HD float2 cdir(float2 w, float dir) { return make_float2(w.x, dir * w.y); }

HD unsigned bitrev(unsigned x, int bits) {
#ifdef __CUDA_ARCH__
    return __brev(x) >> (32 - bits);
#else
    unsigned r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
#endif
}

// thread t handles the pair (i, i+h) of a stage with half size h
HD void radix2_index(unsigned t, unsigned h, unsigned& i, unsigned& k) {
    k = t & (h - 1);
    i = ((t - k) << 1) + k;
}

// sincospif is more accurate than sincosf(pi*x) for large k
HD float2 twiddle_onfly(unsigned k, unsigned m) {
#ifdef __CUDA_ARCH__
    float s, c;
    sincospif(-2.0f * (float)k / (float)m, &s, &c);
    return make_float2(c, s);
#else
    double a = -2.0 * FFT_PI * double(k) / double(m);
    return make_float2((float)std::cos(a), (float)std::sin(a));
#endif
}

HD void butterfly2(float2* a, unsigned i, unsigned h, float2 w) {
    float2 u = a[i];
    float2 v = cmul(a[i + h], w);
    a[i]     = cadd(u, v);
    a[i + h] = csub(u, v);
}

HD void stage2_onfly_thread(float2* a, unsigned t, int s, float dir) {
    unsigned h = 1u << (s - 1), i, k;
    radix2_index(t, h, i, k);
    butterfly2(a, i, h, cdir(twiddle_onfly(k, 2u * h), dir));
}

HD void stage2_table_thread(float2* a, const float2* W, unsigned t, int s, int logn, float dir) {
    unsigned h = 1u << (s - 1), i, k;
    radix2_index(t, h, i, k);
    butterfly2(a, i, h, cdir(W[k << (logn - s)], dir));
}

inline std::vector<float2> make_twiddle_table(unsigned n) {
    std::vector<float2> W(n / 2);
    for (unsigned j = 0; j < n / 2; ++j) {
        double a = -2.0 * FFT_PI * double(j) / double(n);
        W[j] = make_float2((float)std::cos(a), (float)std::sin(a));
    }
    return W;
}

enum class GpuImpl {
    Basic,     // v1: one kernel per stage, twiddles on the fly
    Twiddle,   // v2: twiddle table
    Shared,    // v3: first logB stages in shared memory
    SharedR4   // v4: v3 + remaining stages fused in pairs (radix 2^2)
};

inline const char* impl_name(GpuImpl g) {
    switch (g) {
        case GpuImpl::Basic:    return "gpu_basic";
        case GpuImpl::Twiddle:  return "gpu_twiddle";
        case GpuImpl::Shared:   return "gpu_shared";
        case GpuImpl::SharedR4: return "gpu_shared_r4";
    }
    return "?";
}

inline bool parse_impl(const std::string& s, GpuImpl& out) {
    if (s == "basic"   || s == "gpu_basic")     { out = GpuImpl::Basic;    return true; }
    if (s == "twiddle" || s == "gpu_twiddle")   { out = GpuImpl::Twiddle;  return true; }
    if (s == "shared"  || s == "gpu_shared")    { out = GpuImpl::Shared;   return true; }
    if (s == "shared_r4" || s == "gpu_shared_r4") { out = GpuImpl::SharedR4; return true; }
    return false;
}

struct GpuConfig {
    int threads = 256;      
    int logB = 11;            // tile = 2^logB points
    int sharedThreads = 512;
};

struct Step {
    enum Kind { BitRev, SharedBlock, Radix2OnFly, Radix2Table, Radix4Table } kind;
    int s;  // first stage (1 based), or #stages for SharedBlock
};

std::vector<Step> make_schedule_optimized(GpuImpl impl, int logn, const GpuConfig& cfg);

inline std::vector<Step> make_schedule(GpuImpl impl, int logn, const GpuConfig& cfg) {
    if (impl == GpuImpl::Shared || impl == GpuImpl::SharedR4) return make_schedule_optimized(impl, logn, cfg);
    std::vector<Step> st;
    st.push_back({Step::BitRev, 0});
    for (int s = 1; s <= logn; ++s)
        st.push_back({impl == GpuImpl::Basic ? Step::Radix2OnFly : Step::Radix2Table, s});
    return st;
}

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n  in: %s\n",        \
                         cudaGetErrorString(err__), __FILE__, __LINE__, #call);  \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define CUFFT_CHECK(call)                                                        \
    do {                                                                         \
        cufftResult r__ = (call);                                                \
        if (r__ != CUFFT_SUCCESS) {                                              \
            std::fprintf(stderr, "cuFFT error %d at %s:%d\n  in: %s\n", (int)r__, \
                         __FILE__, __LINE__, #call);                             \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer() { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(a, s)); }
    float stop(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

// average over a batch of launches, median over batches
inline double time_gpu(const std::function<void()>& fn, unsigned n, int batches) {
    const int inner = n <= (1u << 14) ? 100 : (n <= (1u << 18) ? 20 : 5);
    for (int w = 0; w < 3; ++w) fn();  // warm up
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    std::vector<double> v;
    for (int b = 0; b < batches; ++b) {
        t.start();
        for (int i = 0; i < inner; ++i) fn();
        v.push_back(t.stop() / inner);
    }
    return median(v);
}

void gpu_scale(float2* d, unsigned n, float s, cudaStream_t st = 0);

class GpuFFT {
public:
    GpuFFT(unsigned n, GpuImpl impl, const GpuConfig& cfg = GpuConfig());
    ~GpuFFT();
    GpuFFT(const GpuFFT&) = delete;
    GpuFFT& operator=(const GpuFFT&) = delete;

    void forward(const float2* d_in, float2* d_out, cudaStream_t st = 0) { run(d_in, d_out, false, st); }
    void inverse(const float2* d_in, float2* d_out, cudaStream_t st = 0) { run(d_in, d_out, true, st); }

    unsigned size() const { return n_; }
    GpuImpl impl() const { return impl_; }
    const GpuConfig& config() const { return cfg_; }
    int num_launches() const { return (int)sched_.size(); }

private:
    void run(const float2* d_in, float2* d_out, bool inv, cudaStream_t st);
    unsigned n_;
    int logn_;
    GpuImpl impl_;
    GpuConfig cfg_;
    float2* d_W_ = nullptr;  // twiddle table (not used by basic)
    std::vector<Step> sched_;
};

class CufftFFT {
public:
    explicit CufftFFT(unsigned n);
    ~CufftFFT();
    CufftFFT(const CufftFFT&) = delete;
    CufftFFT& operator=(const CufftFFT&) = delete;

    void forward(const float2* d_in, float2* d_out, cudaStream_t st = 0);
    void inverse(const float2* d_in, float2* d_out, cudaStream_t st = 0);
    double plan_ms() const { return plan_ms_; }

private:
    unsigned n_;
    cufftHandle plan_;
    double plan_ms_ = 0.0;
};

void print_device_info();

// implemented with the optimised kernels (gpu_optimized.cuh)
void launch_shared_block(const float2* d_in, float2* d_out, const float2* d_W, unsigned n, int logn,
                         int logB, int sharedThreads, float dir, cudaStream_t st);
void launch_radix4(float2* d_a, const float2* d_W, unsigned n, int logn, int s, int threads, bool inv,
                   cudaStream_t st);

__global__ void k_bitrev(const float2* __restrict__ in, float2* __restrict__ out,
                         unsigned n, int logn) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[bitrev(i, logn)] = in[i];
}

__global__ void k_radix2_onfly(float2* a, unsigned half_n, int s, float dir) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < half_n) stage2_onfly_thread(a, t, s, dir);
}

__global__ void k_radix2_table(float2* a, const float2* __restrict__ W,
                               unsigned half_n, int s, int logn, float dir) {
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < half_n) stage2_table_thread(a, W, t, s, logn, dir);
}

__global__ void k_scale(float2* a, unsigned n, float s) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { a[i].x *= s; a[i].y *= s; }
}

static inline unsigned grid_for(unsigned work, int threads) {
    return (work + threads - 1) / threads;
}

void gpu_scale(float2* d, unsigned n, float s, cudaStream_t st) {
    k_scale<<<grid_for(n, 256), 256, 0, st>>>(d, n, s);
    KERNEL_CHECK();
}

void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int rt = 0, drv = 0, clk_khz = 0, bus_bits = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    double bw = 2.0 * clk_khz * 1e3 * (bus_bits / 8) / 1e9;
    std::printf("# GPU: %s | CC %d.%d | %d SMs | %.1f GB | peak BW %.0f GB/s | runtime %d driver %d\n",
                p.name, p.major, p.minor, p.multiProcessorCount,
                p.totalGlobalMem / 1073741824.0, bw, rt, drv);
}

GpuFFT::GpuFFT(unsigned n, GpuImpl impl, const GpuConfig& cfg)
    : n_(n), logn_(ilog2(n)), impl_(impl), cfg_(cfg) {
    if (n < 2 || !is_pow2(n)) {
        std::fprintf(stderr, "GpuFFT: N must be a power of two >= 2 (got %u)\n", n);
        std::exit(1);
    }
    if (cfg_.logB > 12) cfg_.logB = 12;
    if (cfg_.logB < 1) cfg_.logB = 1;
    if (cfg_.threads < 32) cfg_.threads = 32;
    if (cfg_.threads > 1024) cfg_.threads = 1024;
    if (cfg_.sharedThreads > 1024) cfg_.sharedThreads = 1024;

    sched_ = make_schedule(impl_, logn_, cfg_);

    if (impl_ != GpuImpl::Basic) {
        const unsigned m = n_ / 2;
        std::vector<float2> W = make_twiddle_table(n_);
        CUDA_CHECK(cudaMalloc(&d_W_, m * sizeof(float2)));
        CUDA_CHECK(cudaMemcpy(d_W_, W.data(), m * sizeof(float2), cudaMemcpyHostToDevice));
    }
}

GpuFFT::~GpuFFT() {
    if (d_W_) cudaFree(d_W_);
}

void GpuFFT::run(const float2* d_in, float2* d_out, bool inv, cudaStream_t st) {
    const float dir = inv ? -1.0f : 1.0f;
    const int T = cfg_.threads;
    for (const Step& step : sched_) {
        switch (step.kind) {
            case Step::BitRev:
                k_bitrev<<<grid_for(n_, T), T, 0, st>>>(d_in, d_out, n_, logn_);
                break;
            case Step::SharedBlock:
                launch_shared_block(d_in, d_out, d_W_, n_, logn_, step.s, cfg_.sharedThreads, dir, st);
                break;
            case Step::Radix2OnFly:
                k_radix2_onfly<<<grid_for(n_ / 2, T), T, 0, st>>>(d_out, n_ / 2, step.s, dir);
                break;
            case Step::Radix2Table:
                k_radix2_table<<<grid_for(n_ / 2, T), T, 0, st>>>(d_out, d_W_, n_ / 2, step.s, logn_, dir);
                break;
            case Step::Radix4Table:
                launch_radix4(d_out, d_W_, n_, logn_, step.s, T, inv, st);
                break;
        }
        KERNEL_CHECK();
    }
    if (inv) gpu_scale(d_out, n_, 1.0f / float(n_), st);
}

CufftFFT::CufftFFT(unsigned n) : n_(n) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CpuTimer t;
    t.start();
    CUFFT_CHECK(cufftPlan1d(&plan_, (int)n, CUFFT_C2C, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    plan_ms_ = t.ms();
}

CufftFFT::~CufftFFT() { cufftDestroy(plan_); }

void CufftFFT::forward(const float2* d_in, float2* d_out, cudaStream_t st) {
    CUFFT_CHECK(cufftSetStream(plan_, st));
    CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD));
}

void CufftFFT::inverse(const float2* d_in, float2* d_out, cudaStream_t st) {
    CUFFT_CHECK(cufftSetStream(plan_, st));
    CUFFT_CHECK(cufftExecC2C(plan_, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_INVERSE));
    gpu_scale(d_out, n_, 1.0f / float(n_), st);  // cuFFT inverse is unnormalised
}

namespace gputest {

static int g_fail = 0, g_checks = 0;

static void report(bool ok, const char* impl, unsigned n, const char* sig, double e, double r, double tol) {
    ++g_checks;
    if (!ok) ++g_fail;
    if (!ok || n >= (1u << 16) || n <= 8)
        std::printf("  [%s] %-14s N=%-9u %-10s fwd_err=%.2e roundtrip_err=%.2e tol=%.1e\n",
                    ok ? "PASS" : "FAIL", impl, n, sig, e, r, tol);
}

int run(int argc, char** argv) {
    const int maxlog = (int)arg_int(argc, argv, "--maxlog", 22);
    print_device_info();

    const GpuImpl impls[] = {GpuImpl::Basic, GpuImpl::Twiddle, GpuImpl::Shared, GpuImpl::SharedR4};
    const SignalType sigs[] = {SignalType::Impulse, SignalType::Sine, SignalType::MultiSine, SignalType::Random};

    const unsigned NMAX = 1u << maxlog;
    float2 *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, NMAX * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_y, NMAX * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_z, NMAX * sizeof(float2)));

    for (int lg = 1; lg <= maxlog; ++lg) {
        const unsigned n = 1u << lg;
        const double tol = 1e-6 * (lg + 2);
        for (SignalType sg : sigs) {
            auto x = generate_signal(sg, n);
            std::vector<cdouble> ref = x;
            fft_cpu(ref.data(), n, false);
            auto xf = to_precision<float>(x);
            CUDA_CHECK(cudaMemcpy(d_x, xf.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
            std::vector<cfloat> y(n), z(n);

            auto evaluate = [&](const char* name) {
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float2), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(z.data(), d_z, n * sizeof(float2), cudaMemcpyDeviceToHost));
                double e = rel_l2_error(y.data(), ref.data(), n);
                double r = rel_l2_error(z.data(), x.data(), n);
                report(e < tol && r < tol, name, n, signal_name(sg), e, r, tol);
            };

            for (GpuImpl im : impls) {
                GpuFFT f(n, im);
                CUDA_CHECK(cudaMemset(d_y, 0, n * sizeof(float2)));
                f.forward(d_x, d_y);
                f.inverse(d_y, d_z);
                evaluate(impl_name(im));
            }
            {
                CufftFFT f(n);
                f.forward(d_x, d_y);
                f.inverse(d_y, d_z);
                evaluate("cufft");
            }
        }
    }

    std::printf("== Configuration variants (N = 2^20, random signal) ==\n");
    {
        const unsigned n = 1u << std::min(20, maxlog);
        auto x = generate_signal(SignalType::Random, n);
        std::vector<cdouble> ref = x;
        fft_cpu(ref.data(), n, false);
        auto xf = to_precision<float>(x);
        CUDA_CHECK(cudaMemcpy(d_x, xf.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
        std::vector<cfloat> y(n);
        for (int threads : {32, 128, 1024})
            for (int logB : {6, 9, 12})
                for (int st : {64, 256, 1024}) {
                    GpuConfig c; c.threads = threads; c.logB = logB; c.sharedThreads = st;
                    GpuFFT f(n, GpuImpl::SharedR4, c);
                    f.forward(d_x, d_y);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float2), cudaMemcpyDeviceToHost));
                    double e = rel_l2_error(y.data(), ref.data(), n);
                    bool ok = e < 1e-6 * 22;
                    ++g_checks; if (!ok) ++g_fail;
                    std::printf("  [%s] shared_r4 threads=%-4d logB=%-2d sharedThreads=%-4d err=%.2e\n",
                                ok ? "PASS" : "FAIL", threads, logB, st, e);
                }
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    std::printf("\n%d checks, %s: %d failure(s)\n", g_checks, g_fail ? "FAILED" : "ALL GPU TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}

}  // namespace gputest
