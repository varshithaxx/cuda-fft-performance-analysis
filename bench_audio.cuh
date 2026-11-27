// benchmarks (CPU vs GPU vs cuFFT, transfers) and the audio demo
#pragma once

#include "gpu_optimized.cuh"

namespace bench {

static std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "unknown";
}

struct Row {
    std::string impl;
    unsigned n = 0;
    double kernel_ms = NAN, e2e_ms = NAN, h2d_ms = NAN, d2h_ms = NAN, plan_ms = NAN, rel_err = NAN;
    int launches = 0;
};

static void write_row(std::ofstream& csv, const Row& r) {
    const double fl = fft_flops(r.n);
    auto g = [&](double ms) { return std::isnan(ms) ? NAN : fl / (ms * 1e-3) / 1e9; };
    csv << r.impl << ',' << r.n << ',' << ilog2(r.n) << ',' << r.kernel_ms << ',' << r.e2e_ms << ','
        << r.h2d_ms << ',' << r.d2h_ms << ',' << r.plan_ms << ',' << r.rel_err << ',' << g(r.kernel_ms)
        << ',' << g(r.e2e_ms) << ',' << r.launches << '\n';
    csv.flush();
    std::printf("  %-18s N=2^%-2d kernel=%10.4f ms  e2e=%10.4f ms  err=%.2e\n", r.impl.c_str(), ilog2(r.n),
                r.kernel_ms, r.e2e_ms, r.rel_err);
    std::fflush(stdout);
}

int run_bench(int argc, char** argv) {
    const int minlog = (int)arg_int(argc, argv, "--minlog", 10);
    const int maxlog = (int)arg_int(argc, argv, "--maxlog", 24);
    const int dftmax = (int)arg_int(argc, argv, "--dftmax", 14);
    const int cpumax = (int)arg_int(argc, argv, "--cpumax", 24);
    const int reps = (int)arg_int(argc, argv, "--reps", 9);
    const std::string out = arg_str(argc, argv, "--out", "results/bench.csv");
    const SignalType sig = parse_signal(arg_str(argc, argv, "--signal", "random"));
    GpuConfig cfg;
    cfg.threads = (int)arg_int(argc, argv, "--threads", cfg.threads);
    cfg.logB = (int)arg_int(argc, argv, "--logB", cfg.logB);
    cfg.sharedThreads = (int)arg_int(argc, argv, "--sharedThreads", cfg.sharedThreads);

    std::ofstream csv(out);
    if (!csv) { std::fprintf(stderr, "cannot open %s (does the folder exist?)\n", out.c_str()); return 1; }
    csv << "impl,n,log2n,kernel_ms,e2e_ms,h2d_ms,d2h_ms,plan_ms,rel_err,gflops_kernel,gflops_e2e,launches\n";
    std::printf("# CPU: %s\n# config: threads=%d logB=%d sharedThreads=%d reps=%d signal=%s\n",
                cpu_model().c_str(), cfg.threads, cfg.logB, cfg.sharedThreads, reps, signal_name(sig));

    const unsigned NMAX = 1u << maxlog;
    float2 *d_in, *d_out, *h_in, *h_out;
    CUDA_CHECK(cudaMalloc(&d_in, NMAX * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_out, NMAX * sizeof(float2)));
    CUDA_CHECK(cudaMallocHost(&h_in, NMAX * sizeof(float2)));   // pinned
    CUDA_CHECK(cudaMallocHost(&h_out, NMAX * sizeof(float2)));
    std::vector<float2> pageable(NMAX);

    const GpuImpl impls[] = {GpuImpl::Basic, GpuImpl::Twiddle, GpuImpl::Shared, GpuImpl::SharedR4};

    for (int lg = minlog; lg <= maxlog; ++lg) {
        const unsigned n = 1u << lg;
        const size_t bytes = n * sizeof(float2);
        std::printf("== N = 2^%d = %u ==\n", lg, n);

        auto x = generate_signal(sig, n);
        std::vector<cdouble> ref = x;
        fft_cpu(ref.data(), n, false);
        auto xf = to_precision<float>(x);
        std::memcpy(h_in, xf.data(), bytes);

        if (lg <= dftmax) {
            std::vector<cdouble> y(n);
            std::vector<double> v;
            int r = lg <= 10 ? 5 : 1;
            for (int i = 0; i < r; ++i) {
                CpuTimer t; t.start();
                dft_naive(x.data(), y.data(), n, false);
                v.push_back(t.ms());
            }
            Row row; row.impl = "cpu_dft"; row.n = n;
            row.kernel_ms = row.e2e_ms = median(v);
            row.rel_err = rel_l2_error(y.data(), ref.data(), n);
            write_row(csv, row);
        }

        if (lg <= cpumax) {
            auto tw = make_twiddles<float>(n, false);
            std::vector<cfloat> y;
            std::vector<double> v;
            int r = lg <= 16 ? 11 : (lg <= 20 ? 5 : 3);
            for (int i = 0; i < r; ++i) {
                y = xf;
                CpuTimer t; t.start();
                fft_cpu(y.data(), n, false, &tw);
                v.push_back(t.ms());
            }
            Row row; row.impl = "cpu_fft"; row.n = n;
            row.kernel_ms = row.e2e_ms = median(v);
            row.rel_err = rel_l2_error(y.data(), ref.data(), n);
            write_row(csv, row);
        }

        double h2d_pin, d2h_pin;
        {
            GpuTimer t;
            std::vector<double> a, b, c, d;
            std::memcpy(pageable.data(), xf.data(), bytes);
            for (int i = 0; i < reps; ++i) {
                t.start(); CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice)); a.push_back(t.stop());
                t.start(); CUDA_CHECK(cudaMemcpy(h_out, d_in, bytes, cudaMemcpyDeviceToHost)); b.push_back(t.stop());
                t.start(); CUDA_CHECK(cudaMemcpy(d_in, pageable.data(), bytes, cudaMemcpyHostToDevice)); c.push_back(t.stop());
                t.start(); CUDA_CHECK(cudaMemcpy(pageable.data(), d_in, bytes, cudaMemcpyDeviceToHost)); d.push_back(t.stop());
            }
            h2d_pin = median(a); d2h_pin = median(b);
            Row p; p.impl = "transfer_pinned"; p.n = n; p.h2d_ms = h2d_pin; p.d2h_ms = d2h_pin;
            p.e2e_ms = h2d_pin + d2h_pin; write_row(csv, p);
            Row q; q.impl = "transfer_pageable"; q.n = n; q.h2d_ms = median(c); q.d2h_ms = median(d);
            q.e2e_ms = q.h2d_ms + q.d2h_ms; write_row(csv, q);
        }
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        auto measure = [&](const std::string& name, const std::function<void()>& fwd, int launches, double plan_ms) {
            Row row; row.impl = name; row.n = n; row.launches = launches; row.plan_ms = plan_ms;
            row.kernel_ms = time_gpu(fwd, n, reps);
            GpuTimer t;
            std::vector<double> v;
            for (int i = 0; i < reps; ++i) {
                t.start();
                CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice));
                fwd();
                CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
                v.push_back(t.stop());
            }
            row.e2e_ms = median(v);
            row.h2d_ms = h2d_pin; row.d2h_ms = d2h_pin;
            row.rel_err = rel_l2_error(reinterpret_cast<const cfloat*>(h_out), ref.data(), n);
            write_row(csv, row);
        };

        for (GpuImpl im : impls) {
            GpuFFT f(n, im, cfg);
            measure(impl_name(im), [&] { f.forward(d_in, d_out); }, f.num_launches(), NAN);
        }
        {
            CufftFFT f(n);
            measure("cufft", [&] { f.forward(d_in, d_out); }, 0, f.plan_ms());
        }
    }

    cudaFree(d_in); cudaFree(d_out); cudaFreeHost(h_in); cudaFreeHost(h_out);
    std::printf("wrote %s\n", out.c_str());
    return 0;
}

}  // namespace bench
namespace audio {

enum FilterType { F_NONE = 0, F_LOWPASS, F_HIGHPASS, F_BANDSTOP, F_BANDPASS };

__global__ void k_filter(float2* X, unsigned n, float sr, int type, float f1, float f2) {
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    unsigned kk = k <= n / 2 ? k : n - k;  // fold negative frequencies
    float f = (float)kk * sr / (float)n;
    bool keep = true;
    switch (type) {
        case F_LOWPASS:  keep = f <= f1; break;
        case F_HIGHPASS: keep = f >= f1; break;
        case F_BANDSTOP: keep = !(f >= f1 && f <= f2); break;
        case F_BANDPASS: keep = (f >= f1 && f <= f2); break;
        default: break;
    }
    if (!keep) X[k] = make_float2(0.f, 0.f);
}

static std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
    size_t bytes = (size_t)f.tellg();
    std::vector<float> v(bytes / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
    return v;
}

static void write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}

static std::vector<float> half_magnitude(const std::vector<cfloat>& X) {
    size_t n = X.size();
    std::vector<float> m(n / 2 + 1);
    for (size_t k = 0; k <= n / 2; ++k) m[k] = std::abs(X[k]);
    return m;
}

int run(int argc, char** argv) {
    const std::string in = arg_str(argc, argv, "--in", "results/audio/input.f32");
    const std::string outdir = arg_str(argc, argv, "--outdir", "results/audio");
    const float sr = (float)arg_dbl(argc, argv, "--sr", 44100.0);
    const std::string fname = arg_str(argc, argv, "--filter", "bandstop");
    const float f1 = (float)arg_dbl(argc, argv, "--f1", 2800.0);
    const float f2 = (float)arg_dbl(argc, argv, "--f2", 3200.0);
    GpuImpl impl = GpuImpl::SharedR4;
    if (!parse_impl(arg_str(argc, argv, "--impl", "shared_r4"), impl)) {
        std::fprintf(stderr, "unknown --impl (basic|twiddle|shared|shared_r4)\n");
        return 1;
    }
    int ftype = F_NONE;
    if (fname == "lowpass") ftype = F_LOWPASS;
    else if (fname == "highpass") ftype = F_HIGHPASS;
    else if (fname == "bandstop") ftype = F_BANDSTOP;
    else if (fname == "bandpass") ftype = F_BANDPASS;

    print_device_info();
    std::vector<float> s = read_f32(in);
    const size_t L = s.size();
    if (L < 2) { std::fprintf(stderr, "input too short\n"); return 1; }
    unsigned n = 2;
    while (n < L) n <<= 1;
    std::printf("samples=%zu  sr=%.0f Hz  duration=%.3f s  padded N=%u (2^%d)\n", L, sr, L / sr, n, ilog2(n));

    std::vector<cfloat> x(n, cfloat(0.f, 0.f));
    for (size_t i = 0; i < L; ++i) x[i] = cfloat(s[i], 0.f);

    float2 *d_x, *d_X, *d_y, *d_ref;
    const size_t bytes = n * sizeof(float2);
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_ref, bytes));

    GpuTimer t;
    t.start();
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    float h2d = t.stop();

    GpuFFT fft(n, impl);
    fft.forward(d_x, d_X);  // warm up
    t.start();
    fft.forward(d_x, d_X);
    float fwd_ms = t.stop();

    std::vector<cfloat> X(n), Xc(n);
    CUDA_CHECK(cudaMemcpy(X.data(), d_X, bytes, cudaMemcpyDeviceToHost));

    {
        CufftFFT cf(n);
        cf.forward(d_x, d_ref);
        CUDA_CHECK(cudaMemcpy(Xc.data(), d_ref, bytes, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (unsigned k = 0; k < n; ++k) { num += std::norm(X[k] - Xc[k]); den += std::norm(Xc[k]); }
        std::printf("spectrum difference vs cuFFT (relative L2): %.3e\n", std::sqrt(num / den));
    }

    t.start();
    k_filter<<<(n + 255) / 256, 256>>>(d_X, n, sr, ftype, f1, f2);
    KERNEL_CHECK();
    float filt_ms = t.stop();
    std::vector<cfloat> Y(n);
    CUDA_CHECK(cudaMemcpy(Y.data(), d_X, bytes, cudaMemcpyDeviceToHost));

    t.start();
    fft.inverse(d_X, d_y);
    float inv_ms = t.stop();
    std::vector<cfloat> y(n);
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    double imag_max = 0.0;
    std::vector<float> out(L);
    for (size_t i = 0; i < L; ++i) { out[i] = y[i].real(); imag_max = std::max(imag_max, (double)std::fabs(y[i].imag())); }

    write_f32(outdir + "/spectrum_before.f32", half_magnitude(X));
    write_f32(outdir + "/spectrum_after.f32", half_magnitude(Y));
    write_f32(outdir + "/output.f32", out);

    std::ofstream meta(outdir + "/meta.txt");
    meta << "samples " << L << "\nsr " << sr << "\nn " << n << "\nimpl " << impl_name(impl)
         << "\nfilter " << fname << "\nf1 " << f1 << "\nf2 " << f2 << "\nh2d_ms " << h2d
         << "\nforward_ms " << fwd_ms << "\nfilter_ms " << filt_ms << "\ninverse_ms " << inv_ms
         << "\nmax_imag_after_inverse " << imag_max << "\n";

    std::printf("impl=%s filter=%s [%.0f, %.0f] Hz\n", impl_name(impl), fname.c_str(), f1, f2);
    std::printf("H2D %.3f ms | forward FFT %.3f ms | filter %.3f ms | inverse FFT %.3f ms\n", h2d, fwd_ms, filt_ms, inv_ms);
    std::printf("max |imag| after inverse = %.2e (should be ~1e-7)\n", imag_max);
    std::printf("wrote %s/{spectrum_before,spectrum_after,output}.f32 and meta.txt\n", outdir.c_str());

    cudaFree(d_x); cudaFree(d_X); cudaFree(d_y); cudaFree(d_ref);
    return 0;
}

}  // namespace audio
