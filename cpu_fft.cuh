// CPU reference: direct DFT, iterative radix 2 FFT and CPU tests
#pragma once

#include "common.cuh"

template <typename T>
void bit_reverse_permute(std::complex<T>* a, size_t n) {
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

template <typename T>
std::vector<std::complex<T>> make_twiddles(size_t n, bool inverse) {
    std::vector<std::complex<T>> w(n / 2 > 0 ? n / 2 : 1);
    const double sign = inverse ? 1.0 : -1.0;
    for (size_t k = 0; k < n / 2; ++k) {
        double a = sign * 2.0 * M_PI * double(k) / double(n);
        w[k] = std::complex<T>(T(std::cos(a)), T(std::sin(a)));
    }
    return w;
}

template <typename T>
void fft_cpu(std::complex<T>* a, size_t n, bool inverse,
             const std::vector<std::complex<T>>* tw = nullptr) {
    if (n <= 1) return;
    std::vector<std::complex<T>> local;
    if (!tw) { local = make_twiddles<T>(n, inverse); tw = &local; }
    const std::complex<T>* W = tw->data();

    bit_reverse_permute(a, n);

    for (size_t m = 2; m <= n; m <<= 1) {
        const size_t h = m >> 1;
        const size_t step = n / m;
        for (size_t base = 0; base < n; base += m) {
            for (size_t k = 0; k < h; ++k) {
                const std::complex<T> w = W[k * step];
                const std::complex<T> b = a[base + k + h];
                const T vr = b.real() * w.real() - b.imag() * w.imag();
                const T vi = b.real() * w.imag() + b.imag() * w.real();
                const std::complex<T> u = a[base + k];
                a[base + k]     = std::complex<T>(u.real() + vr, u.imag() + vi);
                a[base + k + h] = std::complex<T>(u.real() - vr, u.imag() - vi);
            }
        }
    }
    if (inverse) {
        const T s = T(1) / T(n);
        for (size_t i = 0; i < n; ++i) a[i] *= s;
    }
}

inline void dft_naive(const cdouble* in, cdouble* out, size_t n, bool inverse) {
    const double sign = inverse ? 1.0 : -1.0;
    std::vector<cdouble> W(n);
    for (size_t j = 0; j < n; ++j) {
        double a = sign * 2.0 * M_PI * double(j) / double(n);
        W[j] = cdouble(std::cos(a), std::sin(a));
    }
    for (size_t k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        size_t idx = 0;
        for (size_t j = 0; j < n; ++j) {
            const cdouble x = in[j], w = W[idx];
            sr += x.real() * w.real() - x.imag() * w.imag();
            si += x.real() * w.imag() + x.imag() * w.real();
            idx += k;
            if (idx >= n) idx -= n;
        }
        out[k] = cdouble(sr, si);
    }
    if (inverse)
        for (size_t k = 0; k < n; ++k) out[k] /= double(n);
}
namespace cputest {

static int g_fail = 0;

static void check(bool ok, const char* what, size_t n, const char* sig, double val, double tol) {
    if (!ok) ++g_fail;
    std::printf("  [%s] %-26s N=%-8zu %-10s value=%.3e tol=%.1e\n",
                ok ? "PASS" : "FAIL", what, n, sig, val, tol);
}

int run() {
    const SignalType sigs[] = {SignalType::Impulse, SignalType::Sine,
                               SignalType::MultiSine, SignalType::Random};

    std::printf("== CPU FFT (double) vs direct DFT ==\n");
    for (int lg = 1; lg <= 12; ++lg) {
        size_t n = size_t(1) << lg;
        for (SignalType s : sigs) {
            auto x = generate_signal(s, n);
            std::vector<cdouble> ref(n), y = x;
            dft_naive(x.data(), ref.data(), n, false);
            fft_cpu(y.data(), n, false);
            double e = rel_l2_error(y.data(), ref.data(), n);
            check(e < 1e-12, "fft<double> vs dft", n, signal_name(s), e, 1e-12);
        }
    }

    std::printf("== CPU FFT (float) vs double reference, and round trip ==\n");
    for (int lg = 1; lg <= 20; lg += (lg < 12 ? 1 : 2)) {
        size_t n = size_t(1) << lg;
        for (SignalType s : sigs) {
            auto x = generate_signal(s, n);
            std::vector<cdouble> ref = x;
            fft_cpu(ref.data(), n, false);
            auto yf = to_precision<float>(x);
            fft_cpu(yf.data(), n, false);
            double e = rel_l2_error(yf.data(), ref.data(), n);
            double tol = 1e-6 * (lg + 1);
            check(e < tol, "fft<float> vs fft<double>", n, signal_name(s), e, tol);

            fft_cpu(yf.data(), n, true);
            double r = rel_l2_error(yf.data(), x.data(), n);
            check(r < tol, "float round trip", n, signal_name(s), r, tol);
        }
    }

    std::printf("== Analytic cases ==\n");
    for (int lg : {3, 10, 16}) {
        size_t n = size_t(1) << lg;
        auto x = generate_signal(SignalType::Impulse, n);
        fft_cpu(x.data(), n, false);
        std::vector<cdouble> ones(n, cdouble(1.0, 0.0));
        double e = max_abs_error(x.data(), ones.data(), n);
        check(e < 1e-12, "impulse -> all ones", n, "impulse", e, 1e-12);

        auto c = generate_signal(SignalType::Sine, n);
        fft_cpu(c.data(), n, false);
        size_t f = sine_bin(n);
        std::vector<cdouble> expect(n, 0.0);
        expect[f] = expect[n - f] = double(n) / 2.0;
        double ec = rel_l2_error(c.data(), expect.data(), n);
        check(ec < 1e-10, "cosine -> spikes at f,N-f", n, "sine", ec, 1e-10);
    }

    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL CPU TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}

}  // namespace cputest
