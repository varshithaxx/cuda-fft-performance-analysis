// shared helpers and test signals
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using cdouble = std::complex<double>;
using cfloat  = std::complex<float>;

inline bool is_pow2(size_t n) { return n != 0 && (n & (n - 1)) == 0; }

inline int ilog2(size_t n) {
    int l = 0;
    while ((size_t(1) << l) < n) ++l;
    return l;
}

struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

inline double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

template <typename T>
double rel_l2_error(const std::complex<T>* a, const cdouble* ref, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dr = double(a[i].real()) - ref[i].real();
        double di = double(a[i].imag()) - ref[i].imag();
        num += dr * dr + di * di;
        den += std::norm(ref[i]);
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

template <typename T>
double max_abs_error(const std::complex<T>* a, const cdouble* ref, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dr = double(a[i].real()) - ref[i].real();
        double di = double(a[i].imag()) - ref[i].imag();
        m = std::max(m, std::sqrt(dr * dr + di * di));
    }
    return m;
}

inline double fft_flops(size_t n) { return 5.0 * double(n) * double(ilog2(n)); }

inline std::string arg_str(int argc, char** argv, const char* name, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == name) return argv[i + 1];
    return def;
}
inline long arg_int(int argc, char** argv, const char* name, long def) {
    std::string s = arg_str(argc, argv, name, "");
    return s.empty() ? def : std::atol(s.c_str());
}
inline double arg_dbl(int argc, char** argv, const char* name, double def) {
    std::string s = arg_str(argc, argv, name, "");
    return s.empty() ? def : std::atof(s.c_str());
}
inline bool arg_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == name) return true;
    return false;
}

enum class SignalType { Impulse, Sine, MultiSine, Random };

inline const char* signal_name(SignalType t) {
    switch (t) {
        case SignalType::Impulse:   return "impulse";
        case SignalType::Sine:      return "sine";
        case SignalType::MultiSine: return "multisine";
        case SignalType::Random:    return "random";
    }
    return "?";
}

inline SignalType parse_signal(const std::string& s) {
    if (s == "impulse")   return SignalType::Impulse;
    if (s == "sine")      return SignalType::Sine;
    if (s == "multisine") return SignalType::MultiSine;
    return SignalType::Random;
}

inline size_t sine_bin(size_t n) { return n >= 8 ? n / 8 + 1 : 1; }

inline std::vector<cdouble> generate_signal(SignalType t, size_t n, unsigned seed = 42) {
    std::vector<cdouble> x(n, cdouble(0.0, 0.0));
    switch (t) {
        case SignalType::Impulse:
            x[0] = 1.0;
            break;
        case SignalType::Sine: {
            size_t f = sine_bin(n);
            for (size_t i = 0; i < n; ++i) x[i] = std::cos(2.0 * M_PI * double(f) * double(i) / double(n));
            break;
        }
        case SignalType::MultiSine: {
            const double fr[3] = {0.031, 0.117, 0.293};
            const double am[3] = {1.0, 0.5, 0.25};
            for (size_t i = 0; i < n; ++i) {
                double v = 0.1;
                for (int k = 0; k < 3; ++k) v += am[k] * std::sin(2.0 * M_PI * fr[k] * double(i));
                x[i] = v;
            }
            break;
        }
        case SignalType::Random: {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> d(-1.0, 1.0);
            for (size_t i = 0; i < n; ++i) x[i] = cdouble(d(rng), d(rng));
            break;
        }
    }
    return x;
}

template <typename T>
std::vector<std::complex<T>> to_precision(const std::vector<cdouble>& x) {
    std::vector<std::complex<T>> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = std::complex<T>(T(x[i].real()), T(x[i].imag()));
    return y;
}
