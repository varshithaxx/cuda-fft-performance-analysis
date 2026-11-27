// GPU accelerated FFT (radix 2 Cooley-Tukey) with CUDA
// build: module load cuda/11.3 && make
// run:   ./fft_project check | test | bench | sweep | audio   (no args prints options)

#include "common.cuh"
#include "cpu_fft.cuh"
#include "gpu_basic.cuh"
#include "gpu_optimized.cuh"
#include "bench_audio.cuh"

static void usage() {
    std::printf(
        "usage: ./fft_project <mode> [options]\n"
        "  check   CPU reference tests + CPU simulation of the GPU kernels (no GPU needed)\n"
        "  test    correctness of every GPU version and cuFFT    [--maxlog 22]\n"
        "  bench   timing of every implementation               [--minlog 10 --maxlog 24 --dftmax 14 --reps 9 --out results/bench.csv]\n"
        "  sweep   block size and shared memory tile sweeps      [--sweeplogs 16,20,24 --reps 9 --out results/sweep.csv]\n"
        "  audio   GPU FFT filtering of an audio signal         [--in results/audio/input.f32 --sr 44100 --filter bandstop --f1 2800 --f2 3200\n"
        "                                                         --impl shared_r4 --outdir results/audio]\n"
        "GPU version options for bench: --threads 256 --logB 11 --sharedThreads 512\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    const std::string mode = argv[1];
    if (mode == "check") {
        int a = cputest::run();
        int b = sim::run(argc, argv);
        return (a || b) ? 1 : 0;
    }
    if (mode == "test")  return gputest::run(argc, argv);
    if (mode == "bench") { print_device_info(); return bench::run_bench(argc, argv); }
    if (mode == "sweep") { print_device_info(); return sweep::run(argc, argv); }
    if (mode == "audio") return audio::run(argc, argv);
    usage();
    return 1;
}
