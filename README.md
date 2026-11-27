# GPU-Accelerated FFT with CUDA

Radix-2 Cooley–Tukey FFT implemented from scratch in CUDA in four optimisation stages, benchmarked against a CPU FFT and NVIDIA cuFFT, with a GPU audio/ECG filtering demo. Built and tested on an NVIDIA A30 (CUDA 11.3, SLURM).

## Implementations

| Version | Idea |
|---|---|
| `cpu_dft`, `cpu_fft` | Direct O(N²) DFT and iterative CPU FFT (accuracy reference and timing baseline) |
| `gpu_basic` (v1) | One kernel per stage, twiddles computed on the fly |
| `gpu_twiddle` (v2) | Precomputed twiddle table |
| `gpu_shared` (v3) | Shared-memory tiles fuse bit reversal and the first stages |
| `gpu_shared_r4` (v4) | Radix-2² fusion, half as many passes over global memory |
| `cufft` | Vendor library for comparison |

## Results

Speedup in compute time over the sequential CPU FFT:

| N | v1 | v3 | v4 | cuFFT |
|---|---:|---:|---:|---:|
| 2¹⁶ | 13.8× | 25.8× | 33.2× | 90× |
| 2²⁰ | 118× | 134× | 181× | 597× |
| 2²⁴ | 123× | 155× | 241× | 1496× |

The GPU beats the CPU from about N = 2¹². All versions reach ~10⁻⁷ relative error. cuFFT is still ~6× faster than v4 at 2²⁴. All figures are in [`results/figs/`](results/figs).

## Demos

Forward FFT → frequency mask → inverse FFT on the GPU: telephone-band filtering of spoken digits ([FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset)), and baseline-wander and 60 Hz removal on MIT-BIH ECG record 100.

## Build and run

Requires an NVIDIA GPU (sm_80; edit `ARCH` in the Makefile otherwise), CUDA with cuFFT, and Python with numpy, pandas and matplotlib for `tools.py`.

```bash
module load cuda/11.3
make
./fft_project check    # CPU tests + CPU simulation of GPU kernels (no GPU needed)
./fft_project test     # GPU correctness vs. double-precision CPU FFT
./fft_project bench    # timings, N = 2^10 .. 2^24 -> results/bench.csv
./fft_project sweep    # block / tile size sweeps -> results/sweep.csv
./fft_project audio    # GPU filtering of an audio signal
python tools.py plot   # CSVs -> figures in results/figs
```

On a SLURM cluster: `sbatch run.sbatch` runs the full study (tests, benchmark, sweep, demos, plots); edit the `#SBATCH` lines for your cluster.

## Layout

```
fft_project.cu     main program
common.cuh         helpers, test signals
cpu_fft.cuh        DFT and CPU FFT
gpu_basic.cuh      v1, v2, cuFFT wrapper
gpu_optimized.cuh  v3, v4, race check, sweeps
bench_audio.cuh    benchmarks and audio demo
tools.py           plotting, audio prep
data/  results/    recordings, CSVs, figures
```

## Group Members

- Prasanna: CPU reference and verification
- Ruthvi: basic CUDA FFT
- Varshitha: shared-memory and radix-2² optimisation
- varsha: benchmarking, audio demo and cluster runs
