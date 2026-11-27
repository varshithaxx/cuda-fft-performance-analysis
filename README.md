# Fast Fourier Analysis: GPU Accelerated FFT Using CUDA

Implementation and performance analysis of the Cooley Tukey radix 2 FFT on NVIDIA GPUs, compared with a direct CPU DFT, a sequential CPU FFT and NVIDIA's cuFFT library, with an audio filtering demonstration. Built for the Bhavani GPU cluster (NVIDIA A30, CUDA 11.3, SLURM).

The CPU side has a direct O(N²) DFT and an iterative radix 2 FFT; the double precision FFT is the accuracy reference for everything else, and the float version is the sequential CPU timing baseline.

There are four CUDA versions, each building on the previous one so the effect of every optimisation can be measured separately. **v1 basic** runs a bit reversal pass and then one kernel per stage, with one thread per butterfly and twiddle factors computed on the fly with `sincospif`. **v2 twiddle** reads the twiddle factors from a table computed once in double precision. **v3 shared** fuses the bit reversal into a shared memory kernel that performs the first logB stages of each tile (2048 points by default) with `__syncthreads()` between stages, so only the wider stages go through global memory. **v4 shared_r4** also fuses the remaining global stages two at a time (a radix 2² butterfly kept in registers), which halves the number of passes over global memory. cuFFT is wrapped behind the same interface for comparison.

The per thread logic of every kernel is written as `__host__ __device__` functions, and the kernel order of each version is defined once. This lets `./fft_project check` run the exact kernel code on the CPU, one simulated thread at a time, and prove that inside every kernel each element is touched by exactly one thread. That check needs no GPU.

## Work division

The project was split four ways along the same lines as the code, so each member owns one file (or two) and the matching part of the report.

**Prasanna, CPU reference and verification** (`common.cuh`, `cpu_fft.cuh`, `data/`). The real recordings used for testing (choosing and preparing the speech clips and the ECG record), test signals, error metrics and helpers, the direct DFT (`dft_naive`), the iterative radix 2 FFT (`bit_reverse_permute`, `make_twiddles`, `fft_cpu`) and the CPU tests. In the report: DFT and FFT background, complexity, and how correctness was verified.

**Ruthvi, basic CUDA FFT** (`gpu_basic.cuh`). The thread to butterfly mapping (`radix2_index`, `butterfly2`, `twiddle_onfly`), the bit reversal and per stage kernels (`k_bitrev`, `k_radix2_onfly`, `k_radix2_table`), the `GpuFFT` host code, the cuFFT wrapper and the GPU tests. In the report: the parallelisation strategy, why each stage is its own kernel launch, and the v1 versus v2 result.

**Varshitha, optimisation** (`gpu_optimized.cuh`). The shared memory kernel (`k_shared_block`), the fused radix 2² pass (`butterfly22`, `k_radix4_table`), the v3 and v4 schedules, the CPU simulation with the race check, and the sweep. In the report: the memory hierarchy, the ablation (fig7), the sweeps (fig8 and fig9) and the bandwidth discussion.

**Varsha, benchmarking, audio and cluster runs** (`bench_audio.cuh`, `fft_project.cu`, `Makefile`, `tools.py`, `run.sbatch`, `results/`). The benchmark driver (timing method, pinned versus pageable transfers), the audio demo with `k_filter` and its runs on the speech and ECG recordings, the main program, the build and job scripts, the plots and the runs on Bhavani. In the report: figures 1 to 6, the transfer overhead discussion and the synthetic, speech and ECG filtering results.

## Real recordings

Besides the synthetic test signals, the GPU filter is applied to two public datasets, both stored in `data/`.

**Speech.** `data/speech/` holds ten clips from the Free Spoken Digit Dataset (speaker jackson saying the digits zero to nine, take 0, 8 kHz). The job joins them into one 6.6 second signal (N = 2^16 after padding) and applies a 300 to 3400 Hz bandpass, the classic telephone band. Dataset: Z. Jackson et al., Free Spoken Digit Dataset, github.com/Jakobovski/free-spoken-digit-dataset, licensed CC BY-SA 4.0.

**ECG.** `data/ecg_100.wav` is record 100 of the MIT-BIH Arrhythmia Database (lead MLII, 360 Hz, 30 minutes, 650,000 samples, so N = 2^20). It shows two common kinds of interference: slow baseline wander (about 5% of the signal energy lies below 0.5 Hz) and a narrow 60 Hz powerline line. The job removes them in two separate runs, a 0.5 Hz highpass and a 59 to 61 Hz bandstop. The WAV file was made with `python tools.py ecg-to-wav --record 100`, which downloads the record from PhysioNet (needs `pip install wfdb`). Dataset: G. B. Moody and R. G. Mark, "The impact of the MIT-BIH Arrhythmia Database", IEEE Engineering in Medicine and Biology 20(3), 2001; and A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet", Circulation 101(23), 2000.

Run them with `sbatch -t 00:20:00 run.sbatch realdata` (or `speech`, `ecg`); they are also part of `sbatch run.sbatch`.

## Modes of the executable

`./fft_project check` runs the CPU reference tests and the CPU simulation of the GPU kernels. `./fft_project test` checks every GPU version and cuFFT against the double precision CPU FFT for all test signals and sizes up to 2^22, forward and inverse. `./fft_project bench` times every implementation from N = 2^10 to 2^24 and writes `results/bench.csv`. `./fft_project sweep` tries different block sizes and shared memory tile sizes at N = 2^16, 2^20 and 2^24 and writes `results/sweep.csv`. `./fft_project audio` filters an audio signal on the GPU. Running `./fft_project` with no mode prints all options.

`python tools.py audio-prep` writes the audio input (a built-in test signal with `--synthetic`, or your own recording with `--wav file.wav`). `python tools.py audio-plot` checks the GPU spectrum against `numpy.fft`, writes `results/audio/output.wav` and plots the demo. `python tools.py plot` turns the CSV files into nine figures and `summary.md` in `results/figs`.

## Running on Bhavani

**1. Copy to the cluster.** On your Mac, from the folder that contains `gpu_project`, pack it and copy it:

```bash
tar czf gpu_project.tar.gz gpu_project
scp gpu_project.tar.gz <username>@<cluster-ip>:~/
```

**2. Log in and unpack.** If an older `gpu_project` folder exists on the cluster, the first command keeps it as `gpu_project_old` instead of mixing the two versions.

```bash
ssh <username>@<cluster-ip>
[ -d ~/gpu_project ] && mv ~/gpu_project ~/gpu_project_old
tar xzf gpu_project.tar.gz
cd ~/gpu_project
```

**3. Build and run the checks that need no GPU** (fine on the login node):

```bash
module load cuda/11.3
make
./fft_project check
```

The last lines should say `ALL CPU TESTS PASSED` and `ALL SIMULATION TESTS PASSED`. The Makefile passes `-ccbin /usr/bin/g++` because CUDA 11.3 rejects the GCC 12 loaded by default on the cluster; the system GCC 8.5 is supported. Load `cuda/11.3` in every new login before running `./fft_project`, because the program needs the cuFFT library from that module.

**4. Run the GPU tests, then everything else.**

```bash
sbatch -t 00:20:00 run.sbatch test
squeue -u $USER
```

`PD` means the job is waiting for a GPU and `R` means it is running. When it no longer appears, read the log with `tail -20 fft_*.out`; it should end with `ALL GPU TESTS PASSED`. Then run the full study (tests, benchmark, sweep, synthetic audio demo, speech and ECG runs, and all plots):

```bash
sbatch run.sbatch
```

Both can also be queued at once, with the full run starting only if the tests pass: `sbatch --dependency=afterok:<test job id> run.sbatch`. Jobs keep waiting and running after you log out. `squeue -u $USER --start` shows the scheduler's estimated start time.

**5. Copy the results back.** On your Mac:

```bash
scp -r <username>@<cluster-ip>:~/gpu_project/results ~/Desktop/gpu_project/results
```

## Outputs

`results/bench.csv` has one row per implementation and size. `kernel_ms` is GPU compute time measured with CUDA events over batches of back to back transforms. `e2e_ms` adds the host to device and device to host copies from pinned memory, and `h2d_ms` and `d2h_ms` are those copies alone. `plan_ms` is cuFFT's plan creation time, reported separately. `rel_err` is the relative L2 error against the double precision CPU FFT, and the GFLOP/s columns use the standard 5 N log₂N operation count. Rows named `transfer_pinned` and `transfer_pageable` compare copies from pinned and ordinary host memory.

`results/figs` receives the nine benchmark figures (compute time, end to end time, speedup, GFLOP/s, accuracy, transfer cost, optimisation ablation and the two sweeps), the four filtering figures (`audio_demo.png`, `speech.png`, `ecg_baseline.png`, `ecg_powerline.png`) and `summary.md`. Each filtering run has its own folder (`results/audio`, `results/speech`, `results/ecg_baseline`, `results/ecg_powerline`) with `input.wav`, the filtered `output.wav` and `meta.txt` with the GPU timings. The synthetic demo removes a loud 3 kHz whistle from a chord with a band stop filter. Other recordings or settings can be tried by changing the `filter_run` lines in `run.sbatch` (filter types: lowpass, highpass, bandstop, bandpass; `f1` and `f2` in Hz).


