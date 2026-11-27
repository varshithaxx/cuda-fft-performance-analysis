#!/usr/bin/env python3
"""Plots and audio helpers for the FFT project.

    python tools.py audio-prep --synthetic     (or --wav clip.wav)
    python tools.py audio-plot
    python tools.py ecg-to-wav --record 100          (pip install wfdb)
    python tools.py plot
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 10,
    "lines.linewidth": 2.0, "lines.markersize": 6,
})
BEFORE, AFTER = "#2a78d6", "#eb6834"  # audio plots: original vs filtered

SERIES = [
    ("cpu_dft",       "CPU DFT O(N²)",        "#2a78d6", "o"),
    ("cpu_fft",       "CPU FFT (sequential)", "#eb6834", "s"),
    ("gpu_basic",     "GPU v1 basic",         "#1baf7a", "^"),
    ("gpu_twiddle",   "GPU v2 twiddle table", "#eda100", "v"),
    ("gpu_shared",    "GPU v3 shared memory", "#e87ba4", "D"),
    ("gpu_shared_r4", "GPU v4 shared + radix 2²", "#008300", "P"),
    ("cufft",         "cuFFT",                "#6250d6", "X"),
]
LABEL = {k: l for k, l, _, _ in SERIES}
COLOR = {k: c for k, _, c, _ in SERIES}
MARK = {k: m for k, _, _, m in SERIES}
GPU = ["gpu_basic", "gpu_twiddle", "gpu_shared", "gpu_shared_r4", "cufft"]


def pow2_axis(ax, ns):
    lg = sorted(set(int(v) for v in ns))
    step = 2 if len(lg) > 8 else 1
    ticks = lg[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$2^{{{t}}}$" for t in ticks])
    ax.set_xlabel("Transform size N")


def line(ax, df, impl, y, **kw):
    d = df[df.impl == impl].sort_values("log2n")
    d = d[d[y].notna()]
    if d.empty:
        return
    ax.plot(d.log2n, d[y], color=COLOR[impl], marker=MARK[impl], label=LABEL[impl],
            markeredgecolor="#fcfcfb", markeredgewidth=0.8, **kw)


def save(fig, out, name):
    fig.tight_layout()
    path = os.path.join(out, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def plot_bench(df, out):
    present = [k for k, *_ in SERIES if k in set(df.impl)]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k in present:
        line(ax, df, k, "kernel_ms")
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_title("Compute time vs transform size")
    pow2_axis(ax, df.log2n)
    ax.legend(ncol=2)
    save(fig, out, "fig1_time_kernel.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k in ["cpu_fft"] + GPU:
        if k in present:
            line(ax, df, k, "e2e_ms")
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_title("End to end time incl. host↔device transfers")
    pow2_axis(ax, df.log2n)
    ax.legend(ncol=2)
    save(fig, out, "fig2_time_e2e.png")

    cpu = df[df.impl == "cpu_fft"].set_index("log2n")["kernel_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, title in [(axes[0], "kernel_ms", "Compute only"), (axes[1], "e2e_ms", "Including transfers")]:
        for k in GPU:
            d = df[df.impl == k].set_index("log2n")[col]
            sp = (cpu / d).dropna()
            if sp.empty:
                continue
            ax.plot(sp.index, sp.values, color=COLOR[k], marker=MARK[k], label=LABEL[k],
                    markeredgecolor="#fcfcfb", markeredgewidth=0.8)
        ax.axhline(1.0, color=AXIS, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_title(f"Speedup over sequential CPU FFT: {title.lower()}")
        pow2_axis(ax, df.log2n)
    axes[0].set_ylabel("Speedup (×, log scale)")
    axes[0].legend()
    save(fig, out, "fig3_speedup.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k in ["cpu_fft"] + GPU:
        if k in present:
            line(ax, df, k, "gflops_kernel")
    ax.set_yscale("log")
    ax.set_ylabel("GFLOP/s (5 N log₂N / time, log scale)")
    ax.set_title("Achieved throughput")
    pow2_axis(ax, df.log2n)
    ax.legend(ncol=2)
    save(fig, out, "fig4_gflops.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for k in present:
        line(ax, df, k, "rel_err")
    ax.set_yscale("log")
    ax.set_ylabel("Relative L2 error vs double precision")
    ax.set_title("Numerical accuracy")
    pow2_axis(ax, df.log2n)
    ax.legend(ncol=2)
    save(fig, out, "fig5_error.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for k in ["gpu_shared_r4", "cufft", "gpu_basic"]:
        d = df[df.impl == k].sort_values("log2n")
        if d.empty:
            continue
        share = 100.0 * (d.e2e_ms - d.kernel_ms) / d.e2e_ms
        ax.plot(d.log2n, share.clip(lower=0, upper=100), color=COLOR[k], marker=MARK[k], label=LABEL[k],
                markeredgecolor="#fcfcfb", markeredgewidth=0.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("(e2e − compute) / e2e  (%)")
    ax.set_title("Share of end to end time not spent computing")
    pow2_axis(ax, df.log2n)
    ax.legend()
    ax = axes[1]
    for k, lab, c, m in [("transfer_pinned", "Pinned host memory", "#2a78d6", "o"),
                         ("transfer_pageable", "Pageable host memory", "#eb6834", "s")]:
        d = df[df.impl == k].sort_values("log2n")
        if d.empty:
            continue
        gbps = (d.n * 8 * 2) / ((d.h2d_ms + d.d2h_ms) * 1e-3) / 1e9
        ax.plot(d.log2n, gbps, color=c, marker=m, label=lab, markeredgecolor="#fcfcfb", markeredgewidth=0.8)
    ax.set_ylabel("Effective PCIe bandwidth (GB/s)")
    ax.set_title("Copy bandwidth: pinned vs pageable")
    pow2_axis(ax, df.log2n)
    ax.legend()
    save(fig, out, "fig6_transfer.png")

    nmax = df[df.impl.isin(GPU)].log2n.max()
    d = df[(df.log2n == nmax) & df.impl.isin(GPU)].set_index("impl").reindex(GPU).dropna(subset=["kernel_ms"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ypos = range(len(d))
    ax.barh(list(ypos), d.kernel_ms, color=[COLOR[k] for k in d.index], height=0.6,
            edgecolor="#fcfcfb", linewidth=2)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([LABEL[k] for k in d.index], color=INK2)
    ax.invert_yaxis()
    base = d.kernel_ms.iloc[0]
    for i, (k, v) in enumerate(d.kernel_ms.items()):
        ax.text(v, i, f"  {v:.3f} ms  ({base / v:.2f}× vs v1)", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("Compute time (ms)")
    ax.set_xlim(0, d.kernel_ms.max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Effect of each optimisation at N = 2^{int(nmax)}")
    save(fig, out, "fig7_ablation.png")


def plot_sweep(sw, out):
    th = sw[sw.sweep == "threads"]
    ns = sorted(th.log2n.unique())
    if ns:
        fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 4.2), squeeze=False)
        for ax, lg in zip(axes[0], ns):
            for k in ["gpu_basic", "gpu_twiddle", "gpu_shared_r4"]:
                d = th[(th.impl == k) & (th.log2n == lg)].sort_values("threads")
                if d.empty:
                    continue
                ax.plot(d.threads, d.kernel_ms, color=COLOR[k], marker=MARK[k], label=LABEL[k],
                        markeredgecolor="#fcfcfb", markeredgewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks([32, 64, 128, 256, 512, 1024])
            ax.set_xticklabels(["32", "64", "128", "256", "512", "1024"])
            ax.set_xlabel("Threads per block")
            ax.set_title(f"N = 2^{int(lg)}")
        axes[0][0].set_ylabel("Compute time (ms)")
        axes[0][0].legend()
        fig.suptitle("Block size sweep for the global stage kernels", fontweight="bold", color=INK)
        save(fig, out, "fig8_threads.png")

    tl = sw[sw.sweep == "tile"]
    ns = sorted(tl.log2n.unique())
    if ns:
        fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 4.2), squeeze=False)
        for ax, lg in zip(axes[0], ns):
            for k in ["gpu_shared", "gpu_shared_r4"]:
                d = tl[(tl.impl == k) & (tl.log2n == lg)]
                if d.empty:
                    continue
                best = d.groupby("logB").kernel_ms.min().sort_index()
                ax.plot(best.index, best.values, color=COLOR[k], marker=MARK[k], label=LABEL[k],
                        markeredgecolor="#fcfcfb", markeredgewidth=0.8)
            ax.set_xticks(range(6, 13))
            ax.set_xticklabels([f"$2^{{{b}}}$" for b in range(6, 13)])
            ax.set_xlabel("Shared memory tile size")
            ax.set_title(f"N = 2^{int(lg)}")
        axes[0][0].set_ylabel("Compute time (ms, best block size)")
        axes[0][0].legend()
        fig.suptitle("Shared memory tile size sweep", fontweight="bold", color=INK)
        save(fig, out, "fig9_tile.png")


def summary(df, sw, out):
    lines = ["# Benchmark summary", ""]
    cpu = df[df.impl == "cpu_fft"].set_index("log2n")
    rows = []
    for lg in sorted(df.log2n.unique()):
        r = {"N": f"2^{int(lg)}"}
        c = cpu.kernel_ms.get(lg, float("nan"))
        r["CPU FFT ms"] = c
        for k in GPU:
            d = df[(df.impl == k) & (df.log2n == lg)]
            if not d.empty:
                r[f"{k} ms"] = d.kernel_ms.iloc[0]
                r[f"{k} speedup"] = c / d.kernel_ms.iloc[0]
        rows.append(r)
    t = pd.DataFrame(rows)
    lines += ["## Compute time and speedup over the sequential CPU FFT", "", t.to_markdown(index=False, floatfmt=".4g"), ""]

    e = df[df.impl.isin(GPU + ["cpu_fft"])].pivot_table(index="log2n", columns="impl", values="rel_err")
    lines += ["## Relative L2 error", "", e.to_markdown(floatfmt=".2e"), ""]

    if sw is not None and not sw.empty:
        best = sw.loc[sw.groupby(["sweep", "impl", "log2n"]).kernel_ms.idxmin()]
        lines += ["## Best configuration found by the sweep", "",
                  best[["sweep", "impl", "log2n", "threads", "logB", "shared_threads", "kernel_ms", "gflops"]]
                  .to_markdown(index=False, floatfmt=".4g"), ""]
    path = os.path.join(out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("wrote", path)


def cmd_plot(a):
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.bench)
    plot_bench(df, a.out)
    sw = None
    if a.sweep and os.path.exists(a.sweep):
        sw = pd.read_csv(a.sweep)
        plot_sweep(sw, a.out)
    try:
        summary(df, sw, a.out)
    except ImportError:
        print("(install 'tabulate' for summary.md: pip install --user tabulate)")


def to_float_mono(x):
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    elif x.dtype.kind == "u":
        x = (x.astype(np.float32) - 128.0) / 128.0
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def synthetic(sr, seconds, seed=0):
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(seed)
    music = (0.30 * np.sin(2 * np.pi * 220 * t)
             + 0.20 * np.sin(2 * np.pi * 277.18 * t)
             + 0.20 * np.sin(2 * np.pi * 329.63 * t)
             + 0.10 * np.sin(2 * np.pi * 440 * t))
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t) ** 2
    whistle = 0.35 * np.sin(2 * np.pi * 3000 * t)  # the unwanted tone
    noise = 0.02 * rng.standard_normal(t.size)
    return (envelope * music + whistle + noise).astype(np.float32)


def cmd_audio_prep(a):
    os.makedirs(a.out, exist_ok=True)

    if a.wav:
        parts, sr = [], None
        for path in a.wav:
            r, x = wavfile.read(path)
            if sr is not None and r != sr:
                raise SystemExit(f"{path}: sample rate {r} Hz differs from {sr} Hz")
            sr = r
            parts += [to_float_mono(x), np.zeros(int(0.15 * r), np.float32)]
        x = np.concatenate(parts[:-1])[: int(sr * a.seconds)]
        src = a.wav[0] if len(a.wav) == 1 else f"{len(a.wav)} files joined ({a.wav[0]} ...)"
    else:
        sr, x = a.sr, synthetic(a.sr, a.seconds)
        src = "synthetic (chord + 3 kHz whistle + noise)"

    peak = float(np.max(np.abs(x))) or 1.0
    x = (0.9 * x / peak).astype(np.float32)

    x.tofile(os.path.join(a.out, "input.f32"))
    with open(os.path.join(a.out, "sr.txt"), "w") as f:
        f.write(f"{sr}\n")
    wavfile.write(os.path.join(a.out, "input.wav"), sr, x)
    print(f"source: {src}\nsamples: {x.size}  sr: {sr} Hz  duration: {x.size / sr:.2f} s")
    print(f"wrote {a.out}/input.f32, input.wav, sr.txt")


def cmd_ecg_to_wav(a):
    import wfdb
    if a.dir:
        rec = wfdb.rdrecord(os.path.join(a.dir, a.record), channels=[a.channel])
    else:
        rec = wfdb.rdrecord(a.record, pn_dir="mitdb", channels=[a.channel])
    x = np.nan_to_num(rec.p_signal[:, 0]).astype(np.float32)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    wavfile.write(a.out, int(rec.fs), x)
    print(f"record {a.record} lead {rec.sig_name[0]}: {x.size} samples at {rec.fs} Hz "
          f"({x.size / rec.fs / 60:.1f} min), wrote {a.out}")


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            k, v = line.split(maxsplit=1)
            meta[k] = v.strip()
    return meta


def db(m):
    return 20 * np.log10(np.maximum(m, 1e-9))


def cmd_audio_plot(a):
    os.makedirs(a.out, exist_ok=True)

    meta = read_meta(os.path.join(a.dir, "meta.txt"))
    sr, n = float(meta["sr"]), int(meta["n"])
    x = np.fromfile(os.path.join(a.dir, "input.f32"), dtype=np.float32)
    y = np.fromfile(os.path.join(a.dir, "output.f32"), dtype=np.float32)
    s0 = np.fromfile(os.path.join(a.dir, "spectrum_before.f32"), dtype=np.float32)
    s1 = np.fromfile(os.path.join(a.dir, "spectrum_after.f32"), dtype=np.float32)

    ref = np.abs(np.fft.fft(x.astype(np.float64), n))[: n // 2 + 1]
    err = np.linalg.norm(s0 - ref) / np.linalg.norm(ref)
    print(f"GPU magnitude spectrum vs numpy.fft: relative L2 error = {err:.3e}")

    wavfile.write(os.path.join(a.dir, "output.wav"), int(sr), np.clip(y, -1, 1).astype(np.float32))
    print(f"wrote {a.dir}/output.wav")

    freqs = np.arange(n // 2 + 1) * sr / n
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    if sr < 2000:
        w0, w1 = int(1 * sr), int(11 * sr)
        t, unit, wtitle = np.arange(w0, w1) / sr, "s", "10 s window"
    else:
        mid = int(np.argmax(np.abs(x)))
        w0 = max(0, mid - int(0.015 * sr))
        w1 = min(x.size, w0 + int(0.03 * sr))
        t, unit, wtitle = np.arange(w0, w1) / sr * 1000, "ms", "30 ms window around the loudest part"
    ax = axes[0]
    ax.plot(t, x[w0:w1], color=BEFORE, linewidth=1.4, label="Original")
    ax.plot(t, y[w0:w1], color=AFTER, linewidth=1.4, label="Filtered (GPU inverse FFT)")
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform, {wtitle}")
    lo_y = min(x[w0:w1].min(), y[w0:w1].min())
    hi_y = max(x[w0:w1].max(), y[w0:w1].max())
    ax.set_ylim(lo_y - 0.05 * (hi_y - lo_y), hi_y + 0.6 * (hi_y - lo_y))  # room for the legend
    ax.legend(loc="upper right", ncol=2)

    fmin = max(freqs[1], sr / 2000)
    label = f"{meta['filter']} {float(meta['f1']):g}" + (
        f"–{float(meta['f2']):g} Hz" if meta["filter"] in ("bandstop", "bandpass") else " Hz")
    for ax, spec, col, title in [(axes[1], s0, BEFORE, "Magnitude spectrum: original"),
                                 (axes[2], s1, AFTER, f"Magnitude spectrum: after {label} filter")]:
        ax.plot(freqs, db(spec), color=col, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(fmin, sr / 2)
        ax.set_ylim(max(db(s0).max() - 120, -60), db(s0).max() + 10)
        ax.set_xlabel("Frequency (Hz, log scale)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(title)
    if meta["filter"] in ("bandstop", "bandpass", "lowpass", "highpass"):
        f1, f2 = float(meta["f1"]), float(meta["f2"])
        span = {"bandstop": (f1, f2), "bandpass": (f1, f2), "lowpass": (f1, sr / 2), "highpass": (fmin, f1)}
        lo, hi = span[meta["filter"]]
        if meta["filter"] == "bandpass":
            for (l, h) in [(fmin, f1), (f2, sr / 2)]:
                axes[2].axvspan(l, h, color=GRID, alpha=0.6, linewidth=0)
        else:
            axes[2].axvspan(lo, hi, color=GRID, alpha=0.6, linewidth=0)
        axes[2].text(0.01, 0.92, "shaded: removed band", transform=axes[2].transAxes, color=INK2, fontsize=9)

    fig.suptitle(f"{a.title}: GPU FFT filtering ({meta['impl']}, N = {n:,}, {sr:g} Hz)  "
                 f"forward {float(meta['forward_ms']):.3f} ms, inverse {float(meta['inverse_ms']):.3f} ms",
                 color=INK, fontsize=11)
    fig.tight_layout()
    path = os.path.join(a.out, a.name + ".png")
    fig.savefig(path, dpi=200)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description="GPU FFT project tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plot", help="figures and summary from bench.csv / sweep.csv")
    p.add_argument("--bench", default="results/bench.csv")
    p.add_argument("--sweep", default="results/sweep.csv")
    p.add_argument("--out", default="results/figs")
    p.set_defaults(fn=cmd_plot)

    p = sub.add_parser("audio-prep", help="write input.f32 for ./fft_project audio")
    p.add_argument("--wav", nargs="+", help="input WAV file(s), joined in order (stereo is averaged to mono)")
    p.add_argument("--synthetic", action="store_true", help="generate the built in test signal")
    p.add_argument("--sr", type=int, default=44100, help="sample rate for --synthetic")
    p.add_argument("--seconds", type=float, default=4.0, help="duration (synthetic) or max length (wav)")
    p.add_argument("--out", default="results/audio")
    p.set_defaults(fn=cmd_audio_prep)

    p = sub.add_parser("audio-plot", help="verify and plot the GPU audio demo")
    p.add_argument("--dir", default="results/audio")
    p.add_argument("--out", default="results/figs")
    p.add_argument("--name", default="audio_demo", help="figure file name (without .png)")
    p.add_argument("--title", default="Synthetic audio")
    p.set_defaults(fn=cmd_audio_plot)

    p = sub.add_parser("ecg-to-wav", help="convert an MIT-BIH ECG record to WAV (needs: pip install wfdb)")
    p.add_argument("--record", default="100")
    p.add_argument("--dir", help="folder with the record files (.hea/.dat); default downloads from PhysioNet")
    p.add_argument("--channel", type=int, default=0, help="0 = MLII lead")
    p.add_argument("--out", default="data/ecg_100.wav")
    p.set_defaults(fn=cmd_ecg_to_wav)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
