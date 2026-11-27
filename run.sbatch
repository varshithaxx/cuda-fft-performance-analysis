#!/bin/bash
#SBATCH -J fft_project
#SBATCH -o fft_%j.out
#SBATCH --partition=normal
#SBATCH --exclude=node004
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -t 01:45:00

# usage: sbatch run.sbatch [test|bench|sweep|audio|speech|ecg|realdata|plot|all]
# speech and ecg use the recordings in data/ (see README)
# node004 has L40s, the rest are A30s, so we skip it to keep results comparable

cd "$SLURM_SUBMIT_DIR"
type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh
module load cuda/11.3
set -e

STEP=${1:-all}
[ -x ./fft_project ] || { echo "fft_project not found: run make on the login node first"; exit 1; }
mkdir -p results/audio results/figs

echo "node: $(hostname)   job: $SLURM_JOB_ID   step: $STEP   started: $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

run_test()  { echo "===== GPU correctness tests ====="; ./fft_project test --maxlog 22; }
run_bench() { echo "===== benchmark ====="; ./fft_project bench --minlog 10 --maxlog 24 --dftmax 14 --reps 9 --out results/bench.csv; }
run_sweep() { echo "===== configuration sweep ====="; ./fft_project sweep --sweeplogs 16,20,24 --reps 9 --out results/sweep.csv; }
run_audio() {
    echo "===== audio demo ====="
    module load anaconda3/2022.10
    python tools.py audio-prep --synthetic
    ./fft_project audio --in results/audio/input.f32 --sr "$(cat results/audio/sr.txt)" \
        --filter bandstop --f1 2800 --f2 3200 --impl shared_r4 --outdir results/audio
    python tools.py audio-plot
}
# real recordings: same GPU filter, different data and settings
filter_run() {   # name, title, filter, f1, f2, wav files...
    local name=$1 title=$2 filter=$3 f1=$4 f2=$5; shift 5
    python tools.py audio-prep --wav "$@" --seconds 3600 --out results/$name
    ./fft_project audio --in results/$name/input.f32 --sr "$(cat results/$name/sr.txt)" \
        --filter $filter --f1 $f1 --f2 $f2 --impl shared_r4 --outdir results/$name
    python tools.py audio-plot --dir results/$name --name $name --title "$title"
}
run_speech() {
    echo "===== speech: Free Spoken Digit Dataset ====="
    module load anaconda3/2022.10
    filter_run speech "Speech (Free Spoken Digit Dataset)" bandpass 300 3400 data/speech/*.wav
}
run_ecg() {
    echo "===== ECG: MIT-BIH record 100 ====="
    module load anaconda3/2022.10
    filter_run ecg_baseline "ECG, MIT-BIH record 100" highpass 0.5 0 data/ecg_100.wav
    filter_run ecg_powerline "ECG, MIT-BIH record 100" bandstop 59 61 data/ecg_100.wav
}
run_realdata() {
    if [ -d data/speech ]; then run_speech; else echo "data/speech missing, skipping speech"; fi
    if [ -f data/ecg_100.wav ]; then run_ecg; else echo "data/ecg_100.wav missing, skipping ECG"; fi
}
run_plot() { echo "===== plots ====="; module load anaconda3/2022.10; python tools.py plot; }

case "$STEP" in
    test)     run_test ;;
    bench)    run_bench ;;
    sweep)    run_sweep ;;
    audio)    run_audio ;;
    speech)   run_speech ;;
    ecg)      run_ecg ;;
    realdata) run_realdata ;;
    plot)     run_plot ;;
    all)      run_test; run_bench; run_sweep; run_audio; run_realdata; run_plot ;;
    *)        echo "unknown step '$STEP' (use test, bench, sweep, audio, speech, ecg, realdata, plot or all)"; exit 1 ;;
esac
echo "finished: $(date)"
