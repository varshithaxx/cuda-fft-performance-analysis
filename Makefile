# CUDA 11.3 does not accept the default GCC 12 on Bhavani, so use the system g++ (8.5)
NVCC  ?= nvcc
CCBIN ?= /usr/bin/g++
ARCH  ?= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
EXTRA ?=

SRC = fft_project.cu common.cuh cpu_fft.cuh gpu_basic.cuh gpu_optimized.cuh bench_audio.cuh

fft_project: $(SRC)
	$(NVCC) -O3 -std=c++14 -ccbin $(CCBIN) $(ARCH) -lineinfo $(EXTRA) fft_project.cu -o fft_project -lcufft
	@mkdir -p results/audio results/figs

clean:
	rm -f fft_project

.PHONY: clean
