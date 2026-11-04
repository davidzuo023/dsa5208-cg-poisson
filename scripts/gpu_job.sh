#!/usr/bin/env bash
# Build and benchmark the GPU code on a Vanda GPU node (batch: pbs/vanda_gpu.pbs;
# interactive: qsub -I -q auto_free -l select=1:ncpus=36:ngpus=1 -l walltime=01:00:00,
# then  cd <project dir> && bash scripts/gpu_job.sh).
cd "$(dirname "$0")/.."
module purge >/dev/null 2>&1
module load nvhpc-hpcx-cuda12/26.1 2>/dev/null || module load nvhpc/26.1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
nvidia-smi
echo "mpicc -> $(mpicc --version 2>&1 | head -2 | tail -1)"
BIN=bin_gpu; mkdir -p $BIN
# serial reference with the same compiler (host only)
nvc -O3 -fast -o $BIN/cg_serial src/cg_serial.c -lm || exit 1
make -B BIN=$BIN GPUCC=mpicc GPUFLAGS="-O3 -fast -mp=gpu -gpu=cc86 -Minfo=mp" $BIN/cg_gpu || exit 1
# smoke test: refuses to run on the host
OMP_TARGET_OFFLOAD=MANDATORY mpirun -np 1 $BIN/cg_gpu -n 64 || { echo "GPU run failed"; exit 1; }
MACHINE=vanda_gpu BIN=$BIN ITERS=50 REPS=3 GPU_SIZES="128 192 256 384 512 640 768" bash scripts/run_gpu.sh
