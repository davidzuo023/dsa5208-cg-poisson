#!/usr/bin/env bash
# GPU benchmarks for bin/cg_gpu (MPI + OpenMP target offload, one GPU per MPI rank).
#   MACHINE=vanda_gpu bash scripts/run_gpu.sh
# Knobs: MACHINE, NGPU (GPUs in the job, default: number of lines in PBS_NODEFILE or 1),
#        GPU_SIZES, GSOLVE_N (time-to-solution grid, 512), ITERS, REPS, MPIRUN, BIN.   Output: results/$MACHINE/gpu.csv
set -euo pipefail
cd "$(dirname "$0")/.."
MACHINE=${MACHINE:-gpu}
BIN=${BIN:-bin}
ITERS=${ITERS:-50}
REPS=${REPS:-3}
MPIRUN=${MPIRUN:-mpirun}
GPU_SIZES=${GPU_SIZES:-"128 192 256 384 512 640 768"}
# GPUs granted to this job: PBS sets CUDA_VISIBLE_DEVICES; nvidia-smi may list the whole node
if [[ -z "${NGPU:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        NGPU=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
    else
        NGPU=$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)
    fi
    (( NGPU >= 1 )) || NGPU=1
fi
# never more ranks than devices the OpenMP runtime can actually see
NDEV=$("$BIN/cg_gpu" -n 8 2>&1 >/dev/null | sed -n 's/.*devices\/node=\([0-9]*\).*/\1/p')
if [[ -n $NDEV ]] && (( NDEV >= 1 && NDEV < NGPU )); then NGPU=$NDEV; fi
OUT=results/$MACHINE
mkdir -p "$OUT"
CSV=$OUT/gpu.csv
[[ -f $CSV ]] && mv "$CSV" "$CSV.bak.$(date +%Y%m%d-%H%M%S)"
echo "exp,rep,$("$BIN/cg_serial" -H)" > "$CSV"
export OMP_TARGET_OFFLOAD=${OMP_TARGET_OFFLOAD:-MANDATORY}   # fail instead of silently running on the host
csv() { grep '^CSV,' | sed 's/^CSV,//'; }
record() {
    local exp=$1 rep=$2 l n; shift 2
    l=$("$@" | csv) || true
    n=$(printf '%s' "$l" | grep -c '' || true)
    if [[ $n -eq 1 ]]; then echo "$exp,$rep,$l" | tee -a "$CSV"; else echo "WARNING: run failed: $*" >&2; fi
}
{
    echo "machine: $MACHINE   date: $(date)"
    echo "gpus: $NGPU   ITERS=$ITERS REPS=$REPS SIZES=$GPU_SIZES"
    hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
    lscpu 2>/dev/null | grep -E "Model name|Socket" || true
    $MPIRUN --version 2>&1 | head -1 || true
    ${MPICC:-mpicc} --version 2>&1 | head -1 || true
} > "$OUT/env.txt"
cat "$OUT/env.txt"

# 1. correctness on the GPU: compare with the serial solution point-wise
"$BIN/cg_serial" -n 40 -q -o /tmp/ser40.$$.bin > /dev/null
for P in $(seq 1 "$NGPU"); do for a in cg cg1; do
    l=$($MPIRUN -np "$P" "$BIN/cg_gpu" -n 40 -a $a -q -o /tmp/gpu40.$$.bin | csv)
    d=$(python3 scripts/compare_bin.py /tmp/gpu40.$$.bin /tmp/ser40.$$.bin)
    echo "validate,1,$l" | tee -a "$CSV"; echo "  max|x_gpu-x_ser|/max|x_ser| = $d (P=$P, $a)" | tee -a "$OUT/validation.txt"
done; done
rm -f /tmp/ser40.$$.bin /tmp/gpu40.$$.bin

# 2. one GPU, growing problem: time per iteration and effective bandwidth
for rep in $(seq 1 "$REPS"); do for N in $GPU_SIZES; do for a in cg cg1; do
    record gsize "$rep" $MPIRUN -np 1 "$BIN/cg_gpu" -n "$N" -a $a -t 0 -m "$ITERS" -q
done; done; done

# 3. several GPUs (if the job has them): strong (N=512) and weak (~384^3 per GPU) scaling
if (( NGPU > 1 )); then
    for rep in $(seq 1 "$REPS"); do for P in $(seq 1 "$NGPU"); do
        record gstrong "$rep" $MPIRUN -np "$P" "$BIN/cg_gpu" -n "${GSTRONG_N:-512}" -t 0 -m "$ITERS" -q
        record gweak "$rep" $MPIRUN -np "$P" "$BIN/cg_gpu" -L "${GWEAK_L:-384}" -t 0 -m "$ITERS" -q
    done; done
fi

# 4. complete solve on one GPU, same problem as the CPU time-to-solution table
for a in cg cg1; do record gsolve 1 $MPIRUN -np 1 "$BIN/cg_gpu" -n "${GSOLVE_N:-512}" -a $a -t 1e-8 -q; done
echo "done -> $CSV"
