# Hybrid MPI + OpenMP Conjugate Gradients for the 3D Poisson equation

DSA5208 project. A matrix-free Conjugate Gradient (CG) solver for

    -Δu = f  on (0,1)^3,  u = 0 on the boundary,

discretised with the 7-point finite-difference stencil on an N³ grid. The project has a serial
reference code and one hybrid MPI+OpenMP code in which every parallelisation decision can be switched
at run time, so you can measure what each choice does to scaling:

| switch | choices | what it changes |
|---|---|---|
| `-d` | `1`, `2`, `3` | slab / pencil / block domain decomposition (MPI Cartesian communicator) |
| `-c` | `blocking`, `overlap` | `MPI_Sendrecv` halo, or `Isend/Irecv` overlapped with the interior stencil |
| `-a` | `cg`, `cg1`, `pipecg` | classic CG (2 Allreduce/iter), Chronopoulos–Gear (1 Allreduce), pipelined CG (1 Iallreduce hidden behind the SpMV) |
| `OMP_NUM_THREADS` | T | OpenMP threads per MPI rank |

The report is `report/report.pdf`. Authors: Wu Yuhan, Zuo Dewei, Zhu Chen. Licence: MIT (`LICENSE`).

## Quick start: from source to report

Run these from the project root, in this order.

```bash
# 0. (containers running as root with Open MPI only)
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# 1. build
make                                      # -> bin/cg_serial  bin/cg_mpi  bin/stream_omp

# 2. a first run: serial and 4 MPI ranks on a 128^3 grid
#    (Open MPI: add --oversubscribe if the machine has fewer than 4 cores)
./bin/cg_serial -n 128
OMP_NUM_THREADS=1 mpirun -np 4 ./bin/cg_mpi -n 128

# 3. correctness (parallel vs serial, edge cases, O(h^2) study, residual histories)
bash scripts/validate.sh full             # -> results/validation/summary.txt  (RESULT: PASS)

# 4. halo-volume measurement for 1D/2D/3D decompositions (machine independent)
bash scripts/comm_volume.sh               # -> results/commvol.csv

# 5. benchmarks on this machine
MACHINE=local NS=256 NW=224 REPS=5 bash scripts/run_suite.sh    # -> results/local/

# 6. figures, tables and the PDF report from those results
make report RESULTS=results/local         # -> figures/*.pdf, report/report.pdf

# 7. on Vanda / Atlas (see the section below): benchmark job, then rebuild the report
qsub pbs/vanda_suite.pbs                  # -> results/vanda/
make report RESULTS=results/vanda
```

## Layout

```
src/cg_serial.c        serial reference solver (classic CG)
src/cg_mpi.c           hybrid MPI+OpenMP solver (all variants)
src/common.h           problem definition (manufactured solution), CSV format
src/stream_omp.c       STREAM-triad bandwidth probe (for the roofline/performance model)
scripts/validate.sh    correctness suite (parallel vs serial, edge cases, O(h^2) study, residual histories)
scripts/check_validation.py   PASS/FAIL evaluation of the validation CSVs
scripts/run_suite.sh   benchmark suite (strong, weak, variants, hybrid, time-to-solution, STREAM)
scripts/comm_volume.sh halo volume per iteration for 1D/2D/3D decompositions
scripts/make_plots.py  figures + LaTeX tables + number macros for the report
pbs/vanda_suite.pbs    PBS job: build, quick validation, full benchmark suite on Vanda
pbs/atlas_suite.pbs    same for Atlas (queue parallel24)
pbs/validate.pbs       full validation as a PBS job
results/               CSV results (local/, validation/, commvol.csv)
figures/, report/      generated figures, report source and PDF
```

## Requirements

* a C11 compiler with OpenMP (GCC ≥ 9; tested with 13.3), reached through the MPI wrapper
  `mpicc`: all three programs are built with it, so the serial baseline and the parallel code
  always use the same compiler
* an MPI-3 library (`MPI_Iallreduce`); tested with Open MPI 4.1.6. Intel MPI and MPICH launch
  lines are also supported by `scripts/run_suite.sh` (`MPI_FLAVOR=intelmpi|mpich`)
* Python ≥ 3.6 with `pandas` and `matplotlib` ≥ 3.3 for the plots
  (`pip install --user pandas matplotlib`), `pdflatex` for the report. The validation scripts
  need only the Python standard library.

## Build

```bash
make                       # -> bin/cg_serial  bin/cg_mpi  bin/stream_omp
make OPT="-O3 -march=x86-64-v3"   # if login and compute nodes have different CPUs
make BIN=bin_vanda         # build into another directory (the PBS scripts do this)
```

## Run

```bash
# serial reference, 128^3 grid, relative residual 1e-8
./bin/cg_serial -n 128

# 8 MPI ranks x 1 thread, 3D blocks, overlapped halo, classic CG
OMP_NUM_THREADS=1 mpirun -np 8 ./bin/cg_mpi -n 256 -d 3 -c overlap -a cg

# 2 ranks x 4 threads, pipelined CG, pinned
mpirun -np 2 --map-by ppr:2:node:PE=4 --bind-to core \
       -x OMP_NUM_THREADS=4 -x OMP_PLACES=cores -x OMP_PROC_BIND=close \
       ./bin/cg_mpi -n 256 -a pipecg

# benchmark mode: exactly 100 iterations (tol 0), print only the CSV line
mpirun -np 4 ./bin/cg_mpi -n 256 -t 0 -m 100 -q

# weak scaling: ~128^3 points per rank
mpirun -np 8 ./bin/cg_mpi -L 128 -t 0 -m 50
```

All options (`./bin/cg_mpi -h`):

```
-n N        global interior grid N^3 (strong scaling)            [128]
-L n        ~n^3 points per rank: global N = round(n*P^(1/3)) (weak scaling)
-a ALG      cg | cg1 | pipecg                                    [cg]
-c COMM     blocking | overlap                                   [overlap]
-d D        decomposition dimensionality 1 | 2 | 3               [3]
-t TOL      relative residual tolerance (0 = run maxit iterations) [1e-8]
-m MAXIT    maximum iterations                                   [50N+100]
-k CHUNKS   interior chunks between MPI progress polls           [4]
-o FILE     write solution (raw float64, C order) with MPI-IO
-r FILE     write residual history (recursive and true) to FILE
-e EVERY    history every EVERY iterations                       [1]
-q          quiet: only print the CSV line
-H          print CSV header and exit
```

Every run prints one line starting with `CSV,` (columns: `./bin/cg_mpi -H`): configuration,
iterations, recursive and true residual, max/L2 error against the exact solution, time split into
stencil / halo / reduction / vector work (max over ranks), halo bytes and messages per iteration.

> Running as root inside a container with Open MPI needs
> `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.

## Validate

```bash
make test                          # quick: ~30 s -> results/validation_quick/
bash scripts/validate.sh full      # 180 variant runs + 30 edge cases + grid study (~3 min on 2 cores)
cat results/validation/summary.txt
```

Checks: every parallel configuration reproduces the serial solution point-wise
(≤ 1e-10 relative for cg/cg1, ≤ 1e-8 for pipecg; observed 4e-15 and 2e-12), same iteration count,
true residual ≤ 1.5·tol, and the discretisation error decreases as O(h²) (observed order 2.000)
in serial and in parallel. Ranks may exceed the number of cores here (Open MPI `--oversubscribe`
is added automatically).

## Benchmarks on Atlas / Vanda (NUS HPC, PBS Pro)

1. Copy the project to the cluster (e.g. `scp -r dsa5208-cg-poisson <nusnet>@vanda.nus.edu.sg:~`).
2. Check the queue and module names: `hpc pbs help`, `qstat -Q`, `module avail 2>&1 | grep -i mpi`,
   and edit the `#PBS -q`, `#PBS -P` and `module load` lines of `pbs/vanda_suite.pbs` (or
   `pbs/atlas_suite.pbs`). The `select=` line sets the number of nodes; the scripts adapt to it.
3. `qsub pbs/vanda_suite.pbs` (from the project root). The job requests whole nodes
   (`place=scatter:excl`; delete that line if the queue rejects it), rebuilds into `bin_vanda/`
   against the loaded MPI, runs the quick validation and then the full suite: STREAM, serial
   baseline, strong scaling on P = 1, 2, 4, …, 32, 36 (one socket), 64, 72 (one node), 144,
   decomposition/halo/algorithm variants (P ≥ 4), hybrid splits, weak scaling and time to
   solution. Default sizes: N = 512 (134 M unknowns); 192³ points per rank for weak
   scaling (larger than the L3 cache, so the one-process baseline is not cache-inflated);
   50 iterations per timing run; 3 repetitions. Expect 1–3 h on 2 nodes.
4. Regenerate the report from the cluster data:
   ```bash
   make report RESULTS=results/vanda      # needs pandas, matplotlib, pdflatex
   ```
   Figures, tables and every number in the text are rebuilt from the CSVs, and the Setup
   paragraph of Section 5 switches from the development machine to the cluster description
   automatically. Then re-read Section 5 and adjust the interpretation where the cluster data
   tells a different story (the paragraphs specific to the 2-core machine are dropped).

Knobs of `scripts/run_suite.sh` (environment variables): `MACHINE`, `NS`, `NW`, `ITERS`, `REPS`,
`PLIST`, `VARIANT_PMIN`, `EXPERIMENTS`, `MPI_FLAVOR`, `MPIRUN`, `BIN`, `APPEND` (see the header of
the script). Every run starts a new `results/$MACHINE/bench.csv` and keeps the previous one as
`bench.csv.bak.<date>`; set `APPEND=1` to add to the existing file instead (e.g. to re-run one
experiment with `EXPERIMENTS=weak`).

**What is timed.** `t_iter` is the time per iteration of the steady-state CG loop. Work vectors
are allocated and first-touched before the clock starts, and the one-off setup SpMV of `cg1` and
`pipecg` is counted in the time to solution (`t_solve`) but not in `t_iter`, so the three
variants are compared iteration against iteration.

## Results included

* `results/vanda/` — the measurements in the report: 36 cores of one NUS Vanda compute node
  (2 × Intel Xeon Gold 6438N, 32 cores per socket), Open MPI 4.1.8, GCC 13.1.0, produced by
  `pbs/vanda36.pbs` (free queue `auto_free`; on Vanda load `gcc/13.1.0 openmpi/gcc/64/4.1.8` and set
  `OMPI_MCA_btl_vader_single_copy_mechanism=none`). `make report` rebuilds the PDF from them.
* `results/local/` — the same suite on the 2-core development machine (`make report RESULTS=results/local`).
* `results/validation/` — the full correctness suite; `results/commvol.csv` — halo volumes.
