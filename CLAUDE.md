# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CUDA implementation of the GPU-based MAX-MIN Ant System (MMAS) for the TSP, from
Skinderowicz, "Implementing a GPU-based parallel MAX-MIN Ant System" (FGCS). It reads
TSPLIB instances, runs the metaheuristic on the GPU, and writes a JSON results file.

This is a **Linux + CUDA** project. The development machine here is Windows, but the code
is built and run on a SLURM GPU cluster (see `run.slurm`) — do not expect `make`/`nvcc` to
work locally on Windows.

## Build & run

```bash
make                 # release build -> ./mmas executable
make mode=debug      # -G -O0 device debug build
make clean
```

Before building you **must** set the target GPU architecture in the `Makefile`: edit
`COMPUTE_CAPABILITY` to one of `$(CC_KEPLER/MAXWELL/PASCAL/VOLTA)`. Building for the wrong
arch produces an executable that silently fails to launch kernels.

On the cluster, submit via SLURM (loads `cuda/12.3` + `gcc10`, builds, runs):

```bash
sbatch run.slurm     # edit INSTANCE= inside to pick the .tsp file
```

Run directly:

```bash
./mmas --instance=ALL_tsp/pr1002.tsp --alg=mmas_wrs_bt_cl
./mmas --help        # full docopt-generated parameter list
```

There is **no test suite**. Correctness is judged by solution cost vs. the optimum, printed
each run as a percentage gap (using `best-known.json`). `--seed=<n>` gives reproducible runs.

Full parameter documentation (in Traditional Chinese) lives in `PARAMETERS.md`.

## Algorithm naming and the key launch constraint

`--alg` names follow `mmas_{select}_{tabu}[_cl]`:
- select: `rwm` (Roulette Wheel) | `wrs` (Weighted Reservoir Sampling)
- tabu:   `lc` (CompressedListTabu) | `ct` (CompactTabu, ~½ the memory) | `bt` (BitmaskTabu)
- `_cl`:  use candidate (nearest-neighbor) lists to speed up node selection

**Critical constraint for `_cl` variants:** `--block-warps × 32` must equal `--cand-list-size`,
and `--cand-list-size` must be a multiple of 32. The program `abort()`s otherwise. This is
because the solution-build kernel maps one warp-lane per candidate-list slot.

## Architecture

Almost all the real logic is in **`src/mmas.cu`** (~3100 lines). The rest is plumbing.

Control flow: `main.cc` parses args (docopt) and loads best-known costs →
`run_mmas_experiment` (bottom of `mmas.cu`) loads the instance, selects an algorithm variant,
and loops over trials → `run_gpu_based_mmas` does all device allocation and the iteration loop.

The algorithm-variant selection is **template-based dispatch**, not virtual calls or runtime
branching in kernels. Each of the 12 variants is a `SolutionConstructionAlgorithm` struct
holding a function pointer to a `build_ant_solution[_using_cand_lists]<Tabu, ChoiceFn>`
template instantiation. The `--alg` string maps to one of these structs in a long if/else
chain near the end of `run_mmas_experiment`. To add a variant you instantiate the template
with a tabu type + choice function and wire it into that chain.

Per-iteration GPU pipeline (in `run_gpu_based_mmas`, each step a kernel launch):
1. `update_pheromone_heuristic_product_cache` — recompute τ^α·η^β cache
2. `update_cand_lists_pheromone_heuristic_product_cache` — same, for candidate lists (cl only)
3. `alg.build_ant_sol_` — each block builds one ant's tour
4. `two_opt_nn` — GPU 2-opt local search with don't-look bits (when `--ls=1`)
5. `update_global_best_and_trail_limits` — track best, recompute MMAS τ_min/τ_max
6. `evaporate_pheromone`
7. `deposit_pheromone` — source solution depends on LS: iteration-best (no LS) vs. a
   randomized mix of iter/reset/global best (with LS)

Pheromone is reset on stagnation (no improvement for `stagnation_period` iters) when LS is on.

Memory note: uses a **full n² pheromone matrix** plus distance, heuristic, and product-cache
matrices, so GPU memory is the hard limit on instance size (≈900 MB for ~7400 cities).

### Supporting files
- `tsp.{h,cc}` — `ProblemInstance` (coords, distance matrix, NN lists), TSPLIB parsing,
  distance functions (`EUC_2D`, `EXPLICIT`, `GEO`, `ATT`, `CEIL_2D`).
- `kd_tree.h` — k-d tree used to build nearest-neighbor lists fast for 2D instances.
- `utils.{h,cc}` — datetime, path creation, results-filename helpers.
- `docopt.*` — vendored CLI parser; the `USAGE` string in `main.cc` is the source of truth
  for parameters.
- `json.hpp` — vendored nlohmann/json for results output.
- `ALL_tsp/` — sample TSPLIB instances. Results are written to `--results-dir` (default `results/`).

## Device helpers

- `device_vector<T>` (in `mmas.cu`) — RAII wrapper over `cudaMalloc`/`cudaMemcpy`; use
  `.as_host_vector()` to read results back. Prefer it over raw CUDA calls for new buffers.
- `CUDA_CHECK(...)` macro wraps CUDA calls with error checking — wrap all new CUDA API calls.
- `Timer` / `GPUTimer` structs accumulate per-stage timings reported at the end of a run.
