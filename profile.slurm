#!/bin/bash
#SBATCH --job-name=mmas-prof
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=prof_%j.out
#SBATCH --error=prof_%j.err
#SBATCH --account=ACD115050

# ─────────────────────────────────────────────────────────────
# 產生 Nsight Systems (.nsys-rep) 與 Nsight Compute (.ncu-rep) 報告
#
# 注意：pla85900 目前會 OOM（見 PARAMETERS.md / OOM 分析），無法 profiling。
# 因此預設用「放得下 32GB V100 的最大實例」pla33810 來看 kernel 行為與
# 記憶體/時間瓶頸的 scaling。等 OOM 修好後，把 INSTANCE 改成 pla85900 再跑一次。
# ─────────────────────────────────────────────────────────────

module purge
module load cuda/12.3
module load gcc10/10.2.1

echo "[$(date)] Building..."
make clean
make -j4 || { echo "Build failed."; exit 1; }

# ── profiling 目標 ────────────────────────────────────────────
INSTANCE=ALL_tsp/pla33810.tsp     # 改成 pla85900 需先修 OOM
ALG=mmas_rwm_bt_cl                  # 與預設一致；_cl 需 block-warps*32 == cand-list-size
ARGS="--instance=${INSTANCE} --alg=${ALG} --iter=50 --trials=1 --seed=42"
TAG="${ALG}_$(basename ${INSTANCE} .tsp)"
mkdir -p profiles

nvidia-smi

# ── 1) Nsight Systems：整體時間軸 / kernel 佔比 / H2D-D2H 傳輸 / API ──
#     輕量，可跑完整 50 iters，先看「哪個 kernel 最花時間、有無傳輸瓶頸」
echo "[$(date)] nsys profiling..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output=profiles/${TAG} \
    ./mmas ${ARGS}
# 產出：profiles/${TAG}.nsys-rep
nsys stats profiles/${TAG}.nsys-rep > profiles/${TAG}_nsys_summary.txt 2>&1 || true

# ── 2) Nsight Compute：單一 kernel 的細部指標（記憶體吞吐 / 佔用率 / 瓶頸）──
#     ncu 很慢（每個 kernel replay 多次），務必限制 iter 與要量測的 kernel。
#     --launch-count 取前幾次啟動即可；--set full 會收完整指標。
echo "[$(date)] ncu profiling (build_ant_solution kernel)..."
ncu \
    --set full \
    --launch-count 3 \
    --kernel-name-base demangled \
    --kernel-name "regex:build_ant_solution" \
    --force-overwrite \
    --export profiles/${TAG}_build \
    ./mmas ${ARGS} --iter=5
# 產出：profiles/${TAG}_build.ncu-rep

# 如需量測 2-opt local search，把上面 --kernel-name 換成 "regex:two_opt_nn"。

echo "[$(date)] Done. 報告在 profiles/：*.nsys-rep / *.ncu-rep"
ls -lh profiles/
