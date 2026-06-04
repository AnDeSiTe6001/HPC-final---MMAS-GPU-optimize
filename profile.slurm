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

set -euo pipefail

module purge
module load cuda/12.8
module load gcc10/10.2.1

# Nsight 工具版本相依：cuda/12.3 不含 nsys/ncu 本體（nsys command not found、
# ncu 只有 wrapper）。改用 cuda/12.8 後 nsys 與 ncu 都內建可用，故這裡載 12.8。
# 若改回較舊的 cuda module，下面的 command -v / ncu --version 偵測會自動略過該段。

echo "[$(date)] Building..."
make clean
make -j4 || { echo "Build failed."; exit 1; }

# ── profiling 目標 ────────────────────────────────────────────
INSTANCE=ALL_tsp/pla33810.tsp     # 改成 pla85900 需先修 OOM
ALG=mmas_rwm_bt_cl                  # 與預設一致；_cl 需 block-warps*32 == cand-list-size
# --iter 不放共用 ARGS：docopt 不接受重複 flag，nsys/ncu 各自帶一次自己的 iter 數。
COMMON_ARGS="--instance=${INSTANCE} --alg=${ALG} --trials=1 --seed=42"
NSYS_ITER=50      # nsys 輕量，可跑完整 iters 看時間軸
NCU_ITER=5        # ncu 每 kernel replay 多次很慢，iter 設很小
TAG="${ALG}_$(basename ${INSTANCE} .tsp)"
mkdir -p profiles

nvidia-smi

# ── 1) Nsight Systems：整體時間軸 / kernel 佔比 / H2D-D2H 傳輸 / API ──
#     輕量，可跑完整 50 iters，先看「哪個 kernel 最花時間、有無傳輸瓶頸」
if command -v nsys >/dev/null 2>&1; then
    echo "[$(date)] nsys profiling..."
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --output=profiles/${TAG} \
        ./mmas ${COMMON_ARGS} --iter=${NSYS_ITER}
    # 產出：profiles/${TAG}.nsys-rep
    nsys stats profiles/${TAG}.nsys-rep > profiles/${TAG}_nsys_summary.txt 2>&1 || true
else
    echo "[$(date)] WARNING: nsys (Nsight Systems) 不可用，略過時間軸分析。"
    echo "  目前的 cuda module 未附帶 nsys。確認已 module load cuda/12.8（內建 nsys），"
    echo "  或 module spider nsight-systems 找獨立 module 後重跑。"
fi

# ── 2) Nsight Compute：單一 kernel 的細部指標（記憶體吞吐 / 佔用率 / 瓶頸）──
#     ncu 很慢（每個 kernel replay 多次），務必限制 iter 與要量測的 kernel。
#     --launch-count 取前幾次啟動即可；--set full 會收完整指標。
# 先確認 ncu 真的能用（本叢集 cuda/12.3 可能只有 wrapper、沒有本體）。
# `ncu --version` 在缺少本體時會印出 nsight-compute not found 並回傳非 0。
if command -v ncu >/dev/null 2>&1 && ncu --version >/dev/null 2>&1; then
    echo "[$(date)] ncu profiling (build_ant_solution kernel)..."
    ncu \
        --set full \
        --launch-count 3 \
        --kernel-name-base demangled \
        --kernel-name "regex:build_ant_solution" \
        --force-overwrite \
        --export profiles/${TAG}_build \
        ./mmas ${COMMON_ARGS} --iter=${NCU_ITER}
    # 產出：profiles/${TAG}_build.ncu-rep
else
    echo "[$(date)] WARNING: ncu (Nsight Compute) 不可用，略過 kernel 細部分析。"
    echo "  目前的 cuda module 只有 ncu wrapper、未裝本體。確認已 module load cuda/12.8"
    echo "  （內建 ncu），或 module spider nsight-compute 找獨立 module 後重跑。"
    echo "  nsys 報告不受影響。"
fi

# 如需量測 2-opt local search，把上面 --kernel-name 換成 "regex:two_opt_nn"。

echo "[$(date)] Done. 報告在 profiles/：*.nsys-rep / *.ncu-rep"
ls -lh profiles/
