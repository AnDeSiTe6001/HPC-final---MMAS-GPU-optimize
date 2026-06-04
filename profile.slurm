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

# 本叢集 cuda/12.3 不含 Nsight 工具本體：nsys 直接 command not found，
# ncu 只有 wrapper。兩者都得靠獨立 module。先查實際名稱：
#   module spider nsight-systems / module spider nsight-compute
#   module avail 2>&1 | grep -i nsight
# 查到後把下面換成完整名稱（如 nsight-systems/2023.4）。找不到就會自動略過該段。

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
if command -v nsys >/dev/null 2>&1; then
    echo "[$(date)] nsys profiling..."
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --output=profiles/${TAG} \
        ./mmas ${ARGS}
    # 產出：profiles/${TAG}.nsys-rep
    nsys stats profiles/${TAG}.nsys-rep > profiles/${TAG}_nsys_summary.txt 2>&1 || true
else
    echo "[$(date)] WARNING: nsys (Nsight Systems) 不可用，略過時間軸分析。"
    echo "  cuda/12.3 未附帶 nsys。請執行 module spider nsight-systems 找獨立 module，"
    echo "  在腳本上方 module load 區塊填入完整名稱後重跑。"
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
        ./mmas ${ARGS} --iter=5
    # 產出：profiles/${TAG}_build.ncu-rep
else
    echo "[$(date)] WARNING: ncu (Nsight Compute) 不可用，略過 kernel 細部分析。"
    echo "  cuda/12.3 只附帶 ncu wrapper、未裝本體。請執行下列指令找獨立 module："
    echo "    module spider nsight-compute"
    echo "    module avail 2>&1 | grep -i nsight"
    echo "  找到後在腳本上方 module load 區塊加入，再重跑即可。nsys 報告不受影響。"
fi

# 如需量測 2-opt local search，把上面 --kernel-name 換成 "regex:two_opt_nn"。

echo "[$(date)] Done. 報告在 profiles/：*.nsys-rep / *.ncu-rep"
ls -lh profiles/
