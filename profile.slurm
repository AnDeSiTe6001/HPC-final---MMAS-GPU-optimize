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
# pla85900 的 OOM 已修好（見 docs/OOM_FIX_pla85900.md），現直接 profiling 它。
# COMMON_ARGS 帶 --ants=0 以啟用「自動填滿 GPU」的 ant 數（方向 2），用來量測
# occupancy 是否從 ~1.95% 提升；若要對照舊行為，把 --ants=0 拿掉即用預設 100。
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
INSTANCE=ALL_tsp/pla85900.tsp
ALG=mmas_rwm_bt_cl                  # 與預設一致；_cl 需 block-warps*32 == cand-list-size
# 實驗（方向 5 評估）：加大候選清單，看 build kernel 的 fallback（候選近鄰用盡時
# 的 O(n) 全域掃描，是 FP32/散讀 gather 的真正來源）是否變少、總時間是否改善。
# 限制：block-warps*32 必須 == cand-list-size，且 cand-list-size 是 32 的倍數。
#   - 基準：CAND_LIST_SIZE=32 BLOCK_WARPS=1（dispatch 走 warp 變體）
#   - 實驗：CAND_LIST_SIZE=64 BLOCK_WARPS=2（dispatch 自動改走 block 變體，支援 >32 候選）
# 注意：2-opt LS kernel 寫死 <=32 鄰居（release 下 assert 被 -DNDEBUG 編掉，不會 abort，
# 但 LS 只會用前 32 個鄰居）。build kernel 會用滿全部 cand-list-size。
CAND_LIST_SIZE=64
BLOCK_WARPS=2
# --iter 不放共用 ARGS：docopt 不接受重複 flag，nsys/ncu 各自帶一次自己的 iter 數。
# --ants=0：自動填滿 GPU 的 ant 數（方向 2）。stderr 會印 "auto-sized to fill GPU"。
COMMON_ARGS="--instance=${INSTANCE} --alg=${ALG} --trials=1 --seed=42 --ants=0 --cand-list-size=${CAND_LIST_SIZE} --block-warps=${BLOCK_WARPS}"
NSYS_ITER=50      # nsys 輕量，可跑完整 iters 看時間軸
NCU_ITER=5        # ncu 每 kernel replay 多次很慢，iter 設很小
TAG="${ALG}_$(basename ${INSTANCE} .tsp)_cl${CAND_LIST_SIZE}"
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

    # ── 2b) compute / memory bound 判讀 ───────────────────────────
    #   ncu --import 只讀剛產生的報告、不需要 GPU。抽出 Speed-Of-Light 的
    #   SM(compute) / Memory 吞吐百分比，以及 achieved occupancy，存成易讀的
    #   摘要並印出一句結論。判讀規則：
    #     SM 高、Memory 低                 → compute bound
    #     Memory 高、SM 低                 → memory bound
    #     兩者都低（且 occupancy 也低）     → latency / occupancy bound
    REP="profiles/${TAG}_build.ncu-rep"
    BOUND="profiles/${TAG}_build_bound.txt"
    echo "[$(date)] 判讀 compute / memory bound..."
    {
        echo "# Speed-Of-Light 摘要：$REP"
        echo "# 取第一個 build kernel launch 的數值"
        echo
        ncu --import "$REP" --page details \
            | grep -E "Compute \(SM\) Throughput|Memory Throughput|Achieved Occupancy|DRAM Throughput|Duration" \
            | head -n 20
    } > "$BOUND" 2>&1 || echo "（details 解析失敗，改看下方 CSV）" >> "$BOUND"

    # 機器可讀的 CSV（值穩定、好後續比較）
    ncu --import "$REP" --csv --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
        >> "$BOUND" 2>&1 || true

    # 用 SM / Memory 吞吐 + occupancy 給一句白話結論。
    # 直接解析上面 details 區段的人類可讀行（每行一個 metric，空白分隔，值在 $NF）——
    # 比解析 --csv 穩：CSV 是「一列一個 metric」的長格式，欄位數因 kernel 名稱含空白/逗號
    # 而不固定。取每個 metric 第一次出現的值（= 第一個 kernel launch）。
    awk '
        function num(v){ gsub(/[^0-9.]/,"",v); return v+0 }
        $0 ~ /Compute \(SM\) Throughput/ && sm=="" { sm=num($NF) }
        $0 ~ /Memory Throughput/ && $(NF-1)=="%" && mem=="" { mem=num($NF) }
        $0 ~ /Achieved Occupancy/ && occ=="" { occ=num($NF) }
        END {
            if (sm=="" && mem=="") { print "（無法解析 SOL 數值，請看上方 details 區段）"; exit }
            print "";
            printf "SM(compute) 吞吐: %.1f%%   Memory 吞吐: %.1f%%   Achieved occupancy: %.1f%%\n", sm, mem, occ;
            if (sm>=60 && sm>mem)        verdict="COMPUTE bound：瓶頸在算術，優先看 fallback 的即時 pow()/sqrt()。";
            else if (mem>=60 && mem>sm)  verdict="MEMORY bound：瓶頸在記憶體吞吐，試 const __restrict__ / __ldg、對齊 gather。";
            else if (occ<50)             verdict="LATENCY / OCCUPANCY bound：SM 多在閒置，瓶頸是相依的 n 步迴圈 + warp 太少（occupancy 低），需從演算法平行度下手。";
            else                         verdict="混合：SM 與 Memory 都未吃滿，先補 occupancy 再看次要瓶頸。";
            print "結論 → " verdict;
        }
    ' "$BOUND" | tee -a "$BOUND"
    echo "[$(date)] 判讀結果：$BOUND"
else
    echo "[$(date)] WARNING: ncu (Nsight Compute) 不可用，略過 kernel 細部分析。"
    echo "  目前的 cuda module 只有 ncu wrapper、未裝本體。確認已 module load cuda/12.8"
    echo "  （內建 ncu），或 module spider nsight-compute 找獨立 module 後重跑。"
    echo "  nsys 報告不受影響。"
fi

# 如需量測 2-opt local search，把上面 --kernel-name 換成 "regex:two_opt_nn"。

echo "[$(date)] Done. 報告在 profiles/：*.nsys-rep / *.ncu-rep"
ls -lh profiles/
