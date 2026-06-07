#!/bin/bash
#SBATCH --job-name=mmas-lsrace
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=lsrace_%j.out
#SBATCH --error=lsrace_%j.err
#SBATCH --account=ACD115050

# ─────────────────────────────────────────────────────────────
# 確認:pla85900 卡 30% 的根因是 2-opt LS 的跨 warp race。
#
# two_opt_nn 在多 warp 時用 atomicMax(&chosen_move_gain) 後「分開」寫
# chosen_move_thread_id —— 兩步非原子,多 warp 交錯會選錯移動。
# ls_warps_per_block = clamp(ceil(n/320),1,32):
#   小實例 → 1 warp(無 race,正常);pla85900 → 32 warp(race 爆發)。
#
# 決定性測試:強制 LS 單 warp(--ls-block-warps=1)→ race 不可能發生。
# bug 在第 0 代就現形,故 --iter=1 即可判:
#   預設         第 0 代 ~108%(重現 bug)
#   --ls-block-warps=1   第 0 代應降到 ~11%(證實就是這個 race)
# ─────────────────────────────────────────────────────────────

set -uo pipefail
module purge
module load cuda/12.8
module load gcc10/10.2.1

INSTANCE=ALL_tsp/pla85900.tsp
COMMON="--instance=${INSTANCE} --alg=mmas_rwm_bt_cl --ants=0 --trials=1 --seed=42 --iter=1"

# 用先前實驗已建好的 noldg binary;沒有就現建(= 7a6c2a7 src)。
if [ ! -x mmas_noldg ]; then
    echo "[$(date)] mmas_noldg 不存在,從 7a6c2a7 建置..."
    cp src/mmas.cu /tmp/mmas.cu.orig.$$
    git show 7a6c2a7:src/mmas.cu > src/mmas.cu
    make clean >/dev/null 2>&1
    make -j4 || { cp /tmp/mmas.cu.orig.$$ src/mmas.cu; echo "build failed"; exit 1; }
    cp /tmp/mmas.cu.orig.$$ src/mmas.cu; rm -f /tmp/mmas.cu.orig.$$
    cp mmas mmas_noldg
fi

nvidia-smi

echo "================================================================"
echo "[A] noldg 預設(ls_warps_per_block 啟發式 = 32 for pla85900)"
echo "================================================================"
./mmas_noldg ${COMMON} 2>&1 | grep -E "ants_count|ls-block-warps|^0:|Final solution cost"

echo
echo "================================================================"
echo "[B] noldg + --ls-block-warps=1(強制單 warp,race 不可能發生)"
echo "================================================================"
./mmas_noldg ${COMMON} --ls-block-warps=1 2>&1 | grep -E "ants_count|ls-block-warps|^0:|Final solution cost"

echo
echo "[$(date)] 判讀:若 [B] 的第 0 代 / Final 從 ~108% 掉到 ~11% → 證實 root cause"
echo "          就是 2-opt LS 的跨 warp race(與 __ldg 無關)。"
