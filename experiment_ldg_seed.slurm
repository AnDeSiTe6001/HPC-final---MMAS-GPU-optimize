#!/bin/bash
#SBATCH --job-name=mmas-ldgexp
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=ldgexp_%j.out
#SBATCH --error=ldgexp_%j.err
#SBATCH --account=ACD115050

# ─────────────────────────────────────────────────────────────
# 對照實驗：__ldg 到底有沒有讓 pla85900 的 gap 從 30% 進步到 5%?
#
# 背景：0a1150f(有 __ldg)實測得到 ~5%,而它之前的 commit(7a6c2a7,無 __ldg)
# 都是 ~30%,兩邊都用 --ants=0、--iter 預設。但 __ldg 是唯讀資料的快取「提示」,
# 理論上不改數值,所以 30%→5% 只可能是：
#   A) 種子變異 —— run.slurm 沒指定 --seed,預設 seed=0=時間隨機,每次種子不同。
#   B) 潛在記憶體 bug 被 __ldg 擾動而意外改變表現(Heisenbug)。
#
# 判別法：兩個 binary 只差 src/mmas.cu(7a6c2a7 vs 0a1150f)。固定同一組 seed,
# 兩邊各跑一次,比同一個 seed 下的 gap：
#   - 同 seed 下「有 ldg=低、無 ldg=高」穩定重現 → B（真 bug，改善可靠，去抓 bug）
#   - 無 ldg 在某些 seed 也很低 / 有 ldg 在某些 seed 也很高 → A（純種子變異，__ldg 無關）
#
# 注意：6 次完整跑可能很久。若 partition 不給 12h，請把 SEEDS 砍成 2 個、或先跑 1 個
# seed 看趨勢。結果是「邊跑邊寫」,中途被 timeout 砍掉也保有已完成的列。
# ─────────────────────────────────────────────────────────────

set -uo pipefail   # 不用 -e：單一 run 失敗不該中斷整批

module purge
module load cuda/12.8
module load gcc10/10.2.1

INSTANCE=ALL_tsp/pla85900.tsp
SEEDS="42 7 123"               # 要省時間就改成 "42" 或 "42 7"
LDG_COMMIT=0a1150f             # 有 __ldg
NOLDG_COMMIT=7a6c2a7           # 無 __ldg（= dir 1b）
COMMON="--instance=${INSTANCE} --alg=mmas_rwm_bt_cl --ants=0 --trials=1"
# --iter 不指定 = 預設 1000，與你拿到 5%/30% 的那兩次一致。
RESULTS_DIR=results/ldg_seed_experiment
SUMMARY=ldgexp_summary.txt

mkdir -p "${RESULTS_DIR}"

# 還原 working tree 的 src/mmas.cu —— 不管 job 怎麼結束都要復原,避免污染原始碼。
SRC=src/mmas.cu
cp "${SRC}" /tmp/mmas.cu.orig.$$
restore_src() {
    cp /tmp/mmas.cu.orig.$$ "${SRC}" && echo "[$(date)] restored ${SRC}"
    rm -f /tmp/mmas.cu.orig.$$
}
trap restore_src EXIT

build_variant() {
    # $1 = commit, $2 = 輸出 binary 名稱
    local commit=$1 outbin=$2
    echo "[$(date)] 取出 ${commit}:${SRC} 並建置 -> ${outbin}"
    git show "${commit}:${SRC}" > "${SRC}" || { echo "git show ${commit} 失敗"; return 1; }
    make clean >/dev/null 2>&1
    if ! make -j4 >"build_${outbin}.log" 2>&1; then
        echo "[$(date)] 建置 ${outbin} 失敗,見 build_${outbin}.log"; return 1
    fi
    cp mmas "${outbin}"
    echo "[$(date)] OK -> ${outbin}"
}

nvidia-smi

# ── 1) 建兩個 binary ────────────────────────────────────────
build_variant "${NOLDG_COMMIT}" mmas_noldg || exit 1
build_variant "${LDG_COMMIT}"   mmas_ldg   || exit 1
restore_src                       # 立刻復原原始碼,後面只用已建好的 binary
trap - EXIT

# ── 2) 跑對照並解析 gap ─────────────────────────────────────
# 程式每次結束印： "Final solution cost: <cost> (<pct>%)"
parse_gap() {  # $1=logfile -> 印 "cost pct"
    local line
    line=$(grep -m1 "Final solution cost:" "$1" 2>/dev/null)
    if [ -z "$line" ]; then echo "NA NA"; return; fi
    local cost pct
    cost=$(echo "$line" | sed -E 's/.*cost:[[:space:]]*([0-9]+).*/\1/')
    pct=$(echo  "$line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    echo "${cost} ${pct}"
}

run_one() {  # $1=binary $2=tag $3=seed
    local bin=$1 tag=$2 seed=$3
    local log="run_${tag}_seed${seed}.log"
    echo "[$(date)] RUN ${tag} seed=${seed} ..."
    ./"${bin}" ${COMMON} --seed=${seed} --results-dir="${RESULTS_DIR}" >"${log}" 2>&1
    local res; res=$(parse_gap "${log}")
    local cost=${res% *} pct=${res#* }
    printf "%-8s | seed=%-4s | cost=%-12s | gap=%s%%\n" "${tag}" "${seed}" "${cost}" "${pct}" \
        | tee -a "${SUMMARY}"
}

: > "${SUMMARY}"
{
    echo "# __ldg 對照實驗  $(date)"
    echo "# instance=${INSTANCE}  ${COMMON} --iter=預設(1000)"
    echo "# noldg=${NOLDG_COMMIT}  ldg=${LDG_COMMIT}"
    echo "# 判別：同 seed 下兩列若穩定不同 → 真 bug(B)；若 noldg 某些 seed 也很低 → 種子變異(A)"
    echo "#--------------------------------------------------------------"
} | tee -a "${SUMMARY}"

for s in ${SEEDS}; do
    run_one mmas_noldg noldg "${s}"
    run_one mmas_ldg   ldg   "${s}"
    echo "#--------------------------------------------------------------" | tee -a "${SUMMARY}"
done

echo "[$(date)] 完成。對照表：${SUMMARY}"
cat "${SUMMARY}"
