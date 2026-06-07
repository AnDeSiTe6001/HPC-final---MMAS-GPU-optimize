#!/bin/bash
#SBATCH --job-name=mmas-ldgexp
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=ldgexp_%j.out
#SBATCH --error=ldgexp_%j.err
#SBATCH --account=ACD115050

# ─────────────────────────────────────────────────────────────
# 對照實驗：__ldg 到底有沒有讓 pla85900 的 gap 從 30% 進步到 5%?
#
# 背景：0a1150f(有 __ldg)實測 ~5%,而 7a6c2a7(無 __ldg)及更早都 ~30%,
# 兩邊都 --ants=0、--iter 預設。但 __ldg 是唯讀快取「提示」,理論上不改數值,
# 所以 30%→5% 只可能是：
#   A) 種子變異 —— 預設 seed=0=時間隨機,30% 是早熟停滯吸引子,ldg 那次抽到好種子。
#   B) 潛在記憶體 bug 被 __ldg 擾動而意外改變表現(Heisenbug)。
# 判別：兩 binary 只差 src/mmas.cu。固定同組 seed,比同 seed 下的 gap。
#
# ★ 此版為「可續跑 + 自動續投」設計,配合 1 小時硬上限(完整 run 約 2000s,一個 job
#   通常只跑得了 1 個 run):
#   - binary 會快取(mmas_noldg / mmas_ldg 已存在就不重建)。
#   - 每個 (variant, seed) 跑完寫進對照表,用唯一鍵 [tag:seed] 判斷已完成並跳過
#     (先前無法跳過是因為 printf 對齊空白讓 grep 比對失敗,已修)。
#   - 逼近 55 分鐘就停止啟動新 run、乾淨收尾。
#   - AUTO_RESUBMIT=1:結束時若還沒跑完會自動再投遞下一個 job,直到全部完成(有安全上限)。
#   => 你只要 sbatch 一次,其餘自動接力;不想自動就把 AUTO_RESUBMIT 設 0、自己重複 sbatch。
# ─────────────────────────────────────────────────────────────

set -uo pipefail   # 不用 -e：單一 run 失敗不該中斷整批

module purge
module load cuda/12.8
module load gcc10/10.2.1

INSTANCE=ALL_tsp/pla85900.tsp
SEEDS="42 7 123"               # 要更快收斂判斷可先留 "42"
LDG_COMMIT=0a1150f             # 有 __ldg
NOLDG_COMMIT=7a6c2a7           # 無 __ldg（= dir 1b）
COMMON="--instance=${INSTANCE} --alg=mmas_rwm_bt_cl --ants=0 --trials=1"
# --iter 不指定 = 預設 1000，與你拿到 5%/30% 的那兩次一致。
RESULTS_DIR=results/ldg_seed_experiment
SUMMARY=ldgexp_summary.txt
DEADLINE=3300                  # 55 分鐘(秒)。$SECONDS 超過就不再啟動新 run。
FIRST_RUN_GUESS=2300           # 還沒測過 run 時長時的保守預留(秒)。完整 run 約 2000s,
                               # 留些 buffer;太大會讓建置後那個 job 不敢啟動 run。
AUTO_RESUBMIT=1                # 1=本 job 結束時若還沒跑完就自動再投遞一個 job 續跑(免手動)。
                               # 設 0 則改回手動:自己重複 sbatch。

mkdir -p "${RESULTS_DIR}"

# ── 1) 建兩個 binary(快取：已存在就跳過)────────────────────
SRC=src/mmas.cu
build_variant() {  # $1=commit  $2=輸出 binary
    local commit=$1 outbin=$2
    if [ -x "${outbin}" ]; then
        echo "[$(date)] ${outbin} 已存在,跳過建置。"; return 0
    fi
    echo "[$(date)] 建置 ${outbin}（取出 ${commit}:${SRC}）..."
    cp "${SRC}" /tmp/mmas.cu.orig.$$            # 備份工作樹原始碼
    git show "${commit}:${SRC}" > "${SRC}" || { cp /tmp/mmas.cu.orig.$$ "${SRC}"; return 1; }
    make clean >/dev/null 2>&1
    local ok=1
    make -j4 >"build_${outbin}.log" 2>&1 || ok=0
    cp /tmp/mmas.cu.orig.$$ "${SRC}"           # 立刻復原原始碼
    rm -f /tmp/mmas.cu.orig.$$
    if [ "${ok}" -ne 1 ]; then echo "[$(date)] 建置 ${outbin} 失敗,見 build_${outbin}.log"; return 1; fi
    cp mmas "${outbin}"
    echo "[$(date)] OK -> ${outbin}"
}

nvidia-smi
build_variant "${NOLDG_COMMIT}" mmas_noldg || exit 1
build_variant "${LDG_COMMIT}"   mmas_ldg   || exit 1

# ── 2) 對照表表頭(只在第一次建立)──────────────────────────
if [ ! -f "${SUMMARY}" ]; then
    {
        echo "# __ldg 對照實驗  $(date)"
        echo "# ${COMMON} --iter=預設(1000)   noldg=${NOLDG_COMMIT} ldg=${LDG_COMMIT}"
        echo "# 同 seed 下兩列穩定不同→真 bug(B);noldg 某些 seed 也很低→種子變異(A)"
        echo "#--------------------------------------------------------------"
    } > "${SUMMARY}"
fi

parse_gap() {  # $1=logfile -> "cost pct"
    local line; line=$(grep -m1 "Final solution cost:" "$1" 2>/dev/null)
    if [ -z "$line" ]; then echo "NA NA"; return; fi
    echo "$(echo "$line" | sed -E 's/.*cost:[[:space:]]*([0-9]+).*/\1/') $(echo "$line" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')"
}

already_done() {  # $1=tag $2=seed ; 成功記錄過就回傳 0
    # 新格式:行尾唯一鍵 [tag:seed],用 grep -F 固定字串比對(不受 printf 對齊空白影響)。
    grep -qF "[${1}:${2}]" "${SUMMARY}" 2>/dev/null && return 0
    # 相容舊格式:行首 tag 與 seed= 之間可能有多個空白(這就是先前無法跳過的 bug)。
    grep -qE "^${1}[[:space:]]+seed=${2}[[:space:]].*gap=[0-9]" "${SUMMARY}" 2>/dev/null
}

run_one() {  # $1=binary $2=tag $3=seed
    local bin=$1 tag=$2 seed=$3
    if already_done "${tag}" "${seed}"; then
        echo "[$(date)] ${tag} seed=${seed} 已完成,跳過。"; return 0
    fi
    local log="run_${tag}_seed${seed}.log"
    echo "[$(date)] RUN ${tag} seed=${seed} ..."
    ./"${bin}" ${COMMON} --seed=${seed} --results-dir="${RESULTS_DIR}" >"${log}" 2>&1
    local res cost pct; res=$(parse_gap "${log}"); cost=${res% *}; pct=${res#* }
    if [ "${pct}" = "NA" ]; then
        echo "[$(date)] ⚠ ${tag} seed=${seed} 沒有結果(可能被 timeout 砍/執行錯誤),不記錄,下次續跑會重試。log: ${log}"
        return 1
    fi
    # 行尾唯一鍵 [tag:seed] 供 already_done 用 grep -F 比對,杜絕對齊空白問題。
    printf "%-6s seed=%-4s cost=%-12s gap=%s%%   [%s:%s]\n" \
        "${tag}" "${seed}" "${cost}" "${pct}" "${tag}" "${seed}" | tee -a "${SUMMARY}"
}

# ── 3) 依序跑,逼近 DEADLINE 就停(讓未跑的留給下次 sbatch)──
LAST=0   # 上一次 run 花的秒數,用來預判下一次跑不跑得完
for s in ${SEEDS}; do
    for v in noldg ldg; do
        bin="mmas_${v}"
        already_done "${v}" "${s}" && { echo "[skip] ${v} seed=${s}"; continue; }
        # 預判：估計這個 run 需要的時間,若(已用 + 估計 + 60s buffer)會超過 DEADLINE 就收工。
        # 第一個 run 還沒有實測值,用保守的 FIRST_RUN_GUESS;之後用上一個 run 的實測 LAST。
        est=${LAST}; [ "${est}" -eq 0 ] && est=${FIRST_RUN_GUESS}
        if [ $(( SECONDS + est + 60 )) -gt "${DEADLINE}" ]; then
            echo "[$(date)] 時間將不足(已用 ${SECONDS}s,預估下個 run ${est}s),停止啟動新 run。"
            echo "  尚未完成的請『再次 sbatch 同一支』續跑(binary 已快取,不會重建)。"
            break 2
        fi
        t0=${SECONDS}
        run_one "${bin}" "${v}" "${s}"
        LAST=$(( SECONDS - t0 ))
        echo "[$(date)] 該 run 花了 ${LAST}s。"
    done
done

echo "[$(date)] 本次結束。目前對照表:"
cat "${SUMMARY}"

# 是否全部完成?(算「有 gap 數字」的列)
DONE=$(grep -cE "seed=.*gap=[0-9]" "${SUMMARY}" 2>/dev/null || echo 0)
NEED=$(( $(echo ${SEEDS} | wc -w) * 2 ))
MAX_ATTEMPTS=$(( NEED * 2 + 3 ))   # 自動續投的安全上限,避免某 run 一直失敗時無限投遞
echo "[$(date)] 進度 ${DONE}/${NEED}。"

if [ "${DONE}" -ge "${NEED}" ]; then
    echo "[$(date)] 全部完成 ✅"
    rm -f ldgexp_attempts
else
    echo "[$(date)] 尚未完成。"
    if [ "${AUTO_RESUBMIT}" = "1" ]; then
        att=$(( $(cat ldgexp_attempts 2>/dev/null || echo 0) + 1 ))
        echo "${att}" > ldgexp_attempts
        if [ "${att}" -le "${MAX_ATTEMPTS}" ] && command -v sbatch >/dev/null 2>&1; then
            echo "[$(date)] 自動再投遞續跑(第 ${att}/${MAX_ATTEMPTS} 次)。binary 已快取、已完成的會跳過。"
            sbatch experiment_ldg_seed.slurm || echo "[$(date)] sbatch 自動續投失敗,請手動再 sbatch。"
        else
            echo "[$(date)] 不自動續投(達上限 ${MAX_ATTEMPTS} 或無 sbatch)。請手動再 sbatch 續跑。"
        fi
    else
        echo "[$(date)] AUTO_RESUBMIT=0 → 請手動再 sbatch 一次續跑。"
    fi
fi
