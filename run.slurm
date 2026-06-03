#!/bin/bash
#SBATCH --job-name=mmas
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00            # 最長 1 小時
#SBATCH --output=mmas_%j.out
#SBATCH --error=mmas_%j.err
#SBATCH --account=ACD115050


# ── 環境設定 ──────────────────────────────────────────────
module purge
module load cuda/12.3
module load gcc10/10.2.1

# ── 編譯 ─────────────────────────────────────────────────
echo "[$(date)] Building..."
make clean
make -j4
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

# ── 確認 GPU ─────────────────────────────────────────────
echo "[$(date)] GPU Info:"
nvidia-smi

# ── 確認 GCC ─────────────────────────────────────────────
echo "Gcc version check:"
gcc --version

# ── 執行 ─────────────────────────────────────────────────
# 請將 --instance 改為你的 .tsp 檔路徑
INSTANCE=ALL_tsp/att48.tsp

echo "[$(date)] Running MMAS on: $INSTANCE"

# 用 cuda-compute-sanitizer 偵測 GPU 記憶體錯誤（比直接執行慢但輸出詳細錯誤）
./mmas --instance=ALL_tsp/att48.tsp

echo "[$(date)] Done."