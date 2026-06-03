# GPU-based MMAS 參數說明

## 基本用法

```bash
./mmas --instance=<path> [options]
```

---

## 必填參數

### `--instance=<path>`
TSPLIB 格式的 `.tsp` 檔路徑（支援 EUC_2D 與 EXPLICIT 格式）。

---

## 演算法選擇

### `--alg=<str>` ｜ 預設：`mmas_rwm_bt_cl`

演算法名稱由三個部分組成：

```
mmas_{選蟻方式}_{Tabu結構}[_cl]
```

| 代號 | 意義 |
|------|------|
| `rwm` | Roulette Wheel Method（輪盤選擇） |
| `wrs` | Weighted Reservoir Sampling |
| `lc`  | CompressedList Tabu |
| `ct`  | Compact Tabu（記憶體用量約 lc 的一半） |
| `bt`  | Bitmask Tabu（每節點 1 bit） |
| `_cl` | 啟用 Candidate List（候選清單加速） |

完整可選值（共 12 種）：

```
不含候選清單：mmas_rwm_lc, mmas_rwm_ct, mmas_rwm_bt
              mmas_wrs_lc, mmas_wrs_ct, mmas_wrs_bt

含候選清單：  mmas_rwm_lc_cl, mmas_rwm_ct_cl, mmas_rwm_bt_cl
              mmas_wrs_lc_cl, mmas_wrs_ct_cl, mmas_wrs_bt_cl
```

> **注意**：使用含 `_cl` 的演算法時，`--block-warps × 32` 必須等於 `--cand-list-size`。

---

## MMAS 核心參數

### `--ants=<n>` ｜ 預設：`100`
螞蟻數量。設為 `0` 時自動等於 TSP 節點數（即每個節點一隻螞蟻）。

### `--iter=<n>` ｜ 預設：`1000`
每次 trial 的迭代次數。設為 `0` 時自動計算為恰好執行 100 萬條解所需的迭代數（`ceil(1000000 / ants)`）。

### `--trials=<n>` ｜ 預設：`1`
獨立重複執行的次數，每次使用不同亂數序列。

### `--rho=<n>` ｜ 預設：`0.9`
費洛蒙保留率。更新公式：`τ ← rho × τ + Δτ`，因此 `1 - rho` 為揮發率。
- `rho=0.9` → 每輪揮發 10%
- 越小費洛蒙揮發越快，探索性越強

### `--cand-list-size=<n>` ｜ 預設：`32`
候選清單（Candidate List）大小，**必須是 32 的倍數**。僅在使用 `_cl` 系列演算法時有效。

### `--seed=<n>` ｜ 預設：`0`
亂數種子。`0` 表示使用系統時間（每次結果不同），設定固定值可重現實驗。

---

## GPU 效能調整參數

### `--block-warps=<n>` ｜ 預設：`1`
解建構 kernel 每個 block 的 warp 數（1 warp = 32 threads）。
使用 `_cl` 演算法時，`block-warps × 32` 必須等於 `cand-list-size`。

### `--ls=<n>` ｜ 預設：`1`
Local Search 演算法：
- `0`：不使用 Local Search
- `1`：使用 GPU 平行化 2-opt

### `--ls-block-warps=<n>` ｜ 預設：`0`
Local Search kernel 每個 block 的 warp 數。`0` 表示使用啟發式自動計算（約 `ceil(dimension / (10 × 32))`，上限 32）。

---

## 輸出設定

### `--results-dir=<path>` ｜ 預設：`results`
結果 JSON 檔的存放目錄。不存在時會自動建立。輸出檔名格式：
```
<alg>-<instance_name>_<日期>_<時間>.json
```

---

## 常用組合範例

```bash
# 基本執行（預設參數）
./mmas --instance=ALL_tsp/pr1002.tsp

# 指定演算法 + 多 trial + 固定種子
./mmas --instance=ALL_tsp/pr1002.tsp \
       --alg=mmas_wrs_bt_cl \
       --trials=5 \
       --iter=500 \
       --seed=42 \
       --results-dir=output

# 停用 Local Search（較快但解品質較低）
./mmas --instance=ALL_tsp/pr1002.tsp \
       --ls=0 \
       --iter=2000

# 迭代至 100 萬條解
./mmas --instance=ALL_tsp/pr1002.tsp \
       --iter=0
```

---

## 參數限制摘要

| 限制 | 說明 |
|------|------|
| `_cl` 演算法 | `block-warps × 32 == cand-list-size` |
| `cand-list-size` | 必須為 32 的倍數 |
| `ls` | 只接受 `0` 或 `1` |
