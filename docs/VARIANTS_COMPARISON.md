# 12 種 MMAS 變體差異對照

`--alg` 由三個正交維度組成：`mmas_{select}_{tabu}[_cl]`，共 `2 × 3 × 2 = 12` 種。本文件對照這 12 種的差異。行號對應 `src/mmas.cu`。

> 搭配閱讀：預設變體的逐行流程見 [`FLOW_mmas_rwm_bt_cl.md`](FLOW_mmas_rwm_bt_cl.md)。

---

## 0. 三個維度速覽

| 維度 | 代號 | 全名 | 核心差異 |
|------|------|------|----------|
| 選蟻 | `rwm` | Roulette Wheel Method | 前綴和輪盤；存正常 `τ·η` |
| 選蟻 | `wrs` | Weighted Reservoir Sampling | 單趟加權水塘抽樣；存 `τ·η` 的**倒數** |
| Tabu | `bt` | BitmaskTabu | 每節點 1 bit；查詢 O(1)、最省記憶體 |
| Tabu | `lc` | CompressedListTabu | 壓縮未拜訪清單；枚舉最快、最耗記憶體 |
| Tabu | `ct` | CompactTabu | 精簡清單；記憶體約 `lc` 一半 |
| 候選 | `_cl` | Candidate List | 每步只看 32 個最近鄰；不配置 `n×n` product cache |

---

## 1. 維度一：選蟻方式（`rwm` vs `wrs`）

| 項目 | `rwm` | `wrs` |
|------|-------|-------|
| 演算法 | 前綴和輪盤（prefix-sum roulette） | Efraimidis–Spirakis 加權水塘抽樣 |
| 非 `_cl` 函式 | `warp_roulette_choice` (`:1532`) / `block_roulette_choice` (`:1628`) | `reservoir_sampling_roulette_choice` (`:1483`) |
| `_cl` 函式 | `warp_roulette_choice_from_cand_list` (`:1221`) / `block_roulette_choice_from_cand_list` (`:1278`) | `reservoir_sampling_roulette_choice_from_cand_list` (`:1336`) |
| **warp/block 雙版本** | **有**（編譯期依 `threads_per_block==WARP_SIZE` 二選一） | **無**（單一實作通吃任何 block 大小） |
| `use_product_reciprocal_` | `false`（存 `τ·η`） | `true`（存 `1/(τ·η)`） |
| 抽樣方式 | 抽一個落點 `r·total`，找累積和越過點的節點 | 對每個候選算 key `= log2f(r)·product`，取最大 |
| 跨 thread 歸約 | `warp_scan`/`block_scan` + `warp_vote`/`block_vote` | `block_arg_max_atomics` (`:300`) |

### 為什麼 `wrs` 要存倒數

WRS 的 key 是 `log2f(r) · product`，而 `log2f(r) < 0`（因 `r∈(0,1]`）。要讓「權重大」的候選 key 大，必須讓 `product` 是 `1/(τ·η)`。因此快取更新 kernel（`update_*_product_cache`，`:830`/`:863`）收到 `calc_reciprocal=true` 時改存倒數（`mmas.cu:850` / `:880`）。`rwm` 直接拿 `τ·η` 做前綴和，故存正常值。

### warp 版 vs block 版（只影響 `rwm`）

`rwm` struct 在編譯期用三元判斷選 kernel（例：`mmas_rwm_bt_cl` 約 `:2822`）：

```cpp
(threads_per_block == WARP_SIZE)
    ? ...<Tabu, warp_roulette_choice[_from_cand_list]<Tabu>>     // 1 warp：純 __shfl，無 shared 往返
    : ...<Tabu, block_roulette_choice[_from_cand_list]<Tabu>>    // 多 warp：經 shared 做跨 warp scan/vote
```

- `_cl` 變體預設 `block-warps=1` ⇒ `threads_per_block=32` ⇒ 走 **warp 版**（最快）。
- 非 `_cl` 變體若 `--block-warps>1` 則走 block 版。
- `wrs` 沒有這個分支，一份程式碼處理所有 block 大小。

---

## 2. 維度二：Tabu 結構（`bt` / `lc` / `ct`）

三者都放在 shared memory，差在記憶體用量與「枚舉未拜訪節點」的效率。

| 項目 | `bt` BitmaskTabu (`:895`) | `lc` CompressedListTabu (`:1001` 附近) | `ct` CompactTabu (`:1093` 附近) |
|------|--------------------------|------------------------------|---------------------------|
| 表示法 | 每節點 1 bit | `unvisited_[]` + `indices_[]` 雙 uint16 陣列 | 單 uint16 陣列（自我索引） |
| shared 記憶體 | `ceil(n/32)·4` ≈ **n/8 bytes** | `2·n·uint16` = **4n bytes** | `n·uint16` = **2n bytes** |
| `is_visited` | 查 bit，O(1) | 查 `indices_[node] >= length_` | 查 `unvisited_[node] > node` |
| `get_length()` | 永遠 `dimension` | 隨拜訪遞減（剩餘未拜訪數） | 隨拜訪遞減 |
| `get_candidate(i)` | 回傳 `i`（**不壓縮**） | 回傳第 i 個未拜訪節點 | 回傳第 i 個未拜訪節點 |
| 枚舉全部未拜訪的成本 | 掃 **全部 n** 個再過濾 | 只掃 `length_` 個（緊密） | 只掃 `length_` 個（緊密） |
| `add_visited` | 設 bit + 延遲批次寫回 | swap-and-pop（與尾端對調） | swap-and-pop（自我索引維護） |
| 路徑寫回 | 延遲批次（register→global） | `parallel_finalize` 反向倒出 | 延遲批次（register→global） |

### 取捨重點

- **`bt`**：記憶體最省、`is_visited` 最快，但**沒有壓縮清單**——`get_candidate(i)==i`，所以「掃全部未拜訪」要遍歷 0..n-1。在非 `_cl` 變體裡，越接近 tour 尾端浪費越多（多數已拜訪節點仍被掃到再丟棄）。
- **`lc`**：`get_candidate(i)` 直接給第 i 個未拜訪節點，枚舉範圍 = 剩餘數，**枚舉最有效率**；代價是 `4n bytes` shared memory（三者最多），shared 吃緊或大實例會受限。
- **`ct`**：`lc` 的省記憶體版（`2n bytes`，約一半），保留「緊密清單」枚舉優勢；代價是 `add_visited` 邏輯較複雜（要分「移除尾端 / 非尾端」維護自我索引不變式，程式碼有大量 `assert`）。

### 與 `_cl` 的交互作用（很重要）

- **`_cl` 變體**：每步主要只在 **32 個最近鄰** 裡選（用 `is_visited` 過濾），枚舉成本本就只有 32 ⇒ **`bt` 的「枚舉慢」缺點幾乎不發作**，反而能享受它最省記憶體的好處。只有「32 個最近鄰全被拜訪」的回退路徑才會掃全部未拜訪節點（`bt` 此時掃全 n，`lc`/`ct` 掃 `length_`）。
- **非 `_cl` 變體**：每步都要對「全部未拜訪節點」做輪盤 ⇒ **`lc`/`ct` 的緊密枚舉優勢明顯**，`bt` 在後期較吃虧。

---

## 3. 維度三：候選清單（`_cl` 有無）

| 項目 | 無 `_cl` | 有 `_cl` |
|------|---------|---------|
| 建構 kernel | `build_ant_solution` (`:1731`) | `build_ant_solution_using_cand_lists` (`:1398`) |
| 每步選擇範圍 | 全部未拜訪節點 | 當前節點的 32 個最近鄰（k-d tree 預建） |
| 機率快取 | `n×n` `product_cache_`（`update_pheromone_heuristic_product_cache` `:863`，`<<<dim,128>>>`） | `n×32` `cand_lists_product_cache_`（`update_cand_lists_..._cache` `:830`，`<<<dim,128>>>`） |
| `n×n` cache 是否配置 | **配置**（`:2353` 的 `use_cand_lists_? 0 : n*n`） | **不配置**（省 `n²·4` bytes） |
| 啟動限制 | 無特別限制 | **`block-warps×32 == cand-list-size`**，且 `cand-list-size` 為 32 倍數，否則 `abort()` |
| thread 對應 | 一 thread 一段節點（chunk） | 一 thread 一個候選 slot |
| 回退機制 | 不需要 | 最近鄰全拜訪時，退化成「掃全未拜訪、取 `τ·η` 最大」的貪婪（`:1436`） |

### 記憶體影響（OOM 關鍵）

`n×n` product cache 對大實例是災難：pla85900 需 `85900² × 4 B ≈ 27.5 GiB`，直接爆掉 32 GiB V100。`_cl` 變體完全不配置它，這是大實例唯一可行的路線（見 memory：pla85900 OOM 根因）。

---

## 4. 12 種變體總表

| # | `--alg` | 建構 kernel | 選蟻函式（預設 1-warp 時） | Tabu shared | recip. | `n×n` cache |
|---|---------|------------|--------------------------|-------------|--------|-------------|
| 1 | `mmas_rwm_bt` | `build_ant_solution` | `warp_roulette_choice` | n/8 B | ✗ | **配置** |
| 2 | `mmas_rwm_lc` | `build_ant_solution` | `warp_roulette_choice` | 4n B | ✗ | **配置** |
| 3 | `mmas_rwm_ct` | `build_ant_solution` | `warp_roulette_choice` | 2n B | ✗ | **配置** |
| 4 | `mmas_wrs_bt` | `build_ant_solution` | `reservoir_sampling_roulette_choice` | n/8 B | ✓ | **配置** |
| 5 | `mmas_wrs_lc` | `build_ant_solution` | `reservoir_sampling_roulette_choice` | 4n B | ✓ | **配置** |
| 6 | `mmas_wrs_ct` | `build_ant_solution` | `reservoir_sampling_roulette_choice` | 2n B | ✓ | **配置** |
| 7 | `mmas_rwm_bt_cl` ⭐ | `build_ant_solution_using_cand_lists` | `warp_roulette_choice_from_cand_list` | n/8 B | ✗ | 不配置 |
| 8 | `mmas_rwm_lc_cl` | `build_ant_solution_using_cand_lists` | `warp_roulette_choice_from_cand_list` | 4n B | ✗ | 不配置 |
| 9 | `mmas_rwm_ct_cl` | `build_ant_solution_using_cand_lists` | `warp_roulette_choice_from_cand_list` | 2n B | ✗ | 不配置 |
| 10 | `mmas_wrs_bt_cl` | `build_ant_solution_using_cand_lists` | `reservoir_sampling_..._from_cand_list` | n/8 B | ✓ | 不配置 |
| 11 | `mmas_wrs_lc_cl` | `build_ant_solution_using_cand_lists` | `reservoir_sampling_..._from_cand_list` | 4n B | ✓ | 不配置 |
| 12 | `mmas_wrs_ct_cl` | `build_ant_solution_using_cand_lists` | `reservoir_sampling_..._from_cand_list` | 2n B | ✓ | 不配置 |

⭐ = 預設（`mmas.cu:main.cc` USAGE 預設 `mmas_rwm_bt_cl`）。`recip.` = `use_product_reciprocal_`。Tabu shared 為近似值（`bt` 實際為 `round_to_multiple_of(4, ceil(n/32)·4)`、`ct` 為 `round_to_multiple_of(4, 2n)`）。

> 註：第 1–3、7–9（`rwm`）若 `--block-warps>1`，選蟻函式自動換成對應的 `block_roulette_choice[_from_cand_list]`；`wrs` 六種不受 block 大小影響。

---

## 5. 與 `--alg` 正交的共同行為（12 種皆同）

這些不因變體而異，但會與變體交互影響效果（細節見 `FLOW_mmas_rwm_bt_cl.md` Step 3–7）：

- **2-opt 局部搜尋**（`--ls=1`，`two_opt_nn` `:2130`）：解品質主力，所有變體共用。
- **trail limits**（`calc_trail_limits` `:663`）：刷新 global best 時重算 τ_min/τ_max。
- **沉積來源**：純費洛蒙模式用 iter best；開 LS 時隨機 10% iter / 9% global / 81% reset best。
- **停滯重置**（僅開 LS）：reset best 停滯 `stagnation_period` 後把費洛蒙填成 τ_max。

---

## 6. 怎麼挑

| 情境 | 建議 | 理由 |
|------|------|------|
| 一般 / 不確定 | `mmas_rwm_bt_cl`（預設） | `_cl` 省時省記憶體；`bt` 在 `_cl` 下缺點不發作；1-warp 走純 warp shuffle 最快 |
| 大型實例（n 很大） | 必用 `_cl` | 避免 `n×n` product cache 的 OOM |
| shared memory / 記憶體吃緊 | Tabu 選 `bt`（最省）或 `ct`（lc 一半） | `bt` n/8 B、`ct` 2n B、`lc` 4n B |
| 非 `_cl` 且要快枚舉 | Tabu 選 `lc` 或 `ct` | 緊密未拜訪清單，後期枚舉不浪費 |
| 想要選蟻不受 block 大小限制 | `wrs` 系列 | 單一實作；`rwm` warp 版受限於 1 warp |

> 論文中 `rwm` 與 `wrs` 解品質相近，主要差在 GPU 執行效率；解品質的主要決定因素是 `--ls`（2-opt）而非選蟻 / Tabu 的選擇。
