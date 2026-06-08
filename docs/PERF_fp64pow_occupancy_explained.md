# 效能優化① 細解 — Achieved Occupancy 與 get_heuristic 的優化範圍

> 本檔是對 `docs/OOM_FIX_pla85900.md`「效能優化① — 砍 FP64 pow」的補充教學,
> 回答兩個問題:
> 1. Achieved Occupancy 的定義、理論值 12.5% 怎麼算、為何看得出 warp 不足?
> 2. 從「`get_heuristic` 被 fallback 熱路徑 + cand-list cache 更新呼叫」能推出這個優化實際改善了哪一步?
>
> 對應程式碼:`src/mmas.cu` 的 `get_heuristic`(約 834 行)、fallback 呼叫點(約 1530 行)、
> cand-list 快取更新呼叫點(約 933 行)。環境:V100-SXM2-32GB / sm_70 / 80 SM。

---

## 一、Achieved Occupancy:定義、12.5% 怎麼算、為何看得出 warp 不足

### 定義

Occupancy(佔用率)= **「一個 SM 上實際同時活躍的 warp 數」 ÷ 「該 SM 硬體能容納的最大 warp 數」**。

要分清楚兩個版本:

| 名稱 | 意義 |
| --- | --- |
| **Theoretical Occupancy(理論)** | 由 kernel 的「靜態資源用量」(每 thread 暫存器數、每 block shared memory)算出的**上限** —— 就算 block 發到爆,也只能到這麼高。 |
| **Achieved Occupancy(實測)** | kernel 真的在跑時,用硬體計數器量到的「平均活躍 warp / 最大 warp」。會 ≤ 理論值。 |

### 理論值 12.5% 怎麼來的

V100(sm_70)每個 SM 最多容納 **64 個 warp**(2048 threads ÷ 32)。

```
理論 12.5% = 8 / 64  → 這支 build kernel 每個 SM 最多只能「常駐 8 個 warp」
```

為什麼卡在 8 而不是 64?因為 build kernel **暫存器用量很重**(register-heavy),每個 block 佔的
暫存器 / shared memory 把「同時常駐的 block 數」壓到只剩 8 個。又因為 bt(BitmaskTabu)變體
一個 block 只有 1 個 warp(`Block (32,1,1)`),8 block = 8 warp = SM 容量的 12.5%。
這就是這支 kernel 自己的天花板。

> 補充:sm_70 每 SM 最多 32 個 block。若 1 warp/block 而資源不受限,理論可達 32 warp = 50%。
> 既然只到 8,代表限制因子是**暫存器 / shared memory**,不是 block 數上限。

#### 「8」是怎麼看出來的:不是手算,是程式直接查 CUDA API

那個「常駐 8 個 block」**不是估的,是執行時由 CUDA runtime 算出來、印在 `.out` 裡的**。
`src/mmas.cu:2677` 呼叫 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`:

```cpp
int max_active_num_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_num_blocks,
    alg.build_ant_sol_,                      // 這支 build kernel
    alg.kernel_cfg_.threads_per_block_,      // 每 block thread 數 (=32)
    alg.kernel_cfg_.shared_buffer_size_);    // 每 block shared memory
cout << "Max active #blocks for ant sol. build kernel: " << max_active_num_blocks << endl;
```

它拿「**kernel 編譯後的暫存器/thread + shared memory/block 用量**」比對「**SM 的資源預算**」,
回傳一個 SM 能同時常駐幾個 block。pla85900 那次跑印出:

```
Max active #blocks for ant sol. build kernel: 8     ← mmas_945599.out 第 49 行
```

`8 blocks × 1 warp = 8 warps/SM`,`8 / 64 = 12.5%` —— 與理論值完全吻合。「8」與「12.5%」是
同一件事的兩種講法。

#### 想確定是哪個資源(暫存器 vs shared memory)把它壓到 8

API 只回「8」,不說 limiter 是誰。要再歸因,看這兩處:

1. **ptxas verbose(編譯期)** — Makefile 的 nvcc 已帶 `-Xptxas="-v"`,會對每支 kernel 印
   `Used N registers, M bytes smem`。拿去比 V100 每 SM 預算(65536 regs、最多 96 KB smem、
   ≤32 blocks、≤64 warps),哪個先用完就是綁住的因子。
2. **ncu 的 Occupancy 區段** — 直接列 `Block Limit Registers / Shared Mem / Warps / SM`
   四個上限,取最小值即理論 occupancy,並標明 limiter。

**用暫存器反推可驗證**:V100 每 SM 65536 個暫存器。若 kernel 編到接近上限 ~255 regs/thread:

```
65536 / (32 threads × 255 regs) ≈ 8 blocks   ← 剛好 8
```

所以幾乎可確定是 **暫存器** 把常駐 block 壓到 8(build kernel 本就 register-heavy,這也呼應
修法⑥裡 2-opt 因 `1024 × regs > 65536` 啟動失敗的同一類限制)。確切數字以 ptxas `-v` /
ncu Block Limit 為準。

### 實測值 1.95% 怎麼來的(這就是「看出 warp 不足」的地方)

預設 `--ants=100`,而 `blocks_count = ants_count`,且每 block 1 warp → **整顆 GPU 只發了 100 個 warp**。
V100 有 **80 個 SM**:

```
平均每 SM 的 warp = 100 / 80 ≈ 1.25 個
Achieved Occupancy = 1.25 / 64 ≈ 1.95%   ← ncu 報告裡的數字
```

**判讀**:實測 1.95% 連 kernel 自己的天花板 12.5% 都摸不到 —— 代表「**根本沒發足夠的 warp**」。
每個 warp scheduler 平均手上不到 2 個 warp 可調度。GPU 藏延遲的唯一手段就是「一個 warp 卡在
長延遲指令(FP64 `sqrt`、分散的 global memory 讀)時,馬上切到別的 warp 去算」;warp 太少就
沒得切,長延遲指令直接曝露成空等 → 這就是報告判定 **latency / occupancy bound** 的依據。

> 這也是為什麼效能優化②(`--ants=0` 自動開到 ~640 隻)能把實測從 1.95% 拉到 12.3% ——
> 把 warp 數補滿到逼近 12.5% 那道天花板。但要再往上,得改 kernel 本身的暫存器用量,
> 或改成「一個 ant 多 warp / 一個 block 多 ant」。

---

## 二、「get_heuristic 被 fallback 熱路徑 + cand-list cache 更新呼叫」推得出什麼

`get_heuristic` 在 `_cl` 變體(預設 `mmas_rwm_bt_cl`)只會在這兩處跑到:

1. **`src/mmas.cu:1530`** — `build_ant_solution_using_cand_lists` 的 **fallback**:候選清單近鄰
   全被走過時,改掃整個 unvisited list,每個節點都呼叫 `get_heuristic`。
   這屬於 **solution-build kernel**(每輪 pipeline 第 3 步:螞蟻建路徑)。
   ```cpp
   const float product = pheromone[node] * instance.get_heuristic(current_node, node);
   ```
2. **`src/mmas.cu:933`** — `update_cand_lists_pheromone_heuristic_product_cache` kernel
   (每輪 pipeline 第 2 步:更新候選清單快取)。
   ```cpp
   const auto product = instance.get_heuristic(src_node, node) * pheromone[node];
   ```

> (還有 `src/mmas.cu:963` 的完整 product cache 更新也呼叫 `get_heuristic`,但那只給**非** `_cl`
> 變體用,`_cl` 已 gate 掉不執行;預設大實例走不到。)

### 推論

- **受惠的不是只有「螞蟻找路徑」那一步**,而是**同時涵蓋候選清單快取更新那一步**。
  只是後者極小(`cand-list-product-cache-update` 才 ~0.00015 s/iter),實務上幾乎全部好處
  集中在 solution-build kernel(佔 GPU ~97%)。
- **更關鍵**:這兩個呼叫點**都在每輪迭代迴圈裡** → 這個優化只會反映在
  **mean-iteration-time / solution-build kernel 時間**上,**完全不會碰到 `initialization-time`**。
  座標型實例在初始化階段根本不建 heuristic 矩陣、也不呼叫 `get_heuristic`,
  所以 FP64 pow 的改動在初始化階段**在結構上就不可能有作用**。

> 這點可用來反推「某次跑的 initialization-time 變化」絕不是 FP64 pow 造成的 ——
> 例如 pla85900 的 `initialization-time` 從 29.11 s → 0.248 s,那是修法④(device fill kernel
> 省掉 host 暫存 + 27.5 GiB H2D 拷貝)的功勞,不是這個優化;FP64 pow 的效果要看
> `solution-build` kernel 時間(且須用 `profile.slurm` 的 ncu 報告佐證)。

---

## 三、判讀指南 — 如何對照完整跑的報告,判定 FP64 pow 優化是否成功

跑完整 1000 iter 後,結果可能來自兩種格式:JSON 結果檔(`results/*.json`)或 SLURM 的
`.out` 文字輸出。`.out` 通常只印一行 `Build sol. kernel mean time [ms]`,但**這正好就是該看的
指標** —— 因為 FP64 pow 只動到 `get_heuristic`,它只在 build kernel 裡跑(見第二節)。

### 對照步驟

1. **換算單位再比同一支 kernel。** JSON 的 `kernel-mean-times.solution-build` 單位是**秒**,
   × 1000 換成 ms,再去比 `.out` 的 `Build sol. kernel mean time [ms]`。

   ```
   JSON solution-build (s) × 1000  ⇄  .out  "Build sol. kernel mean time [ms]"
   ```

2. **先確認兩份報告設定一致(否則不可比)**:`--alg`、`--instance`、`--ants`、`--block-warps`、
   `--cand-list-size`、`--iter` 必須相同。JSON 的設定看頂層 `arguments`;`.out` 看開頭的
   `Running MMAS on:` 與 `Max active #blocks` 等行。

### 實例(pla85900,1000 iter,mmas_rwm_bt_cl / ants=100 / block-warps=1 / cand-list-size=32)

| 指標 | 優化前(`...2026-6-4_18_57_2.json`) | 優化後(`mmas_945599.out`,6-5) | 變化 |
| --- | --- | --- | --- |
| **Build sol. kernel mean** | 2530.11 ms | **1655.52 ms** | **↓ 34.6%** |
| Mean iter. time | 2634.91 ms | 1760.25 ms | ↓ 33.2% |

`2530.11 = JSON solution-build 2.530113517063 s × 1000`;`1655.52` 直接讀 `.out`。
降幅 ~34.6%,與本優化記載的「~36%」一致 → **判定優化成功**。

### 三個常見陷阱

1. **看 `Build sol. kernel mean`,不要看 `total-time` / `initialization-time`。** 後兩者混進與本
   優化無關的初始化與其他 kernel,會稀釋甚至誤導訊號。build kernel 是唯一作用點、又佔 GPU ~97%,
   訊號最乾淨。`Mean iter. time` 可當輔助佐證(會一起降),但主指標是 build kernel mean。
2. **要 like-for-like 同來源比。** 上表兩數都來自完整跑的 `GPUTimer`(CUDA event),直接可比。
   **不要**拿它去比文件裡 ncu 的單次 launch duration(例如 4.04 s)—— 那帶 profiling 額外開銷、
   來源不同、不可混比。同類比同類。
3. **解品質(gap)不是這個優化的判準。** gap 微幅變動(此例 33.54% → 34.12%,差 ~0.6pp)是預期內:
   heuristic 改 float 連乘後選點權重**微幅**改變 → 走出不同 tour;但 tour cost 仍以精確 double
   距離量測,成本公平。單一 seed 下 0.6pp 屬雜訊等級,**不代表退步**;本優化追求速度,品質要比
   得用多個 seed 取平均。
