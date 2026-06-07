# pla85900 OOM 修復報告

> 目標:讓 `pla85900.tsp`（n = 85900，CEIL_2D）在 32 GiB Tesla V100 上跑得起來。
> 環境:Taiwania 2 叢集 / V100-SXM2-32GB / sm_70 / CUDA 12.3 / GCC 10.2.1。
> 本機是 Windows 無 GPU,無法編譯與執行;所有驗證一律 `sbatch run.slurm` 在叢集上做。
>
> **維護慣例**:本檔是 pla85900 修復 + 後續效能優化的單一事實來源。
> 每當對相關程式碼(記憶體配置、即時計算路徑、build kernel 等)有改動,
> **務必同步更新本檔**對應章節與程式片段,勿留過時程式碼。

---

## 0. 問題根因

`run_gpu_based_mmas`（`src/mmas.cu`）原本會配置 **4 個完整 n×n float 矩陣**,
每個 = n² × 4 B = **27.49 GiB**:

| 矩陣 | 用途 | 是否真的必要 |
| --- | --- | --- |
| `d_pheromone` | 費洛蒙儲存、evaporate / deposit、候選清單耗盡時 fallback | 必要 |
| `d_heuristic` | fallback 用 | 可由座標即時算 `1/dist^β` |
| `d_dist_matrix` | 2-opt 與成本計算 | 座標型實例可由座標即時算 |
| `d_product_cache` | **只有非 `_cl` 變體會讀**;`_cl` 變體每輪更新卻從不讀 → 純浪費 | `_cl` 不需要 |

4 個合計約 110 GiB ≫ 32 GiB。預設演算法 `mmas_rwm_bt_cl` 正是 `_cl` 變體,
所以 `d_product_cache` 完全是浪費。下面四項修法依序拆掉這些負擔。

---

## 修法① — `_cl` 變體移除 `d_product_cache`(省 27.5 GiB)

`_cl`（候選清單）變體用的是 `d_cand_lists_product_cache`(n × cand_list_size,約 11 MB),
從來不讀完整的 `d_product_cache`。因此只在**非** `_cl` 時才配置該矩陣。

**配置處（`src/mmas.cu:2485`）**
```cpp
// 完整 n*n product cache 只有非 _cl 的 build kernel 會讀;_cl 變體用
// d_cand_lists_product_cache 取代,配置它純粹浪費 27.5 GiB → 只在需要時配。
device_vector<float> d_product_cache(
    alg.use_cand_lists_ ? size_t{0} : static_cast<size_t>(dimension) * dimension);
```

**每輪更新 kernel 也跟著 gate（`src/mmas.cu:2557`）**
```cpp
for (uint32_t iter = 0; iter < iterations; ++iter) {
    // 完整 product cache 只餵非 _cl build kernel,_cl 不更新(也沒配)。
    if (!alg.use_cand_lists_) {
        product_cache_update_timer.start();
        update_pheromone_heuristic_product_cache<<<dimension, 128>>>
            (instance_ctx, mmas_ctx, alg.use_product_reciprocal_);
        product_cache_update_timer.accumulate_elapsed();
    }
    ...
}
```

省下 27.5 GiB 記憶體 + 每輪一支全 n×n 的更新 kernel。

---

## 修法② — heuristic / distance 改即時計算(再省 2 × 27.5 GiB)

座標型實例（非 EXPLICIT）不再上傳 `d_heuristic` / `d_dist_matrix`,
改在 GPU 上由座標即時計算。

**新增 `device_node_distance()`（`src/mmas.cu:683`）** — 用 `double` 精確鏡像
`tsp.cc` 的距離公式,確保與 host 位元級一致(EUC_2D / CEIL_2D / ATT / GEO):
```cpp
__device__ inline
float device_node_distance(EdgeWeightType type, const double *coords,
                           uint32_t from, uint32_t to) {
    const double x1 = coords[2*from],   y1 = coords[2*from+1];
    const double x2 = coords[2*to],     y2 = coords[2*to+1];
    switch (type) {
        case EUC_2D:  { double dx=x2-x1,dy=y2-y1;
                        return (float)(int32_t)(sqrt(dx*dx+dy*dy)+0.5); }
        case CEIL_2D: { double dx=x2-x1,dy=y2-y1;
                        return (float)(int32_t)ceil(sqrt(dx*dx+dy*dy)); }
        case ATT:     { double xd=x1-x2,yd=y1-y2;
                        double rij=sqrt((xd*xd+yd*yd)/10.0);
                        int32_t tij=(int32_t)rij;
                        return (float)(tij<rij?tij+1:tij); }
        case GEO:     { /* 完整 geo 鏡像,kPi、cos/acos */ }
        default:      return 0.0f;  // EXPLICIT 必須用 distance_matrix_
    }
}
```

**`InstanceContext` 改寫（`src/mmas.cu:742`）** — 矩陣指標為 null 時即時算;
座標改存 `double`：
```cpp
struct InstanceContext {
    uint32_t dimension_ = 0;
    double*  coordinates_ = nullptr;       // 改成 double,與 host 位元對齊
    float*   distance_matrix_ = nullptr;   // null ⇒ 即時算
    float*   heuristic_matrix_ = nullptr;  // null ⇒ 即時算
    float    heuristic_weight_ = 2;
    uint32_t cand_list_size_ = 0;
    uint32_t* cand_lists_ = nullptr;
    EdgeWeightType edge_weight_type_ = EUC_2D;

    __device__ float get_distance(uint32_t from, uint32_t to) const {
        assert(from < dimension_ && to < dimension_);
        if (distance_matrix_ != nullptr)
            return distance_matrix_[static_cast<size_t>(from) * dimension_ + to];
        return device_node_distance(edge_weight_type_, coordinates_, from, to);
    }
    __device__ float get_heuristic(uint32_t from, uint32_t to) const {
        assert(from < dimension_ && to < dimension_);
        if (heuristic_matrix_ != nullptr)
            return heuristic_matrix_[static_cast<size_t>(from) * dimension_ + to];
        // ⚠ 已於「效能優化①(砍 FP64 pow)」與「效能優化③(方向 1b:dist 改
        //   float sqrtf)」改寫;此處為原始版,完整版見下方兩章節。
        const double dist = get_distance(from, to);
        return dist > 0
             ? (float)(1.0 / pow(dist, (double)heuristic_weight_)) : 1.0f;
    }
};
```

**配置端依旗標決定要不要上傳矩陣（`src/mmas.cu:2411`）**
```cpp
const bool compute_dist_on_the_fly =
    (instance.edge_weight_type_ != EXPLICIT) && !instance.coordinates_.empty();

// create_heuristic_matrix() 只有真的要矩陣時才被求值、才配 n*n host 向量。
device_vector<float> d_heuristic(
    compute_dist_on_the_fly ? vector<float>()
                            : create_heuristic_matrix(instance, params));

vector<float> dist_matrix;
if (!compute_dist_on_the_fly) { /* double→float 轉填 */ }
device_vector<float> d_dist_matrix(dist_matrix);

// 座標存 double,讓即時距離與 host(tsp.cc)位元一致。
vector<double> coordinates; /* 攤平成 x,y,x,y... */
device_vector<double> d_coordinates(coordinates);

InstanceContext instance_ctx {
    dimension, d_coordinates,
    compute_dist_on_the_fly ? nullptr : d_dist_matrix.data(),
    compute_dist_on_the_fly ? nullptr : d_heuristic.data(),
    (float)params.beta_, params.cand_list_size_,
    d_cand_lists, instance.edge_weight_type_
};
```

**host 端略過建距離矩陣（`src/mmas.cu:~3189`）** — 只有 EXPLICIT 才呼叫
`init_distance_matrix()`,座標型實例改走 NN list（kd-tree）+ 即時 `calculate_distance`,
省下 55 GiB host double 矩陣與 O(n²) 建構時間。

- **代價**:2-opt 與路徑成本改即時 sqrt,計算量上升(記憶體換計算)。
- **EXPLICIT 實例**:沒有座標,維持原本的距離 / heuristic 矩陣不變。

---

## 修法③(bug 修) — n > 65535 時 uint32 索引溢位

修①②讓 OOM 消失、job 944491 成功進到迭代迴圈,但 pla85900 立刻噴:
```
CUDA assertion failed: operation not supported on global/shared address space
at: src/mmas.cu:576  (cudaErrorInvalidAddressSpace 717)
```

**根因**:n > 65535 時所有 n×n 的大小 / 位移在 **uint32 溢位**。
- `dimension * dimension`(uint32):85900² = 7.38×10⁹ > uint32 上限 4.29×10⁹。
  取模後 ≈ 3.08×10⁹,所以 `d_pheromone` **只配了 ~11.5 GB**(這也是「居然沒 OOM」的真正原因之一)。
- device 端 `src_node * dimension` 之類索引可達 ~4.29×10⁹ > 實配 3.08×10⁹ → 越界 → 717。
- ⚠️ 出錯行 `src/mmas.cu:576` 是 `GPUTimer` 的 `cudaEventSynchronize`,**所有 timer 共用此行**;
  錯誤是非同步的,在第一個 sync 點才被捕捉,**所以這行不代表出錯的 kernel**。

**修法**:把所有 n×n 的大小與位移改成 64-bit(`static_cast<size_t>(...)`),涵蓋:
`get_distance` / `get_heuristic`、兩支 product-cache 更新 kernel、`_cl` fallback、
三支非 `_cl` choice fn、evaporate、deposit、`create_heuristic_matrix`,以及 host 端
`d_pheromone` / `dist_matrix` / 非 `_cl` product_cache 的配置大小。

> 受 `ants`(≤100)/`cand_list_size`(32) 限制的 `*dimension` 不會溢位,未動。

範例(`src/mmas.cu:764`):
```cpp
return distance_matrix_[static_cast<size_t>(from) * dimension_ + to];
```

修正後 `d_pheromone` 正確為 27.49 GiB。

---

## 修法④ — `d_pheromone` 改用 device fill kernel 初始化

修③之後 `d_pheromone` 用 `device_vector(size, value)` 建構,會在 **host 暫存 29.5 GB**
再 H2D 拷貝。改成「只配置 + 在 GPU 上填值」,徹底拿掉 host 暫存與那次拷貝。

**新增 fill kernel（`src/mmas.cu:1952`）** — grid-stride、`size_t` count 支援 > 2³² 元素:
```cpp
__global__
void fill_float_array(float *data, size_t count, float value) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        data[i] = value;
    }
}
```

**配置 + 填值（`src/mmas.cu:2392`）**
```cpp
const size_t pheromone_count = static_cast<size_t>(dimension) * dimension;
device_vector<float> d_pheromone(pheromone_count);   // 未初始化
{
    const int fill_threads = 256;
    const size_t threads_wanted = (pheromone_count + fill_threads - 1) / fill_threads;
    const int fill_blocks = static_cast<int>(std::min<size_t>(threads_wanted, 65535));
    fill_float_array<<<fill_blocks, fill_threads>>>(
        d_pheromone.data(), pheromone_count, trail_limits.max_);  // 初值 = τ_max
    CUDA_CHECK(cudaGetLastError());
}
```
blocks 取 `min(需要的 thread 數 / 256, 65535)`,靠 grid-stride loop 覆蓋全部 7.38×10⁹ 元素。
填值在迴圈前、default stream 上依序執行,迭代 kernel 用到時已填好。
(stagnation reset 走的是另一支 `evaporate_pheromone(reset=true)`,不受影響。)

---

## 效能優化① — 砍 FP64 pow(方向 1,2026-06-05)

> OOM 解決後的第一個效能瓶頸。屬於「修法②(座標即時計算)」用記憶體換計算的後續帳單。

### 剖析證據(`sbatch profile.slurm`,pla85900)

nsys 顯示 `build_ant_solution_using_cand_lists` 佔 **~97% GPU 時間**,其餘 kernel(evaporate 2.8% 等)受 Amdahl 限制不值得碰。ncu Speed-Of-Light 對該 kernel 實測:

| 指標 | 數值 | 判讀 |
| --- | --- | --- |
| Compute (SM) 吞吐 | 6.6% | 算術單元閒置 |
| Memory / DRAM 吞吐 | 1.1% / 0.95% | **非 memory bound** |
| Achieved Occupancy | 1.95%(理論 12.5%) | warp 嚴重不足 |
| Grid / Block | (100,1,1) / (32,1,1) | 全 GPU 僅 100 warps |
| FP64 指令數 / launch | **~28e9**(FP32 僅 0.34e9) | FP64 是主要的功 |
| Roofline | ~3% FP64 峰值(V100 FP64 = FP32 的 ½) | — |

→ 判定 **latency bound**:warp 太少無法掩蓋高延遲指令,而那些長延遲指令正是
座標即時計算的 **FP64 `sqrt`/`pow`**。`pow` 主要來自 fallback 路徑(候選清單近鄰用盡時
掃整個 unvisited list,每個都呼叫 `get_heuristic`),且預設 β=2 卻在跑 `pow(dist, 2.0)`。

### 改法

對**整數 β** 改用 float 連乘取代 `pow`,非整數才退回單精度 `__powf`。
距離仍走 `get_distance`(double `sqrt`)以保 tour cost 與 TSPLIB 取整一致 ——
heuristic 只是選點權重,float 足夠。刻意**不**開全域 `--use_fast_math`,避免波及
pheromone / 距離精度,改用外科手術式的 `__powf`。

**device 端 `get_heuristic`(`src/mmas.cu:773`)**
```cpp
const float dist = static_cast<float>(get_distance(from, to));
if (dist <= 0.0f) return 1.0f;
const int int_beta = static_cast<int>(heuristic_weight_);
if (static_cast<float>(int_beta) == heuristic_weight_ && int_beta >= 0) {
    float denom = 1.0f;
    for (int b = 0; b < int_beta; ++b) denom *= dist;   // dist^β,無 pow
    return 1.0f / denom;
}
return __powf(dist, -heuristic_weight_);                  // 非整數 β:單精度
```

**host 端 `create_heuristic_matrix`(`src/mmas.cu:1961`)** — 鏡像同樣的整數 β 特化,
讓「有矩陣/無矩陣」「`_cl`/非 `_cl`」兩條路徑權重一致(僅小實例會用到矩陣)。

### 影響範圍

`get_heuristic` 被 fallback 熱路徑(`src/mmas.cu:1446`)與 cand-list cache 更新
(`src/mmas.cu:849`)呼叫,兩處都受惠。tour cost / 路徑長度路徑不變(仍 double)。

### 實測結果(`sbatch profile.slurm`,pla85900 build kernel,2026-06-05)

對照舊(FP64 pow)vs 新(方向 1):

| 指標 | 舊 (pow/FP64) | 新 (方向 1) | 變化 |
| --- | --- | --- | --- |
| Duration / launch | ~4.04 s | **~2.57 s** | **↓ ~36%** |
| FP64 指令數 / launch | ~28.0e9 | **~3.38e9** | **↓ ~88%** |
| FP32 指令數 / launch | ~0.34e9 | ~2.03e9 | ↑(pow → float 乘法) |
| Compute (SM) 吞吐 | 6.6% | 3.4% | ↓ |
| fixed-latency 依賴 stall | ~50% | ~37% | ↓ |

→ **方向 1 有效**:`pow` 砍掉後 FP64 降 ~88%,單顆 build kernel 快約 1/3。

### 瓶頸位移(下一步依據)

新報告 stall 結構改變,整體**仍是 latency / occupancy bound**(occupancy 1.95%、
eligible warps 0.12、0.2 wave),但主因從「FP64 compute 延遲」變成兩者並列:

- **fixed-latency execution dependency 37%** — 剩下的 double `sqrt` + 算術依賴鏈。
- **scoreboard L1TEX 記憶體 stall 34%** — 分散的 `pheromone[node]` / cand gather 的全域記憶體延遲(舊報告非主角,現在浮上來)。

剩下的 ~3.38e9 FP64 幾乎都是 `get_distance` 的 double `sqrt`(fallback 掃 unvisited
+ route length 都會呼叫)。後續方向:

- **方向 1b(小):** 給 heuristic 一條 float `sqrtf` 距離(cost 路徑維持 double),
  砍掉剩餘 double `sqrt`。報酬遞減但因 latency bound、串行依賴鏈仍會直接縮時。
- **方向 2(大、治本):** compute 已不肥,memory latency(34%)+ occupancy 1.95%
  成天花板 → 增加在飛行的 warp 數(每 ant 多 warp / 一 block 多 ant)。
  單純降 shared memory 無效(block 總數 100 已 < SM 容量)。
- **跳過方向 3:** uncoalesced 雖 44% 但 memory 吞吐才 1.8%,非瓶頸。

> 註:`*_build_bound.txt` 底部那行 awk 一句話結論在舊版 profile.slurm 有解析 bug
> (誤印 SM/occupancy 0.0%),已改為解析 details 區段;以表格上方 details 數字為準。

---

## 效能優化② — 增加在飛行的 warp 數(方向 2,2026-06-05)

> 方向 1 把 compute 砍瘦後,瓶頸剩 occupancy 1.95% 太低(latency bound)。

### 根因鏈

- **warp 數 = ant 數**:`blocks_count = params.ants_count_`(`src/mmas.cu:2493`、`2930`),
  一隻 ant = 一個 block。pla85900 為避免 per-ant buffer 爆記憶體只用 `--ants=100`
  → 全 GPU 僅 100 warps,分散在 80 SM,occupancy 1.95%(理論天花板 12.5%,受 shared mem 限)。
- **ant 數被記憶體綁死**:每隻 ant 約佔 `ant_routes`(n×4)等;LS 開時再加
  `pos_in_routes`(n×4)+`dont_look_bits`(n×1)。
- **發現一個死碼**:`d_temp_ant_routes`(ants×n×4)配置後**從未被使用**。

### 改法(`--ants=0` 自動填滿 GPU + gate LS buffer)

1. **刪除死碼 `d_temp_ant_routes`**(`src/mmas.cu:~2491`)。
2. **gate LS 專用 buffer**:`d_pos_in_routes` / `d_dont_look_bits` 只在
   `use_local_search` 時配置,否則 size 0(`src/mmas.cu:~2488`)。LS 關閉時每隻 ant
   省 n×5 B,騰出記憶體塞更多 ant。
3. **`--ants=0` ⇒ 自動填滿 GPU**:新增 `compute_auto_ant_count()`
   (`src/mmas.cu:~2902`),取
   `min(占用率目標, 記憶體預算, n)`:
   - 占用率目標 = `SM 數 × 8 warps ÷ block-warps`(8 warp/SM = BitmaskTabu 的
     shared-mem occupancy 天花板,剛好填滿、不超額認購);
   - 記憶體預算 = `cudaMemGetInfo 的 free − 固定配置(主要 n×n 費洛蒙)− 1 GiB 安全邊際`,
     再除以每隻 ant 的位元組數。
   - **預設仍為 100,不影響既有 benchmark**;`--ants=0` 才啟用(取代原本會 OOM 的
     「0 ⇒ dimension」語意)。`main.cc` 的 USAGE 已更新。

pla85900 預估:占用率目標 80×8=**640 隻 ant**(記憶體可塞 >10000,n 也夠),
occupancy → 接近天花板 12.5%。

> 沿革:`target_warps_per_sm` 初版為 16(→1280 ant),下方實測證實 1280 是 2× 超額認購
> (occupancy 已達天花板,多的只排隊),故 2026-06-06 調為 8(→640 ant):同樣的
> per-ant throughput 與 occupancy,但每代 launch 時間約減半。

### 實測結果(`sbatch profile.slurm --ants=0`,pla85900,2026-06-06)

> 此次量測時 `target_warps_per_sm` 仍為 16,故 auto 選了 1280 ant;之後已調為 8
> (→640 ant),per-ant 數字不變、launch 時間約減半。下表保留 1280 的歷史數據。

auto 選了 **1280 ant**(grid 1280)。**關鍵:ncu Duration 是「一次 launch = 全部 ant」**,
ant 數 ×12.8,所以要看 per-ant throughput,不是 launch 時間:

| 指標 | dir 1(100 ant) | dir 2(1280 ant) | 變化 |
| --- | --- | --- | --- |
| Duration / launch | 2.57 s | 5.95 s | ↑ 2.3× |
| **時間 / ant** | 25.7 ms | **4.65 ms** | **↓ 5.5×** |
| Achieved occupancy | 1.95% | **12.3%** | ↑ 6.3×(≈ 天花板 12.5%) |
| SM 吞吐 | 3.4% | 19.2% | ↑ |
| FP64 指令 / launch | 3.38e9 | 43.4e9 | ↑ 12.8×(= ant 比,per-ant 不變) |

→ **方向 2 有效**:per-ant 快 5.5×、occupancy 逼近 shared-mem 天花板。Duration 變長
只是每代多建 12.8× 的 tour。**別把 ncu Duration 當絕對時間**(replay/鎖頻會灌水);
端到端請看程式 stdout 的 `Build sol. kernel mean time` / `Elapsed` / final gap%。

### 後續

- ✅ 已做(2026-06-06):`target_warps_per_sm` 16 → 8 → auto 改選 ~640 ant
  (occupancy 天花板 8 warp/SM × 80),每代 launch 約減半、per-ant throughput 不變。
- ⚠ 更正:**不能用 `bt → ct` 突破天花板**。各 Tabu 的 shared mem/block:bt=`ceil(n/8)`
  ≈10.7 KB(1 bit/node,最省)、ct=`2n`≈171.8 KB、lc=`4n`≈343.6 KB。CLAUDE.md 說的
  「ct ~½ 記憶體」是相對 lc,不是相對 bt。ct 在 pla85900 需 171.8 KB > V100 每 block
  上限 96 KB → **kernel 無法啟動**;且 shared 更大只會讓 occupancy 更差。**bt 已是
  occupancy 最佳的 tabu,12.5% 就是此 kernel 設計的天花板。**
- 真要突破 12.5%:把 visited bitmask 從 shared 移到 global(L2 cache)以釋放 shared →
  更多 block/SM,但會增加全域記憶體流量(目前 L1TEX stall 已 40%),得失未定、風險高。
- 更務實:occupancy 已到頂,改攻剩餘 stall —— 方向 1b(double `sqrt`→float `sqrtf`,
  佔 ~32%)與 gather 的合併存取 / `__ldg`(L1TEX 記憶體 stall,佔 ~40%)。
- 剩餘 stall:L1TEX 記憶體 scoreboard 40% + fixed-latency(double sqrt)32%
  → 對應方向 1b(heuristic 改 float sqrtf)與 gather 的合併存取改善。

---

## 效能優化③ — heuristic 距離改 float sqrtf(方向 1b,2026-06-06)

> occupancy 已到 bt 的天花板(12.5%),改攻剩餘 stall 的另一半:double `sqrt`
> (dir 1 之後的 fixed-latency stall ~32%)。

### 根因

dir 1 砍掉 FP64 pow 後,剩餘 ~3.38e9 FP64/launch(100 ant)幾乎全是 `get_distance`
的 double `sqrt`。熱點是 **fallback 路徑**:候選清單近鄰用盡時掃整個 unvisited,
每個都呼叫 `get_heuristic` → `get_distance`(double sqrt)。cand-list product cache
更新(`src/mmas.cu:849`)也會呼叫。

### 改法

新增**只給 heuristic 用**的單精度距離 `device_node_distance_f()`(`src/mmas.cu:~736`,
用 `sqrtf`/`ceilf`/`cosf`/`acosf`),`get_heuristic`(`src/mmas.cu:~800`)改呼叫它:

```cpp
const float dist = (coordinates_ != nullptr)
    ? device_node_distance_f(edge_weight_type_, coordinates_, from, to)  // 1b
    : static_cast<float>(get_distance(from, to));                        // 退路
```

- **tour cost 路徑完全不動**:route length / 2-opt 仍走 double 的 `get_distance` /
  `device_node_distance`,成本仍對齊 TSPLIB 取整與 best-known。heuristic 只是選點權重,
  float 足夠。
- coords 不存在(理論上只剩 EXPLICIT,但它早從 `heuristic_matrix_` 回傳)時退回精確距離。

### 實測結果(`sbatch profile.slurm --ants=0`,pla85900,2026-06-06)

`profiles/sqrtf/mmas_rwm_bt_cl_pla85900_build_bound.txt`(640 ant):

- **FP64 幾乎歸零**:roofline 報「achieved … close to 0% of its fp64 peak」,FP64 指令
  僅 ~2.06e7/launch(dir 1 後是 ~3.38e9),剩的只是零星整數轉型。dir 1b 成功。
- SM 吞吐 20.1%、Memory 吞吐 10.5%、occupancy 12.2%(到 bt 天花板),仍 **latency bound**。
- stall 結構位移:**L1TEX scoreboard(記憶體相依)39.8% 升為最大宗**,fixed-latency
  執行相依降到 32.1%。→ 下一步主攻記憶體 stall(見 ④)。

---

## 效能優化④ —(已試)read-only data cache `__ldg`:**實測無效,已回退**

> dir 1b 後最大宗 stall 是 L1TEX scoreboard 39.8%。曾試把 build kernel 期間唯讀的散讀
> (`coordinates_`、`cand_lists_`、`cand_lists_product_cache`、fallback 的 `pheromone[node]`)
> 全換成 `__ldg(&...)`,想用 read-only data cache 提命中率、縮短 scoreboard 延遲鏈。

### 實測結果(`profiles/L1TEX scoreboard/…build_bound.txt` vs `profiles/sqrtf/…`,640 ant)

| 指標 | dir 1b(sqrtf) | + `__ldg` | 變化 |
| --- | --- | --- | --- |
| Duration | 2.78/2.74/2.73 s | 2.80/2.75/2.74 s | 持平(噪音內,甚至略升) |
| L1TEX scoreboard stall | 39.78% | 39.89% | **沒降** |
| Uncoalesced excessive sectors | 19,820,053,952(45%) | **19,820,053,952(45%)** | **逐位元相同** |
| FP32 指令 | 18.37e9 | 18.37e9 | 相同 |

**結論:`__ldg` 無效**,已 `git checkout 7a6c2a7 -- src/mmas.cu` 回退程式碼(本節保留作負面記錄)。

### 為什麼沒用(重要,避免重蹈)

1. **流量沒變**:excessive sector 數逐位元相同 —— `__ldg` 只換 load 走的快取路徑,
   **不改位址、不減少抓取的 sector**。
2. **Volta 的 read-only cache 就是同一塊 L1**:sm_70 的 L1 與 texture/read-only cache 是
   統一的實體 SRAM,`__ldg` 只多「唯讀不變」提示、不增容量。
3. **工作集 >> 快取**:`pheromone[node]` 是對 **27.5 GiB** 矩陣的隨機 gather,遠超任何
   快取;scoreboard 等的是 **DRAM/L2 延遲**,`__ldg` 動不到。

→ 教訓:這個 kernel 的記憶體 stall 是「working set 爆快取的 DRAM 延遲」,純快取提示無解;
要嘛**減少散讀流量**(改資料結構,見 ⑤),要嘛**提高 occupancy 用更多 warp 蓋延遲**
(但 bt 已在 12.5% 天花板)。

---

## 效能優化⑤(評估中)— 加大候選清單,壓低 fallback 頻率

> 18.4e9 FP32 與 45% 散讀的真正來源是 **fallback 路徑**:候選近鄰(預設 32)用盡時改
> 掃**整個** unvisited(O(n)),每個都 `get_heuristic` + `pheromone[node]` 散讀。tour 後期
> 頻繁觸發。先用「不改程式碼」的旗標實驗量化它能不能被壓下去。

### 零成本實驗(`profile.slurm` 已設定)

`--cand-list-size=64 --block-warps=2`(基準是 32/1):

- dispatch 在 `threads_per_block != 32` 時自動改走 **block 變體**
  (`block_roulette_choice_from_cand_list`,支援 >32 候選),不需改碼。
- NN list 由 `init_nn_lists(cand_list_size)` 建滿 64 鄰居,kd-tree 路徑(CEIL_2D)OK。
- ⚠ 2-opt LS kernel 寫死 `cand_list_size_ <= WARP_SIZE`:release 下 `assert` 被 `-DNDEBUG`
  編掉、不會 abort,但 LS 只會用前 32 個鄰居(build kernel 用滿 64)。
- 量測目標:fallback 觸發次數↓ → FP32 指令↓、散讀 sector↓、總時間↓?同時看 gap 變化
  (候選清單變大會改變解品質,需一併確認)。

### 待驗證(`sbatch profile.slurm`)

- 比對 `profiles/…_cl64_build_bound.txt` 與 `profiles/sqrtf/…`(cl32)的 FP32/sector/Duration。
- 若 fallback 確實是主因 → 有效,再考慮把 LS 也擴到 >32、或正式調大預設。
- 若無感 → fallback 不是瓶頸,改走 ⑥(候選清單式費洛蒙,從根本砍掉 27.5 GiB 隨機 gather)。

---

## 修法⑤(bug 修) — 2-opt LS 在大實例「靜默不啟動」(pla85900 卡 30% 的真因,2026-06-07)

> 這是調查「`__ldg` 為何讓 gap 從 30%→5%」時挖出來的真正 bug。結論:**`__ldg` 從不是修復**,
> 它只是意外降低暫存器用量,讓原本啟動失敗的 LS kernel 得以啟動。正解是直接保證 LS 能啟動。

### 症狀與定位(逐步縮小)

1. 固定 seed=42 對照:無 `__ldg` 34.3% vs 有 `__ldg` 5.7%(ant 都 640)→ 排除種子變異。
2. 收斂曲線第 0 代就分岔(108% vs 11%),費洛蒙未作用 → bug 在「建解 + 2-opt LS」。
3. `--ls-block-warps=1`(強制單 warp)→ 第 0 代 108.66%→**9.97%** → 指向多 warp LS。
4. **關鍵**:完整跑的 GPUTimer 顯示 **`Local search time ≈ 0.0038 ms`(event 計時、為真)**
   → 多 warp LS **根本沒執行**。

### 根因(`two_opt_nn` 啟動失敗,`src/mmas.cu`)

`ls_warps_per_block = clamp(ceil(n/320), 1, 32)`(`mmas.cu:~3408`):
- att48 等小實例 → 1 warp(32 threads)→ 啟動 OK → LS 正常。
- **pla85900 → 32 warp = 1024 threads/block**。`two_opt_nn` 暫存器很重(> 64 regs/thread),
  `1024 × regs > 65536`(V100 每 block 暫存器檔上限)→ 啟動回傳
  `cudaErrorLaunchOutOfResources`、**kernel 沒跑**。而 LS launch **沒有任何錯誤檢查**,
  失敗被靜默吞掉 → tour 不被優化 → 卡在建解的 ~108%/34%。
- **`__ldg` 的真正機制**:read-only load 釋放暫存器,把 `two_opt_nn` 壓到 ≤ 64 regs/thread →
  1024-thread LS 終於能啟動 → 5.7%。這才是 Heisenbug 的本質(與「選錯移動」無關)。

這解釋了**為何所有舊版本在 pla85900 都卡 30%**:大實例 → 多 warp LS → 啟動失敗。

### 確認(`confirm_ls_race.slurm`,--iter=1)

| | 第 0 代 gap |
| --- | --- |
| noldg 預設(32 LS warp,LS 啟動失敗) | 108.66% |
| noldg `--ls-block-warps=1`(32 threads,LS 啟動成功) | **9.97%** |

`Local search time ≈ 0.0038 ms`(完整跑)= LS 沒在做事的鐵證。

### 修法(三項)

1. **`__launch_bounds__(32*WARP_SIZE)` 加到 `two_opt_nn`**(主修):強制編譯器把暫存器壓到
   ≤ 64/thread,**保證 1024-thread 的 LS 一定能啟動**(必要時 spill,但 LS 本來等於沒跑,
   能跑就是巨大改善)。
2. **LS launch 後加 `CUDA_CHECK(cudaGetLastError())`**:啟動失敗從此會大聲報錯,不再靜默。
3. **`(gain, thread_id)` 打包進單一 64-bit `atomicMax`**(原 race 修法):LS 現在會多 warp
   執行,所以原本「`atomicMax(gain)` 後分開寫 `thread_id`」的跨 warp race 必須一併修掉,
   否則會選錯移動。winner 一致讀回(`packed & 0xFFFFFFFF`),gain 同值由較大 tid 決勝。
   另外補上 `reverse_route_segment` 第二 case 缺的 `__syncthreads`(多 warp 才會 race)。

全部 race-free、對所有實例/seed/建置正確、不需 `__ldg`、不犧牲 LS 多 warp 平行度。

### 待驗證(`sbatch run.slurm`,pla85900,`--ants=0`,預設 32 LS warp)

- gap 應達 ~5–6%(與單 warp / `__ldg` 同級),且 **`Local search time` 不再是 ~0**。
- 若 `__launch_bounds__` 仍不足以啟動,第 2 項的 `CUDA_CHECK` 會明確報
  `cudaErrorLaunchOutOfResources`(而非靜默卡 30%)。
- 編譯時 `-Xptxas=-v` 會印 `two_opt_nn` 的暫存器數(在 `.err`),可核對是否 ≤ 64。

---

## 結果與記憶體估算

pla85900 在預設 `mmas_rwm_bt_cl` + `--ants=100` 下,GPU 端只剩:

| 項目 | 大小 |
| --- | --- |
| `d_pheromone`(n×n) | 27.49 GiB |
| 雜項(座標、cand lists、ant routes、RNG…) | ~0.3 GiB |
| CUDA context | ~0.5 GiB |
| **合計** | **~30.3 GB < 34.36 GB(32 GiB)** |

→ 應可放進 V100 並正確執行。要再大(例如完整 n² 費洛蒙塞不下)需 **修法⑤:
候選清單式費洛蒙(n × cand_list_size 取代 n² 費洛蒙)**,即論文針對大型實例的做法。

---

## 待驗證

本機無 GPU,以下需在叢集 `sbatch run.slurm` 確認:
1. **可編譯**(sm_70)。
2. **att48 等小實例結果不變**(64-bit cast 對小 n 是 identity;double 鏡像給位元一致距離)。
3. **pla85900 跑得完、gap 合理**。

跑完若要看瓶頸:`sbatch profile.slurm`(目標改 pla85900、帶 `--ants=100`)。
