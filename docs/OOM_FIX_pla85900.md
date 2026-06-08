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

## 0.5 迭代 pipeline 速查(本檔多處引用的 build / LS kernel 是哪一步)

`run_gpu_based_mmas` 每輪 iteration 依序發 7 支 kernel(每步一支),全檔常提到的
「build kernel」「LS kernel」分別是第 3、4 步:

| 步驟 | kernel | 暱稱 | 做什麼 | JSON 計時欄位 |
| --- | --- | --- | --- | --- |
| 1 | `update_pheromone_heuristic_product_cache` | | 重算 τ^α·η^β 快取(僅非 `_cl`,`_cl` 跳過) | `product-cache-update` |
| 2 | `update_cand_lists_pheromone_heuristic_product_cache` | | 候選清單版快取更新(`_cl` 用) | `cand-list-product-cache-update` |
| **3** | `build_ant_solution[_using_cand_lists]` | **Build kernel** | **每個 block 建一隻螞蟻的 tour**(看 cand_list + 費洛蒙挑下一城市) | `solution-build` |
| **4** | `two_opt_nn` | **LS kernel** | **對剛建好的 tour 做 2-opt 局部搜尋**(配 don't-look bits) | `local-search` |
| 5 | `update_global_best_and_trail_limits` | | 更新 global best、重算 τ_min/τ_max | `best-solution-update` |
| 6 | `evaporate_pheromone` | | 費洛蒙蒸發 | `pheromone-evaporate` |
| 7 | `deposit_pheromone` | | 把費洛蒙灑在 best 螞蟻的路徑上 | `pheromone-deposit` |

- **Build(第 3 步)** 佔 GPU 最大宗 —— 效能優化①/③ 動的是它裡面的 `get_heuristic`;占用率討論
  (1 warp/block、achieved 1.95%)講的也是它。
- **LS(第 4 步)** 是接力的下一支獨立 kernel —— 修法⑥ 修的是它在大實例靜默不啟動;它每 block
  最多 32 warp,與 build 的 1 warp/block 是兩回事。
- 兩者**前後接力**:第 3 步先把 tour 建出來,第 4 步再拿那條 tour 去 2-opt。

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
        // ⚠ 已於「效能優化① — 砍 FP64 pow」改寫:整數 β 用 float 連乘、
        //   非整數退 __powf,移除 FP64 pow。完整版見下方該章節。
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

## 效能優化② — 自動填滿 GPU 的 ant 數(方向 2,2026-06-06)

> build kernel 是 latency / occupancy bound:warp 數 = ant 數(`blocks_count = ants_count`),
> 預設只有 100 隻 → 每個 scheduler 平均 < 2 active warp,achieved occupancy 僅 ~1.95%
> (理論天花板 12.5%),長延遲指令完全藏不住。

### 改法(`src/mmas.cu` / `src/main.cc`)

1. **`--ants=0` ⇒ 自動依 GPU 大小決定 ant 數**(新增 `compute_auto_ant_count`):取
   `min(占用率目標 SM×8/block-warps, 記憶體預算, n)`。預設仍 100,既有 benchmark 不受影響;
   大實例請用 `--ants=0` 開到滿。
2. **LS 專用緩衝依 `--ls` gate**:`d_pos_in_routes` / `d_dont_look_bits` 在不開 LS 時配置 0,
   並刪掉死碼 `d_temp_ant_routes`。省下 `ants×n×(4+1) B`,讓自動 ant 數能開更大。

### 實測(`sbatch profile.slurm --ants=0`,pla85900)

- `--ants=0` → 640 隻 ant:achieved occupancy 1.95% → **12.3%**(逼近 bt 天花板 12.5%);
  per-ant 時間 ↓ ~5.5×。
- ⚠ ncu 的 Duration 是「一次 launch = 全部 ant」,ant 變多 launch 時間變長是正常,要看
  **per-ant throughput**。`target_warps_per_sm = 8`(≈640 ant)即足以吃滿占用率,不需 2× 超額。

### 端到端結果分析(`sbatch run.slurm`,pla85900,1000 iter,LS 開,2026-06-07)

⚠ profile 的 occupancy/per-ant 紅利**不等於端到端加速**。完整跑(修法⑥ LS 已正常)對照
`--ants=100` vs `--ants=0`(640 隻):

| 指標 | `--ants=100` | `--ants=0`(640) | 變化 |
| --- | --- | --- | --- |
| **gap** | 5.745% | 5.782% | 幾乎相同(100 還略好) |
| **total / mean-iter time** | 1307 s / 1.307 s | 2631 s / 2.631 s | **↑ ~2.01×** |
| solution-build kernel | 0.921 s | 1.051 s | ↑ 1.14× |
| **local-search kernel** | 0.281 s | **1.475 s** | **↑ 5.25×** |

→ **品質持平、時間翻倍**,而翻倍幾乎全來自 LS。原因是 build 與 LS 對 ant 數的反應相反:

- **build kernel(+14%)= 優化②的紅利兌現**:ant 100→640 是 6.4× 工作,卻只多 14% →
  per-ant `6.4/1.14 ≈ 5.6×` 加速,正是 occupancy 1.95%→12.3% 的效果(本來每 block 1 warp、
  SM 閒置,多塞 ant 幾乎免費吃下)。
- **LS kernel(×5.25)拿不到這種免費午餐**:2-opt LS「一 block 顧一條完整路徑」,且 pla85900
  每 block 已開 **32 warp(1024 thread)、本就不缺 warp**(非 occupancy-bound)。多 6.4× 隻螞蟻
  = 多 6.4× 條路徑要做完整 2-opt,是**實打實的額外計算**,只能被有限 SM 近乎線性消化
  (6.4× 工作 → 5.25× 時間)。build 省的那點完全補不回來 → 端到端翻倍。

**為什麼品質沒變好**(更多螞蟻 = 同一分布抽更多 i.i.d. 樣本,**沒改變分布的品質天花板**):

- **MMAS 每輪只用「最佳那一隻」螞蟻 deposit 費洛蒙**,不是全部螞蟻。多養螞蟻**不會加速費洛蒙
  學習**,唯一作用是「每輪挑 iteration-best 時樣本池更大」,而 **best-of-N 對 N 報酬遞減**:
  100→640 最佳值只邊際變好。
- 每條路徑都被 **2-opt 打磨到相近的 local optimum**,640 個起點幾乎都掉進同一批盆地 →
  **best-of-640 ≈ best-of-100**,故 gap 幾乎一字不差(5.745 vs 5.782)。品質由「2-opt 盆地 +
  費洛蒙動態」決定,**不是樣本數量**;多抽逃不出盆地。

> **結論**:優化② 是 build kernel 的「吞吐/占用率」旋鈕,**不是品質槓桿**(故預設維持 100)。
> `--ants=0` 只有在「build kernel 是唯一瓶頸、且不開或極輕量 LS」時才划算;一旦 LS 正常運作
> (修法⑥),它隨 ant 數線性膨脹,`--ants=0` 反而品質持平、時間翻倍。

---

## 效能優化③ — heuristic 距離改 float `sqrtf`(方向 1b,2026-06-06)

> 效能優化① 砍掉 FP64 `pow` 後,build kernel 剩餘的 FP64 幾乎都是 `get_distance` 的
> double `sqrt`,熱點在「候選清單近鄰用盡 → 掃整個 unvisited」的 fallback,每個節點都呼叫。

### 改法(`src/mmas.cu`)

新增**只給 heuristic 用**的單精度距離 `device_node_distance_f()`(用 `sqrtf`/`ceilf`/`cosf`
/`acosf`),`get_heuristic` 改用它:

```cpp
const float dist = (coordinates_ != nullptr)
    ? device_node_distance_f(edge_weight_type_, coordinates_, from, to)  // 1b
    : static_cast<float>(get_distance(from, to));                        // 退路
```

- **tour cost 路徑完全不動**:route length / 2-opt 仍走 double 的 `get_distance` /
  `device_node_distance`,成本仍對齊 TSPLIB 取整與 best-known。heuristic 只是選點權重,
  float 足夠。

### 實測(`sbatch profile.slurm --ants=0`,pla85900,640 ant)

- FP64 指令 ~3.38e9 → **~2.06e7/launch**(roofline 報 fp64「close to 0%」),dir 1b 成功。
- 瓶頸位移到記憶體 latency(L1TEX scoreboard ~40%);曾試 `__ldg` 但**實測無效**
  (Volta read-only cache 即同一塊 L1,工作集遠超快取),已捨棄。

---

## 修法⑥(bug 修) — 2-opt LS 在大實例「靜默不啟動」(pla85900 解品質卡 30% 的真因,2026-06-07)

OOM 修好後,pla85900 仍長期卡在 **~30% gap**,而小實例(att48…)正常。這其實是一個與
記憶體無關、長期存在的 **2-opt local search bug**。

### 症狀與定位

- 完整跑的 GPUTimer(CUDA event 計時、為真)顯示 **`Local search time ≈ 0.0038 ms`**
  → 多 warp 的 LS **根本沒執行**;iter 0 的 ~108% 就是「純建解、完全沒 local search」。
- 強制單 warp(`--ls-block-warps=1`)時:iter 0 從 108.66% → **9.97%** → LS 一旦能跑就正常。

> **「iter 0 = 108%」怎麼讀**:LS launch 被默默吞掉時,tour 完全沒被 2-opt 改善,iter 0 印出的
> 108% **數值上等價於「沒套 2-opt 的純建解品質」**。>100% 代表長度超過最佳解兩倍多,在大實例、
> iter 0(費洛蒙還全是 τ_max、幾乎只靠 heuristic+隨機建解)且無 local search 時很正常。注意這
> **不是**刻意的 no-LS 跑法 ——程式開了 `--ls=1`、以為在跑 LS,只是 launch 失敗;但結果與不跑
> 2-opt 一致。佐證:強制單 warp LS(能啟動)→ iter 0 108.66% → 9.97%;修好後 final gap
> 34.62% → 5.78%。

### 根因(`two_opt_nn` 啟動失敗,`src/mmas.cu`)

`ls_warps_per_block = clamp(ceil(n/320), 1, 32)`(`src/main.cc:3439`,`--ls-block-warps=0` 自動模式):
- 小實例 → 1 warp(32 threads)→ 啟動 OK。
- **pla85900 → 32 warp = 1024 threads/block**。`two_opt_nn` 暫存器很重(> 64 regs/thread),
  `1024 × regs > 65536`(V100 每 block 暫存器檔上限)→ 啟動回傳
  `cudaErrorLaunchOutOfResources`、**kernel 沒跑**。而 LS launch **沒有任何錯誤檢查**,
  失敗被靜默吞掉 → tour 不被優化 → 卡在建解的 ~30%。

> **別把 LS 的 warp 數和 build kernel 搞混**:這支 `two_opt_nn` 與建解 kernel 是**兩支不同的
> kernel**,block 配置也不同 —— LS 的 32 warp 跟 `cand_list_size` 無關。
>
> | | build kernel(建解) | `two_opt_nn`(2-opt LS) |
> | --- | --- | --- |
> | 一個 block 算什麼 | 一隻螞蟻建一條 tour | 對一條 tour 做 2-opt 改善 |
> | block 大小 | `warps_per_block×32` = `cand_list_size` = **1 warp(32 threads)** | `ls_warps_per_block×32`,**最多 32 warp(1024 threads)** |
> | warp 數由誰決定 | `cand_list_size`(固定 32,一 lane 對一候選 slot) | **n**:`clamp(ceil(n/320),1,32)`,用多 warp 平行掃整條路徑 |
> | 大/小實例差別 | 都是 1 warp | att48→`ceil(48/320)`=1 warp;pla85900→`ceil(85900/320)`→clamp=32 warp |
>
> 所以「pla85900 → 32 warp」是 **LS** 因路徑長(85900 節點、~10 節點/thread)配到上限,
> 1024-thread block 才撐爆暫存器檔;小實例只配 1 warp 遠低於上限,故一直正常。build kernel
> 那 1 warp/block 是另一回事(也是占用率討論裡 warp 不足的主角),別與此處混為一談。

#### `ceil(n/320)` 怎麼拆 —— `10*32` 的真正語意

`src/main.cc:3439` 的註解寫「1 warp per 10 nodes」其實**不精確**;工作分配的單位是 **thread**,
不是 warp。`10 * 32 = 320` 應該這樣拆:

```
10 * 32  =  (每個 thread 顧 ~10 個 node)  ×  (每個 warp 32 個 thread = WARP_SIZE)
         =  一個 warp 能顧的 node 數  =  320

ls_warps_per_block = ceil( n / 320 ) = 「一個 block 要幾個 warp 才能掃完整條 n 節點路徑」
```

- 一個 block 處理一隻螞蟻的整條路徑(n 節點)→ 想「1 thread 顧 ~10 節點」就需要 `n/10` 個 thread。
- thread 打包成 warp,**每 warp 32 thread** → 需要 `(n/10)/32 = n/320` 個 warp。
- ⚠ 這裡的 **`/32` 是「thread→warp 的硬體單位換算(WARP_SIZE)」,不是「假設 block 固定 32 warp」**。
  算出來的 `n/320` 本身就是 warps-per-block,不需另外假設 block 多大。
- 真正寫死 32 的是後面的 `min(32u, ...)`:**一個 block 上限 1024 thread = 32 warp** 是硬體上限。
  對大實例 `n/320` 早超過 32,故永遠夾到 32 → 等於「直接開最大 block」。

合理性:**方向對但很粗**。節點越多→越多 thread 平行掃 2-opt 候選,直到 block 容量上限,這是對的;
但「10 節點/thread」只是經驗粒度、無理論最佳值,所以才開 `--ls-block-warps` 讓你手動覆寫。對
pla85900 這條公式其實退化成「永遠 1024 thread」,真正的調校空間在那個 clamp 上限。

#### V100 的兩種暫存器上限 —— 撞到的是「每 block」那條

per-thread 與 per-block 是**兩個不同層級**的限制,都存在;這個 bug 撞的是 **per-block**:

| 限制 | V100(sm_70) | 意義 |
| --- | --- | --- |
| 每 **block** thread 數上限 | **1024** threads/block | 一個 block 最多開幾個 thread(= 32 warp) |
| 每 **thread** 暫存器上限 | **255** regs/thread | 單一 thread 最多用幾個暫存器(架構硬上限) |
| 每 **block** 暫存器上限 | **65536** regs/block | 一個 block 內所有 thread 加總(= 整個 SM 的暫存器檔大小) |
| 每 **SM** 暫存器檔 | 65536 個 32-bit(256 KB) | 一個 SM 的暫存器總量 |

```
1024 threads/block × (>64 regs/thread)  >  65536  → 超過「每 block」上限 → cudaErrorLaunchOutOfResources
```

- **不是** per-thread 255 那條被違反 —— 每 thread 可能才 ~80 regs(遠低於 255),問題是 **1024 個
  thread 加總**超過 block 的 65536 總額。
- `__launch_bounds__(32 * WARP_SIZE)`(=1024)之所以有效:它告訴編譯器「block 最多 1024 thread」,
  編譯器便把暫存器壓到 **≤ 65536 / 1024 = 64 regs/thread**,保證「block 總暫存器 ≤ 65536」成立 →
  1024-thread LS 一定能啟動(代價是必要時 spill 到 local memory)。

#### 釐清「踩到上限」vs「超過上限」—— bug 不是因為 thread 數沒限制

常見誤解:「V100 沒限制 block 能開幾個 thread,才害 1024-thread 撐爆」。**錯。** V100 有 thread
數上限(1024),程式的 `min(32u, ...)` 正是在尊重它。三條限制各自獨立、檢查時機也不同,要分清
「踩到(=,合法)」與「超過(>,違法)」:

| 限制 | 規則 | pla85900 的 1024-thread block | 結果 |
| --- | --- | --- | --- |
| 每 block thread 數 | ≤ 1024 | 1024 = 1024 | **踩到上限但沒超過 → 合法** |
| 每 thread 暫存器 | ≤ 255 | ~80 | 沒超過 → 合法 |
| 每 block 暫存器 | ≤ 65536 | 1024 × (>64) > 65536 | **超過 → 啟動失敗** |

- **thread 數合法是必要、不是充分條件**:1024 正好頂到天花板(等號成立仍合法),所以 thread 數
  **不是**失敗主因;真正**被超過**的是「每 block 暫存器」這條獨立限制。
- 但兩條會互相牽制:正因 thread 數**踩滿 1024**(無法再少),暫存器那條的分母被鎖死,只要每 thread
  暫存器 >64 就必爆。也就是「thread 數頂到 1024」本身合法,卻把暫存器的容錯空間壓到零。
- 所以正確因果是「**滿足 thread 數上限 ≠ 滿足暫存器上限,兩者是分開的約束**」,而非「缺少 thread 數
  上限」。`__launch_bounds__` 就是把暫存器那一側壓回 ≤64,讓三條同時成立。

### 修法(三項,皆在 `two_opt_nn` / LS launch 附近)

1. **`__launch_bounds__(32 * WARP_SIZE)` 加到 `two_opt_nn`**(主修):強制編譯器把暫存器壓到
   ≤ 64/thread,**保證 1024-thread 的 LS 一定能啟動**(必要時 spill;LS 本來等於沒跑,能跑
   就是巨大改善)。
2. **LS launch 後加 `CUDA_CHECK(cudaGetLastError())`**:啟動失敗從此會立刻報錯,不再靜默。
3. **best-move 選擇改用單一 64-bit `atomicMax`**:LS 現在會多 warp 執行,原本
   「`atomicMax(&chosen_move_gain)` 後*分開*寫 `chosen_move_thread_id`」兩步非原子的跨 warp
   race(多 warp 才會觸發)必須一併修掉 —— 把 `(gain 的 float 位元樣式, thread id)` 打包進
   一個 64-bit 值,一次原子定出 gain+擁有者,winner 一致讀回。另補上 `reverse_route_segment`
   第二 case 缺的 `__syncthreads`(多 warp 才會 race)。

### 實測結果(`sbatch run.slurm`,pla85900,2026-06-07)

| | 修法前 | 修法後 |
| --- | --- | --- |
| Final gap | 34.62% | **5.78%** |
| iter 0 | 108.66% | 11.25% |
| `Local search time` | 0.0038 ms(沒跑) | 1373.6 ms(正常) |

→ 30% 老 bug 解決;大實例的 LS 終於正常運作。小實例行為不變(本來就單 warp,只是現在也
完全確定性)。

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
