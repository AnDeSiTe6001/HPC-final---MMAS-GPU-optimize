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

### 待驗證(`sbatch profile.slurm` 後比對 `profiles/*_build_bound.txt`)

- 預期 FP64 指令數大降、Compute 吞吐下降、Duration 縮短。
- 若瓶頸轉到 double `sqrt`:再加 heuristic 專用的 float 距離(cost 路徑維持 double)。
- 若 Duration 改善有限、occupancy 仍 ~2%:確認是「warp 太少」的結構問題 →
  進入**方向 2**(每 ant 多 warp / 一 block 多 ant / 提高 ant 數);
  注意單純降 shared memory 無效(block 總數 100 已 < SM 容量)。
- 跳過方向 3:uncoalesced 雖 44% 但 memory 吞吐才 1%,非瓶頸。

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
