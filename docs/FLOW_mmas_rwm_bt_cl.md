# `mmas_rwm_bt_cl` 逐行流程追蹤

本文件針對**預設演算法 `mmas_rwm_bt_cl`** 在**預設參數**下，把一次完整 trial 中「每個迭代（per-step / per-iteration）」實際執行的每一個 kernel、每一行關鍵程式碼追一遍。

所有行號對應 `src/mmas.cu`（撰寫時的版本）。

---

## 0. 前置條件：這個變體是什麼

`mmas_rwm_bt_cl` 拆解為三個維度 + 一個後綴：

| 部分 | 值 | 意義 |
|------|----|------|
| select | `rwm` | Roulette Wheel Method（傳統前綴和輪盤） |
| tabu | `bt` | `BitmaskTabu`（每節點 1 bit） |
| `_cl` | 有 | 用 candidate list（最近鄰候選清單）加速選點 |

### 0.1 預設參數（決定走哪條程式碼路徑）

```
--ants=100  --iter=1000  --rho=0.9
--cand-list-size=32  --block-warps=1  --ls=1  --seed=0
```

關鍵推導：

- `_cl` 的硬限制：`block-warps × 32 == cand-list-size`，且 `cand-list-size` 為 32 倍數。
- `block-warps=1` ⇒ `threads_per_block = 1 × 32 = 32 = WARP_SIZE`。
- 因此**一個 block = 一隻螞蟻 = 剛好一個 warp（32 threads）**，且**每個 thread 對應候選清單的一個 slot**。

### 0.2 樣板分派與旗標（`mmas.cu:2822` 一帶的 `mmas_rwm_bt_cl` struct）

因為 `threads_per_block == WARP_SIZE`，編譯期選到的是 **warp 版**選點函式：

```cpp
SolutionConstructionAlgorithm mmas_rwm_bt_cl {
    (threads_per_block == WARP_SIZE)
        ? build_ant_solution_using_cand_lists<
              BitmaskTabu,
              warp_roulette_choice_from_cand_list<BitmaskTabu>   // ← 走這支
          >
        : build_ant_solution_using_cand_lists<
              BitmaskTabu,
              block_roulette_choice_from_cand_list<BitmaskTabu>
          >,
    /* use_cand_lists_ */       true,    // ← 用候選清單快取，不配置 n×n product cache
    /* use_product_reciprocal_ */ false, // ← rwm 用「正常」product，不存倒數（倒數是 wrs 用的）
    { blocks_count, threads_per_block,
      BitmaskTabu::get_required_shared_memory_size_in_bytes(dimension)
        + common_shared_mem_size }
};
```

- `use_cand_lists_ = true` ⇒ 整個 run **不配置** `n×n` 的 `product_cache_`（`mmas.cu:2353` 附近 `use_cand_lists_ ? 0 : n*n`），改用小很多的 `n × cand_list_size` 候選快取 `cand_lists_product_cache_`。這是大型實例不 OOM 的關鍵。
- `use_product_reciprocal_ = false` ⇒ 快取存 `τ^α·η^β` 本身（rwm 直接拿來做前綴和），不像 wrs 存倒數。

### 0.3 一次迭代的 7 步 pipeline（主迴圈 `mmas.cu:2556` 起）

每個 `iter` 依序啟動下列 kernel（`!use_cand_lists_` 那支不會跑，因為我們是 `_cl`）：

1. **`update_cand_lists_pheromone_heuristic_product_cache`** — 更新候選清單的 `τ·η` 快取
2. **`build_ant_solution_using_cand_lists`** — 100 隻螞蟻各建一條 tour
3. **`two_opt_nn`** — 每條 tour 跑 2-opt 局部搜尋（`--ls=1`）
4. **`update_global_best_and_trail_limits`** — 找該輪最佳、更新 global/reset best、重算 τ_min/τ_max
5. **（host 端）停滯偵測 → 視情況觸發 pheromone reset**
6. **`evaporate_pheromone`** — 揮發（或 reset 時填成 τ_max）
7. **`deposit_pheromone`** — 沉積（LS 模式下隨機從 iter/global/reset best 三選一）

以下逐步逐行。

---

## Step 1 — `update_cand_lists_pheromone_heuristic_product_cache`

**啟動**：`mmas.cu:2568` `<<<dimension, 128>>>`，即一個 block 負責一個來源節點，128 threads 平行掃該節點的候選清單。
**定義**：`mmas.cu:830`。

```cpp
const auto src_node = blockIdx.x;                       // 此 block 負責的來源城市
const auto cand_list_size = instance.cand_list_size_;   // = 32

auto *pheromone   = ctx.pheromone_matrix_ + src_node * dimension;        // 該行費洛蒙
auto *cand_list   = instance.cand_lists_ + src_node * cand_list_size;     // 該節點的 32 個最近鄰
auto *product_cache = ctx.cand_lists_product_cache_ + src_node*cand_list_size;

for (uint32_t i = threadIdx.x; i < cand_list_size; i += blockDim.x) {
    const auto node    = cand_list[i];                                   // 第 i 個最近鄰
    const auto product = instance.get_heuristic(src_node, node) * pheromone[node];
    product_cache[i]   = (calc_reciprocal && product > 0) ? 1.0f/product : product;
}
```

逐行：

- `get_heuristic(src_node, node)` 回傳 `η = 1/dist^β`（`mmas.cu:773`；座標型實例會即時算，矩陣存在就直接讀）。
- 乘上 `pheromone[node]`（即 `τ`，其實已含 α=1 的效果）得到 `product = τ·η`。
- 因為本變體 `calc_reciprocal == false`（`use_product_reciprocal_=false`），所以**存 product 本身**。
- 結果：`cand_lists_product_cache_` 這個 `n×32` 表，每格是「從 src_node 走到它第 i 個最近鄰」的選擇權重，供 Step 2 直接查。

> 對照：非 `_cl` 變體會改跑 `update_pheromone_heuristic_product_cache`（`mmas.cu:863`，更新整個 `n×n`），本變體**不會**進那支（`mmas.cu:2559` 的 `if (!alg.use_cand_lists_)` 為假）。

---

## Step 2 — `build_ant_solution_using_cand_lists`（建構 tour）

**啟動**：`mmas.cu:2575` `<<<100, 32, shared>>>`，每 block 一隻螞蟻、32 threads（一 warp）。
**定義**：`mmas.cu:1398`。樣板參數 `Tabu = BitmaskTabu`、`roulette_choice_fn = warp_roulette_choice_from_cand_list`。

### 2.1 初始化（`mmas.cu:1404–1423`）

```cpp
extern __shared__ char shared_mem[];      // 動態 shared memory
__shared__ uint32_t chosen_node;

const auto dimension = instance.dimension_;
const auto ant_idx   = blockIdx.x;                       // 這隻螞蟻的編號 0..99
uint32_t *ant_route  = ctx.ant_routes_ + ant_idx * dimension;  // 它的輸出路徑緩衝（global）
rand_state_t rng     = rng_states[get_global_thread_id()];     // 載入本 thread 的 PRNG 狀態

Tabu tabu(dimension, shared_mem, ant_route);   // BitmaskTabu，visited_mask 落在 shared_mem
tabu.parallel_init();                          // 把 bitmask 清 0（見下）
char *shared_mem_buffer = shared_mem + BitmaskTabu::get_required_shared_memory_size_in_bytes(dimension);
                                               // bitmask 之後的 shared 區段，供歸約暫存

if (threadIdx.x == 0)
    chosen_node = get_random_uint32(rng, 0, dimension - 1);  // 隨機起點城市
__syncthreads();
tabu.add_visited(chosen_node);                 // 標記起點已拜訪，並寫入路徑第 0 格
auto current_node = chosen_node;
```

`BitmaskTabu::parallel_init()`（`mmas.cu` 內 `BitmaskTabu`）把 `ceil(n/32)` 個 uint32 字組全清 0、`visited_count_=0`、`out_count_=0`，再 `__syncthreads()`。

`BitmaskTabu::add_visited(node)`（位於 `BitmaskTabu` 結構）做兩件事：

1. thread 0 把 `visited_mask_[node/32] |= 1<<(node%32)` 設位；`++visited_count_`。
2. **延遲批次寫回**：用 `out_node_/out_count_` 把節點暫存在各 thread 暫存器，湊滿 `blockDim.x`（=32）個才一次寫進 `out_route_`（global memory 的 `ant_route`），減少 global 寫入次數。

### 2.2 主建構迴圈（`mmas.cu:1425–1455`）：走 `dimension-1` 步

```cpp
for (uint32_t i = 1; i < dimension; ++i) {
    auto cand_node = roulette_choice_fn(instance, rng, current_node,
                                        ctx.cand_lists_product_cache_, tabu);
    // ↑ 即 warp_roulette_choice_from_cand_list，見 Step 2.3

    if (cand_node == dimension) {        // 候選清單裡的最近鄰全被拜訪過 → 回退
        auto *pheromone = ctx.pheromone_matrix_ + current_node * dimension;
        const auto length = tabu.get_length();   // = dimension（bitmask 不壓縮，永遠掃全部）
        float cand_product = -1;
        for (uint32_t i = threadIdx.x; i < length; i += blockDim.x) {
            const auto node = tabu.get_candidate(i);          // BitmaskTabu: get_candidate(i)==i
            if (tabu.is_candidate_unvisited(node)) {          // 等於 !is_visited(node)
                const float product = pheromone[node] * instance.get_heuristic(current_node, node);
                cand_product = max(cand_product, product);
                cand_node = cand_product == product ? node : cand_node;
            }
        }
        cand_node = block_arg_max(cand_node, cand_product, shared_mem_buffer);  // 取 product 最大者（貪婪）
    }
    tabu.add_visited(cand_node);   // 標記、寫回路徑
    current_node = cand_node;
}
```

關鍵理解：

- **正常路徑**：每步只在當前城市的 32 個最近鄰裡用輪盤抽（Step 2.3），極快。
- **回退路徑**（`cand_node == dimension`）：當 32 個最近鄰全拜訪過，才退化成「掃全部未拜訪節點、直接挑 `τ·η` 最大者」的**貪婪**選擇（不是輪盤）。這裡用到 `BitmaskTabu` 的特性：它不維護壓縮清單，`get_length()` 永遠是 `dimension`、`get_candidate(i)==i`，所以這個回退迴圈會掃過 0..n-1 全部編號再用 bitmask 過濾。
- `block_arg_max`（`mmas.cu:239`）對一個 warp 做 argmax 歸約（內部用 `warp_reduce_arg_max` + `warp_bcast`），讓全 block 得到同一個贏家節點。

### 2.3 選點核心：`warp_roulette_choice_from_cand_list`（`mmas.cu:1221`）

這是 `rwm + _cl + warp` 的精華。**每個 thread 對應候選清單的一個 slot**（`assert(blockDim.x == cand_list_size_)`、`assert(get_warp_count()==1)`）。

```cpp
const auto offset    = cand_list_size_ * current_node;
const auto cand_list = instance.cand_lists_ + offset;        // 當前城市的 32 個最近鄰
const auto cand_index = threadIdx.x;                         // 本 thread 負責第 threadIdx.x 個
const auto cand_node  = cand_list[cand_index];
const bool is_visited = tabu.is_visited(cand_node);          // 查 bitmask（O(1)）
const auto product_cache = cand_lists_product_cache + offset;
const auto product = is_visited ? 0 : product_cache[cand_index];  // 已拜訪者權重歸 0
```

逐行（全部用 warp shuffle，不碰 shared memory）：

```cpp
const auto prefix_sum = warp_scan(product);   // 32-lane inclusive 前綴和（mmas.cu:113，shfl_up 階梯）
const auto unvisited_mask = __ballot_sync(FULL_MASK, is_visited == false);  // 哪些 slot 還沒拜訪
const auto last_unvisited_pos = 32 - __clz(unvisited_mask);  // 最後一個未拜訪 slot 的位置（1-based）

if (last_unvisited_pos) {                      // 還有可選的最近鄰
    float chosen_point;
    if (threadIdx.x + 1 == last_unvisited_pos) {     // 「總和」就在最後一個未拜訪 lane 的 prefix
        const auto total = prefix_sum;
        const auto r = get_random_float(rng);
        chosen_point = min(total, total * r);        // 在 [0, total) 抽一點
    }
    chosen_point = warp_bcast<float>(chosen_point, last_unvisited_pos - 1);  // 廣播給整個 warp

    const bool is_valid = (chosen_point <= prefix_sum);            // 我的累積和是否已越過抽樣點
    const auto chosen_pos = warp_vote(is_valid, unvisited_mask);   // 第一個越過的「未拜訪」lane（mmas.cu:194，__ffs∘__ballot）
    chosen_node = warp_bcast<uint32_t>(cand_node, chosen_pos - 1); // 那個 lane 的節點即為選中
}
return chosen_node;     // 若 last_unvisited_pos==0（全拜訪），回傳 dimension → 觸發 2.2 的回退
```

要點：

- 這是道地的輪盤法：以 `product` 為權重做前綴和，抽一個落點，找第一個累積和 ≥ 落點且「未拜訪」的 slot。
- 整個過程只用 `__shfl_*` / `__ballot_sync`（warp 內建），所以 1-warp 配置最有效率——這正是預設 `block-warps=1` 讓它走 warp 版而非 block 版的理由。
- `warp_vote` 用 `unvisited_mask` 當 active mask，確保不會選到已拜訪 lane（即使浮點誤差讓某個已拜訪 lane 的 prefix 也 ≥ 落點）。

### 2.4 收尾：寫回路徑與算長度（`mmas.cu:1456–1470`）

```cpp
tabu.parallel_finalize();   // BitmaskTabu: 把暫存器裡尚未湊滿的尾段節點 flush 進 ant_route

float my_sum = 0;           // 平行算 tour 長度
for (uint32_t i = threadIdx.x; i < dimension; i += blockDim.x) {
    const auto node = ant_route[i];
    const auto next_node = ant_route[(i + 1) % dimension];
    my_sum += instance.distance_matrix_[node * dimension + next_node];
}
my_sum = block_sum(my_sum, shared_mem_buffer);   // warp/block 歸約成總長
if (threadIdx.x == 0) ctx.ant_route_costs_[ant_idx] = my_sum;   // 寫回這隻螞蟻的成本
rng_states[get_global_thread_id()] = rng;        // 回存 PRNG 狀態（下一輪續用）
```

Step 2 結束後：`ant_routes_`（100×n）裝著 100 條 tour、`ant_route_costs_` 裝著對應成本。

---

## Step 3 — `two_opt_nn`（GPU 2-opt 局部搜尋，`--ls=1`）

**啟動**：`mmas.cu:2586` `<<<100, ls_threads, shared>>>`，每 block 一條 tour。`ls_warps_per_block` 由 `--ls-block-warps` 決定（0 時用啟發式 `ceil(n/(10·32))`，上限 32）。
**定義**：`mmas.cu:2130`。是帶 **don't-look bits** 的最近鄰 2-opt。

### 3.1 初始化（`mmas.cu:2145–2162`）

```cpp
uint32_t *route        = all_routes + blockIdx.x * dimension;   // 這條 tour
int32_t  *pos_in_route = all_pos_in_route + blockIdx.x*dimension; // node→它在 route 的索引（反查表）
int8_t   *dont_look_bits = all_dont_look_bits + blockIdx.x*dimension;

for (i = threadIdx.x; i < dimension; i += blockDim.x) {
    pos_in_route[route[i]] = i;     // 建反查表
    dont_look_bits[i] = 0;          // 全部清 0（每條 tour 重新開始）
}
// thread 0 重置：chosen_move_thread_id=blockDim.x(哨兵)、chosen_move_gain=0、any_candidate_nodes=false
```

### 3.2 改善迴圈（`mmas.cu:2170` 起的 `while(improvement_found)`）

核心思想：對每個「還沒被標 don't-look」的節點 `a`，檢查它的候選清單（最近鄰）`b`，看能否做出讓 tour 變短的 2-opt 交換（反轉一段）。

每個 warp 認領一批節點 `a`；對每個 `a`：

```cpp
const auto a_succ = route[i+1];  const auto dist_a_to_succ = dist(a, a_succ);  // a 的後繼邊
const auto a_pred = route[i-1];  const auto dist_a_to_pred = dist(a, a_pred);  // a 的前繼邊
const auto cand_list = instance.cand_lists_ + cand_list_size_ * a;
assert(cand_list_size_ <= WARP_SIZE);            // 每個 lane 檢查一個最近鄰 b

if (cl_i < cand_list_size_) {
    const auto b = cand_list[cl_i];
    const auto dist_ab = dist(a, b);
    const auto b_pos = pos_in_route[b];

    if (dist_a_to_succ > dist_ab) {              // 只有「新邊更短」才可能有增益（NN 剪枝）
        // 嘗試斷 (a,a_succ)+(b,b_succ)、接 (a,b)+(a_succ,b_succ)
        diff = dist_a_to_succ + dist(b,b_succ) - dist_ab - dist(a_succ,b_succ);
        if (diff > max_gain) { left = min(i,b_pos)+1; right = max(i,b_pos)+1; max_gain = diff; }
    }
    if (dist_a_to_pred > dist_ab) {              // 對稱地檢查前繼方向
        diff = dist_a_to_pred + dist(b_pred,b) - dist_ab - dist(a_pred,b_pred);
        if (diff > max_gain) { left = min(i,b_pos); right = max(i,b_pos); max_gain = diff; }
    }
}

if (warp_vote(max_gain > 0) == 0) dont_look_bits[a] = 1;   // a 找不到任何改善 → 設 don't-look
else {                                                      // 有增益：選 warp 內最大者，再 atomicMax 競選 block 冠軍
    auto warp_max = warp_all_reduce_max(max_gain);
    if (max_gain == warp_max && max_gain > 0)
        if (atomicMax(&chosen_move_gain, max_gain) < max_gain) chosen_move_thread_id = threadIdx.x;
}
```

選出全 block 增益最大的一步後（`mmas.cu:2291` 起）：

```cpp
if (chosen_move_thread_id != blockDim.x) {        // 有人提出有效 move
    // 冠軍 thread 把要反轉的區段 [left,right) 寫進 move_beg/move_end（shared）
    reverse_route_segment(left, right, dimension, route, pos_in_route);  // 平行反轉該段並同步更新 pos_in_route
    if (threadIdx.x == 0) {
        // 反轉端點附近 4 個節點的 don't-look bit 清 0（它們的鄰接變了，要重新檢查）
        dont_look_bits[route[left]] = 0; dont_look_bits[route[right-1]] = 0;
        dont_look_bits[route[left-1]] = 0; dont_look_bits[route[right]] = 0;
        chosen_move_thread_id = blockDim.x; chosen_move_gain = 0; any_candidate_nodes = false;
    }
    improvement_found = true;  break;   // 套用一步後跳出內層、外層再掃一遍
} else {
    // 沒人提出 move：若這輪根本沒有候選節點（全部 don't-look）就結束，否則繼續找
    if (!any_candidate_nodes) break;
}
```

### 3.3 收尾（`mmas.cu:2337–2342`）

```cpp
const auto final_cost = par_calculate_route_length(instance, route, shared_mem);
if (threadIdx.x == 0) *route_cost = final_cost;   // 用 2-opt 後的新長度覆蓋 ant_route_costs_[blockIdx.x]
```

要點：

- 一次只套用「全 block 增益最大」的一步，反轉後清掉受影響端點的 don't-look bit，重新掃；直到沒有任何改善為止（局部最優）。
- don't-look bits 大幅減少重複檢查，是 2-opt 在大實例上能跑的關鍵。
- 完成後 `ant_route_costs_` 已是 LS 後的成本，餵給 Step 4。

---

## Step 4 — `update_global_best_and_trail_limits`

**啟動**：`mmas.cu:2601` `<<<1, 32, shared>>>`，單 block 單 warp。
**定義**：`mmas.cu:1808`。

```cpp
// 4.1 平行掃 100 個成本，找該輪最小（iteration best）
float min_cost = route_costs[0]; uint32_t iter_best_index = 0;
for (i = threadIdx.x; i < route_count; i += blockDim.x)
    if (min_cost > route_costs[i]) { min_cost = route_costs[i]; iter_best_index = i; }
const auto best_route_idx = block_arg_max(iter_best_index, -min_cost, shared_mem);  // argmin = argmax(-cost)

if (threadIdx.x == 0) {
    const auto best_cost = route_costs[best_route_idx];
    *ctx.iter_best_cost_ = best_cost;
    // 複製該輪最佳路徑到 iter_best_route_
    for (i...) ctx.iter_best_route_[i] = best_route[i];

    // 4.2 更新 global best（且重算 trail limits！）
    if (best_cost < *ctx.global_best_cost_) {
        *ctx.global_best_cost_ = best_cost;
        for (i...) ctx.global_best_route_[i] = best_route[i];
        update_best_solutions_log(ctx, best_cost, iteration);            // 記錄 (cost, iter) 供輸出
        *trail_limits = calc_trail_limits(dimension, best_cost,
                                          params.get_evaporation_rate(), params.p_best_);
    }

    // 4.3 更新 reset best（自上次 pheromone reset 以來的最佳）
    if (best_cost < *ctx.reset_best_cost_) {
        *ctx.reset_best_cost_ = best_cost;
        for (i...) ctx.reset_best_route_[i] = best_route[i];
    }
}
```

`calc_trail_limits`（`mmas.cu:663`，MMAS 論文 Eq.11）：

```cpp
tau_max = 1 / (best_cost * evaporation_rate);             // 上限隨「全域最佳成本」變動
avg     = dimension / 2.0f;
p       = pow(p_best, 1.0 / dimension);
tau_min = min(tau_max * (1-p) / ((avg-1)*p), tau_max);    // 下限
```

要點：MMAS 的招牌——費洛蒙被夾在 `[τ_min, τ_max]`，且**只有在刷新 global best 時才重算上下限**（因為 τ_max 綁定 global best 成本）。Step 6/7 會用到這組 limits。

> 注意：`iter_best_cost_`、`reset_best_cost_`、`global_best_cost_` 三者同時維護，後面沉積要從中三選一。

---

## Step 5 —（Host 端）停滯偵測與 pheromone reset 決策

**位置**：`mmas.cu:2609–2643`，在 CPU 上判斷（每 10 個 iter 讀一次 device 成本）。

```cpp
if (iter % 10 == 0) {
    best_cost_since_reset = d_reset_best_cost.as_host_vector().front();
    ...
    if (prev_best_cost == best_cost_since_reset) ++stagnation_counter;   // reset best 沒進步 → 累加
    else { stagnation_counter = 0; prev_best_cost = best_cost_since_reset; }
}

bool reset_pheromone = false;
if (use_local_search
    && stagnation_counter * 10 >= stagnation_period          // 停滯夠久
    && (iter - reset_iter) <= (iterations - iter)) {         // 且離結束還有足夠時間值得 reset
    reset_pheromone = true;
    stagnation_counter = 0; reset_iter = iter;
    zero_reset_route_cost<<<1,32>>>(mmas_ctx);               // 把 reset_best_cost_ 設回 FLT_MAX
    cout << "Resetting pheromone at: " << reset_iter << "\n";
}
```

要點：

- 只有開 LS 時才有 reset 機制（純費洛蒙模式沒有）。
- 連續 `stagnation_period` 迭代「自上次 reset 以來最佳」毫無進步 ⇒ 視為陷入局部最優，準備重置費洛蒙重新探索。
- `zero_reset_route_cost`（`mmas.cu:1875`）把 `reset_best_cost_` 歸回無限大，讓下一輪重新累積 reset best。

---

## Step 6 — `evaporate_pheromone`（揮發 / 重置）

**啟動**：`mmas.cu:2646` `<<<dimension, 256>>>`，一 block 一列。
**定義**：`mmas.cu:1890`。

```cpp
const auto node = blockIdx.x;
const float min_trail = trail_limits->min_;
for (endpoint = threadIdx.x; endpoint < dimension; endpoint += blockDim.x) {
    if (reset) {
        pheromone[node*dim + endpoint] = trail_limits->max_;          // reset：整張表填成 τ_max
    } else {
        const auto pher = pheromone[node*dim + endpoint];
        pheromone[node*dim + endpoint] = max(min_trail, pher * (1 - evaporation_rate));  // 揮發並夾下限
    }
}
```

要點：

- 一般情況：`τ ← max(τ_min, τ·(1-ρ_evap))`。預設 `--rho=0.9`，`get_evaporation_rate()` 對應 `1-rho=0.1`（每輪揮發 10%），且不低於 `τ_min`。
- 若 Step 5 觸發 reset：整張費洛蒙表直接填 `τ_max`（最大化探索，等效「重新開始」）。

---

## Step 7 — `deposit_pheromone`（沉積）

**定義**：`mmas.cu:1918`。沉積量 `Δτ = 1 / route_cost`，加到該解經過的每條邊上並夾上限 `τ_max`；`is_symmetric=true` 時雙向都加。

```cpp
const auto deposit = 1 / *route_cost;
const auto max_trail = trail_limits->max_;
for (i = threadIdx.x; i < dimension; i += blockDim.x) {
    auto node = route[i], next = route[(i+1)%dimension];
    auto trail = fmin(max_trail, pheromone[node*dim+next] + deposit);
    pheromone[node*dim+next] = trail;
    if (is_symmetric) pheromone[next*dim+node] = trail;
}
```

**來源解的選擇**（`mmas.cu:2656–2685`，與 LS 連動，是這個變體最關鍵的演算法決策之一）：

```cpp
if (!use_local_search) {
    // 純費洛蒙模式：固定用「該輪最佳（iteration best）」
    deposit_pheromone<<<1,256>>>(dim, d_pheromone, d_iter_best_cost, d_iter_best_route, limits, true);
}
else if (!reset_pheromone) {     // 開 LS 且本輪不是 reset
    bool use_iter_best   = float_distribution(rng) < 0.1;   // 10%
    bool use_global_best = float_distribution(rng) < 0.1;   // （iter 不中時再）10%
    deposit_pheromone<<<1,256>>>(dim, d_pheromone,
        use_iter_best ? d_iter_best_cost
                      : (use_global_best ? d_global_best_cost : d_reset_best_cost),
        use_iter_best ? d_iter_best_route
                      : (use_global_best ? d_global_best_route : d_reset_best_route),
        limits, true);
}
// 若本輪 reset_pheromone==true：跳過沉積（剛把表填成 τ_max，不再額外加）
```

預設（開 LS）下的機率混合：

- 約 **10%** 用 iteration best（鼓勵探索當輪新解）
- 約 **9%**（剩 90% 再取 10%）用 global best（強化全域最佳）
- 約 **81%** 用 reset best（自上次 reset 以來最佳——LS 模式下最有效的主力來源）

這正是論文建議「搭配 LS 時主要沉積在 reset-best 上」的實作。

---

## 一次迭代結束

迴圈回到 Step 1，用更新後的費洛蒙重算候選快取、再建 100 條新 tour……重複 `--iter`（預設 1000）次。

### 全部跑完後（`mmas.cu:2687` 之後）

```cpp
CUDA_CHECK(cudaDeviceSynchronize());
final_cost = d_global_best_cost.front();                   // 取 global best 成本
final_rel_err = get_error_relative_to_optimum(...);        // 對照 best-known.json 算 gap%
final_route = d_global_best_route.as_host_vector();
if (!instance.is_route_valid(final_route)) throw ...;      // 驗證是合法 tour
assert(instance.calculate_route_length(final_route) == final_cost);  // CPU 重算成本一致
// → 連同各階段 Timer、global_best_log 寫成 results/<alg>-<inst>_<時間>.json
```

---

## 附錄：本變體一次迭代的資料流摘要

```
pheromone_matrix_ (n×n)
        │  Step1: × heuristic
        ▼
cand_lists_product_cache_ (n×32)
        │  Step2: warp roulette（每 thread 一個候選 slot）
        ▼
ant_routes_ (100×n) ─ Step3 two-opt ─► 改善後的 ant_routes_ + ant_route_costs_
        │
        ▼  Step4: argmin
iter_best / global_best / reset_best (cost + route)  ─► trail_limits (τ_min/τ_max)
        │
        ▼  Step6 evaporate（×(1-ρ) 或填 τ_max）
        ▼  Step7 deposit（10% iter / 9% global / 81% reset best，+1/cost，夾 τ_max）
pheromone_matrix_  ← 回到下一輪 Step1
```

### 為什麼預設挑這個組合

- `_cl`：每步只看 32 個最近鄰，且不配置 `n×n` product cache，省時間又省記憶體（大實例不 OOM 的關鍵）。
- `bt`（BitmaskTabu）：`is_visited` O(1)、記憶體最省（n bits）；搭 `_cl` 時枚舉範圍只有 32，bitmask「不壓縮、枚舉慢」的缺點不發作。
- `rwm` + `block-warps=1`：走純 warp shuffle 的輪盤實作，無 shared memory 往返，1-warp 配置下最快。
- `--ls=1`：2-opt 是解品質的主要來源；連帶啟用 reset-best 主導的沉積策略與停滯重置。
