# 修法④ 細解 — fill kernel、grid-stride loop 與 blocks 取值

> 本檔是對 `docs/OOM_FIX_pla85900.md` 修法④ 的補充教學,專門回答三個問題:
> 1. 這個 fill kernel 到底在做什麼?
> 2. 什麼是 grid-stride loop?
> 3. 配置 + 填值時為何 `blocks = min(需要的 thread 數 / 256, 65535)`?
>
> 對應程式碼:`src/mmas.cu` 的 `fill_float_array`(約 2036 行)與 `d_pheromone` 配置處(約 2521 行)。

---

## 0. 先講背景:這支 kernel 是來取代什麼的

修③之後,`d_pheromone` 是一個 **n×n 的 float 矩陣**。pla85900 時 n = 85900,所以:

```
元素數 = 85900 × 85900 = 7,378,810,000 ≈ 7.38×10⁹ 個 float
位元組 = 7.38×10⁹ × 4 B ≈ 27.49 GiB
```

原本的寫法是 `device_vector<float> d_pheromone(size, value)`,這個建構子的流程是:

1. 在 **host(CPU 記憶體)** 先開一個 `std::vector<float>`,塞滿 `value`(初值 τ_max)。
2. 用 `cudaMemcpy` 把這 **整段 27.5 GB 從 host 拷貝到 device**(H2D)。

問題:光是 host 那份暫存就要 ~29.5 GB 的 RAM,而且還得搬一次 27.5 GB 過 PCIe,
又慢又佔記憶體。但我們其實只是想把整塊 GPU 記憶體「全部填成同一個值 τ_max」而已 ——
**根本不需要在 CPU 上先準備好資料再搬過去**。

修法④的做法:
- `device_vector<float> d_pheromone(pheromone_count);` — **只 `cudaMalloc`,不初始化、不從 host 拷貝**。此時 GPU 上那塊 27.5 GB 是垃圾值。
- 再發一支 GPU kernel(`fill_float_array`),讓 **GPU 自己** 把那塊記憶體每一格都寫成 τ_max。

省掉了 host 29.5 GB 暫存 + 一次 27.5 GB 的 H2D 拷貝。這就是「device fill kernel」。

---

## 1. fill kernel 在做什麼

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

逐一拆解:

- `__global__`:這是一支 **kernel**,由 CPU 用 `<<<blocks, threads>>>` 語法發射,在 GPU 上由成千上萬個 thread 同時執行。
- 參數:
  - `data` — 要填的那塊 GPU 記憶體(就是 `d_pheromone.data()`)。
  - `count` — 一共有幾個元素(`pheromone_count = 7.38×10⁹`)。**型別是 `size_t`(64-bit)很重要**,因為 7.38×10⁹ 已經超過 32-bit 上限 4.29×10⁹,用 `int`/`uint32` 會溢位(這正是修③踩過的雷)。
  - `value` — 要填的值(τ_max)。
- 函式體就是一個迴圈,把 `data[i] = value` 對「這個 thread 負責的每一個 i」做一次。

### CUDA 的 thread 編號

每個 thread 都能讀到幾個內建變數,用來算出「我是第幾號 thread」:

| 變數 | 意義 |
| --- | --- |
| `threadIdx.x` | 我在自己 block 內的編號(0 ~ blockDim.x−1) |
| `blockIdx.x` | 我的 block 在 grid 內的編號(0 ~ gridDim.x−1) |
| `blockDim.x` | 一個 block 有幾個 thread(這裡 = 256) |
| `gridDim.x` | 一個 grid 有幾個 block(這裡 = fill_blocks) |

所以一個 thread 的「全域唯一編號」就是:

```
global_id = blockIdx.x * blockDim.x + threadIdx.x
```

這正是迴圈裡 `i` 的初始值。

---

## 2. 什麼是 grid-stride loop

### 2.1 為什麼不能「一個 thread 配一個元素」就好

最直覺的寫法是:發射剛好 `count` 個 thread,每個 thread 處理一個元素:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < count) data[i] = value;   // 沒有迴圈
```

問題有兩個:

1. **count 太大,thread 開不出來。** 7.38×10⁹ 個 thread 需要 `7.38e9 / 256 ≈ 2880 萬個 block`,
   但 CUDA 對 `gridDim.x` 有 **上限 65535**(見第 3 節)。一次根本發不出這麼多 block。
2. 就算硬體允許,「元素數 = thread 數」也讓 kernel 失去彈性 —— 換一台 GPU、換一個 n 都要重算。

### 2.2 grid-stride loop 的核心想法

**grid-stride(grid 跨步)** 的精神是:

> 不要求「thread 數 = 元素數」。而是發射「適量」的 thread,然後讓 **每個 thread 用迴圈處理多個元素**,每次往後跳「一整個 grid 的 thread 數」這麼遠。

關鍵那一行:

```cpp
i += static_cast<size_t>(gridDim.x) * blockDim.x;
```

`gridDim.x * blockDim.x` = 「整個 grid 一共有幾個 thread」,這就是 **stride(步幅)**。

### 2.3 具體例子

假設 `count = 1000`,我們只發射 `gridDim.x = 2` 個 block、每 block `blockDim.x = 4` 個 thread,
所以一共 `stride = 2 × 4 = 8` 個 thread,全域編號 0 ~ 7。

來看 **thread 0** 做了什麼:

```
i = 0  → data[0] = value;   i += 8
i = 8  → data[8] = value;   i += 8
i = 16 → data[16] = value;  i += 8
...
i = 992 → data[992] = value; i += 8
i = 1000 → 1000 < 1000 為 false,迴圈結束
```

thread 1 則處理 1, 9, 17, …;thread 7 處理 7, 15, 23, …。
**8 個 thread 像梳子一樣交錯,把 1000 個元素全部覆蓋**,沒有一格漏掉、沒有一格重複。

圖示(每格標記負責它的 thread 編號):

```
索引:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 ...
thread: 0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7 ...
                                 └──── 第二圈,每個 thread 往後跳 stride=8 ────┘
```

### 2.4 grid-stride 的好處

- **解耦「資料量」與「thread 數」。** 不管 count 是 1000 還是 7.38×10⁹,同一支 kernel、同一組
  發射參數都跑得動 —— 只是迴圈多轉幾圈而已。這就是它能覆蓋 7.38×10⁹ 元素卻只發 65535 個 block 的原因。
- **可攜性 / 可調效能。** block 數可以按 GPU 大小自由選,不必等於資料量。
- **記憶體存取連續(coalesced)。** 同一個 block 內相鄰 thread(threadIdx 0,1,2,…)在同一圈寫的是
  相鄰位址 `data[base+0], data[base+1], …`,GPU 能把它們合併成少數幾筆寬記憶體交易,頻寬利用率高。

---

## 3. 為何 `blocks = min(需要的 thread 數 / 256, 65535)`

對應程式碼:

```cpp
const int fill_threads = 256;                                  // 每 block 256 thread
const size_t threads_wanted =
    (pheromone_count + fill_threads - 1) / fill_threads;       // 理想上需要幾個 block
const int fill_blocks =
    static_cast<int>(std::min<size_t>(threads_wanted, 65535)); // 但夾在 65535 以內
```

拆成兩半看。

### 3.1 `threads_wanted = (count + 255) / 256` 是什麼

`fill_threads = 256` 是每個 block 的 thread 數(256 是常見好選擇:32 的倍數、占用率高、暫存器壓力低)。

`(pheromone_count + fill_threads - 1) / fill_threads` 是 **無條件進位除法**(ceiling division):

```
(count + 255) / 256
```

它算的是「如果要讓 thread 數 ≥ count(理想上一個 thread 配一個元素),至少需要幾個 block」。
`+255` 是為了在不能整除時把商往上進位,避免最後不足一個 block 的元素被漏掉。

> 注意這裡變數名叫 `threads_wanted`,但它的單位其實是 **block 數**(= 理想 thread 數 ÷ 每 block thread 數)。
> 也就是使用者問題裡說的「需要的 thread 數 / 256」。

以 pla85900:

```
threads_wanted = (7.38×10⁹ + 255) / 256 ≈ 28,823,477 個 block
```

### 3.2 為什麼要 `min(..., 65535)`

CUDA 規定 **`gridDim.x`(grid 第一維的 block 數)的硬體上限是 65535**。
(這是 x 維的限制;y、z 維在新架構上更大,但這支 kernel 只用一維。)

而上面算出來理想 block 數是 ~2882 萬,**遠遠超過 65535**。如果直接拿去發射:

```cpp
fill_float_array<<<28823477, 256>>>(...);   // ❌ gridDim.x 超過 65535
```

kernel 會 **啟動失敗**(invalid configuration argument),`cudaGetLastError()` 會回報錯誤。

所以用 `std::min<size_t>(threads_wanted, 65535)` 把 block 數 **夾在合法上限 65535 以內**:

```
fill_blocks = min(28,823,477, 65535) = 65535
```

於是實際發射:

```cpp
fill_float_array<<<65535, 256>>>(...);   // ✅ 合法
```

這時候 grid 一共有 `65535 × 256 = 16,776,960` 個 thread。
但元素有 7.38×10⁹ 個,thread 不夠一格配一個怎麼辦?—— **這正是 grid-stride loop 派上用場的地方**:

```
每個 thread 要處理的圈數 ≈ 7.38×10⁹ / 16,776,960 ≈ 440 圈
```

每個 thread 自動多轉約 440 圈,把全部元素填完,**一格不漏**。

> `std::min<size_t>` 特別寫成 `size_t` 版本,是為了讓比較在 64-bit 下進行 ——
> `threads_wanted`(2882 萬)雖然放得進 32-bit,但養成用 `size_t` 比較的習慣可避免大數溢位。
> 夾完之後結果 ≤ 65535,再 `static_cast<int>` 轉回 `int` 給 `<<<>>>` 用就安全了。

### 3.3 小結:三者如何配合

| 角色 | 做的事 |
| --- | --- |
| `fill_threads = 256` | 固定每 block thread 數 |
| `threads_wanted` | 算「理想」block 數(一格一 thread 要幾個 block) |
| `min(..., 65535)` | 把 block 數壓進硬體合法範圍 |
| **grid-stride loop** | 補上「block 不夠」的缺口:每 thread 多轉幾圈,涵蓋所有元素 |

換句話說:**`min(..., 65535)` 負責讓 kernel 合法啟動,grid-stride loop 負責讓不足的 thread 仍能覆蓋全部資料。** 兩者搭配,才能用區區 65535 個 block 填滿 7.38×10⁹ 個元素。

---

## 4. 補充:執行時機與正確性

- 填值 kernel 發射在 **迭代迴圈開始之前**,且在 **default stream** 上。CUDA 同一條 stream 內的工作
  依序執行,所以等到後面迭代 kernel 真正讀 `d_pheromone` 時,τ_max 早已填好,**不需要額外 `cudaDeviceSynchronize`**。
- 發射後立刻 `CUDA_CHECK(cudaGetLastError())`,確保「配置參數非法」或「啟動失敗」這類錯誤不會被靜默吞掉
  (呼應修法⑥的教訓:kernel launch 一定要查錯)。
- stagnation reset 重置費洛蒙走的是另一支 `evaporate_pheromone(reset=true)`,不經過這支 fill kernel,互不影響。

---

## 5. 一句話總結

> fill kernel 用 **grid-stride loop**,讓「適量的 thread」用迴圈反覆跨步、各自處理多個元素,
> 因此即使 CUDA 限制 `gridDim.x ≤ 65535`、塞不下「一格一 thread」所需的 2882 萬個 block,
> 也能用 `min(理想 block 數, 65535)` 個合法 block 把整塊 27.5 GiB 費洛蒙矩陣直接在 GPU 上填成 τ_max,
> 徹底省掉 host 29.5 GB 暫存與一次 H2D 拷貝。
