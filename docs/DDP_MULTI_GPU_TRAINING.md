# Multi-GPU Distributed Training with DDP

本文檔總結了為 OMAGD 專案添加的多 GPU 跨卡訓練功能。

## 概述

使用 PyTorch 的 **DistributedDataParallel (DDP)** 實現多 GPU 訓練，支持：
- Stage 1: Tokenizer 預訓練
- Stage 2: Mask Predictor 預訓練  
- Stage 3: QMIX 強化學習訓練

## 新增文件列表

### 1. 分散式訓練工具 (`src/utils/dist_utils.py`)

提供 DDP 訓練所需的基礎工具函數：

```python
# 初始化分散式環境
setup_distributed(rank, world_size, backend='nccl', port='12355')

# 清理分散式環境
cleanup_distributed()

# 檢查是否為主進程（只有主進程負責日誌和模型保存）
is_main_process()

# 獲取當前進程的 rank（GPU 編號）
get_rank()

# 獲取總進程數（GPU 數量）
get_world_size()

# 跨 GPU 平均張量
reduce_mean(tensor)

# 用 DDP 包裝模型
wrap_model_ddp(model, device_id, find_unused_parameters=True)

# 從 DDP 包裝中取出原始模型
unwrap_model(model)
```

### 2. DDP 版 NQLearner (`src/learners/nq_learner_ddp.py`)

標準 Q-Learning 的 DDP 版本：

**主要改動：**
- 使用 `wrap_model_ddp()` 包裝 agent 和 mixer 模型
- 梯度在 `backward()` 時自動跨 GPU 同步
- 只有主進程（rank=0）執行日誌記錄和模型保存
- 支持從 DDP 模型中正確保存/加載權重

```python
class NQLearnerDDP:
    def __init__(self, mac, scheme, logger, args):
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()
        # ...
    
    def _wrap_ddp(self):
        """在 cuda() 之後調用，用 DDP 包裝模型"""
        self.mac.agent = wrap_model_ddp(self.mac.agent, self.rank)
        self.mixer = wrap_model_ddp(self.mixer, self.rank)
```

### 3. DDP 版 NQGraphLearner (`src/learners/nq_graph_learner_ddp.py`)

繼承 `NQLearnerDDP`，添加 Graph Reconstructer 的 DDP 支持：

**主要功能：**
- 支持 Stage 1/2 預訓練的 DDP
- 支持 Stage 3 QMIX 訓練的 DDP
- Graph reconstructer 根據訓練階段選擇性包裝

```python
class NQGraphLearnerDDP(NQLearnerDDP):
    def _wrap_ddp_graph(self):
        """根據訓練階段包裝對應組件"""
        if stage == 'stage1':
            self.graph_reconstructer.tokenizer = wrap_model_ddp(...)
        elif stage == 'stage2':
            self.graph_reconstructer.stage2_model = wrap_model_ddp(...)
```

### 4. DDP 運行腳本 (`src/run/run_ddp.py`)

完整的 DDP 訓練流程控制：

**核心流程：**
```python
def run(_run, _config, _log):
    if use_ddp and world_size > 1:
        # 使用 mp.spawn 啟動多進程
        mp.spawn(ddp_worker, args=(...), nprocs=world_size)
    else:
        # 單 GPU 訓練
        run_single(_run, _config, _log)

def ddp_worker(rank, ...):
    setup_distributed(rank, world_size)  # 初始化 DDP
    try:
        run_single(_run, _config, _log, rank=rank)
    finally:
        cleanup_distributed()  # 清理
```

### 5. 啟動腳本

#### `run_gnn_stage3_ddp.sh` - Stage 3 多 GPU 訓練
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
NUM_GPUS=2

python src/main.py \
    --config=omagd \
    --env-config=sc2 \
    with \
    run=ddp \                           # 使用 DDP 運行模式
    learner=nq_graph_learner_ddp \      # 使用 DDP learner
    use_ddp=True \                      # 啟用 DDP
    world_size=${NUM_GPUS} \            # GPU 數量
    ddp_port=12355                      # DDP 通信端口
```

#### `run_pretrain_ddp.sh` - Stage 1/2 預訓練
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
STAGE="stage1"  # 或 "stage2"

python src/main.py \
    --config=omagd_origin \
    --env-config=sc2 \
    with \
    recontructer_stage=${STAGE} \
    run=ddp \
    learner=nq_graph_learner_ddp \
    use_ddp=True \
    world_size=${NUM_GPUS}
```

## 修改的現有文件

### 1. `src/learners/__init__.py`
新增 DDP learner 的註冊：
```python
from .nq_learner_ddp import NQLearnerDDP
from .nq_graph_learner_ddp import NQGraphLearnerDDP

REGISTRY["nq_learner_ddp"] = NQLearnerDDP
REGISTRY["nq_graph_learner_ddp"] = NQGraphLearnerDDP
```

### 2. `src/run/__init__.py`
新增 DDP run 模式的註冊：
```python
from .run_ddp import run as run_ddp
REGISTRY["ddp"] = run_ddp
```

### 3. `src/utils/logging.py`
添加 `minimal` 模式，讓非主進程跳過日誌記錄：
```python
class Logger:
    def __init__(self, console_logger, minimal=False):
        self.minimal = minimal  # 非主進程設為 True
    
    def log_stat(self, key, value, t, to_sacred=True):
        if self.minimal:
            return  # 跳過非主進程的日誌
        # ...
```

## 使用方法

### 基本使用（2 GPU）

```bash
cd /home/marl2025/new_omagd

# Stage 3 訓練
./run_gnn_stage3_ddp.sh

# 或手動運行
CUDA_VISIBLE_DEVICES=0,1 python src/main.py \
    --config=omagd --env-config=sc2 \
    with run=ddp learner=nq_graph_learner_ddp \
    use_ddp=True world_size=2
```

### 使用 4 GPU

修改啟動腳本：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
```

### 配置參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `run` | 運行模式 | `ddp` |
| `learner` | 學習器類型 | `nq_graph_learner_ddp` |
| `use_ddp` | 是否啟用 DDP | `True` |
| `world_size` | GPU 數量 | 自動檢測 |
| `ddp_port` | 進程通信端口 | `12355` |

## 架構說明

```
┌─────────────────────────────────────────────────────────────────┐
│                        主進程 (GPU 0)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   環境      │  │   Buffer    │  │   Learner   │              │
│  │ (Runner)    │  │             │  │   (DDP)     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                           │                      │
│  日誌記錄 ✓   W&B ✓   模型保存 ✓          │ 梯度同步             │
└───────────────────────────────────────────┼─────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       ↓                       │
┌───────────────────┴─────────────────────────────────────────────┐
│                       工作進程 (GPU 1+)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   環境      │  │   Buffer    │  │   Learner   │              │
│  │ (Runner)    │  │             │  │   (DDP)     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  日誌記錄 ✗   W&B ✗   模型保存 ✗                                 │
└──────────────────────────────────────────────────────────────────┘
```

**每個 GPU 的工作流程：**
1. 運行獨立的環境 (StarCraft II)
2. 收集經驗到各自的 Replay Buffer
3. 從 Buffer 採樣進行訓練
4. 計算本地梯度
5. DDP 自動同步梯度（`loss.backward()` 時）
6. 更新模型參數

## 性能預期

- **吞吐量**：理論上 N 個 GPU 可達到接近 N 倍的訓練速度
- **批次大小**：每個 GPU 維持原本的批次大小，總有效批次 = batch_size × N
- **通信開銷**：主要在梯度同步時產生，NCCL 後端已優化

## 注意事項

1. **端口衝突**：如果 12355 端口被占用，修改 `ddp_port` 參數
2. **GPU 記憶體**：每個 GPU 需要足夠記憶體運行完整模型
3. **環境變量**：確保 `CUDA_VISIBLE_DEVICES` 與 `world_size` 一致
4. **模型保存**：只有主進程保存模型，避免重複寫入

## 文件結構

```
new_omagd/
├── docs/
│   └── DDP_MULTI_GPU_TRAINING.md    # 本文檔
├── src/
│   ├── learners/
│   │   ├── __init__.py              # 已修改：新增 DDP learner 註冊
│   │   ├── nq_learner_ddp.py        # 新增：DDP 版 NQLearner
│   │   └── nq_graph_learner_ddp.py  # 新增：DDP 版 NQGraphLearner
│   ├── run/
│   │   ├── __init__.py              # 已修改：新增 DDP run 註冊
│   │   └── run_ddp.py               # 新增：DDP 運行腳本
│   └── utils/
│       ├── dist_utils.py            # 新增：分散式訓練工具
│       └── logging.py               # 已修改：新增 minimal 模式
├── run_gnn_stage3_ddp.sh            # 新增：Stage 3 DDP 啟動腳本
└── run_pretrain_ddp.sh              # 新增：預訓練 DDP 啟動腳本
```
