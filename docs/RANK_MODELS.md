# Rank 模型支持情况

## 概述

Reckit 支持多种常用的排序（Rank）模型，包括 DNN、Wide&Deep、DIN/DIEN、两塔等大厂主流模型。

## 模型列表

| 模型 | 文件 | Node | 特点 | 使用场景 |
|------|------|------|------|----------|
| DNN | `model/dnn.go` | `rank.DNNNode` | 深度神经网络，自动学习特征交互 | 大规模特征推荐 |
| Wide&Deep | `model/wide_deep.go` | `rank.WideDeepNode` | Wide（记忆）+ Deep（泛化） | 大厂主流推荐系统 |
| DIN | `model/din.go` | `rank.DINNode` | 行为序列 + 注意力机制 | 电商推荐（淘宝等） |
| 两塔 | `model/two_tower.go` | `rank.TwoTowerNode` | User Tower + Item Tower | 召回 + 排序两阶段 |

## 1. DNN 模型

### 核心思想

深度神经网络（Deep Neural Network），通过多层全连接网络自动学习特征交互。

### 工程特征

- **实时性**：好（本地推理）
- **计算复杂度**：中等（多层全连接）
- **可解释性**：弱（黑盒模型）
- **特征交互**：强（自动学习特征交互）

### 使用示例

```go
// 创建 DNN 模型
dnnModel := model.NewDNNModel([]int{128, 64, 32, 1}) // 层结构

// 在 Pipeline 中使用
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        &feature.EnrichNode{...},
        &rank.DNNNode{Model: dnnModel},
    },
}
```

### 模型结构

```
输入特征 → [128] → [64] → [32] → [1] → Sigmoid → 输出分数
```

## 2. Wide&Deep 模型

### 核心思想

Wide&Deep 模型结合了线性模型（Wide）和深度模型（Deep）：
- **Wide 部分**：线性模型，记忆（memorization）用户-物品交互
- **Deep 部分**：DNN 模型，泛化（generalization）特征交互
- **联合训练**：Wide + Deep，结合记忆和泛化

### 工程特征

- **实时性**：好（本地推理）
- **计算复杂度**：中等（线性 + DNN）
- **可解释性**：中等（Wide 部分可解释）
- **特征交互**：强（Wide 显式交互 + Deep 隐式交互）

### 使用示例

```go
// 创建 Wide&Deep 模型
wideDeepModel := model.NewWideDeepModel(
    []string{"user_age_x_item_ctr", "user_gender_x_item_category"}, // Wide 特征（交叉特征）
    []string{"user_age", "user_gender", "item_ctr", "item_cvr"},    // Deep 特征（原始特征）
    []int{128, 64, 32, 1}, // Deep 层结构
)

// 设置 Wide 权重
wideDeepModel.WideWeights["user_age_x_item_ctr"] = 0.5
wideDeepModel.WideBias = 0.1

// 在 Pipeline 中使用
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        &feature.EnrichNode{...},
        &rank.WideDeepNode{Model: wideDeepModel},
    },
}
```

### 模型结构

```
输入特征
  ├─ Wide 部分（线性模型）→ 记忆用户-物品交互
  └─ Deep 部分（DNN）→ 泛化特征交互
       ↓
   联合输出（Wide + Deep）→ Sigmoid → 输出分数
```

## 3. DIN 模型

### 核心思想

Deep Interest Network（深度兴趣网络）：
- **用户行为序列**：利用用户历史行为序列（点击、购买等）
- **注意力机制**：计算候选物品与历史行为的注意力权重
- **兴趣提取**：根据注意力权重聚合历史行为，得到用户兴趣表示

### 工程特征

- **实时性**：中等（需要处理行为序列）
- **计算复杂度**：较高（注意力计算）
- **可解释性**：中等（注意力权重可解释）
- **特征交互**：强（行为序列 + 注意力机制）

### 使用示例

```go
// 创建 DIN 模型
dinModel := model.NewDINModel(
    32,              // 物品嵌入维度
    []int{64, 32},  // 注意力网络层结构
    []int{128, 64, 32, 1}, // MLP 层结构
)

// 初始化物品嵌入（实际应该从训练好的模型加载）
dinModel.ItemEmbeddings[itemID] = embeddingVector

// 在 Pipeline 中使用（需要用户行为序列）
userProfile := core.NewUserProfile(userID)
userProfile.AddRecentClick(itemID1, 10)
userProfile.AddRecentClick(itemID2, 10)

rctx := &core.RecommendContext{
    User: userProfile,
}

p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        &feature.EnrichNode{...},
        &rank.DINNode{
            Model: dinModel,
            MaxBehaviorSeqLen: 10, // 最大行为序列长度
        },
    },
}
```

### 模型结构

```
候选物品嵌入
  ↓
用户行为序列嵌入
  ↓
注意力机制（计算候选物品与历史行为的注意力权重）
  ↓
加权聚合历史行为 → 用户兴趣表示
  ↓
拼接（候选物品嵌入 + 用户兴趣表示）
  ↓
MLP → Sigmoid → 输出分数
```

## 4. 两塔模型

### 核心思想

两塔模型（User Tower + Item Tower）：
- **User Tower**：学习用户表示（User Embedding）
- **Item Tower**：学习物品表示（Item Embedding）
- **相似度计算**：User Embedding 和 Item Embedding 的内积/余弦相似度

### 工程特征

- **实时性**：好（可以离线计算 Item Embedding）
- **计算复杂度**：低（向量内积）
- **可解释性**：中等（可以分析用户/物品向量）
- **特征交互**：强（塔内特征交互）

### 使用示例

```go
// 创建两塔模型
twoTowerModel := model.NewTwoTowerModel(
    []int{128, 64, 32}, // User Tower 层结构
    []int{128, 64, 32}, // Item Tower 层结构
    32,                 // Embedding 维度
)
twoTowerModel.SimilarityType = "dot" // 或 "cosine"

// 在 Pipeline 中使用（需要 user_ 和 item_ 前缀的特征）
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        &feature.EnrichNode{
            UserFeaturePrefix:  "user_", // 用户特征前缀
            ItemFeaturePrefix:  "item_", // 物品特征前缀
        },
        &rank.TwoTowerNode{Model: twoTowerModel},
    },
}
```

### 模型结构

```
用户特征 → User Tower (DNN) → User Embedding
                                    ↓
物品特征 → Item Tower (DNN) → Item Embedding
                                    ↓
                            相似度计算（内积/余弦）
                                    ↓
                              Sigmoid → 输出分数
```

## 完整示例

完整示例请参考：`examples/rank_models/main.go`

运行示例：
```bash
go run ./examples/rank_models
```

输出示例：
```
=== 1. DNN 模型 ===
模型: DNN
  1. 物品 5 (分数: 0.6299) [dnn]
  2. 物品 4 (分数: 0.6101) [dnn]
  ...

=== 2. Wide&Deep 模型 ===
模型: Wide&Deep
  1. 物品 5 (分数: 0.5745) [wide_deep]
  ...

=== 3. DIN 模型（行为序列） ===
模型: DIN
  1. 物品 5 (分数: 0.6512) [din]
  ...

=== 4. 两塔模型（User Tower + Item Tower） ===
模型: TwoTower
  1. 物品 5 (分数: 0.5270) [two_tower]
  ...
```

## 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 大规模特征推荐 | DNN | 自动学习特征交互 |
| 大厂主流推荐系统 | Wide&Deep | 记忆+泛化平衡 |
| 电商推荐（有行为序列） | DIN | 利用用户行为序列 |
| 召回+排序两阶段 | 两塔 | 快速推理，可离线计算 |
| 需要可解释性 | Wide&Deep | Wide 部分可解释 |
| 需要快速推理 | 两塔 | 向量内积计算快 |

## 4. DeepFM 模型（通过 RPC 调用 PyTorch 模型）

### 核心思想

DeepFM 结合了 **Factorization Machine (FM)** 和 **Deep Neural Network (DNN)**：
- **FM 部分**：处理二阶特征交互（显式交互）
- **Deep 部分**：处理高阶非线性交互（隐式交互）
- **联合输出**：FM + Deep + Bias

### 工程特征

- **实时性**：中等（需要 RPC 调用 PyTorch 服务）
- **计算复杂度**：中等（FM + DNN）
- **可解释性**：弱（黑盒模型）
- **特征交互**：强（自动学习低阶和高阶交互）

### 使用示例

```go
// 1. 训练 DeepFM 模型（Python）
// cd python
// python train/train_deepfm.py --version v1.0.0

// 2. 启动 DeepFM 服务（Python）
// uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080

// 3. 在 Golang 中使用
deepfmModel := model.NewRPCModel(
    "deepfm",
    "http://localhost:8080/predict",
    5*time.Second,
)

p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        &feature.EnrichNode{...},
        &rank.RPCNode{Model: deepfmModel},
    },
}
```

### Python 训练

```bash
cd python
pip install -r requirements.txt  # 包含 torch>=2.0.0

# 训练模型
python train/train_deepfm.py --version v1.0.0 --epochs 50 --batch-size 32

# 启动服务
uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080
```

**详细文档**：`python/train/DEEPFM_README.md`

**完整示例**：`examples/deepfm/main.go`

## 扩展方向

1. **DIEN（Deep Interest Evolution Network）**：DIN 的改进版，考虑兴趣演化
2. **xDeepFM**：结合 CIN 和 DNN
3. **AutoInt**：使用自注意力机制的特征交互

## 注意事项

1. **模型训练**：这些模型需要先训练才能使用，示例中的权重是随机初始化的
2. **特征工程**：不同模型对特征的要求不同，需要根据模型特点准备特征
3. **性能优化**：实际应用中需要考虑模型加载、批量预测等性能优化
4. **模型版本管理**：建议支持模型版本管理和 A/B 测试
