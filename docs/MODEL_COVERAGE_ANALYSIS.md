# 推荐模型覆盖度分析

本文档分析当前工程支持的推荐模型，评估是否满足工业级推荐系统的需求。

---

## 一、当前模型支持情况

### 1.1 召回阶段（Recall）

| 模型 | 状态 | 实现位置 | 适用场景 | 优先级 |
|------|------|----------|----------|--------|
| **双塔模型 (Two-Tower)** | ✅ 已实现 | `recall/two_tower_recall.go` | 大规模初筛、跨域召回 | **P0**（工业界标准） |
| **Word2Vec/Item2Vec** | ✅ 已实现 | `recall/word2vec_recall.go` | 文本/序列召回、I2I | P1 |
| **BERT** | ✅ 已实现 | `recall/bert_recall.go` | 文本语义召回、搜索推荐 | P1 |
| **协同过滤 (User-CF)** | ✅ 已实现 | `recall/collaborative_filtering.go` | 相似用户推荐 | P1 |
| **协同过滤 (Item-CF)** | ✅ 已实现 | `recall/collaborative_filtering.go` | I2I 召回 | P1 |
| **矩阵分解 (MF/ALS)** | ✅ 已实现 | `recall/matrix_factorization.go` | 隐式反馈召回 | P1 |
| **内容推荐 (Content)** | ✅ 已实现 | `recall/content.go` | 基于物品特征召回 | P1 |
| **热门推荐 (Hot)** | ✅ 已实现 | `recall/hot.go` | 热门物品召回 | P0（必选） |
| **用户历史 (UserHistory)** | ✅ 已实现 | `recall/user_history.go` | 基于行为历史召回 | P0（必选） |
| **ANN/Embedding** | ✅ 已实现 | `recall/ann.go` | 向量检索召回 | P0（与双塔配合） |
| **RPC 召回** | ✅ 已实现 | `recall/rpc_recall.go` | 外部模型服务召回 | P2（可扩展） |
| **YouTube DNN** | ✅ 已实现 | `recall/youtube_dnn_recall.go` + Python | 视频/内容流召回 | P2（特定场景） |
| **DSSM** | ✅ 已实现 | `recall/dssm_recall.go` + Python | 搜索推荐、语义匹配 | P2（特定场景） |
| **GraphSAGE/Node2vec** | ✅ 已实现 | `recall/graph_recall.go` + Python | 社交推荐、关注页 | P2（特定场景） |

**召回覆盖度**：**100%**（13/13 核心模型已实现）

### 1.2 排序阶段（Rank）

| 模型 | 状态 | 实现位置 | 适用场景 | 优先级 |
|------|------|----------|----------|--------|
| **LR (逻辑回归)** | ✅ 已实现 | `model/lr.go` + `rank/lr_node.go` | 快速 baseline | P0（必选） |
| **DNN** | ✅ 已实现 | `model/dnn.go` + `rank/dnn_node.go` | 自动特征交互 | P1 |
| **Wide&Deep** | ✅ 已实现 | `model/wide_deep.go` + `rank/wide_deep_node.go` | 记忆+泛化 | P1 |
| **DIN** | ✅ 已实现 | `model/din.go` + `rank/din_node.go` | 行为序列+注意力 | P1 |
| **Two-Tower** | ✅ 已实现 | `model/two_tower.go` + `rank/two_tower_node.go` | 快速推理 | P1 |
| **XGBoost** | ✅ 已实现 | `model/rpc.go` + `rank/rpc_node.go` + Python | 树模型，训练快 | **P0**（工业界常用） |
| **DeepFM** | ✅ 已实现 | `model/rpc.go` + `rank/rpc_node.go` + Python | 特征自动交叉 | **P0**（工业界标准） |
| **DIEN** | ❌ 未实现 | - | 动态兴趣演化 | P2（DIN 增强版） |
| **xDeepFM** | ❌ 未实现 | - | CIN + DNN | P2（DeepFM 增强版） |

**排序覆盖度**：**88%**（7/9 核心模型已实现）

### 1.3 重排阶段（Rerank）

| 模型 | 状态 | 实现位置 | 适用场景 | 优先级 |
|------|------|----------|----------|--------|
| **Diversity** | ✅ 已实现 | `rerank/diversity.go` | 多样性去重、作者打散 | P0（必选） |
| **TopN** | ✅ 已实现 | `rerank/topn.go` | Top-N 截断 | P0（必选） |
| **MMoE** | ✅ 已实现 | `rerank/mmoe_node.go` + Python | 多目标优化（CTR+时长+GMV） | P1（多目标场景） |

**重排覆盖度**：**100%**（3/3 核心模型已实现）

---

## 二、工业级推荐系统模型需求

### 2.1 核心模型（必须）

| 阶段 | 模型 | 当前状态 | 说明 |
|------|------|----------|------|
| 召回 | 双塔模型 | ✅ | 工业界召回标准，QPS 高、延迟低 |
| 召回 | 热门推荐 | ✅ | 冷启动、新用户必备 |
| 召回 | 用户历史 | ✅ | 个性化基础 |
| 排序 | XGBoost | ✅ | 树模型，训练快、效果好 |
| 排序 | DeepFM | ✅ | 特征自动交叉，工业界 baseline |
| 排序 | LR | ✅ | 快速 baseline、可解释 |
| 重排 | Diversity | ✅ | 多样性保证 |

**核心模型覆盖度**：**100%**（7/7 已实现）

### 2.2 进阶模型（推荐）

| 阶段 | 模型 | 当前状态 | 说明 |
|------|------|----------|------|
| 召回 | Word2Vec/Item2Vec | ✅ | 文本/序列语义召回 |
| 召回 | BERT | ✅ | 深度语义理解 |
| 召回 | 协同过滤 | ✅ | 经典算法 |
| 排序 | DIN | ✅ | 行为序列建模 |
| 排序 | Wide&Deep | ✅ | 记忆+泛化 |
| 排序 | DNN | ✅ | 深度特征交互 |

**进阶模型覆盖度**：**100%**（6/6 已实现）

### 2.3 特定场景模型（可选）

| 阶段 | 模型 | 当前状态 | 说明 |
|------|------|----------|------|
| 召回 | YouTube DNN | ✅ | 视频/内容流平台（如 YouTube、抖音） |
| 召回 | DSSM | ✅ | 搜索推荐、Query-Doc 匹配 |
| 召回 | GraphSAGE/Node2vec | ✅ | 社交推荐、关注页 |
| 排序 | DIEN | ❌ | DIN 增强版（动态兴趣演化） |
| 排序 | xDeepFM | ❌ | DeepFM 增强版（CIN 网络） |
| 重排 | MMoE | ✅ | 多目标优化（CTR + 时长 + GMV） |

**特定场景模型覆盖度**：**67%**（4/6 已实现）

---

## 三、评估结论

### 3.1 整体覆盖度

- **核心模型**：✅ **100%**（7/7）
- **进阶模型**：✅ **100%**（6/6）
- **特定场景模型**：✅ **67%**（4/6）
- **总体覆盖度**：**89%**（17/19）

### 3.2 是否够用？

#### ✅ **对于通用推荐系统：够用**

当前工程已覆盖：
- ✅ **召回**：双塔、Word2Vec、BERT、CF、MF、Content、Hot、UserHistory、ANN
- ✅ **排序**：LR、DNN、Wide&Deep、DIN、Two-Tower、XGBoost、DeepFM
- ✅ **重排**：Diversity、TopN

**可满足 80%+ 的工业级推荐场景**，包括：
- 电商推荐（淘宝、京东模式）
- 内容推荐（新闻、文章）
- 视频推荐（B站、YouTube 基础版）
- 广告推荐（CTR 预估）

#### ✅ **对于特定场景：已覆盖**

以下场景已实现：

1. **视频/内容流平台**（如 YouTube、抖音）：✅ YouTube DNN
2. **搜索推荐**（Query-Doc 匹配）：✅ DSSM
3. **社交推荐**（关注页、好友推荐）：✅ GraphSAGE/Node2vec
4. **多目标优化**（CTR + 时长 + GMV）：✅ MMoE

---

## 四、新增模型使用说明

### MMoE（多目标重排）

- **训练**：`python train/train_mmoe.py [--data-path data/mmoe_train_data.csv] [--epochs 50]`
- **服务**：`uvicorn service.mmoe_server:app --host 0.0.0.0 --port 8081`
- **Golang**：`rerank.MMoENode{ Endpoint: "http://localhost:8081/predict", WeightCTR: 1, WeightWatchTime: 0.01, WeightGMV: 1e-6 }`

### YouTube DNN（视频/内容流召回）

- **训练**：`python train/train_youtube_dnn.py [--data data/youtube_dnn_data.csv] [--epochs 20]`
- **服务**：`uvicorn service.youtube_dnn_server:app --host 0.0.0.0 --port 8082`
- **Golang**：`recall.YouTubeDNNRecall{ UserEmbeddingURL: "http://localhost:8082/user_embedding", VectorService, TopK, Collection }`；需将 Item Embeddings 导入 VectorService。

### DSSM（Query-Doc 语义召回）

- **训练**：`python train/train_dssm.py [--data data/dssm_data.csv] [--epochs 20]`
- **服务**：`uvicorn service.dssm_server:app --host 0.0.0.0 --port 8083`
- **Golang**：`recall.DSSMRecall{ QueryEmbeddingURL: "http://localhost:8083/query_embedding", VectorService, TopK, Collection }`；`rctx.Params["query_features"]` 提供 query 特征。

### GraphRecall（Node2vec 社交/关注页召回）

- **训练**：`python train/train_node2vec.py [--edges data/graph_edges.csv] [--dim 64] [--epochs 10]`
- **服务**：`uvicorn service.graph_recall_server:app --host 0.0.0.0 --port 8084`
- **Golang**：`recall.GraphRecall{ Endpoint: "http://localhost:8084/recall", TopK: 20 }`；按 `user_id` 召回相似用户。

---

## 五、总结

### 当前状态

✅ **核心模型齐全**：双塔、XGBoost、DeepFM、LR、Diversity 等工业界标准模型均已实现

✅ **进阶模型完整**：DIN、Wide&Deep、Word2Vec、BERT 等进阶模型均已实现

✅ **可扩展性强**：RPC 召回/排序支持接入外部模型服务

### 是否够用？

**对于 80%+ 的通用推荐场景：✅ 够用**

- 电商推荐 ✅
- 内容推荐 ✅
- 视频推荐（基础版）✅
- 广告推荐 ✅

**对于特定场景：✅ 已覆盖**

- 视频流平台（YouTube DNN）✅
- 搜索推荐（DSSM）✅
- 社交推荐（GraphRecall / Node2vec）✅
- 多目标优化（MMoE）✅

### 建议

1. **当前阶段**：✅ **够用**，MMoE、YouTube DNN、DSSM、Node2vec 均已实现
2. **未来扩展**：可按业务需求补充 DIEN、xDeepFM 等增强模型
3. **架构优势**：通过 RPC 接口可快速接入新模型，无需修改核心代码

---

## 六、模型选型参考

详见：[模型选型指南](./MODEL_SELECTION.md)

**选型口诀**：
> **量大快跑选双塔（召回）**  
> **不知道选啥选 DeepFM（精排）**  
> **历史行为多用 DIN（提升精准度）**  
> **又要点击又要买选 MMoE（重排）** ✅ 已实现
