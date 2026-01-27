# 召回模块字段冗余与统一分析

本文档分析了召回模块中所有字段，识别冗余和可以统一的概念。

## 分析结果总结

### ✅ 已统一的字段

| 字段概念 | 统一命名 | 涉及召回源 | 状态 |
|---------|---------|-----------|------|
| 返回数量 | `TopK` | 所有召回源 | ✅ 已统一 |
| 距离度量 | `Metric` | 所有召回源 | ✅ 已统一（原 `SimilarityMetric`） |
| 向量集合 | `Collection` | 向量召回源 | ✅ 已统一 |
| 用户向量字段 | `UserEmbedding` | ANN, MF | ✅ 已统一（原 `UserVector`） |
| 用户向量变量 | `userEmbedding` | 所有向量召回源 | ✅ 已统一（原 `userVector`） |
| 查询向量变量 | `queryEmbedding` | DSSM | ✅ 已统一（原 `emb`） |
| HTTP 端点 | `Endpoint` | YouTubeDNN, DSSM, Graph, RPC | ✅ 已统一（原 `UserEmbeddingURL`, `QueryEmbeddingURL`） |
| HTTP 超时 | `Timeout` | HTTP 召回源 | ✅ 已统一 |
| HTTP 客户端 | `Client` | HTTP 召回源 | ✅ 已统一 |
| Metric Label | `recall_metric` | 所有召回源 | ✅ 已统一（原 `content_metric`, `ann_metric`） |

### ✅ 保持语义区分的字段（合理）

| 字段概念 | 命名 | 说明 | 涉及召回源 |
|---------|------|------|-----------|
| 特征提取器 | `UserFeatureExtractor` | 用户特征提取器 | TwoTower, YouTubeDNN |
| 特征提取器 | `QueryFeatureExtractor` | 查询特征提取器 | DSSM |
| 特征提取器 | `UserPreferencesExtractor` | 用户偏好提取器 | Content |
| 特征提取器 | `UserEmbeddingExtractor` | 用户向量提取器 | ANN, MF |
| 特征提取器 | `HistoryExtractor` | 历史序列提取器 | YouTubeDNN |
| Context Key | `UserPreferencesKey` | 用户偏好 key | Content |
| Context Key | `UserEmbeddingKey` | 用户向量 key | MF |
| Context Key | `UserHistoryKey` | 用户历史 key | ItemBasedCF |
| 存储 Key | `Key` | 简单 key | ANN, Hot |
| 存储 Key | `KeyPrefix` | key 前缀（需拼接 UserID） | UserHistory |

**说明**：这些字段虽然概念相似，但语义不同，保持区分是合理的。

---

## 字段详细分析

### 1. HTTP 相关字段（已统一）

#### Endpoint

**统一前**：
- `YouTubeDNNRecall.UserEmbeddingURL`
- `DSSMRecall.QueryEmbeddingURL`
- `GraphRecall.Endpoint`
- `RPCRecall.Endpoint`

**统一后**：
- 所有 HTTP 召回源统一使用 `Endpoint`

**理由**：都是 HTTP 服务端点，概念相同，统一命名更清晰。

#### Timeout

**状态**：✅ 已统一

所有 HTTP 召回源都使用 `Timeout`，默认值 `5 * time.Second`。

#### Client

**状态**：✅ 已统一

所有 HTTP 召回源都使用 `Client *http.Client`。

---

### 2. 向量相关字段（已统一）

#### UserEmbedding

**统一前**：
- `ANN.UserVector`
- `MFRecall.UserVector`（字段）
- 代码中 `userVector`（变量）

**统一后**：
- 字段：`UserEmbedding []float64`
- 变量：`userEmbedding []float64`

**理由**：Embedding 是更准确的术语，与工业标准一致。

#### QueryEmbedding

**统一前**：
- `DSSMRecall` 中 `emb`（变量）

**统一后**：
- 变量：`queryEmbedding []float64`

**理由**：语义明确，区分用户向量和查询向量。

---

### 3. 特征提取器字段（保持语义区分）

虽然都是提取器，但提取的内容不同：

| 提取器类型 | 提取内容 | 返回类型 | 使用场景 |
|-----------|---------|---------|---------|
| `UserFeatureExtractor` | 用户特征 | `map[string]float64` | TwoTower, YouTubeDNN |
| `QueryFeatureExtractor` | 查询特征 | `map[string]float64` | DSSM |
| `UserPreferencesExtractor` | 用户偏好 | `map[string]float64` | Content |
| `UserEmbeddingExtractor` | 用户向量 | `[]float64` | ANN, MF |
| `HistoryExtractor` | 历史序列 | `[]string` | YouTubeDNN |

**结论**：保持语义区分是合理的，因为：
1. 提取的内容不同（特征 vs 向量 vs 序列）
2. 返回类型不同（map vs slice）
3. 使用场景不同

---

### 4. Context Key 字段（保持语义区分）

虽然都是从 Context 获取数据，但数据语义不同：

| Key 类型 | 数据语义 | 使用场景 |
|---------|---------|---------|
| `UserPreferencesKey` | 用户偏好特征 | Content 召回 |
| `UserEmbeddingKey` | 用户向量 | MF 召回 |
| `UserHistoryKey` | 用户历史物品 | ItemBasedCF 召回 |

**结论**：保持语义区分是合理的，因为数据语义不同。

---

### 5. 存储 Key 字段（保持语义区分）

| Key 类型 | 使用方式 | 使用场景 |
|---------|---------|---------|
| `Key` | 直接使用 | ANN, Hot |
| `KeyPrefix` | 拼接 UserID 使用 | UserHistory |

**结论**：保持区分是合理的，因为使用方式不同。

---

## 冗余字段分析

### 无冗余字段

经过分析，所有字段都有其存在的必要性：

1. **HTTP 字段**：`Endpoint`, `Timeout`, `Client` - 已统一
2. **向量字段**：`UserEmbedding`, `Collection`, `Metric` - 已统一
3. **特征提取器**：语义不同，保持区分
4. **Context Key**：语义不同，保持区分
5. **存储 Key**：使用方式不同，保持区分

---

## 统一化建议

### 已完成 ✅

1. ✅ 统一 `Metric` 字段名
2. ✅ 统一 `UserEmbedding` 字段和变量名
3. ✅ 统一 `Endpoint` 字段名
4. ✅ 统一 `recall_metric` Label 命名

### 无需统一（保持语义区分）

1. ✅ 特征提取器字段 - 语义不同
2. ✅ Context Key 字段 - 语义不同
3. ✅ 存储 Key 字段 - 使用方式不同

---

## 总结

经过全面分析和统一化改造：

1. **已统一**：10 个字段/概念已统一命名
2. **保持区分**：9 个字段因语义不同保持区分（合理）
3. **无冗余**：所有字段都有存在的必要性

当前命名规范：
- ✅ **统一性**：相同概念使用统一命名
- ✅ **区分性**：不同语义保持区分
- ✅ **清晰性**：命名清晰，易于理解

详见 [召回模块专有名词字典](./RECALL_TERMINOLOGY.md)。
