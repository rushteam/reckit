# 训练与在线特征一致性指南

本文档详细说明如何保证训练阶段和在线推理阶段的特征处理一致性，避免 train-serving skew 问题。

## 目录

1. [核心原则](#核心原则)
2. [训练与在线特征处理流程](#训练与在线特征处理流程)
3. [一致性保证机制](#一致性保证机制)
4. [常见问题与风险](#常见问题与风险)
5. [最佳实践](#最佳实践)
6. [代码示例](#代码示例)

---

## 核心原则

**训练和在线必须使用完全相同的特征处理逻辑**，包括：

- ✅ **特征名称**：训练时使用的特征名，在线推理时必须完全一致
- ✅ **特征顺序**：特征向量的顺序必须一致
- ✅ **特征标准化**：如果训练时做了标准化，在线也必须做相同的标准化
- ✅ **缺失值处理**：缺失值的填充策略必须一致
- ✅ **交叉特征**：交叉特征的计算公式必须一致

**违反一致性会导致 train-serving skew，模型性能严重下降！**

---

## 训练与在线特征处理流程

### 训练阶段（Python）

#### 1. 特征定义

在 `python/train/features.py` 中定义特征列。**默认与 RPCNode 不去掉前缀对齐**，使用带前缀特征名：

```python
FEATURE_COLUMNS = [
    "item_ctr", "item_cvr", "item_price",
    "user_age", "user_gender",
    "cross_age_x_ctr", "cross_gender_x_price",
]
```

#### 2. 数据准备

训练数据（CSV）包含所有特征列，特征名**带前缀**（与 EnrichNode 产出一致）：

```csv
item_ctr,item_cvr,item_price,user_age,user_gender,cross_age_x_ctr,cross_gender_x_price,label
0.15,0.08,99.0,25.0,1.0,3.75,99.0,1
0.12,0.05,150.0,30.0,2.0,3.6,300.0,0
```

#### 3. 特征标准化（可选）

如果使用 `--normalize` 参数，训练时会进行 Z-score 标准化：

```python
# python/train/train_xgb.py
if normalize:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # 计算 mean 和 std
```

#### 4. 保存特征元数据

训练完成后保存：

- **`feature_meta.json`**：特征列名、顺序、模型版本等
- **`feature_scaler.json`**（如果使用了标准化）：每列的 mean 和 std

```json
// feature_meta.json
{
  "feature_columns": ["item_ctr", "item_cvr", "item_price", "user_age", "user_gender", "cross_age_x_ctr", "cross_gender_x_price"],
  "feature_count": 7,
  "normalized": true,
  "model_version": "v1.0.0"
}

// feature_scaler.json
{
  "item_ctr": {"mean": 0.15, "std": 0.08},
  "item_cvr": {"mean": 0.05, "std": 0.03},
  ...
}
```

### 在线推理阶段（Go → Python）

#### 1. 特征注入（EnrichNode）

`EnrichNode` 为物品注入特征，**带前缀**：

```go
// feature/enrich.go
item.Features["user_age"] = 25.0
item.Features["item_ctr"] = 0.15
item.Features["cross_age_x_ctr"] = 3.75
```

#### 2. 特征名与 RPCNode 的 StripFeaturePrefix 开关

**默认（`StripFeaturePrefix == false`）**：不去掉前缀，直接传递 EnrichNode 产出的特征名（如 `user_age`、`item_ctr`、`cross_age_x_ctr`）。训练时 `FEATURE_COLUMNS` 使用带前缀名称，与在线一致。

**`StripFeaturePrefix == true`**：去掉 `user_`、`item_`、`cross_`、`scene_` 等前缀后再发给模型，适用于训练时 `FEATURE_COLUMNS` 为无前缀的旧模型或外部模型。

```go
// rank/rpc_node.go
rpcNode := &rank.RPCNode{
    Model:             xgbModel,
    StripFeaturePrefix: false, // 默认：不去掉，与 FEATURE_COLUMNS 带前缀对齐
}
```

若设为 `true`，内部会调用 `stripFeaturePrefix`，例如：
- `user_age` → `age`
- `item_ctr` → `ctr`
- `cross_age_x_ctr` → `age_x_ctr`

#### 3. Python 服务处理

Python 服务（`ModelLoader`）接收特征后：

1. **特征验证**：根据 `feature_meta.json` 的 `feature_columns` 验证
2. **缺失值填充**：缺失的特征填充为 `0.0`
3. **特征标准化**：如果 `feature_scaler.json` 存在，应用相同的标准化公式

```python
# python/service/model_loader.py
def _normalize_features(self, features: dict) -> dict:
    if self.feature_scaler is None:
        return features
    
    normalized = features.copy()
    for col in self.feature_columns:
        if col in normalized and col in scaler:
            mean = scaler[col].get("mean", 0.0)
            std = scaler[col].get("std", 1.0)
            if std > 0:
                normalized[col] = (normalized[col] - mean) / std
    return normalized
```

---

## 一致性保证机制

### 1. 特征名称一致性

| 阶段 | 特征名格式 | 保证机制 |
|------|-----------|---------|
| 训练 | `item_ctr`, `user_age`, `cross_age_x_ctr`（带前缀） | `FEATURE_COLUMNS` 定义 |
| 在线 | `item_ctr`, `user_age`, `cross_age_x_ctr`（带前缀） | `EnrichNode` 产出；`RPCNode` 默认不去掉前缀 |
| 模型输入 | `item_ctr`, `user_age`, `cross_age_x_ctr`（带前缀） | `feature_meta.json` 的 `feature_columns` |

**关键**：默认情况下 `RPCNode.StripFeaturePrefix == false`，不去掉前缀；`FEATURE_COLUMNS` 使用带前缀名称，与 EnrichNode 及在线特征一致。若使用无前缀的旧模型，可设置 `StripFeaturePrefix: true`。

### 2. 特征标准化一致性

如果训练时使用了 `--normalize`：

| 阶段 | 标准化操作 | 参数来源 |
|------|-----------|---------|
| 训练 | `StandardScaler().fit_transform(X)` | 从训练数据计算 mean/std |
| 在线 | `(x - mean) / std` | 从 `feature_scaler.json` 读取 |

**关键**：
- ✅ 训练时用 `--normalize` → 必须生成并部署 `feature_scaler.json`
- ✅ 在线服务必须加载 `feature_scaler.json` 并应用相同的标准化
- ❌ 训练时没用 `--normalize` → 在线也不能做标准化

### 3. 缺失值处理一致性

| 阶段 | 处理方式 | 实现 |
|------|---------|------|
| 训练 | 假设数据完整（或预处理） | CSV 数据准备 |
| 在线 | 缺失特征填充为 `0.0` | `ModelLoader._validate_features` |

### 4. 交叉特征一致性

训练数据中的交叉特征必须与在线生成方式一致：

**训练数据**（带前缀，与 FEATURE_COLUMNS 一致）：
```csv
user_age,item_ctr,cross_age_x_ctr
25.0,0.15,3.75  # cross_age_x_ctr = user_age * item_ctr = 25 * 0.15
```

**在线生成**（`EnrichNode.defaultCrossFeatures`）：
```go
// 生成交叉特征：用户特征 × 物品特征；写入 item 时加 cross_ 前缀
key := fmt.Sprintf("%s_x_%s", uk, ik)  // "age_x_ctr"
crossFeatures[key] = uv * iv            // 25.0 * 0.15 = 3.75 → 最终 item.Features["cross_age_x_ctr"]
```

**关键**：
- ✅ 交叉特征的**命名规则**必须一致（`{user_feat}_x_{item_feat}`）
- ✅ 交叉特征的**计算公式**必须一致（乘法）
- ✅ 使用的**原始特征**必须一致（`KeyUserFeatures` 和 `KeyItemFeatures` 必须对齐）

### 5. 特征顺序一致性

特征向量的顺序由 `feature_meta.json` 的 `feature_columns` 决定：

```python
# 训练和在线都按相同顺序构建特征向量
feature_vector = [normalized_features[col] for col in self.feature_columns]
```

---

## 常见问题与风险

### ⚠️ 问题 1：新增场景特征未在训练中使用

**现象**：
- `EnrichNode` 添加了 `scene_*` 特征（如 `scene_id`, `scene_id_hash`）
- 但训练数据中没有这些特征，`FEATURE_COLUMNS` 也不包含

**影响**：
- 在线会生成 `scene_*` 特征
- 但模型训练时没见过这些特征，不会使用
- 如果 `RPCNode` 没有去掉 `scene_` 前缀，会传给 Python 服务，但会被 `feature_columns` 过滤掉

**解决方案**：
1. 在训练数据中添加场景特征
2. 更新 `FEATURE_COLUMNS` 包含场景特征
3. 重新训练模型
4. 确保 `RPCNode.normalizeFeatures` 也去掉 `scene_` 前缀

### ⚠️ 问题 2：RPCNode 前缀列表不完整（仅当 StripFeaturePrefix == true 时）

**现象**：
- `stripFeaturePrefix` 未处理 `scene_` 等新增前缀

**影响**：
- 特征名会带前缀传给模型，与无前缀的 `FEATURE_COLUMNS` 不一致

**解决方案**：
- 默认不去掉前缀时无需处理。
- 若 `StripFeaturePrefix == true`，确保 `stripFeaturePrefix` 的 `prefixes` 包含所有使用的前缀（当前含 `item_`, `user_`, `cross_`, `scene_`）。

### ⚠️ 问题 3：训练和在线标准化不一致

**现象**：
- 训练时用了 `--normalize`，但在线没有加载 `feature_scaler.json`
- 或训练时没用 `--normalize`，但在线做了标准化

**影响**：
- **严重的 train-serving skew**
- 模型性能急剧下降

**解决方案**：
- ✅ 训练时用 `--normalize` → 必须部署 `feature_scaler.json`
- ✅ 在线服务必须检查 `feature_meta.json` 的 `normalized` 字段
- ✅ 如果 `normalized: true`，必须加载并应用 `feature_scaler.json`

### ⚠️ 问题 4：Go 侧和 Python 侧双重标准化

**现象**：
- Go 侧使用 `feature.ZScoreNormalizer` 做了标准化
- Python 服务又用 `feature_scaler.json` 做了一次标准化

**影响**：
- 特征被标准化了两次，分布完全错误
- 模型预测结果错误

**解决方案**：
- **统一标准化位置**：要么全在 Go，要么全在 Python
- 当前架构：**标准化统一在 Python 服务**（使用 `feature_scaler.json`）
- Go 侧只做特征提取和组合，不做标准化

### ⚠️ 问题 5：交叉特征计算不一致

**现象**：
- 训练数据中 `cross_age_x_ctr` 等的计算方式与在线不同
- 或使用的原始特征不同

**影响**：
- 交叉特征值不一致
- 模型性能下降

**解决方案**：
- 在 `features.py` 中明确定义交叉特征的计算公式
- `EnrichNode` 的 `KeyUserFeatures` 和 `KeyItemFeatures` 必须与训练数据对齐
- 交叉特征的命名规则必须一致

---

## 最佳实践

### 1. 单一特征配置源

**问题**：训练和在线各自维护特征配置，容易不一致

**解决方案**：使用单一配置源

```python
# python/train/features.py（作为配置源，带前缀与默认 RPCNode 对齐）
FEATURE_COLUMNS = [
    "item_ctr", "item_cvr", "item_price",
    "user_age", "user_gender",
    "cross_age_x_ctr", "cross_gender_x_price",
]

# Go 代码从配置生成特征；Key 为逻辑名，交叉特征生成 cross_* 与上面一致
enrichNode := &feature.EnrichNode{
    KeyUserFeatures: []string{"age", "gender"},
    KeyItemFeatures: []string{"ctr", "cvr", "price"},
}
```

### 2. 特征合约（Feature Contract）

**将 `feature_meta.json` + `feature_scaler.json` 视为模型版本的"特征合约"**

- ✅ 训练时生成的特征元数据，在线必须完全遵守
- ✅ 同一模型版本必须使用相同的特征合约
- ✅ 不同模型版本可以有不同的特征合约

```json
// feature_meta.json（特征合约）
{
  "feature_columns": [...],      // 必须使用的特征
  "normalized": true,            // 是否标准化
  "model_version": "v1.0.0"      // 模型版本
}
```

### 3. 标准化标志检查

**在线服务必须检查 `normalized` 标志**

```python
# python/service/model_loader.py
if feature_meta.get("normalized", False):
    if not os.path.exists(scaler_path):
        raise ValueError(f"模型需要标准化，但找不到 feature_scaler.json")
    # 加载并应用标准化
```

### 4. 交叉特征显式定义

**在配置中明确定义交叉特征**

```python
# python/train/features.py（交叉特征对应 cross_*，公式与 defaultCrossFeatures 一致）
CROSS_FEATURES = {
    "cross_age_x_ctr": ("age", "ctr", lambda a, c: a * c),
    "cross_gender_x_price": ("gender", "price", lambda g, p: g * p),
}
```

Go 侧确保使用相同的公式：

```go
// feature/enrich.go
key := fmt.Sprintf("%s_x_%s", uk, ik)
crossFeatures[key] = uv * iv  // 与训练时的公式一致；写入时加 cross_ 前缀
```

### 5. 特征验证与监控

**在线推理时记录特征使用情况**

```python
# 记录缺失的特征
missing_features = [col for col in feature_columns if col not in features]
if missing_features:
    logger.warning(f"缺失特征: {missing_features}")

# 记录特征分布（用于监控）
feature_stats = {
    col: {"mean": np.mean(values), "std": np.std(values)}
    for col, values in feature_vectors.T
}
```

### 6. 集成测试

**使用金标样本验证一致性**

```python
# python/tests/test_integration.py
def test_feature_consistency():
    # 1. 从训练集采样
    sample = training_data.sample(10)
    
    # 2. 训练管线处理
    train_features = prepare_features(sample)
    train_pred = model.predict(train_features)
    
    # 3. 在线管线处理（模拟 Go → Python）
    online_features = prepare_online_features(sample)
    online_pred = online_model.predict(online_features)
    
    # 4. 验证特征一致性
    assert train_features.columns.tolist() == online_features.columns.tolist()
    assert np.allclose(train_features.values, online_features.values)
    
    # 5. 验证预测一致性
    assert np.allclose(train_pred, online_pred, atol=1e-5)
```

---

## 代码示例

### 完整的训练流程

```bash
# 1. 训练模型（使用标准化）
cd python
python train/train_xgb.py --normalize --version v1.0.0

# 2. 检查生成的文件
ls model/
# feature_meta.json      # 特征元数据
# feature_scaler.json    # 标准化参数（因为用了 --normalize）
# xgb_model.json         # 模型文件
```

### 完整的在线流程

```go
// examples/rpc_xgb/main.go

// 1. 创建特征注入节点
enrichNode := &feature.EnrichNode{
    UserFeaturePrefix:  "user_",
    ItemFeaturePrefix:  "item_",
    CrossFeaturePrefix: "cross_",
    KeyUserFeatures:    []string{"age", "gender"},       // 逻辑名，与 UserProfile / 交叉生成对齐
    KeyItemFeatures:    []string{"ctr", "cvr", "price"}, // 逻辑名，与训练数据对齐
}

// 2. 创建 RPC 模型（调用 Python 服务）
xgbModel := model.NewRPCModel(
    "xgboost",
    "http://localhost:8080/predict",
    5*time.Second,
)

// 3. 构建 Pipeline
p := &pipeline.Pipeline{
    Nodes: []pipeline.Node{
        &recall.Fanout{...},
        enrichNode,                    // 特征注入（带前缀）
        &rank.RPCNode{Model: xgbModel}, // 特征转换 + 预测（去掉前缀）
    },
}
```

### Python 服务特征处理

```python
# python/service/model_loader.py

# 1. 加载模型和特征元数据
loader = ModelLoader(model_path, feature_meta_path)
loader.load()

# 2. 预测时处理特征
def predict(features_list):
    for features in features_list:
        # 验证特征（填充缺失值）
        validated = loader._validate_features(features)
        
        # 标准化（如果配置了）
        normalized = loader._normalize_features(validated)
        
        # 按 feature_columns 顺序构建向量
        vector = [normalized[col] for col in loader.feature_columns]
        
        # 预测
        score = model.predict(vector)
```

---

## 检查清单

在部署新模型版本前，请检查：

- [ ] 训练时使用的 `FEATURE_COLUMNS` 与在线特征名一致（默认带前缀；若 `StripFeaturePrefix == true` 则与去掉前缀后的名称一致）
- [ ] 如果训练时用了 `--normalize`，在线服务已加载 `feature_scaler.json`
- [ ] 如果训练时没用 `--normalize`，在线服务不做标准化
- [ ] `EnrichNode` 的 `KeyUserFeatures` 和 `KeyItemFeatures` 与训练数据对齐
- [ ] 交叉特征的命名规则和计算公式与训练数据一致
- [ ] 若 `StripFeaturePrefix == true`，`RPCNode` 的 strip 逻辑处理了所有使用的前缀（`user_`, `item_`, `cross_`, `scene_`）
- [ ] 特征顺序与 `feature_meta.json` 的 `feature_columns` 一致
- [ ] 缺失值处理策略一致（默认填充 `0.0`）

---

## 总结

**核心原则**：训练和在线必须使用完全相同的特征处理逻辑。

**关键机制**：
1. **特征名转换**：`RPCNode` 去掉前缀，匹配训练时的特征名
2. **特征元数据**：`feature_meta.json` 定义特征列和顺序
3. **标准化参数**：`feature_scaler.json` 保存训练时的 mean/std，在线应用相同标准化
4. **交叉特征对齐**：确保命名规则和计算公式一致

**最佳实践**：
- 单一特征配置源
- 特征合约（feature contract）
- 标准化标志检查
- 集成测试验证一致性

遵循这些原则和最佳实践，可以有效避免 train-serving skew，保证模型在线性能。
