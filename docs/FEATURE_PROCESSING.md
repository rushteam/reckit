# 推荐系统特征处理方法

本文档整理推荐系统中常用的特征处理方法，包括归一化、编码、特征工程等技术，以及 Reckit 中的 Go 实现。

## 目录

1. [数值特征归一化/标准化](#数值特征归一化标准化)
2. [分类特征编码](#分类特征编码)
3. [特征工程](#特征工程)
4. [缺失值处理](#缺失值处理)
5. [特征选择](#特征选择)
6. [在 Reckit 中的使用](#在-reckit-中的使用)

---

## 数值特征归一化/标准化

### 1. Z-score 标准化（Standardization）

**公式**：`z = (x - μ) / σ`

**特点**：
- 均值变为 0，标准差变为 1
- 适用于数据近似正态分布
- 对异常值敏感

**适用场景**：
- 特征量纲差异大
- 使用线性模型（LR、SVM）
- 神经网络输入

**Reckit 实现**：

```go
import "github.com/rushteam/reckit/feature"

// 创建 Z-score 标准化器
mean := map[string]float64{
    "age": 30.0,
    "price": 100.0,
}
std := map[string]float64{
    "age": 10.0,
    "price": 50.0,
}
normalizer := feature.NewZScoreNormalizer(mean, std)

// 标准化特征
features := map[string]float64{
    "age": 25.0,
    "price": 150.0,
}
normalized := normalizer.Normalize(features)
// 结果: {"age": -0.5, "price": 1.0}
```

### 2. Min-Max 归一化（Normalization）

**公式**：`x' = (x - min) / (max - min)`

**特点**：
- 将值缩放到 [0, 1] 区间
- 对异常值敏感（min/max 受异常值影响）
- 保持原始分布形状

**适用场景**：
- 神经网络输入
- 需要固定范围的算法（如 KNN）
- 图像处理

**Reckit 实现**：

```go
// 创建 Min-Max 归一化器
min := map[string]float64{
    "age": 18.0,
    "price": 10.0,
}
max := map[string]float64{
    "age": 60.0,
    "price": 200.0,
}
normalizer := feature.NewMinMaxNormalizer(min, max)

// 归一化特征
features := map[string]float64{
    "age": 25.0,
    "price": 150.0,
}
normalized := normalizer.Normalize(features)
// 结果: {"age": 0.166, "price": 0.737}
```

### 3. Robust 标准化

**公式**：`x' = (x - median) / IQR`

**特点**：
- 使用中位数和四分位距（IQR）
- 对异常值鲁棒
- 适用于有异常值的数据

**Reckit 实现**：

```go
// 创建 Robust 标准化器
median := map[string]float64{
    "age": 30.0,
    "price": 100.0,
}
iqr := map[string]float64{
    "age": 15.0,
    "price": 60.0,
}
normalizer := feature.NewRobustNormalizer(median, iqr)

// 标准化特征
normalized := normalizer.Normalize(features)
```

### 4. Log 变换

**公式**：`x' = log(x + 1)`

**特点**：
- 处理长尾分布
- 压缩大值，扩展小值
- 适用于计数类特征（点击量、曝光量）

**Reckit 实现**：

```go
// 创建 Log 变换器
normalizer := feature.NewLogNormalizer()

// 变换特征
features := map[string]float64{
    "click_count": 1000.0,
    "view_count": 5000.0,
}
normalized := normalizer.Normalize(features)
// 结果: {"click_count": 6.908, "view_count": 8.517}
```

### 5. 平方根变换

**公式**：`x' = sqrt(x)`

**特点**：
- 处理长尾分布，比 Log 变换更温和
- 适用于计数类特征

**Reckit 实现**：

```go
// 创建平方根变换器
normalizer := feature.NewSqrtNormalizer()

// 变换特征
normalized := normalizer.Normalize(features)
```

---

## 分类特征编码

### 1. One-Hot 编码（独热编码）

**原理**：
- 将类别特征转换为二进制向量
- 每个类别对应一个维度
- 只有一个维度为 1，其他为 0

**示例**：
```
性别: 男、女、未知
编码后:
  男   -> [1, 0, 0]
  女   -> [0, 1, 0]
  未知 -> [0, 0, 1]
```

**特点**：
- 维度高（类别多时）
- 无大小关系
- 适合类别数量少（< 50）

**Reckit 实现**：

```go
// 创建 One-Hot 编码器
categories := map[string][]string{
    "gender": {"male", "female", "unknown"},
    "city":   {"beijing", "shanghai", "guangzhou"},
}
encoder := feature.NewOneHotEncoder(categories).WithPrefix("onehot")

// 编码特征
features := map[string]interface{}{
    "gender": "male",
    "city":   "beijing",
}
encoded := encoder.EncodeFeatures(features)
// 结果: {
//   "onehot_gender_0": 1.0, "onehot_gender_1": 0.0, "onehot_gender_2": 0.0,
//   "onehot_city_0": 1.0, "onehot_city_1": 0.0, "onehot_city_2": 0.0,
// }
```

### 2. Label 编码（标签编码）

**原理**：
- 将类别映射为整数（0, 1, 2, ...）
- 保持顺序关系（如果存在）

**特点**：
- 维度低（只有一个特征）
- 可能引入虚假的顺序关系
- 适合有序类别

**Reckit 实现**：

```go
// 创建 Label 编码器
labelMap := map[string]map[string]int{
    "level": {"low": 0, "medium": 1, "high": 2},
}
encoder := feature.NewLabelEncoder(labelMap)

// 编码特征
features := map[string]interface{}{
    "level": "high",
}
encoded := encoder.EncodeFeatures(features)
// 结果: {"level": 2.0}
```

### 3. Hash 编码（哈希编码）

**原理**：
- 使用哈希函数将类别映射到固定维度
- 维度可控（通过哈希桶数量）

**特点**：
- 维度固定
- 可能有哈希冲突
- 适合高基数类别（百万级）

**Reckit 实现**：

```go
// 创建 Hash 编码器
encoder := feature.NewHashEncoder(1000).WithPrefix("hash")

// 编码特征
features := map[string]interface{}{
    "user_id": "user_12345",
    "item_id": "item_67890",
}
encoded := encoder.EncodeFeatures(features)
// 结果: {
//   "hash_user_id_hash_123": 1.0,
//   "hash_item_id_hash_456": 1.0,
// }
```

### 4. Target 编码（目标编码）

**原理**：
- 用目标变量的统计量（均值）编码类别
- 考虑类别与目标的关系

**特点**：
- 维度低
- 可能过拟合（需要交叉验证）
- 适合高基数类别

**Reckit 实现**：

```go
// 创建 Target 编码器
encodings := map[string]map[string]float64{
    "category": {
        "electronics": 0.15,  // 该类别的平均 CTR
        "clothing":    0.08,
        "books":      0.05,
    },
}
encoder := feature.NewTargetEncoder(encodings)

// 编码特征
features := map[string]interface{}{
    "category": "electronics",
}
encoded := encoder.EncodeFeatures(features)
// 结果: {"category_target": 0.15}
```

### 5. Frequency 编码（频率编码）

**原理**：
- 用类别出现的频率编码
- 反映类别的常见程度

**特点**：
- 维度低
- 简单有效
- 适合高基数类别

**Reckit 实现**：

```go
// 创建频率编码器
frequencies := map[string]map[string]float64{
    "category": {
        "electronics": 0.3,  // 该类别的出现频率
        "clothing":    0.5,
        "books":      0.2,
    },
}
encoder := feature.NewFrequencyEncoder(frequencies)

// 编码特征
encoded := encoder.EncodeFeatures(features)
// 结果: {"category_freq": 0.3}
```

### 6. Binary 编码（二进制编码）

**原理**：
- 将整数类别转换为二进制表示
- 维度 = log2(类别数)

**特点**：
- 维度比 One-Hot 低
- 保留部分顺序信息
- 适合中等基数类别

**Reckit 实现**：

```go
// 创建二进制编码器
encoder := feature.NewBinaryEncoder(8) // 8 位

// 编码特征
features := map[string]interface{}{
    "category_id": 5,
}
encoded := encoder.EncodeFeatures(features)
// 结果: {
//   "category_id_bit_0": 1.0,
//   "category_id_bit_1": 0.0,
//   "category_id_bit_2": 1.0,
//   ...
// }
```

---

## 特征工程

### 1. 交叉特征（Cross Features）

**原理**：
- 组合多个特征生成新特征
- 捕捉特征间的交互效应

**常见形式**：
- 乘积：`feature1 × feature2`
- 比值：`feature1 / feature2`
- 差值：`feature1 - feature2`
- 组合：`feature1 + feature2`

**Reckit 实现**：

```go
// 创建交叉特征生成器
generator := feature.NewCrossFeatureGenerator(
    []string{"age", "gender"},      // 用户特征
    []string{"ctr", "price"},       // 物品特征
).WithOperations([]string{"multiply", "divide"})

// 生成交叉特征
userFeatures := map[string]float64{
    "age":    25.0,
    "gender": 1.0,
}
itemFeatures := map[string]float64{
    "ctr":   0.15,
    "price": 99.0,
}
crossFeatures := generator.Generate(userFeatures, itemFeatures)
// 结果: {
//   "age_x_ctr": 3.75,
//   "age_x_price": 2475.0,
//   "gender_x_ctr": 0.15,
//   "gender_x_price": 99.0,
//   "age_div_ctr": 166.67,
//   "age_div_price": 0.253,
//   ...
// }
```

### 2. 分桶（Binning）

**原理**：
- 将连续值离散化为区间
- 减少异常值影响

**方法**：
- 等宽分桶：固定区间宽度
- 等频分桶：每个桶样本数相同
- 自定义分桶：根据业务逻辑

**Reckit 实现**：

```go
// 等宽分桶
min := map[string]float64{"age": 18.0, "price": 10.0}
max := map[string]float64{"age": 60.0, "price": 200.0}
numBins := map[string]int{"age": 5, "price": 10}
binner := feature.NewEqualWidthBinner(min, max, numBins)

features := map[string]float64{
    "age":   25.0,
    "price": 150.0,
}
binned := binner.BinFeatures(features)
// 结果: {"age": 1, "price": 7}

// 自定义分桶
bins := map[string][]float64{
    "age":   {0, 18, 30, 50, 100},
    "price": {0, 50, 100, 200, 500},
}
customBinner := feature.NewCustomBinner(bins)
binned = customBinner.BinFeatures(features)
```

---

## 缺失值处理

### 1. 填充固定值

**方法**：
- 数值特征：填充 0、-1、均值、中位数
- 分类特征：填充 "unknown"、众数

**Reckit 实现**：

```go
// 创建缺失值处理器
handler := feature.NewMissingValueHandler("zero", 0.0).
    WithDefaultValues(map[string]float64{
        "age":    25.0,  // age 缺失时填充 25
        "gender": 0.0,   // gender 缺失时填充 0
    })

// 处理缺失值
features := map[string]float64{
    "age": 30.0,
    // gender 缺失
}
requiredFeatures := []string{"age", "gender", "city"}
handled := handler.Handle(features, requiredFeatures)
// 结果: {"age": 30.0, "gender": 0.0, "city": 0.0}
```

---

## 特征选择

### 1. 特征选择器

**方法**：
- 选择指定特征
- 排除指定特征

**Reckit 实现**：

```go
// 创建特征选择器
selector := feature.NewFeatureSelector([]string{"age", "gender", "ctr"}).
    WithExcludedFeatures([]string{"debug_feature"})

// 选择特征
features := map[string]float64{
    "age":          25.0,
    "gender":       1.0,
    "ctr":          0.15,
    "debug_feature": 1.0,
    "other":        100.0,
}
selected := selector.Select(features)
// 结果: {"age": 25.0, "gender": 1.0, "ctr": 0.15}
```

---

## 特征统计

### 1. 计算特征统计信息

**Reckit 实现**：

```go
// 计算统计信息
values := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
stats := feature.ComputeStatistics(values)

fmt.Printf("均值: %.2f\n", stats.Mean)     // 55.00
fmt.Printf("标准差: %.2f\n", stats.Std)    // 30.28
fmt.Printf("最小值: %.2f\n", stats.Min)     // 10.00
fmt.Printf("最大值: %.2f\n", stats.Max)     // 100.00
fmt.Printf("中位数: %.2f\n", stats.Median)  // 55.00
fmt.Printf("P25: %.2f\n", stats.P25)        // 32.50
fmt.Printf("P75: %.2f\n", stats.P75)         // 77.50
fmt.Printf("P95: %.2f\n", stats.P95)        // 95.00
fmt.Printf("P99: %.2f\n", stats.P99)        // 99.00
```

---

## 在 Reckit 中的使用

### 1. 完整示例

```go
package main

import (
    "fmt"
    "github.com/rushteam/reckit/feature"
)

func main() {
    // 1. 准备原始特征
    userFeatures := map[string]float64{
        "age":    25.0,
        "gender": 1.0, // 1=男, 2=女
    }
    itemFeatures := map[string]float64{
        "ctr":   0.15,
        "price": 99.0,
    }

    // 2. 归一化数值特征
    mean := map[string]float64{"age": 30.0, "price": 100.0}
    std := map[string]float64{"age": 10.0, "price": 50.0}
    normalizer := feature.NewZScoreNormalizer(mean, std)
    normalizedUser := normalizer.Normalize(userFeatures)
    normalizedItem := normalizer.Normalize(itemFeatures)

    // 3. 编码分类特征
    categories := map[string][]string{
        "gender": {"male", "female"},
    }
    encoder := feature.NewOneHotEncoder(categories)
    genderEncoded := encoder.EncodeWithKey("gender", 1) // 假设 1=male

    // 4. 生成交叉特征
    generator := feature.NewCrossFeatureGenerator(
        []string{"age"},
        []string{"ctr", "price"},
    )
    crossFeatures := generator.Generate(normalizedUser, normalizedItem)

    // 5. 合并所有特征
    finalFeatures := make(map[string]float64)
    for k, v := range normalizedUser {
        finalFeatures["user_"+k] = v
    }
    for k, v := range normalizedItem {
        finalFeatures["item_"+k] = v
    }
    for k, v := range genderEncoded {
        finalFeatures[k] = v
    }
    for k, v := range crossFeatures {
        finalFeatures["cross_"+k] = v
    }

    fmt.Printf("最终特征: %v\n", finalFeatures)
}
```

### 2. 在 EnrichNode 中使用

```go
// 在特征注入节点中使用特征处理
enrichNode := &feature.EnrichNode{
    FeatureService: featureService,
    // 可以结合特征处理工具类进行预处理
}
```

### 3. 特征处理流程

```
原始特征
  ↓
缺失值处理 (MissingValueHandler)
  ↓
分类特征编码 (Encoder: OneHot/Label/Hash)
  ↓
特征工程 (CrossFeatureGenerator, Binner)
  ↓
数值特征归一化 (Normalizer: ZScore/MinMax/Log)
  ↓
特征选择 (FeatureSelector)
  ↓
最终特征
```

---

## 最佳实践

### 1. 训练/测试一致性

- 训练和推理使用相同的处理参数（均值、标准差、分桶边界等）
- 将处理参数保存到配置文件或模型元数据中

### 2. 性能考虑

- 在线服务中避免复杂的特征处理
- 使用缓存存储处理后的特征
- 批量处理特征以提高效率

### 3. 特征版本管理

- 不同版本的特征需要兼容处理
- 记录每个特征的处理方法

### 4. 监控特征分布

- 定期检查特征分布变化
- 检测特征漂移（Feature Drift）

### 5. 推荐组合

- **线性模型**：Z-score 标准化 + One-Hot 编码
- **树模型**：无需归一化，可使用 Target 编码
- **深度学习**：Min-Max 归一化 + Embedding 编码
- **高基数类别**：Hash 编码或 Target 编码
- **计数特征**：Log 变换

---

## 总结

Reckit 提供了完整的特征处理工具类，包括：

1. **归一化**：Z-score、Min-Max、Robust、Log、Sqrt
2. **编码**：One-Hot、Label、Hash、Target、Frequency、Binary
3. **特征工程**：交叉特征、分桶
4. **缺失值处理**：填充固定值
5. **特征选择**：选择/排除特征
6. **统计信息**：均值、标准差、分位数等

所有工具类都采用接口设计，易于扩展和组合使用。
