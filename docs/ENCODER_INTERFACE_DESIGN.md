# 编码器接口设计说明

## 接口设计原则

### 为什么所有编码都需要特征名？

在推荐系统中，**所有编码方法都需要特征名**，原因如下：

1. **配置查找**：编码器需要通过特征名查找对应的配置
   - One-Hot: 需要查找类别列表 `Categories[key]`
   - Label: 需要查找映射表 `LabelMap[key]`
   - Hash: 需要特征名来区分不同特征（避免冲突）
   - Target/Frequency/Embedding: 需要查找编码表 `Encodings[key]`

2. **特征区分**：不同特征可能有相同的值，需要特征名区分
   ```go
   // 例如：user_id 和 item_id 都可能是 "12345"
   // 但它们的编码方式可能不同
   encoder.EncodeWithKey("user_id", "12345")
   encoder.EncodeWithKey("item_id", "12345")
   ```

3. **特征命名**：编码后的特征名通常包含原始特征名
   ```go
   // One-Hot 编码后: "gender_0", "gender_1", "gender_2"
   // Hash 编码后: "user_id_hash_123"
   // Target 编码后: "category_target"
   ```

## 接口设计演进

### 初始设计（有问题）

```go
type Encoder interface {
    Encode(value interface{}) map[string]float64  // ❌ 无法实现，需要特征名
    EncodeFeatures(features map[string]interface{}) map[string]float64
}
```

**问题**：
- `Encode` 方法无法实现（所有编码都需要特征名）
- 所有实现都返回空 map，造成接口和实现脱节
- 用户容易混淆，不知道应该用哪个方法

### 重构后的设计（正确）

```go
type Encoder interface {
    // EncodeWithKey 编码单个值（指定特征名）
    // 这是编码的核心方法，因为所有编码都需要通过特征名查找配置
    EncodeWithKey(key string, value interface{}) map[string]float64
    
    // EncodeFeatures 编码特征字典（批量编码）
    // 内部会调用 EncodeWithKey 对每个特征进行编码
    EncodeFeatures(features map[string]interface{}) map[string]float64
}
```

**优势**：
- 接口清晰，所有方法都有实际实现
- 明确要求特征名，避免误用
- 符合实际使用场景

## 使用场景

### 场景1：批量编码（最常见）

```go
encoder := feature.NewOneHotEncoder(categories)
features := map[string]interface{}{
    "gender": "male",
    "city":   "beijing",
}
encoded := encoder.EncodeFeatures(features)
```

### 场景2：单个值编码

```go
encoder := feature.NewOneHotEncoder(categories)
encoded := encoder.EncodeWithKey("gender", "male")
```

### 场景3：在循环中编码

```go
encoder := feature.NewOneHotEncoder(categories)
for key, value := range features {
    encoded := encoder.EncodeWithKey(key, value)
    // 处理编码结果
}
```

## 为什么没有 "Encode(value)" 方法？

**答案**：因为所有编码都需要特征名，所以没有单独编码值的场景。

即使是最简单的 Hash 编码，虽然理论上可以只编码值：
```go
// 理论上可以这样
hash := hashValue("user_12345")  // 只编码值

// 但实际使用中，我们仍然需要特征名
encoder.EncodeWithKey("user_id", "user_12345")  // 需要特征名来区分不同特征
```

## 设计模式

### 接口方法设计

- **EncodeWithKey**: 核心方法，所有编码器必须实现
- **EncodeFeatures**: 便利方法，内部调用 EncodeWithKey

### 实现模式

```go
// 所有编码器都遵循这个模式
func (e *XxxEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
    encoded := make(map[string]float64)
    for k, v := range features {
        encodedFeatures := e.EncodeWithKey(k, v)  // 调用核心方法
        for ek, ev := range encodedFeatures {
            encoded[ek] = ev
        }
    }
    return encoded
}
```

## 总结

1. **所有编码都需要特征名**：因为需要通过特征名查找配置
2. **接口设计要符合实际**：移除无法实现的 `Encode` 方法
3. **明确核心方法**：`EncodeWithKey` 是核心，`EncodeFeatures` 是便利方法
4. **避免接口和实现脱节**：接口中的每个方法都应该有实际用途
