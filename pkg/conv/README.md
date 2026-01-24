# pkg/conv — 类型转换与泛型工具

提供类型转换、map/slice 转换、配置读取等泛型工具，用于简化各模块中的重复逻辑。

## 类型转换

```go
import "github.com/rushteam/reckit/pkg/conv"

// 数值
v, ok := conv.ToFloat64(anyVal)  // 支持 float64/float32/int/int64/int32/bool
n, ok := conv.ToInt(anyVal)      // 支持 int/int64/int32/float64/float32

// 字符串
s, ok := conv.ToString(anyVal)

// 泛型类型断言
t, ok := conv.TypeAssert[MyType](v)
```

## Map / Slice 转换

```go
// map[string]any -> map[string]float64（仅保留可转换的 value）
out := conv.MapToFloat64(configMap)

// []any -> []string（string 保留，数字格式化为 "%.0f"）
ids := conv.SliceAnyToString(yamlSlice)

// 通用
out := conv.ConvertMap(m, func(v V1) (V2, bool) { ... })
out := conv.ConvertSlice(s, func(t T) (U, bool) { ... })
```

## 配置读取

```go
// 从 map[string]any（如 YAML/JSON）按 key 取 T，否则返回默认值
x := conv.ConfigGet[float64](config, "bias", 0.0)
s := conv.ConfigGet[string](config, "label_key", "category")

// 兼容 int/int64/float64 的 int64 读取
n := conv.ConfigGetInt64(config, "timeout", 5)
```

## 与 UserProfile 扩展属性配合

```go
import "github.com/rushteam/reckit/core"

// 精确类型匹配时使用泛型
tags, ok := core.GetExtraAs[[]string](userProfile, "custom_tags")

// 需数值转换时使用 GetExtraFloat64 / GetExtraInt
vip, _ := userProfile.GetExtraFloat64("vip_level")
```

## 使用位置

- `config`：ConfigGet、ConfigGetInt64、SliceAnyToString、MapToFloat64
- `core.UserProfile`：GetExtraFloat64/Int/String 内部使用 ToFloat64/ToInt/ToString
- `recall`：TwoTowerRecall 使用 ToFloat64；ContentRecall 使用 MapToFloat64
