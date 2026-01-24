// Package conv 提供类型转换、map/slice 转换等泛型工具，用于简化各模块中的重复逻辑。
package conv

import "fmt"

// ToFloat64 将 any 转为 float64。
// 支持 float64、float32、int、int64、int32；bool 视为 1.0/0.0。
func ToFloat64(v any) (float64, bool) {
	if v == nil {
		return 0, false
	}
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case int32:
		return float64(val), true
	case bool:
		if val {
			return 1.0, true
		}
		return 0.0, true
	default:
		return 0, false
	}
}

// ToInt 将 any 转为 int。
// 支持 int、int64、int32、float64、float32。
func ToInt(v any) (int, bool) {
	if v == nil {
		return 0, false
	}
	switch val := v.(type) {
	case int:
		return val, true
	case int64:
		return int(val), true
	case int32:
		return int(val), true
	case float64:
		return int(val), true
	case float32:
		return int(val), true
	default:
		return 0, false
	}
}

// ToString 将 any 转为 string。
// 仅支持 string 类型，否则返回 ("", false)。
func ToString(v any) (string, bool) {
	if v == nil {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// TypeAssert 对 v 做类型断言为 T，等价于 v.(T) 的 (val, ok) 形式。
func TypeAssert[T any](v any) (T, bool) {
	t, ok := v.(T)
	return t, ok
}

// ConvertMap 将 map[K]V1 按 convert 转为 map[K]V2，convert 返回 false 的条目被跳过。
func ConvertMap[K comparable, V1, V2 any](m map[K]V1, convert func(V1) (V2, bool)) map[K]V2 {
	if m == nil {
		return nil
	}
	out := make(map[K]V2, len(m))
	for k, v := range m {
		if v2, ok := convert(v); ok {
			out[k] = v2
		}
	}
	return out
}

// MapToFloat64 将 map[string]any 转为 map[string]float64，仅保留可转为 float64 的 value。
func MapToFloat64(m map[string]any) map[string]float64 {
	return ConvertMap(m, func(v any) (float64, bool) { return ToFloat64(v) })
}

// ConvertSlice 将 []T 按 convert 转为 []U，convert 返回 false 的元素被跳过。
func ConvertSlice[T, U any](s []T, convert func(T) (U, bool)) []U {
	if s == nil {
		return nil
	}
	out := make([]U, 0, len(s))
	for _, v := range s {
		if u, ok := convert(v); ok {
			out = append(out, u)
		}
	}
	return out
}

// SliceAnyToString 将 []any（即 []interface{}）转为 []string。
// 元素为 string 直接保留，为数字时格式化为 "%.0f"。
func SliceAnyToString(v any) []string {
	if v == nil {
		return nil
	}
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	return ConvertSlice(raw, func(e any) (string, bool) {
		if s, ok := e.(string); ok {
			return s, true
		}
		if f, ok := ToFloat64(e); ok {
			return fmt.Sprintf("%.0f", f), true
		}
		return "", false
	})
}

// ConfigGet 从 map[string]any（如 YAML/JSON 解析结果）按 key 取 T，取不到或类型不符时返回 defaultVal。
func ConfigGet[T any](m map[string]any, key string, defaultVal T) T {
	if m == nil {
		return defaultVal
	}
	v, ok := m[key]
	if !ok {
		return defaultVal
	}
	t, ok := v.(T)
	if !ok {
		return defaultVal
	}
	return t
}

// ConfigGetInt64 从 config 取 int64。YAML/JSON 常得到 int 或 float64，此处兼容并统一为 int64。
func ConfigGetInt64(m map[string]any, key string, defaultVal int64) int64 {
	if m == nil {
		return defaultVal
	}
	v, ok := m[key]
	if !ok {
		return defaultVal
	}
	switch val := v.(type) {
	case int:
		return int64(val)
	case int64:
		return val
	case float64:
		return int64(val)
	case float32:
		return int64(val)
	default:
		return defaultVal
	}
}
