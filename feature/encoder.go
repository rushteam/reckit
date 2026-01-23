package feature

import (
	"fmt"
	"hash/fnv"
)

// Encoder 是特征编码器接口
type Encoder interface {
	// Encode 编码单个值
	Encode(value interface{}) map[string]float64
	// EncodeFeatures 编码特征字典
	EncodeFeatures(features map[string]interface{}) map[string]float64
}

// OneHotEncoder One-Hot 编码（独热编码）
// 将类别特征转换为二进制向量，每个类别对应一个维度
type OneHotEncoder struct {
	Categories map[string][]string // 每个特征名对应的类别列表
	Prefix     string              // 特征名前缀
}

// NewOneHotEncoder 创建 One-Hot 编码器
func NewOneHotEncoder(categories map[string][]string) *OneHotEncoder {
	return &OneHotEncoder{
		Categories: categories,
		Prefix:     "",
	}
}

// WithPrefix 设置特征名前缀
func (e *OneHotEncoder) WithPrefix(prefix string) *OneHotEncoder {
	e.Prefix = prefix
	return e
}

// Encode 编码单个值
func (e *OneHotEncoder) Encode(value interface{}) map[string]float64 {
	// One-Hot 编码需要知道特征名和类别列表
	// 使用 EncodeWithKey 方法
	return make(map[string]float64)
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *OneHotEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	categories, ok := e.Categories[key]
	if !ok {
		return encoded
	}

	valStr := fmt.Sprintf("%v", value)
	prefix := e.Prefix
	if prefix != "" {
		prefix = prefix + "_"
	}

	for i, cat := range categories {
		featureName := fmt.Sprintf("%s%s_%d", prefix, key, i)
		if cat == valStr {
			encoded[featureName] = 1.0
		} else {
			encoded[featureName] = 0.0
		}
	}

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *OneHotEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}

// LabelEncoder Label 编码（标签编码）
// 将类别映射为整数（0, 1, 2, ...）
type LabelEncoder struct {
	LabelMap map[string]map[string]int // 每个特征名对应的类别到整数的映射
}

// NewLabelEncoder 创建 Label 编码器
func NewLabelEncoder(labelMap map[string]map[string]int) *LabelEncoder {
	return &LabelEncoder{
		LabelMap: labelMap,
	}
}

// Encode 编码单个值
func (e *LabelEncoder) Encode(value interface{}) map[string]float64 {
	// Label 编码需要知道特征名
	// 使用 EncodeWithKey 方法
	return make(map[string]float64)
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *LabelEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	labelMap, ok := e.LabelMap[key]
	if !ok {
		return encoded
	}

	valStr := fmt.Sprintf("%v", value)
	if label, ok := labelMap[valStr]; ok {
		encoded[key] = float64(label)
	} else {
		encoded[key] = 0.0 // 未知类别默认为 0
	}

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *LabelEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}

// HashEncoder Hash 编码（哈希编码）
// 使用哈希函数将类别映射到固定维度
type HashEncoder struct {
	NumBuckets int    // 哈希桶数量
	Prefix     string // 特征名前缀
}

// NewHashEncoder 创建 Hash 编码器
func NewHashEncoder(numBuckets int) *HashEncoder {
	return &HashEncoder{
		NumBuckets: numBuckets,
		Prefix:     "",
	}
}

// WithPrefix 设置特征名前缀
func (e *HashEncoder) WithPrefix(prefix string) *HashEncoder {
	e.Prefix = prefix
	return e
}

// Encode 编码单个值
func (e *HashEncoder) Encode(value interface{}) map[string]float64 {
	// Hash 编码需要知道特征名
	// 使用 EncodeWithKey 方法
	return make(map[string]float64)
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *HashEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	valStr := fmt.Sprintf("%v", value)

	// 计算哈希值
	h := fnv.New32a()
	h.Write([]byte(valStr))
	bucket := int(h.Sum32()) % e.NumBuckets
	if bucket < 0 {
		bucket = -bucket
	}

	prefix := e.Prefix
	if prefix != "" {
		prefix = prefix + "_"
	}
	featureName := fmt.Sprintf("%s%s_hash_%d", prefix, key, bucket)
	encoded[featureName] = 1.0

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *HashEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}

// BinaryEncoder 二进制编码
// 将整数类别转换为二进制表示
type BinaryEncoder struct {
	MaxBits int // 最大位数
}

// NewBinaryEncoder 创建二进制编码器
func NewBinaryEncoder(maxBits int) *BinaryEncoder {
	return &BinaryEncoder{
		MaxBits: maxBits,
	}
}

// Encode 编码单个值
func (e *BinaryEncoder) Encode(value interface{}) map[string]float64 {
	// 二进制编码需要知道特征名
	// 使用 EncodeWithKey 方法
	return make(map[string]float64)
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *BinaryEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)

	// 将值转换为整数
	var intVal int
	switch v := value.(type) {
	case int:
		intVal = v
	case int64:
		intVal = int(v)
	case float64:
		intVal = int(v)
	default:
		return encoded
	}

	// 转换为二进制
	for i := 0; i < e.MaxBits; i++ {
		bit := (intVal >> i) & 1
		featureName := fmt.Sprintf("%s_bit_%d", key, i)
		encoded[featureName] = float64(bit)
	}

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *BinaryEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}

// TargetEncoder Target 编码（目标编码）
// 用目标变量的统计量（均值）编码类别
type TargetEncoder struct {
	Encodings map[string]map[string]float64 // 每个特征名对应的类别到目标均值的映射
}

// NewTargetEncoder 创建 Target 编码器
func NewTargetEncoder(encodings map[string]map[string]float64) *TargetEncoder {
	return &TargetEncoder{
		Encodings: encodings,
	}
}

// Encode 编码单个值
func (e *TargetEncoder) Encode(value interface{}) map[string]float64 {
	// Target 编码需要知道特征名
	// 使用 EncodeWithKey 方法
	return make(map[string]float64)
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *TargetEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	encodingMap, ok := e.Encodings[key]
	if !ok {
		return encoded
	}

	valStr := fmt.Sprintf("%v", value)
	if encoding, ok := encodingMap[valStr]; ok {
		encoded[key+"_target"] = encoding
	} else {
		// 未知类别使用全局均值或 0
		encoded[key+"_target"] = 0.0
	}

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *TargetEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}

// FrequencyEncoder 频率编码
// 用类别出现的频率编码
type FrequencyEncoder struct {
	Frequencies map[string]map[string]float64 // 每个特征名对应的类别到频率的映射
}

// NewFrequencyEncoder 创建频率编码器
func NewFrequencyEncoder(frequencies map[string]map[string]float64) *FrequencyEncoder {
	return &FrequencyEncoder{
		Frequencies: frequencies,
	}
}

// Encode 编码单个值
func (e *FrequencyEncoder) Encode(value interface{}) map[string]float64 {
	// 频率编码需要知道特征名
	// 使用 EncodeWithKey 方法
	return make(map[string]float64)
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *FrequencyEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	freqMap, ok := e.Frequencies[key]
	if !ok {
		return encoded
	}

	valStr := fmt.Sprintf("%v", value)
	if freq, ok := freqMap[valStr]; ok {
		encoded[key+"_freq"] = freq
	} else {
		encoded[key+"_freq"] = 0.0
	}

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *FrequencyEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}
