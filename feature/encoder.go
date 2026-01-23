package feature

import (
	"fmt"
	"hash/fnv"
)

// Encoder 是特征编码器接口
// 所有编码都需要特征名才能正确编码（因为需要通过特征名查找对应的配置）
type Encoder interface {
	// EncodeWithKey 编码单个值（指定特征名）
	// 这是编码的核心方法，因为所有编码都需要通过特征名查找配置
	EncodeWithKey(key string, value interface{}) map[string]float64
	// EncodeFeatures 编码特征字典（批量编码）
	// 内部会调用 EncodeWithKey 对每个特征进行编码
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

// EmbeddingEncoder Embedding 编码（嵌入编码）
// 使用预训练的 Embedding 表将类别映射到低维稠密向量
// Embedding 是通过神经网络学习得到的，可以捕捉语义相似性
type EmbeddingEncoder struct {
	Embeddings map[string]map[string][]float64 // 每个特征名对应的类别到 embedding 向量的映射
	Prefix     string                          // 特征名前缀
}

// NewEmbeddingEncoder 创建 Embedding 编码器
func NewEmbeddingEncoder(embeddings map[string]map[string][]float64) *EmbeddingEncoder {
	return &EmbeddingEncoder{
		Embeddings: embeddings,
		Prefix:     "",
	}
}

// WithPrefix 设置特征名前缀
func (e *EmbeddingEncoder) WithPrefix(prefix string) *EmbeddingEncoder {
	e.Prefix = prefix
	return e
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *EmbeddingEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	embeddingMap, ok := e.Embeddings[key]
	if !ok {
		return encoded
	}

	valStr := fmt.Sprintf("%v", value)
	embedding, ok := embeddingMap[valStr]
	if !ok {
		// 未知类别使用零向量或默认 embedding
		return encoded
	}

	prefix := e.Prefix
	if prefix != "" {
		prefix = prefix + "_"
	}

	// 将 embedding 向量展开为特征
	for i, val := range embedding {
		featureName := fmt.Sprintf("%s%s_emb_%d", prefix, key, i)
		encoded[featureName] = val
	}

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *EmbeddingEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}

// OrdinalEncoder 有序编码（Ordinal Encoding）
// 将有序类别映射为整数，保持顺序关系
// 与 Label 编码类似，但更明确地表示有序关系
type OrdinalEncoder struct {
	OrderMap map[string][]string // 每个特征名对应的有序类别列表（从小到大）
}

// NewOrdinalEncoder 创建有序编码器
func NewOrdinalEncoder(orderMap map[string][]string) *OrdinalEncoder {
	return &OrdinalEncoder{
		OrderMap: orderMap,
	}
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *OrdinalEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	order, ok := e.OrderMap[key]
	if !ok {
		return encoded
	}

	valStr := fmt.Sprintf("%v", value)
	for i, cat := range order {
		if cat == valStr {
			encoded[key] = float64(i)
			return encoded
		}
	}

	// 未知类别默认为 0
	encoded[key] = 0.0
	return encoded
}

// EncodeFeatures 编码特征字典
func (e *OrdinalEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}

// CountEncoder 计数编码
// 用类别出现的次数编码
// 与频率编码类似，但使用绝对计数而不是相对频率
type CountEncoder struct {
	Counts map[string]map[string]int64 // 每个特征名对应的类别到计数的映射
}

// NewCountEncoder 创建计数编码器
func NewCountEncoder(counts map[string]map[string]int64) *CountEncoder {
	return &CountEncoder{
		Counts: counts,
	}
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *CountEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	countMap, ok := e.Counts[key]
	if !ok {
		return encoded
	}

	valStr := fmt.Sprintf("%v", value)
	if count, ok := countMap[valStr]; ok {
		encoded[key+"_count"] = float64(count)
	} else {
		encoded[key+"_count"] = 0.0
	}

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *CountEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}

// WOEEncoder Weight of Evidence (WoE) 编码
// 证据权重编码，用于衡量类别对目标变量的预测能力
// 公式: WoE = ln((Good% / Bad%) / (Total Good% / Total Bad%))
type WOEEncoder struct {
	WOEMap map[string]map[string]float64 // 每个特征名对应的类别到 WoE 值的映射
}

// NewWOEEncoder 创建 WoE 编码器
func NewWOEEncoder(woeMap map[string]map[string]float64) *WOEEncoder {
	return &WOEEncoder{
		WOEMap: woeMap,
	}
}

// EncodeWithKey 编码单个值（指定特征名）
func (e *WOEEncoder) EncodeWithKey(key string, value interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	woeMap, ok := e.WOEMap[key]
	if !ok {
		return encoded
	}

	valStr := fmt.Sprintf("%v", value)
	if woe, ok := woeMap[valStr]; ok {
		encoded[key+"_woe"] = woe
	} else {
		encoded[key+"_woe"] = 0.0
	}

	return encoded
}

// EncodeFeatures 编码特征字典
func (e *WOEEncoder) EncodeFeatures(features map[string]interface{}) map[string]float64 {
	encoded := make(map[string]float64)
	for k, v := range features {
		encodedFeatures := e.EncodeWithKey(k, v)
		for ek, ev := range encodedFeatures {
			encoded[ek] = ev
		}
	}
	return encoded
}
