package feature

import (
	"fmt"
	"math"
	"sort"
)

// FeatureProcessor 是特征处理器的统一接口
type FeatureProcessor interface {
	// Process 处理特征，返回处理后的特征
	Process(features map[string]float64) map[string]float64
}

// Normalizer 是特征归一化/标准化接口
type Normalizer interface {
	// Normalize 归一化特征
	Normalize(features map[string]float64) map[string]float64
	// NormalizeValue 归一化单个值
	NormalizeValue(value float64) float64
}

// ZScoreNormalizer Z-score 标准化（Standardization）
// 公式: z = (x - μ) / σ
// 特点: 均值变为 0，标准差变为 1
type ZScoreNormalizer struct {
	Mean map[string]float64 // 特征均值
	Std  map[string]float64 // 特征标准差
}

// NewZScoreNormalizer 创建 Z-score 标准化器
func NewZScoreNormalizer(mean, std map[string]float64) *ZScoreNormalizer {
	return &ZScoreNormalizer{
		Mean: mean,
		Std:  std,
	}
}

// Normalize 标准化特征
func (n *ZScoreNormalizer) Normalize(features map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	for k, v := range features {
		normalized[k] = n.NormalizeValueWithKey(k, v)
	}
	return normalized
}

// NormalizeValue 标准化单个值（使用默认均值0和标准差1）
func (n *ZScoreNormalizer) NormalizeValue(value float64) float64 {
	return value // 需要指定特征名，使用 NormalizeValueWithKey
}

// NormalizeValueWithKey 标准化单个值（指定特征名）
func (n *ZScoreNormalizer) NormalizeValueWithKey(key string, value float64) float64 {
	mean := n.Mean[key]
	std := n.Std[key]
	if std > 0 {
		return (value - mean) / std
	}
	return value
}

// MinMaxNormalizer Min-Max 归一化
// 公式: x' = (x - min) / (max - min)
// 特点: 将值缩放到 [0, 1] 区间
type MinMaxNormalizer struct {
	Min map[string]float64 // 特征最小值
	Max map[string]float64 // 特征最大值
}

// NewMinMaxNormalizer 创建 Min-Max 归一化器
func NewMinMaxNormalizer(min, max map[string]float64) *MinMaxNormalizer {
	return &MinMaxNormalizer{
		Min: min,
		Max: max,
	}
}

// Normalize 归一化特征
func (n *MinMaxNormalizer) Normalize(features map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	for k, v := range features {
		normalized[k] = n.NormalizeValueWithKey(k, v)
	}
	return normalized
}

// NormalizeValue 归一化单个值
func (n *MinMaxNormalizer) NormalizeValue(value float64) float64 {
	return value // 需要指定特征名，使用 NormalizeValueWithKey
}

// NormalizeValueWithKey 归一化单个值（指定特征名）
func (n *MinMaxNormalizer) NormalizeValueWithKey(key string, value float64) float64 {
	min := n.Min[key]
	max := n.Max[key]
	rangeVal := max - min
	if rangeVal > 0 {
		return (value - min) / rangeVal
	}
	return value
}

// RobustNormalizer Robust 标准化
// 公式: x' = (x - median) / IQR
// 特点: 对异常值鲁棒
type RobustNormalizer struct {
	Median map[string]float64 // 特征中位数
	IQR    map[string]float64 // 四分位距 (Q75 - Q25)
}

// NewRobustNormalizer 创建 Robust 标准化器
func NewRobustNormalizer(median, iqr map[string]float64) *RobustNormalizer {
	return &RobustNormalizer{
		Median: median,
		IQR:    iqr,
	}
}

// Normalize 标准化特征
func (n *RobustNormalizer) Normalize(features map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	for k, v := range features {
		normalized[k] = n.NormalizeValueWithKey(k, v)
	}
	return normalized
}

// NormalizeValue 标准化单个值
func (n *RobustNormalizer) NormalizeValue(value float64) float64 {
	return value // 需要指定特征名，使用 NormalizeValueWithKey
}

// NormalizeValueWithKey 标准化单个值（指定特征名）
func (n *RobustNormalizer) NormalizeValueWithKey(key string, value float64) float64 {
	median := n.Median[key]
	iqr := n.IQR[key]
	if iqr > 0 {
		return (value - median) / iqr
	}
	return value
}

// LogNormalizer Log 变换
// 公式: x' = log(x + 1)
// 特点: 处理长尾分布，压缩大值
type LogNormalizer struct{}

// NewLogNormalizer 创建 Log 变换器
func NewLogNormalizer() *LogNormalizer {
	return &LogNormalizer{}
}

// Normalize 变换特征
func (n *LogNormalizer) Normalize(features map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	for k, v := range features {
		normalized[k] = n.NormalizeValue(v)
	}
	return normalized
}

// NormalizeValue 变换单个值
func (n *LogNormalizer) NormalizeValue(value float64) float64 {
	if value < 0 {
		return 0 // Log 变换要求值 >= 0
	}
	return math.Log1p(value) // log(x + 1)
}

// SqrtNormalizer 平方根变换
// 公式: x' = sqrt(x)
// 特点: 处理长尾分布，比 Log 变换更温和
type SqrtNormalizer struct{}

// NewSqrtNormalizer 创建平方根变换器
func NewSqrtNormalizer() *SqrtNormalizer {
	return &SqrtNormalizer{}
}

// Normalize 变换特征
func (n *SqrtNormalizer) Normalize(features map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	for k, v := range features {
		normalized[k] = n.NormalizeValue(v)
	}
	return normalized
}

// NormalizeValue 变换单个值
func (n *SqrtNormalizer) NormalizeValue(value float64) float64 {
	if value < 0 {
		return 0
	}
	return math.Sqrt(value)
}

// Binner 是特征分桶接口
type Binner interface {
	// Bin 将值分桶
	Bin(value float64) int
	// BinFeatures 将特征分桶
	BinFeatures(features map[string]float64) map[string]int
}

// EqualWidthBinner 等宽分桶
type EqualWidthBinner struct {
	Min     map[string]float64 // 特征最小值
	Max     map[string]float64 // 特征最大值
	NumBins map[string]int     // 每个特征的桶数
}

// NewEqualWidthBinner 创建等宽分桶器
func NewEqualWidthBinner(min, max map[string]float64, numBins map[string]int) *EqualWidthBinner {
	return &EqualWidthBinner{
		Min:     min,
		Max:     max,
		NumBins: numBins,
	}
}

// Bin 将值分桶
func (b *EqualWidthBinner) Bin(value float64) int {
	// 需要指定特征名，使用 BinWithKey
	return 0
}

// BinWithKey 将值分桶（指定特征名）
func (b *EqualWidthBinner) BinWithKey(key string, value float64) int {
	min := b.Min[key]
	max := b.Max[key]
	numBins := b.NumBins[key]
	if numBins <= 0 {
		return 0
	}

	if value < min {
		return 0
	}
	if value > max {
		return numBins - 1
	}

	binWidth := (max - min) / float64(numBins)
	bin := int((value - min) / binWidth)
	if bin >= numBins {
		bin = numBins - 1
	}
	return bin
}

// BinFeatures 将特征分桶
func (b *EqualWidthBinner) BinFeatures(features map[string]float64) map[string]int {
	binned := make(map[string]int)
	for k, v := range features {
		binned[k] = b.BinWithKey(k, v)
	}
	return binned
}

// CustomBinner 自定义分桶（指定分桶边界）
type CustomBinner struct {
	Bins map[string][]float64 // 每个特征的分桶边界（升序）
}

// NewCustomBinner 创建自定义分桶器
func NewCustomBinner(bins map[string][]float64) *CustomBinner {
	// 确保每个特征的边界是升序的
	for key, boundaries := range bins {
		sorted := make([]float64, len(boundaries))
		copy(sorted, boundaries)
		sort.Float64s(sorted)
		bins[key] = sorted
	}
	return &CustomBinner{
		Bins: bins,
	}
}

// Bin 将值分桶
func (b *CustomBinner) Bin(value float64) int {
	// 需要指定特征名，使用 BinWithKey
	return 0
}

// BinWithKey 将值分桶（指定特征名）
func (b *CustomBinner) BinWithKey(key string, value float64) int {
	boundaries, ok := b.Bins[key]
	if !ok || len(boundaries) == 0 {
		return 0
	}

	// 二分查找找到合适的桶
	for i := 0; i < len(boundaries)-1; i++ {
		if value >= boundaries[i] && value < boundaries[i+1] {
			return i
		}
	}

	// 值 >= 最后一个边界，返回最后一个桶
	if value >= boundaries[len(boundaries)-1] {
		return len(boundaries) - 1
	}

	// 值 < 第一个边界，返回第一个桶
	return 0
}

// BinFeatures 将特征分桶
func (b *CustomBinner) BinFeatures(features map[string]float64) map[string]int {
	binned := make(map[string]int)
	for k, v := range features {
		binned[k] = b.BinWithKey(k, v)
	}
	return binned
}

// CrossFeatureGenerator 交叉特征生成器
type CrossFeatureGenerator struct {
	// UserFeatures 用户特征列表
	UserFeatures []string
	// ItemFeatures 物品特征列表
	ItemFeatures []string
	// Operations 操作类型：multiply, divide, subtract, add
	Operations []string
}

// NewCrossFeatureGenerator 创建交叉特征生成器
func NewCrossFeatureGenerator(userFeatures, itemFeatures []string) *CrossFeatureGenerator {
	return &CrossFeatureGenerator{
		UserFeatures: userFeatures,
		ItemFeatures: itemFeatures,
		Operations:   []string{"multiply"}, // 默认只做乘积
	}
}

// WithOperations 设置操作类型
func (g *CrossFeatureGenerator) WithOperations(ops []string) *CrossFeatureGenerator {
	g.Operations = ops
	return g
}

// Generate 生成交叉特征
func (g *CrossFeatureGenerator) Generate(userFeatures, itemFeatures map[string]float64) map[string]float64 {
	crossFeatures := make(map[string]float64)

	for _, userKey := range g.UserFeatures {
		userVal, userOk := userFeatures[userKey]
		if !userOk {
			continue
		}

		for _, itemKey := range g.ItemFeatures {
			itemVal, itemOk := itemFeatures[itemKey]
			if !itemOk {
				continue
			}

			// 生成不同操作的交叉特征
			for _, op := range g.Operations {
				var value float64
				var featureName string

				switch op {
				case "multiply":
					value = userVal * itemVal
					featureName = fmt.Sprintf("%s_x_%s", userKey, itemKey)
				case "divide":
					if itemVal != 0 {
						value = userVal / itemVal
						featureName = fmt.Sprintf("%s_div_%s", userKey, itemKey)
					} else {
						continue
					}
				case "subtract":
					value = userVal - itemVal
					featureName = fmt.Sprintf("%s_sub_%s", userKey, itemKey)
				case "add":
					value = userVal + itemVal
					featureName = fmt.Sprintf("%s_add_%s", userKey, itemKey)
				default:
					continue
				}

				crossFeatures[featureName] = value
			}
		}
	}

	return crossFeatures
}

// MissingValueHandler 缺失值处理器
type MissingValueHandler struct {
	// DefaultValues 默认值映射
	DefaultValues map[string]float64
	// DefaultValue 全局默认值（当特征不在 DefaultValues 中时使用）
	DefaultValue float64
	// Strategy 处理策略：zero, mean, median, mode
	Strategy string
}

// NewMissingValueHandler 创建缺失值处理器
func NewMissingValueHandler(strategy string, defaultValue float64) *MissingValueHandler {
	return &MissingValueHandler{
		DefaultValues: make(map[string]float64),
		DefaultValue:  defaultValue,
		Strategy:      strategy,
	}
}

// WithDefaultValues 设置特征特定的默认值
func (h *MissingValueHandler) WithDefaultValues(defaults map[string]float64) *MissingValueHandler {
	h.DefaultValues = defaults
	return h
}

// Handle 处理缺失值
func (h *MissingValueHandler) Handle(features map[string]float64, requiredFeatures []string) map[string]float64 {
	handled := make(map[string]float64)

	// 复制已有特征
	for k, v := range features {
		handled[k] = v
	}

	// 处理缺失的特征
	for _, key := range requiredFeatures {
		if _, ok := handled[key]; !ok {
			// 使用特征特定的默认值
			if val, ok := h.DefaultValues[key]; ok {
				handled[key] = val
			} else {
				// 使用全局默认值
				handled[key] = h.DefaultValue
			}
		}
	}

	return handled
}

// FeatureSelector 特征选择器
type FeatureSelector struct {
	// SelectedFeatures 选中的特征列表
	SelectedFeatures []string
	// ExcludedFeatures 排除的特征列表
	ExcludedFeatures []string
}

// NewFeatureSelector 创建特征选择器
func NewFeatureSelector(selectedFeatures []string) *FeatureSelector {
	return &FeatureSelector{
		SelectedFeatures: selectedFeatures,
		ExcludedFeatures: []string{},
	}
}

// WithExcludedFeatures 设置排除的特征
func (s *FeatureSelector) WithExcludedFeatures(excluded []string) *FeatureSelector {
	s.ExcludedFeatures = excluded
	return s
}

// Select 选择特征
func (s *FeatureSelector) Select(features map[string]float64) map[string]float64 {
	selected := make(map[string]float64)

	// 如果指定了选中特征列表，只选择这些特征
	if len(s.SelectedFeatures) > 0 {
		for _, key := range s.SelectedFeatures {
			if val, ok := features[key]; ok {
				selected[key] = val
			}
		}
	} else {
		// 否则选择所有特征（除了排除的）
		excludedMap := make(map[string]bool)
		for _, key := range s.ExcludedFeatures {
			excludedMap[key] = true
		}

		for k, v := range features {
			if !excludedMap[k] {
				selected[k] = v
			}
		}
	}

	return selected
}

// FeatureStatistics 特征统计信息
type FeatureStatistics struct {
	Mean   float64
	Std    float64
	Min    float64
	Max    float64
	Median float64
	P25    float64
	P75    float64
	P95    float64
	P99    float64
}

// ComputeStatistics 计算特征统计信息
func ComputeStatistics(values []float64) *FeatureStatistics {
	if len(values) == 0 {
		return &FeatureStatistics{}
	}

	// 复制并排序
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	stats := &FeatureStatistics{
		Min: sorted[0],
		Max: sorted[len(sorted)-1],
	}

	// 计算均值
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	stats.Mean = sum / float64(len(values))

	// 计算标准差
	variance := 0.0
	for _, v := range values {
		variance += (v - stats.Mean) * (v - stats.Mean)
	}
	stats.Std = math.Sqrt(variance / float64(len(values)))

	// 计算分位数
	stats.Median = computePercentile(sorted, 0.5)
	stats.P25 = computePercentile(sorted, 0.25)
	stats.P75 = computePercentile(sorted, 0.75)
	stats.P95 = computePercentile(sorted, 0.95)
	stats.P99 = computePercentile(sorted, 0.99)

	return stats
}

// computePercentile 计算分位数
func computePercentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	index := p * float64(len(sorted)-1)
	lower := int(index)
	upper := lower + 1
	if upper >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}
