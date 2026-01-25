package feature

import (
	"encoding/json"
	"fmt"
	"os"
)

// FeatureMetadata 特征元数据，对应 feature_meta.json
type FeatureMetadata struct {
	// FeatureColumns 特征列名列表（按顺序）
	FeatureColumns []string `json:"feature_columns"`
	// FeatureCount 特征数量
	FeatureCount int `json:"feature_count"`
	// LabelColumn 标签列名
	LabelColumn string `json:"label_column"`
	// ModelVersion 模型版本
	ModelVersion string `json:"model_version"`
	// Normalized 是否使用了特征标准化
	Normalized bool `json:"normalized"`
	// CreatedAt 创建时间
	CreatedAt string `json:"created_at"`
}

// FeatureScaler 特征标准化器，对应 feature_scaler.json
// 每个特征对应一个 ScalerParams，包含 mean 和 std
type FeatureScaler map[string]ScalerParams

// ScalerParams 标准化参数
type ScalerParams struct {
	// Mean 均值
	Mean float64 `json:"mean"`
	// Std 标准差
	Std float64 `json:"std"`
}

// LoadFeatureMetadata 从文件加载特征元数据
//
// 内部使用 FileMetadataLoader 实现。
//
// 用法：
//
//	meta, err := feature.LoadFeatureMetadata("python/model/feature_meta.json")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("模型版本: %s\n", meta.ModelVersion)
//	fmt.Printf("特征列: %v\n", meta.FeatureColumns)
//	fmt.Printf("是否标准化: %v\n", meta.Normalized)
//
// 推荐使用接口方式：
//
//	loader := feature.NewFileMetadataLoader()
//	meta, err := loader.Load(ctx, "python/model/feature_meta.json")
func LoadFeatureMetadata(path string) (*FeatureMetadata, error) {
	return LoadFeatureMetadataFromFile(path)
}

// LoadFeatureMetadataFromFile 从文件加载特征元数据（内部实现）
func LoadFeatureMetadataFromFile(path string) (*FeatureMetadata, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("读取特征元数据文件失败: %w", err)
	}

	var meta FeatureMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("解析特征元数据失败: %w", err)
	}

	return &meta, nil
}

// LoadFeatureScaler 从文件加载特征标准化器
//
// 内部使用 FileScalerLoader 实现。
//
// 用法：
//
//	scaler, err := feature.LoadFeatureScaler("python/model/feature_scaler.json")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	// 对特征进行标准化
//	normalized := scaler.Normalize(features)
//
// 推荐使用接口方式：
//
//	loader := feature.NewFileScalerLoader()
//	scaler, err := loader.Load(ctx, "python/model/feature_scaler.json")
func LoadFeatureScaler(path string) (FeatureScaler, error) {
	return LoadFeatureScalerFromFile(path)
}

// LoadFeatureScalerFromFile 从文件加载特征标准化器（内部实现）
func LoadFeatureScalerFromFile(path string) (FeatureScaler, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("读取特征标准化器文件失败: %w", err)
	}

	var scaler FeatureScaler
	if err := json.Unmarshal(data, &scaler); err != nil {
		return nil, fmt.Errorf("解析特征标准化器失败: %w", err)
	}

	return scaler, nil
}

// Normalize 使用标准化器对特征进行标准化（Z-score）
//
// 公式：normalized = (x - mean) / std
//
// 如果特征不在 scaler 中，则保持不变。
// 如果 std <= 0，则返回原值。
func (s FeatureScaler) Normalize(features map[string]float64) map[string]float64 {
	normalized := make(map[string]float64)
	for k, v := range features {
		normalized[k] = v
		if params, ok := s[k]; ok {
			if params.Std > 0 {
				normalized[k] = (v - params.Mean) / params.Std
			}
		}
	}
	return normalized
}

// NormalizeValue 对单个特征值进行标准化
func (s FeatureScaler) NormalizeValue(featureName string, value float64) float64 {
	if params, ok := s[featureName]; ok {
		if params.Std > 0 {
			return (value - params.Mean) / params.Std
		}
	}
	return value
}

// ValidateFeatures 根据特征元数据验证特征，填充缺失值
//
// 用法：
//
//	meta, _ := feature.LoadFeatureMetadata("python/model/feature_meta.json")
//	validated := meta.ValidateFeatures(features)
//	// 缺失的特征会被填充为 0.0
func (m *FeatureMetadata) ValidateFeatures(features map[string]float64) map[string]float64 {
	validated := make(map[string]float64)
	for _, col := range m.FeatureColumns {
		if v, ok := features[col]; ok {
			validated[col] = v
		} else {
			validated[col] = 0.0 // 缺失值填充为 0.0
		}
	}
	return validated
}

// GetMissingFeatures 返回缺失的特征列
func (m *FeatureMetadata) GetMissingFeatures(features map[string]float64) []string {
	var missing []string
	for _, col := range m.FeatureColumns {
		if _, ok := features[col]; !ok {
			missing = append(missing, col)
		}
	}
	return missing
}

// BuildFeatureVector 按 feature_columns 顺序构建特征向量
//
// 用法：
//
//	meta, _ := feature.LoadFeatureMetadata("python/model/feature_meta.json")
//	vector := meta.BuildFeatureVector(features)
//	// vector 是按 feature_columns 顺序的 []float64
func (m *FeatureMetadata) BuildFeatureVector(features map[string]float64) []float64 {
	vector := make([]float64, len(m.FeatureColumns))
	for i, col := range m.FeatureColumns {
		if v, ok := features[col]; ok {
			vector[i] = v
		} else {
			vector[i] = 0.0 // 缺失值填充为 0.0
		}
	}
	return vector
}

// ProcessFeatures 完整的特征处理流程：验证 + 标准化（如果配置了）
//
// 用法：
//
//	meta, _ := feature.LoadFeatureMetadata("python/model/feature_meta.json")
//	scaler, _ := feature.LoadFeatureScaler("python/model/feature_scaler.json")
//	processed := meta.ProcessFeatures(features, scaler)
func (m *FeatureMetadata) ProcessFeatures(features map[string]float64, scaler FeatureScaler) map[string]float64 {
	// 1. 验证特征（填充缺失值）
	validated := m.ValidateFeatures(features)

	// 2. 标准化（如果配置了）
	if m.Normalized && scaler != nil {
		return scaler.Normalize(validated)
	}

	return validated
}
