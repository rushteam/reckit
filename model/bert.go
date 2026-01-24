package model

import (
	"context"
	"fmt"
	"math"

	"github.com/rushteam/reckit/core"
)

// BERTModel 是 BERT 文本编码模型。
//
// 核心思想：
//   - 使用预训练的 BERT 模型将文本编码为稠密向量
//   - 通过外部 ML 服务（TorchServe、TensorFlow Serving）进行推理
//   - 支持批量编码以提高效率
//
// 使用场景：
//   - 文本特征向量化：物品标题、描述、标签 → 向量
//   - 语义理解：理解用户查询、物品描述的语义
//   - 文本相似度计算：基于语义相似度进行召回
//
// 工程特征：
//   - 实时性：中等（需要 RPC 调用，但支持批量）
//   - 计算复杂度：高（BERT 模型较大）
//   - 可解释性：中等（可以分析语义相似度）
//   - 语义理解：强（BERT 捕捉深层语义）
type BERTModel struct {
	// Service ML 服务接口（用于调用外部 BERT 服务）
	Service core.MLService

	// ModelName 模型名称（可选）
	ModelName string

	// ModelVersion 模型版本（可选）
	ModelVersion string

	// Dimension 向量维度（BERT 通常为 768 或 1024）
	Dimension int

	// MaxLength 最大序列长度（BERT 通常为 512）
	MaxLength int

	// PoolingStrategy 池化策略：cls（[CLS] token）、mean（平均池化）、max（最大池化）
	PoolingStrategy string
}

// NewBERTModel 创建一个新的 BERT 模型。
func NewBERTModel(service core.MLService, dimension int) *BERTModel {
	if dimension == 0 {
		dimension = 768 // BERT-base 默认维度
	}
	return &BERTModel{
		Service:         service,
		Dimension:       dimension,
		MaxLength:       512, // BERT 默认最大长度
		PoolingStrategy: "cls", // 默认使用 [CLS] token
	}
}

// WithModelName 设置模型名称。
func (m *BERTModel) WithModelName(name string) *BERTModel {
	m.ModelName = name
	return m
}

// WithModelVersion 设置模型版本。
func (m *BERTModel) WithModelVersion(version string) *BERTModel {
	m.ModelVersion = version
	return m
}

// WithMaxLength 设置最大序列长度。
func (m *BERTModel) WithMaxLength(maxLength int) *BERTModel {
	m.MaxLength = maxLength
	return m
}

// WithPoolingStrategy 设置池化策略：cls / mean / max。
func (m *BERTModel) WithPoolingStrategy(strategy string) *BERTModel {
	m.PoolingStrategy = strategy
	return m
}

// EncodeText 将单个文本编码为向量。
func (m *BERTModel) EncodeText(ctx context.Context, text string) ([]float64, error) {
	vectors, err := m.EncodeTexts(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("empty response")
	}
	return vectors[0], nil
}

// EncodeTexts 批量编码文本为向量。
// 通过批量调用提高效率。
func (m *BERTModel) EncodeTexts(ctx context.Context, texts []string) ([][]float64, error) {
	if m.Service == nil {
		return nil, fmt.Errorf("ML service is not set")
	}

	if len(texts) == 0 {
		return [][]float64{}, nil
	}

	// 构建请求
	// BERT 服务通常期望的格式：
	//   - 文本列表：["text1", "text2", ...]
	//   - 或者特征字典：[{"text": "text1"}, {"text": "text2"}, ...]
	req := &core.MLPredictRequest{
		ModelName:    m.ModelName,
		ModelVersion: m.ModelVersion,
		Params: map[string]interface{}{
			"texts":            texts,
			"max_length":       m.MaxLength,
			"pooling_strategy": m.PoolingStrategy,
		},
	}

	// 如果服务支持 Features 格式，使用 Features
	features := make([]map[string]float64, 0, len(texts))
	for range texts {
		// 将文本转换为特征字典（如果服务需要）
		// 这里假设服务可以直接处理文本列表
		features = append(features, map[string]float64{})
	}
	req.Features = features

	// 调用 ML 服务
	resp, err := m.Service.Predict(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("BERT encoding failed: %w", err)
	}

	// 解析响应
	// BERT 服务通常返回向量列表
	vectors, err := m.parseVectors(resp)
	if err != nil {
		return nil, fmt.Errorf("parse BERT response: %w", err)
	}

	if len(vectors) != len(texts) {
		return nil, fmt.Errorf("vector count mismatch: expected %d, got %d", len(texts), len(vectors))
	}

	return vectors, nil
}

// parseVectors 解析 ML 服务响应中的向量。
// 支持多种响应格式：
//   - Outputs 为向量数组：[[0.1, 0.2, ...], [0.3, 0.4, ...]]
//   - Outputs 为 map：{"embeddings": [[0.1, 0.2, ...], ...]}
//   - Predictions 为向量（如果服务直接返回向量）
func (m *BERTModel) parseVectors(resp *core.MLPredictResponse) ([][]float64, error) {
	vectors := make([][]float64, 0)

	// 尝试从 Outputs 解析
	if resp.Outputs != nil {
		switch v := resp.Outputs.(type) {
		case []interface{}:
			// 数组格式：[[0.1, 0.2, ...], [0.3, 0.4, ...]]
			for _, item := range v {
				if vec, ok := m.parseVector(item); ok {
					vectors = append(vectors, vec)
				}
			}
		case map[string]interface{}:
			// Map 格式：{"embeddings": [[0.1, 0.2, ...], ...]}
			if embeddings, ok := v["embeddings"]; ok {
				if arr, ok := embeddings.([]interface{}); ok {
					for _, item := range arr {
						if vec, ok := m.parseVector(item); ok {
							vectors = append(vectors, vec)
						}
					}
				}
			} else if vectorsRaw, ok := v["vectors"]; ok {
				// 或者 {"vectors": [[0.1, 0.2, ...], ...]}
				if arr, ok := vectorsRaw.([]interface{}); ok {
					for _, item := range arr {
						if vec, ok := m.parseVector(item); ok {
							vectors = append(vectors, vec)
						}
					}
				}
			}
		}
	}

	// 如果从 Outputs 解析失败，尝试从 Predictions 解析
	// 注意：Predictions 通常是分数，不是向量，但某些服务可能返回向量
	if len(vectors) == 0 && len(resp.Predictions) > 0 {
		// 如果只有一个预测值，可能是单个向量的第一个元素
		// 这种情况下，需要从 Outputs 获取完整向量
		return nil, fmt.Errorf("unable to parse vectors from response")
	}

	if len(vectors) == 0 {
		return nil, fmt.Errorf("no vectors found in response")
	}

	return vectors, nil
}

// parseVector 解析单个向量。
func (m *BERTModel) parseVector(v interface{}) ([]float64, bool) {
	switch val := v.(type) {
	case []float64:
		return val, true
	case []interface{}:
		vector := make([]float64, 0, len(val))
		for _, item := range val {
			switch fv := item.(type) {
			case float64:
				vector = append(vector, fv)
			case float32:
				vector = append(vector, float64(fv))
			case int:
				vector = append(vector, float64(fv))
			case int64:
				vector = append(vector, float64(fv))
			default:
				return nil, false
			}
		}
		return vector, true
	default:
		return nil, false
	}
}

// Similarity 计算两个向量的余弦相似度。
func (m *BERTModel) Similarity(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) {
		return 0.0
	}

	var dot, norm1, norm2 float64
	for i := 0; i < len(vec1); i++ {
		dot += vec1[i] * vec2[i]
		norm1 += vec1[i] * vec1[i]
		norm2 += vec2[i] * vec2[i]
	}

	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}

	return dot / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// Name 返回模型名称。
func (m *BERTModel) Name() string {
	return "bert"
}

// BERTServiceAdapter 是 BERT 服务的适配器接口。
// 用于将文本转换为 ML 服务可以理解的格式。
type BERTServiceAdapter interface {
	// Encode 将文本列表编码为特征格式
	Encode(ctx context.Context, texts []string) (*core.MLPredictRequest, error)
}

// DefaultBERTAdapter 是默认的 BERT 服务适配器。
// 假设服务接受文本列表作为输入。
type DefaultBERTAdapter struct {
	ModelName    string
	ModelVersion string
	MaxLength    int
}

// Encode 将文本列表编码为 ML 预测请求。
func (a *DefaultBERTAdapter) Encode(ctx context.Context, texts []string) (*core.MLPredictRequest, error) {
	// 构建特征字典列表
	features := make([]map[string]float64, 0, len(texts))
	for range texts {
		// 这里可以根据实际服务需求调整格式
		// 某些服务可能需要将文本转换为 token IDs
		// 文本通过 Params["texts"] 传递
		features = append(features, map[string]float64{})
	}

	return &core.MLPredictRequest{
		ModelName:    a.ModelName,
		ModelVersion: a.ModelVersion,
		Features:     features,
		Params: map[string]interface{}{
			"texts":      texts,
			"max_length": a.MaxLength,
		},
	}, nil
}
