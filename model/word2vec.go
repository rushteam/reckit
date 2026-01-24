package model

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// Word2VecModel 是 Word2Vec 词向量模型。
//
// 核心思想：
//   - 将文本/序列中的词（或物品ID）映射为稠密向量
//   - 通过词向量的平均或加权平均得到文本/序列的向量表示
//   - 支持 OOV（Out-of-Vocabulary）处理
//
// 使用场景：
//   - 文本特征向量化：物品标题、描述、标签 → 向量
//   - 序列向量化：用户行为序列（点击的物品ID序列）→ 向量
//   - I2I 召回：基于物品相似度的召回
//
// 工程特征：
//   - 实时性：好（预加载词向量表，O(1) 查找）
//   - 计算复杂度：低（向量平均）
//   - 可解释性：中等（可以分析词向量相似度）
type Word2VecModel struct {
	// WordVectors 词向量表：word -> vector
	WordVectors map[string][]float64

	// Dimension 向量维度
	Dimension int

	// OOVVector OOV（Out-of-Vocabulary）词的默认向量
	// 如果为 nil，OOV 词使用零向量
	OOVVector []float64

	// AggregationMethod 聚合方法：mean（平均）或 sum（求和）
	AggregationMethod string
}

// NewWord2VecModel 创建一个新的 Word2Vec 模型。
func NewWord2VecModel(wordVectors map[string][]float64, dimension int) *Word2VecModel {
	if dimension <= 0 && len(wordVectors) > 0 {
		// 从第一个向量推断维度
		for _, vec := range wordVectors {
			dimension = len(vec)
			break
		}
	}

	return &Word2VecModel{
		WordVectors:       wordVectors,
		Dimension:         dimension,
		AggregationMethod: "mean",
	}
}

// WithOOVVector 设置 OOV 词的默认向量。
func (m *Word2VecModel) WithOOVVector(oovVector []float64) *Word2VecModel {
	m.OOVVector = oovVector
	return m
}

// WithAggregationMethod 设置聚合方法：mean（平均）或 sum（求和）。
func (m *Word2VecModel) WithAggregationMethod(method string) *Word2VecModel {
	m.AggregationMethod = method
	return m
}

// GetWordVector 获取单个词的向量。
// 如果词不存在，返回 OOVVector 或零向量。
func (m *Word2VecModel) GetWordVector(word string) []float64 {
	if vec, ok := m.WordVectors[word]; ok {
		return vec
	}
	if m.OOVVector != nil {
		return m.OOVVector
	}
	// 返回零向量
	return make([]float64, m.Dimension)
}

// EncodeText 将文本编码为向量（通过词向量的聚合）。
// 文本会被按空格分割为词列表。
func (m *Word2VecModel) EncodeText(text string) []float64 {
	words := strings.Fields(strings.ToLower(strings.TrimSpace(text)))
	return m.EncodeWords(words)
}

// EncodeWords 将词列表编码为向量（通过词向量的聚合）。
func (m *Word2VecModel) EncodeWords(words []string) []float64 {
	if len(words) == 0 {
		return make([]float64, m.Dimension)
	}

	// 聚合所有词的向量
	aggregated := make([]float64, m.Dimension)
	validCount := 0

	for _, word := range words {
		vec := m.GetWordVector(word)
		if len(vec) != m.Dimension {
			continue
		}
		validCount++
		for i := 0; i < m.Dimension; i++ {
			aggregated[i] += vec[i]
		}
	}

	if validCount == 0 {
		return make([]float64, m.Dimension)
	}

	// 根据聚合方法处理
	switch m.AggregationMethod {
	case "sum":
		return aggregated
	case "mean":
		fallthrough
	default:
		for i := 0; i < m.Dimension; i++ {
			aggregated[i] /= float64(validCount)
		}
		return aggregated
	}
}

// EncodeSequence 将序列（物品ID列表）编码为向量。
// 适用于用户行为序列（点击的物品ID序列）向量化。
func (m *Word2VecModel) EncodeSequence(sequence []string) []float64 {
	return m.EncodeWords(sequence)
}

// Similarity 计算两个向量的余弦相似度。
func (m *Word2VecModel) Similarity(vec1, vec2 []float64) float64 {
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
func (m *Word2VecModel) Name() string {
	return "word2vec"
}

// LoadWord2VecFromMap 从 map 加载词向量（用于从 JSON/YAML 等格式加载）。
func LoadWord2VecFromMap(data map[string]interface{}) (*Word2VecModel, error) {
	wordVectors := make(map[string][]float64)
	dimension := 0

	for word, vecInterface := range data {
		vec, ok := vecInterface.([]interface{})
		if !ok {
			continue
		}

		vector := make([]float64, 0, len(vec))
		for _, v := range vec {
			switch val := v.(type) {
			case float64:
				vector = append(vector, val)
			case int:
				vector = append(vector, float64(val))
			case int64:
				vector = append(vector, float64(val))
			}
		}

		if len(vector) > 0 {
			if dimension == 0 {
				dimension = len(vector)
			} else if len(vector) != dimension {
				return nil, fmt.Errorf("inconsistent vector dimension: word %s has dimension %d, expected %d", word, len(vector), dimension)
			}
			wordVectors[word] = vector
		}
	}

	if dimension == 0 {
		return nil, fmt.Errorf("no valid vectors found")
	}

	return NewWord2VecModel(wordVectors, dimension), nil
}

// Word2VecLoader 是 Word2Vec 模型加载器接口。
// 支持从不同来源加载模型（文件、HTTP、S3 等）。
type Word2VecLoader interface {
	// Load 加载 Word2Vec 模型
	Load(ctx context.Context, source string) (*Word2VecModel, error)
}
