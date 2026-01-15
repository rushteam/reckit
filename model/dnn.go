package model

import (
	"math"
)

// DNNModel 是深度神经网络模型（Deep Neural Network）。
//
// 工程特征：
//   - 实时性：好（本地推理）
//   - 计算复杂度：中等（多层全连接）
//   - 可解释性：弱（黑盒模型）
//   - 特征交互：强（自动学习特征交互）
//
// 使用场景：
//   - 大规模特征推荐场景
//   - 需要自动学习特征交互的场景
//   - 对可解释性要求不高的场景
type DNNModel struct {
	// Layers 是每层的神经元数量，例如 [128, 64, 32, 1] 表示 4 层网络
	Layers []int

	// Weights 是每层的权重矩阵
	// weights[layer][neuron][input] = weight
	Weights [][][]float64

	// Biases 是每层的偏置
	// biases[layer][neuron] = bias
	Biases [][]float64

	// FeatureEmbeddings 是特征嵌入（可选，用于稀疏特征）
	// key: feature_name, value: embedding_vector
	FeatureEmbeddings map[string][]float64

	// EmbeddingDim 是嵌入维度（如果使用嵌入）
	EmbeddingDim int
}

// NewDNNModel 创建一个新的 DNN 模型。
// layers: 每层的神经元数量，例如 [128, 64, 32, 1]
func NewDNNModel(layers []int) *DNNModel {
	if len(layers) < 2 {
		layers = []int{128, 64, 32, 1} // 默认结构
	}

	model := &DNNModel{
		Layers:            layers,
		Weights:           make([][][]float64, len(layers)),
		Biases:            make([][]float64, len(layers)),
		FeatureEmbeddings: make(map[string][]float64),
		EmbeddingDim:      16,
	}

	// 初始化权重和偏置（简化实现，实际应该从训练好的模型加载）
	// 这里使用随机初始化作为示例
	for i := 0; i < len(layers); i++ {
		prevSize := 0
		if i == 0 {
			prevSize = layers[0] // 输入层大小
		} else {
			prevSize = layers[i-1]
		}

		model.Weights[i] = make([][]float64, layers[i])
		model.Biases[i] = make([]float64, layers[i])

		for j := 0; j < layers[i]; j++ {
			model.Weights[i][j] = make([]float64, prevSize)
			// 简单初始化：Xavier 初始化
			for k := 0; k < prevSize; k++ {
				model.Weights[i][j][k] = (math.Sqrt(2.0 / float64(prevSize+layers[i]))) * 0.1
			}
			model.Biases[i][j] = 0.0
		}
	}

	return model
}

func (m *DNNModel) Name() string {
	return "dnn"
}

// Predict 使用 DNN 模型进行预测。
func (m *DNNModel) Predict(features map[string]float64) (float64, error) {
	if len(features) == 0 {
		return 0.0, nil
	}

	// 1. 特征嵌入（如果有）
	embeddedFeatures := m.embedFeatures(features)

	// 2. 前向传播
	output := m.forward(embeddedFeatures)

	// 3. Sigmoid 激活（输出概率）
	return sigmoid(output), nil
}

// embedFeatures 将特征嵌入到固定维度。
func (m *DNNModel) embedFeatures(features map[string]float64) []float64 {
	// 如果没有嵌入，直接使用特征值
	if len(m.FeatureEmbeddings) == 0 {
		// 将特征转换为固定大小的向量
		dim := m.Layers[0]
		embedded := make([]float64, dim)
		idx := 0
		for _, v := range features {
			if idx >= dim {
				break
			}
			embedded[idx] = v
			idx++
		}
		// 如果特征不足，用 0 填充
		return embedded
	}

	// 使用嵌入
	dim := m.EmbeddingDim
	embedded := make([]float64, dim)
	for name, value := range features {
		if emb, ok := m.FeatureEmbeddings[name]; ok {
			// 加权求和
			for i := 0; i < len(emb) && i < dim; i++ {
				embedded[i] += emb[i] * value
			}
		} else {
			// 如果没有嵌入，使用原始值
			hash := hashFeature(name) % dim
			embedded[hash] += value
		}
	}
	return embedded
}

// forward 前向传播。
func (m *DNNModel) forward(input []float64) float64 {
	if len(m.Layers) == 0 {
		return 0.0
	}

	// 确保输入维度匹配
	current := make([]float64, len(input))
	copy(current, input)
	if len(current) > m.Layers[0] {
		current = current[:m.Layers[0]]
	} else if len(current) < m.Layers[0] {
		// 填充 0
		padded := make([]float64, m.Layers[0])
		copy(padded, current)
		current = padded
	}

	// 逐层前向传播
	for layer := 0; layer < len(m.Layers); layer++ {
		next := make([]float64, m.Layers[layer])
		for j := 0; j < m.Layers[layer]; j++ {
			sum := m.Biases[layer][j]
			prevSize := len(current)
			if layer > 0 {
				prevSize = m.Layers[layer-1]
			}
			for k := 0; k < prevSize && k < len(current); k++ {
				if layer < len(m.Weights) && j < len(m.Weights[layer]) && k < len(m.Weights[layer][j]) {
					sum += m.Weights[layer][j][k] * current[k]
				}
			}
			// ReLU 激活（最后一层除外）
			if layer < len(m.Layers)-1 {
				next[j] = relu(sum)
			} else {
				next[j] = sum // 最后一层不激活
			}
		}
		current = next
	}

	if len(current) > 0 {
		return current[0]
	}
	return 0.0
}

// relu ReLU 激活函数。
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// sigmoid Sigmoid 激活函数。
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// hashFeature 简单的特征哈希（用于未嵌入的特征）。
func hashFeature(name string) int {
	hash := 0
	for _, c := range name {
		hash = hash*31 + int(c)
	}
	if hash < 0 {
		hash = -hash
	}
	return hash
}
