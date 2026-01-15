package model

import (
	"fmt"
	"math"
)

// DINModel 是 Deep Interest Network（深度兴趣网络）模型。
//
// 核心思想：
//   - 用户行为序列：利用用户历史行为序列（点击、购买等）
//   - 注意力机制：计算候选物品与历史行为的注意力权重
//   - 兴趣提取：根据注意力权重聚合历史行为，得到用户兴趣表示
type DINModel struct {
	// ItemEmbeddingDim 是物品嵌入维度
	ItemEmbeddingDim int

	// ItemEmbeddings 是物品嵌入表
	// key: item_id, value: embedding_vector
	ItemEmbeddings map[string][]float64

	// AttentionLayers 是注意力网络的层结构
	AttentionLayers []int

	// AttentionWeights 是注意力网络的权重
	AttentionWeights [][][]float64

	// AttentionBiases 是注意力网络的偏置
	AttentionBiases [][]float64

	// MLPLayers 是 MLP 网络的层结构（用于最终预测）
	MLPLayers []int

	// MLP 模型（用于最终预测）
	MLP *DNNModel
}

// NewDINModel 创建一个新的 DIN 模型。
func NewDINModel(itemEmbeddingDim int, attentionLayers, mlpLayers []int) *DINModel {
	if attentionLayers == nil {
		attentionLayers = []int{64, 32}
	}
	if mlpLayers == nil {
		mlpLayers = []int{128, 64, 32, 1}
	}

	model := &DINModel{
		ItemEmbeddingDim: itemEmbeddingDim,
		ItemEmbeddings:   make(map[string][]float64),
		AttentionLayers:  attentionLayers,
		MLPLayers:       mlpLayers,
		MLP:              NewDNNModel(mlpLayers),
	}

	// 初始化注意力网络权重（简化实现）
	model.AttentionWeights = make([][][]float64, len(attentionLayers))
	model.AttentionBiases = make([][]float64, len(attentionLayers))
	for i := 0; i < len(attentionLayers); i++ {
		prevSize := itemEmbeddingDim * 2 // 候选物品嵌入 + 历史物品嵌入
		if i > 0 {
			prevSize = attentionLayers[i-1]
		}
		model.AttentionWeights[i] = make([][]float64, attentionLayers[i])
		model.AttentionBiases[i] = make([]float64, attentionLayers[i])
		for j := 0; j < attentionLayers[i]; j++ {
			model.AttentionWeights[i][j] = make([]float64, prevSize)
			for k := 0; k < prevSize; k++ {
				model.AttentionWeights[i][j][k] = 0.01
			}
			model.AttentionBiases[i][j] = 0.0
		}
	}

	return model
}

func (m *DINModel) Name() string {
	return "din"
}

// Predict 使用 DIN 模型进行预测。
func (m *DINModel) Predict(features map[string]float64) (float64, error) {
	// 1. 获取候选物品 ID
	// 注意：这里由于 features 只能存 float64，复杂的 ID 传递需要特殊处理
	// 简单起见，假设 ID 是数值字符串
	var candidateItemID string
	if id, ok := features["candidate_item_id"]; ok {
		candidateItemID = fmt.Sprintf("%.0f", id)
	}

	// 2. 获取候选物品嵌入
	candidateEmb := m.getItemEmbedding(candidateItemID)

	// 3. 获取用户行为序列
	behaviorSeq := m.extractBehaviorSequence(features)

	if len(behaviorSeq) == 0 {
		return m.MLP.Predict(features)
	}

	// 4. 计算注意力权重
	attentionWeights := m.computeAttention(candidateEmb, behaviorSeq)

	// 5. 加权聚合历史行为
	userInterest := m.aggregateWithAttention(behaviorSeq, attentionWeights)

	// 6. 拼接特征：候选物品嵌入 + 用户兴趣表示
	combinedFeatures := make(map[string]float64)
	for k, v := range features {
		combinedFeatures[k] = v
	}
	// 添加用户兴趣特征
	for i, v := range userInterest {
		combinedFeatures[fmt.Sprintf("user_interest_%d", i)] = v
	}

	// 7. 使用 MLP 预测
	return m.MLP.Predict(combinedFeatures)
}

// getItemEmbedding 获取物品嵌入。
func (m *DINModel) getItemEmbedding(itemID string) []float64 {
	if itemID != "" {
		if emb, ok := m.ItemEmbeddings[itemID]; ok {
			return emb
		}
	}
	// 如果没有嵌入，使用随机初始化
	return m.randomEmbedding(m.ItemEmbeddingDim)
}

// extractBehaviorSequence 从 features 中提取用户行为序列。
func (m *DINModel) extractBehaviorSequence(features map[string]float64) [][]float64 {
	seq := make([][]float64, 0)
	for i := 0; i < 10; i++ { // 最多 10 个历史行为
		key := fmt.Sprintf("behavior_item_%d", i)
		if id, ok := features[key]; ok && id > 0 {
			itemID := fmt.Sprintf("%.0f", id)
			emb := m.getItemEmbedding(itemID)
			seq = append(seq, emb)
		}
	}
	return seq
}

// computeAttention 计算注意力权重。
func (m *DINModel) computeAttention(candidateEmb []float64, behaviorSeq [][]float64) []float64 {
	if len(behaviorSeq) == 0 {
		return nil
	}

	attentionScores := make([]float64, len(behaviorSeq))

	for i, histEmb := range behaviorSeq {
		concat := append(candidateEmb, histEmb...)
		if len(concat) > len(candidateEmb)+len(histEmb) {
			concat = concat[:len(candidateEmb)+len(histEmb)]
		}

		score := m.forwardAttention(concat)
		attentionScores[i] = score
	}

	return softmax(attentionScores)
}

func (m *DINModel) forwardAttention(input []float64) float64 {
	current := make([]float64, len(input))
	copy(current, input)

	for layer := 0; layer < len(m.AttentionLayers); layer++ {
		next := make([]float64, m.AttentionLayers[layer])
		for j := 0; j < m.AttentionLayers[layer]; j++ {
			sum := m.AttentionBiases[layer][j]
			prevSize := len(current)
			if layer > 0 {
				prevSize = m.AttentionLayers[layer-1]
			}
			for k := 0; k < prevSize && k < len(current); k++ {
				if layer < len(m.AttentionWeights) && j < len(m.AttentionWeights[layer]) && k < len(m.AttentionWeights[layer][j]) {
					sum += m.AttentionWeights[layer][j][k] * current[k]
				}
			}
			next[j] = relu(sum)
		}
		current = next
	}

	if len(current) > 0 {
		return current[0]
	}
	return 0.0
}

func (m *DINModel) aggregateWithAttention(behaviorSeq [][]float64, attentionWeights []float64) []float64 {
	if len(behaviorSeq) == 0 {
		return make([]float64, m.ItemEmbeddingDim)
	}

	dim := len(behaviorSeq[0])
	aggregated := make([]float64, dim)

	for i, histEmb := range behaviorSeq {
		weight := 0.0
		if i < len(attentionWeights) {
			weight = attentionWeights[i]
		}
		for j := 0; j < dim && j < len(histEmb); j++ {
			aggregated[j] += weight * histEmb[j]
		}
	}

	return aggregated
}

func (m *DINModel) randomEmbedding(dim int) []float64 {
	emb := make([]float64, dim)
	for i := range emb {
		emb[i] = (float64(i%10) - 5) / 10.0
	}
	return emb
}

func softmax(scores []float64) []float64 {
	if len(scores) == 0 {
		return nil
	}

	maxScore := scores[0]
	for _, s := range scores {
		if s > maxScore {
			maxScore = s
		}
	}

	expScores := make([]float64, len(scores))
	sum := 0.0
	for i, s := range scores {
		expScores[i] = math.Exp(s - maxScore)
		sum += expScores[i]
	}

	if sum == 0 {
		for i := range expScores {
			expScores[i] = 1.0 / float64(len(expScores))
		}
		return expScores
	}

	for i := range expScores {
		expScores[i] /= sum
	}

	return expScores
}
