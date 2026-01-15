package model

import "math"

// TwoTowerModel 是两塔模型（User Tower + Item Tower）。
//
// 核心思想：
//   - User Tower：学习用户表示（User Embedding）
//   - Item Tower：学习物品表示（Item Embedding）
//   - 相似度计算：User Embedding 和 Item Embedding 的内积/余弦相似度
//
// 工程特征：
//   - 实时性：好（可以离线计算 Item Embedding）
//   - 计算复杂度：低（向量内积）
//   - 可解释性：中等（可以分析用户/物品向量）
//   - 特征交互：强（塔内特征交互）
//
// 使用场景：
//   - 召回 + 排序（两阶段推荐）
//   - 大规模推荐系统
//   - 需要快速推理的场景
type TwoTowerModel struct {
	// UserTower 是用户塔（DNN）
	UserTower *DNNModel

	// ItemTower 是物品塔（DNN）
	ItemTower *DNNModel

	// UserTowerLayers 是用户塔的层结构
	UserTowerLayers []int

	// ItemTowerLayers 是物品塔的层结构
	ItemTowerLayers []int

	// EmbeddingDim 是最终嵌入维度
	EmbeddingDim int

	// SimilarityType 是相似度计算方式：dot（内积）或 cosine（余弦）
	SimilarityType string
}

// NewTwoTowerModel 创建一个新的两塔模型。
func NewTwoTowerModel(userTowerLayers, itemTowerLayers []int, embeddingDim int) *TwoTowerModel {
	if userTowerLayers == nil {
		userTowerLayers = []int{128, 64, 32}
	}
	if itemTowerLayers == nil {
		itemTowerLayers = []int{128, 64, 32}
	}
	if embeddingDim == 0 {
		embeddingDim = 32
	}

	// 确保最后一层是 embeddingDim
	if len(userTowerLayers) > 0 && userTowerLayers[len(userTowerLayers)-1] != embeddingDim {
		userTowerLayers[len(userTowerLayers)-1] = embeddingDim
	}
	if len(itemTowerLayers) > 0 && itemTowerLayers[len(itemTowerLayers)-1] != embeddingDim {
		itemTowerLayers[len(itemTowerLayers)-1] = embeddingDim
	}

	return &TwoTowerModel{
		UserTower:      NewDNNModel(userTowerLayers),
		ItemTower:      NewDNNModel(itemTowerLayers),
		UserTowerLayers: userTowerLayers,
		ItemTowerLayers: itemTowerLayers,
		EmbeddingDim:    embeddingDim,
		SimilarityType:  "dot", // 默认使用内积
	}
}

func (m *TwoTowerModel) Name() string {
	return "two_tower"
}

// Predict 使用两塔模型进行预测。
// features 需要包含用户特征和物品特征（通过前缀区分）
func (m *TwoTowerModel) Predict(features map[string]float64) (float64, error) {
	// 1. 提取用户特征（user_ 前缀）
	userFeatures := m.extractUserFeatures(features)

	// 2. 提取物品特征（item_ 前缀）
	itemFeatures := m.extractItemFeatures(features)

	// 3. User Tower：得到用户嵌入
	userEmb, err := m.getUserEmbedding(userFeatures)
	if err != nil {
		return 0, err
	}

	// 4. Item Tower：得到物品嵌入
	itemEmb, err := m.getItemEmbedding(itemFeatures)
	if err != nil {
		return 0, err
	}

	// 5. 计算相似度
	similarity := m.computeSimilarity(userEmb, itemEmb)

	// 6. Sigmoid 激活（转换为概率）
	return sigmoid(similarity), nil
}

// extractUserFeatures 提取用户特征。
func (m *TwoTowerModel) extractUserFeatures(features map[string]float64) map[string]float64 {
	userFeatures := make(map[string]float64)
	for k, v := range features {
		if hasPrefix(k, "user_") {
			userFeatures[k] = v
		}
	}
	return userFeatures
}

// extractItemFeatures 提取物品特征。
func (m *TwoTowerModel) extractItemFeatures(features map[string]float64) map[string]float64 {
	itemFeatures := make(map[string]float64)
	for k, v := range features {
		if hasPrefix(k, "item_") {
			itemFeatures[k] = v
		}
	}
	return itemFeatures
}

// getUserEmbedding 通过 User Tower 得到用户嵌入。
func (m *TwoTowerModel) getUserEmbedding(userFeatures map[string]float64) ([]float64, error) {
	// 使用 User Tower 的前向传播，但不经过最后一层的激活
	// 这里简化处理，直接使用 DNN 的输出
	score, err := m.UserTower.Predict(userFeatures)
	if err != nil {
		return nil, err
	}

	// 简化实现：将 score 转换为 embedding（实际应该从中间层获取）
	emb := make([]float64, m.EmbeddingDim)
	for i := range emb {
		emb[i] = score * 0.1 // 简单的映射
	}
	return emb, nil
}

// getItemEmbedding 通过 Item Tower 得到物品嵌入。
func (m *TwoTowerModel) getItemEmbedding(itemFeatures map[string]float64) ([]float64, error) {
	// 使用 Item Tower 的前向传播
	score, err := m.ItemTower.Predict(itemFeatures)
	if err != nil {
		return nil, err
	}

	// 简化实现：将 score 转换为 embedding（实际应该从中间层获取）
	emb := make([]float64, m.EmbeddingDim)
	for i := range emb {
		emb[i] = score * 0.1 // 简单的映射
	}
	return emb, nil
}

// computeSimilarity 计算用户嵌入和物品嵌入的相似度。
func (m *TwoTowerModel) computeSimilarity(userEmb, itemEmb []float64) float64 {
	if len(userEmb) != len(itemEmb) {
		return 0.0
	}

	switch m.SimilarityType {
	case "cosine":
		return cosineSimilarity(userEmb, itemEmb)
	case "dot":
		fallthrough
	default:
		return dotProduct(userEmb, itemEmb)
	}
}

// dotProduct 计算向量内积。
func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// cosineSimilarity 计算余弦相似度。
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}
	dot := dotProduct(a, b)
	normA := 0.0
	normB := 0.0
	for i := range a {
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0.0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// hasPrefix 检查字符串是否有指定前缀。
func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}
