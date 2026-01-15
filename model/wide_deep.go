package model

// WideDeepModel 是 Wide&Deep 模型（大厂主流推荐模型）。
//
// 核心思想：
//   - Wide 部分：线性模型，记忆（memorization）用户-物品交互
//   - Deep 部分：DNN 模型，泛化（generalization）特征交互
//   - 联合训练：Wide + Deep，结合记忆和泛化
//
// 工程特征：
//   - 实时性：好（本地推理）
//   - 计算复杂度：中等（线性 + DNN）
//   - 可解释性：中等（Wide 部分可解释）
//   - 特征交互：强（Wide 显式交互 + Deep 隐式交互）
//
// 使用场景：
//   - 大规模推荐系统（Google Play 等）
//   - 需要记忆和泛化平衡的场景
//   - 对可解释性有一定要求的场景
type WideDeepModel struct {
	// Wide 部分：线性模型（类似 LR）
	WideWeights map[string]float64
	WideBias    float64

	// Deep 部分：DNN 模型
	Deep *DNNModel

	// WideFeatures 是 Wide 部分使用的特征（通常是交叉特征）
	// 例如：["user_id_x_item_id", "user_age_x_item_category"]
	WideFeatures []string

	// DeepFeatures 是 Deep 部分使用的特征（原始特征）
	// 例如：["user_age", "item_ctr", "item_cvr"]
	DeepFeatures []string
}

// NewWideDeepModel 创建一个新的 Wide&Deep 模型。
func NewWideDeepModel(wideFeatures, deepFeatures []string, deepLayers []int) *WideDeepModel {
	if deepLayers == nil {
		deepLayers = []int{128, 64, 32, 1}
	}

	return &WideDeepModel{
		WideWeights:  make(map[string]float64),
		WideBias:     0.0,
		Deep:          NewDNNModel(deepLayers),
		WideFeatures:  wideFeatures,
		DeepFeatures:  deepFeatures,
	}
}

func (m *WideDeepModel) Name() string {
	return "wide_deep"
}

// Predict 使用 Wide&Deep 模型进行预测。
func (m *WideDeepModel) Predict(features map[string]float64) (float64, error) {
	// 1. Wide 部分：线性模型
	wideScore := m.widePredict(features)

	// 2. Deep 部分：DNN 模型
	deepScore, err := m.deepPredict(features)
	if err != nil {
		return 0, err
	}

	// 3. 联合输出：Wide + Deep（加权求和）
	// 实际应用中可以使用学习到的权重，这里简化为平均
	combinedScore := 0.5*wideScore + 0.5*deepScore

	// 4. Sigmoid 激活
	return sigmoid(combinedScore), nil
}

// widePredict Wide 部分预测（线性模型）。
func (m *WideDeepModel) widePredict(features map[string]float64) float64 {
	score := m.WideBias

	// 使用 Wide 特征（通常是交叉特征）
	for _, featName := range m.WideFeatures {
		if value, ok := features[featName]; ok {
			if weight, ok := m.WideWeights[featName]; ok {
				score += weight * value
			}
		}
	}

	// 如果没有指定 WideFeatures，使用所有特征
	if len(m.WideFeatures) == 0 {
		for name, value := range features {
			// 只使用交叉特征（包含 "_x_" 的特征）
			if contains(name, "_x_") {
				if weight, ok := m.WideWeights[name]; ok {
					score += weight * value
				}
			}
		}
	}

	return score
}

// deepPredict Deep 部分预测（DNN 模型）。
func (m *WideDeepModel) deepPredict(features map[string]float64) (float64, error) {
	// 使用 Deep 特征（原始特征）
	deepFeatures := make(map[string]float64)
	if len(m.DeepFeatures) > 0 {
		for _, featName := range m.DeepFeatures {
			if value, ok := features[featName]; ok {
				deepFeatures[featName] = value
			}
		}
	} else {
		// 如果没有指定 DeepFeatures，使用所有非交叉特征
		for name, value := range features {
			if !contains(name, "_x_") {
				deepFeatures[name] = value
			}
		}
	}

	return m.Deep.Predict(deepFeatures)
}

// contains 检查字符串是否包含子串。
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
