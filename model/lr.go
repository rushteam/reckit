package model

import (
	"encoding/json"
	"math"
	"os"
)

// LRModel 实现了逻辑回归 (Logistic Regression) 模型。
// 它是推荐系统中点击率预估 (CTR) 最基础也最经典的算法。
//
// 预测原理：
// 1. 线性加权求和: z = Bias + sum(Weight_i * Feature_i)
// 2. Sigmoid 变换: P = 1 / (1 + exp(-z))
//
// 最终输出值 P 代表概率（如点击概率），范围在 (0, 1) 之间。
type LRModel struct {
	Bias    float64            // 偏置项 (Bias / Intercept)
	Weights map[string]float64 // 特征权重 (Weights / Coefficients)
}

func LoadLRModel(path string) (*LRModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Bias    float64            `json:"bias"`
		Weights map[string]float64 `json:"weights"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	return &LRModel{Bias: raw.Bias, Weights: raw.Weights}, nil
}

func (m *LRModel) Name() string { return "lr" }

func (m *LRModel) Predict(features map[string]float64) (float64, error) {
	score := m.Bias
	for k, v := range features {
		if w, ok := m.Weights[k]; ok {
			score += w * v
		}
	}
	return 1 / (1 + math.Exp(-score)), nil
}
