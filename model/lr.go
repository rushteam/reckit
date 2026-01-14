package model

import (
	"encoding/json"
	"math"
	"os"
)

type LRModel struct {
	Bias    float64
	Weights map[string]float64
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
