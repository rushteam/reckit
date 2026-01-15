package core

import "reckit/pkg/utils"

// Item 是推荐链路中的统一承载结构：特征、分数、元信息、标签。
// Labels 用于解释与策略驱动；Score 用于排序决策。
//
// ID 类型设计：
//   - 使用 string 类型（通用，支持所有 ID 格式）
type Item struct {
	ID       string // 使用 string 类型（通用，支持所有 ID 格式）
	Score    float64
	Features map[string]float64
	Meta     map[string]any
	Labels   map[string]utils.Label
	
	// LabelMergeStrategy 自定义 Label 合并策略（可选）
	// 如果为 nil，则使用默认策略
	LabelMergeStrategy utils.LabelMergeStrategy
}

// NewItem 创建一个新的 Item
func NewItem(id string) *Item {
	return &Item{
		ID:       id,
		Score:    0,
		Features: make(map[string]float64),
		Meta:     make(map[string]any),
		Labels:   make(map[string]utils.Label),
	}
}

// PutLabel 写入 Label；若已存在同名 key，则按合并策略合并。
// 如果 Item 设置了 LabelMergeStrategy，则使用自定义策略；否则使用默认策略。
func (it *Item) PutLabel(key string, lbl utils.Label) {
	if it.Labels == nil {
		it.Labels = make(map[string]utils.Label)
	}
	if old, ok := it.Labels[key]; ok {
		var merged utils.Label
		if it.LabelMergeStrategy != nil {
			merged = it.LabelMergeStrategy.Merge(old, lbl)
		} else {
			merged = utils.MergeLabel(old, lbl)
		}
		it.Labels[key] = merged
		return
	}
	it.Labels[key] = lbl
}
