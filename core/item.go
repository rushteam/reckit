package core

import "reckit/pkg/utils"

// Item 是推荐链路中的统一承载结构：特征、分数、元信息、标签。
// Labels 用于解释与策略驱动；Score 用于排序决策。
type Item struct {
	ID       int64
	Score    float64
	Features map[string]float64
	Meta     map[string]any
	Labels   map[string]utils.Label
}

func NewItem(id int64) *Item {
	return &Item{
		ID:       id,
		Score:    0,
		Features: make(map[string]float64),
		Meta:     make(map[string]any),
		Labels:   make(map[string]utils.Label),
	}
}

// PutLabel 写入 Label；若已存在同名 key，则按默认 Merge 规则累积。
func (it *Item) PutLabel(key string, lbl utils.Label) {
	if it.Labels == nil {
		it.Labels = make(map[string]utils.Label)
	}
	if old, ok := it.Labels[key]; ok {
		it.Labels[key] = utils.MergeLabel(old, lbl)
		return
	}
	it.Labels[key] = lbl
}
