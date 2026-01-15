package rank

import (
	"context"
	"sort"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// SortStrategy 是排序策略接口，用于自定义物品排序逻辑。
type SortStrategy interface {
	// Sort 对物品列表进行排序（原地排序）
	Sort(items []*core.Item)
}

// ScoreDescSortStrategy 是按分数降序排序（默认策略）。
type ScoreDescSortStrategy struct{}

func (s *ScoreDescSortStrategy) Sort(items []*core.Item) {
	sort.SliceStable(items, func(i, j int) bool {
		if items[i] == nil {
			return false
		}
		if items[j] == nil {
			return true
		}
		return items[i].Score > items[j].Score
	})
}

// ScoreAscSortStrategy 是按分数升序排序。
type ScoreAscSortStrategy struct{}

func (s *ScoreAscSortStrategy) Sort(items []*core.Item) {
	sort.SliceStable(items, func(i, j int) bool {
		if items[i] == nil {
			return false
		}
		if items[j] == nil {
			return true
		}
		return items[i].Score < items[j].Score
	})
}

// MultiFieldSortStrategy 是多字段排序策略。
type MultiFieldSortStrategy struct {
	Fields []SortField
}

type SortField struct {
	Key       string // 特征名或 Label key
	IsLabel   bool   // true 表示从 Label 读取，false 表示从 Features 读取
	Ascending bool   // true 表示升序，false 表示降序
}

func (s *MultiFieldSortStrategy) Sort(items []*core.Item) {
	sort.SliceStable(items, func(i, j int) bool {
		if items[i] == nil {
			return false
		}
		if items[j] == nil {
			return true
		}
		
		for _, field := range s.Fields {
			var valI, valJ float64
			var okI, okJ bool
			
			if field.IsLabel {
				if _, ok := items[i].Labels[field.Key]; ok {
					// 简化：假设 Label Value 是数值字符串
					// 实际使用时需要更完善的解析
					valI, okI = 0, true // 占位
				}
				if _, ok := items[j].Labels[field.Key]; ok {
					valJ, okJ = 0, true // 占位
				}
			} else {
				valI, okI = items[i].Features[field.Key]
				valJ, okJ = items[j].Features[field.Key]
			}
			
			if !okI && !okJ {
				continue
			}
			if !okI {
				return false
			}
			if !okJ {
				return true
			}
			
			if field.Ascending {
				if valI < valJ {
					return true
				}
				if valI > valJ {
					return false
				}
			} else {
				if valI > valJ {
					return true
				}
				if valI < valJ {
					return false
				}
			}
		}
		return false
	})
}

// LRNode 是一个使用 RankModel 的排序 Node 示例（不限定模型类型，LR 只是默认实现之一）。
// - 写入 labels：rank_model
// - 更新 item.Score 并按分数降序排序
type LRNode struct {
	Model       model.RankModel
	SortStrategy SortStrategy // 必需，如果为 nil 则使用默认降序排序
}

func (n *LRNode) Name() string        { return "rank.model" }
func (n *LRNode) Kind() pipeline.Kind { return pipeline.KindRank }

func (n *LRNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Model == nil || len(items) == 0 {
		return items, nil
	}

	for _, it := range items {
		if it == nil {
			continue
		}
		score, err := n.Model.Predict(it.Features)
		if err != nil {
			return nil, err
		}
		it.Score = score
		it.PutLabel("rank_model", utils.Label{Value: n.Model.Name(), Source: "rank"})
	}

	// 使用排序策略
	if n.SortStrategy == nil {
		n.SortStrategy = &ScoreDescSortStrategy{} // 默认降序
	}
	n.SortStrategy.Sort(items)
	
	return items, nil
}
